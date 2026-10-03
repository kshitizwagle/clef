"""Runs llama-server as a child process for the lifetime of the web app."""

import asyncio
import collections
import ctypes
import logging
import os
import shlex
import signal
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

log = logging.getLogger("clef.llama")

DEFAULT_BINARY = Path.home() / "src/llama.cpp/build/bin/llama-server"
DEFAULT_MODEL = "ggml-org/Clef-Flash-GGUF:Q8_0"
DEFAULT_LOG = Path.home() / "clef-server.log"


def _die_with_parent() -> None:
    # Linux PR_SET_PDEATHSIG: the child gets SIGTERM if this process dies,
    # even on SIGKILL, so llama-server is never left holding the GPU.
    PR_SET_PDEATHSIG = 1
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


class LlamaServer:
    def __init__(self, upstream_url: str) -> None:
        url = urlparse(upstream_url)
        self.host = url.hostname or "127.0.0.1"
        self.port = url.port or 8080
        self.binary = Path(os.environ.get("CLEF_LLAMA_SERVER", DEFAULT_BINARY)).expanduser()
        self.model = os.environ.get("CLEF_MODEL", DEFAULT_MODEL)
        self.extra_args = shlex.split(os.environ.get("CLEF_LLAMA_ARGS", ""))
        self.log_path = Path(os.environ.get("CLEF_SERVER_LOG", DEFAULT_LOG)).expanduser()
        # The first start downloads ~10 GB, so allow plenty of time.
        self.ready_timeout = float(os.environ.get("CLEF_READY_TIMEOUT", "1800"))
        self.proc: asyncio.subprocess.Process | None = None
        self._pump: asyncio.Task | None = None
        self._tail: collections.deque[bytes] = collections.deque(maxlen=40)

    @property
    def command(self) -> list[str]:
        return [
            str(self.binary),
            "-hf", self.model,
            "-ngl", "99",
            "--host", self.host,
            "--port", str(self.port),
            "-lv", "4",
            *self.extra_args,
        ]

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def start(self, client: httpx.AsyncClient) -> None:
        if await self._healthy(client):
            log.warning(
                "a model server is already running at %s:%s, using it instead of starting one",
                self.host, self.port,
            )
            return
        if not self.binary.is_file():
            raise RuntimeError(f"llama-server not found at {self.binary} (set CLEF_LLAMA_SERVER)")

        log.info("starting: %s", shlex.join(self.command))
        log.info("llama-server log: %s", self.log_path)
        self.proc = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            preexec_fn=_die_with_parent,
        )
        self._pump = asyncio.create_task(self._pump_output())

        deadline = time.monotonic() + self.ready_timeout
        while True:
            if not self.running:
                await self._pump
                raise RuntimeError(
                    f"llama-server exited with code {self.proc.returncode} before it was ready. "
                    f"Last output:\n{self._tail_text()}"
                )
            if await self._healthy(client):
                log.info("llama-server is ready (pid %s)", self.proc.pid)
                return
            if time.monotonic() > deadline:
                await self.stop()
                raise RuntimeError(f"llama-server was not ready after {self.ready_timeout:.0f}s")
            await asyncio.sleep(1)

    async def stop(self) -> None:
        if self.running:
            log.info("stopping llama-server (pid %s)", self.proc.pid)
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=15)
            except TimeoutError:
                log.warning("llama-server did not stop after SIGTERM, killing it")
                self.proc.kill()
                await self.proc.wait()
        if self._pump is not None:
            await self._pump

    async def _healthy(self, client: httpx.AsyncClient) -> bool:
        # llama-server answers 503 while the model is still loading.
        try:
            return (await client.get("/health", timeout=2)).status_code == 200
        except httpx.HTTPError:
            return False

    async def _pump_output(self) -> None:
        # Raw chunks rather than lines, so the \r download progress bar still
        # renders in the terminal.
        with self.log_path.open("wb") as f:
            while chunk := await self.proc.stdout.read(4096):
                sys.stderr.buffer.write(chunk)
                sys.stderr.buffer.flush()
                f.write(chunk)
                f.flush()
                self._tail.append(chunk)

    def _tail_text(self) -> str:
        text = b"".join(self._tail).decode(errors="replace").replace("\r", "\n")
        return "\n".join(text.splitlines()[-20:])
