import logging
import os
import socket


def _lan_ip() -> str:
    """Address of the interface that routes outward, or loopback if there is none."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))  # UDP connect sends nothing
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


class _ShowRealAddress(logging.Filter):
    """Uvicorn logs the bind address; swap a wildcard for the address clients use."""

    def __init__(self, shown: str) -> None:
        super().__init__()
        self.shown = shown

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(
                self.shown if a == "0.0.0.0" else a for a in record.args
            )
        return True


def main() -> None:
    import uvicorn

    host = os.environ.get("CLEF_HOST", "0.0.0.0")
    if host == "0.0.0.0":
        logging.getLogger("uvicorn.error").addFilter(_ShowRealAddress(_lan_ip()))

    uvicorn.run(
        "clef.app:app",
        host=host,
        port=int(os.environ.get("CLEF_PORT", "8000")),
    )
