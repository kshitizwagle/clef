"""FastAPI front end for a llama.cpp server running Clef-Flash.

Each question has a list of candidate answers. They are sent to the
upstream /v1/systemone endpoint as `choice` questions, and the response is
returned with each question's answers ranked by probability and the questions
ranked by confidence.

/api/evaluate scores the model against a CSV of labelled examples, see
clef.evaluate for the format.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from clef.evaluate import CsvError, EvalRow, parse_csv, summarize
from clef.llama import LlamaServer

UPSTREAM_URL = os.environ.get("CLEF_UPSTREAM", "http://127.0.0.1:8080")
UPSTREAM_TIMEOUT = float(os.environ.get("CLEF_UPSTREAM_TIMEOUT", "120"))
STATIC_DIR = Path(__file__).parent / "static"
# Set CLEF_MANAGE_SERVER=0 to use a llama-server started some other way.
MANAGE_SERVER = os.environ.get("CLEF_MANAGE_SERVER", "1") != "0"
# Requests in flight during an evaluation; llama-server runs 4 slots by default.
EVAL_CONCURRENCY = int(os.environ.get("CLEF_EVAL_CONCURRENCY", "4"))
MAX_CSV_BYTES = 20 * 1024 * 1024

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

# Clef accepts at most 255 options per choice question.
MAX_ANSWERS = 255


class Question(BaseModel):
    question: str = Field(min_length=1)
    answers: list[str] = Field(min_length=2, max_length=MAX_ANSWERS)

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v

    @field_validator("answers")
    @classmethod
    def clean_answers(cls, v: list[str]) -> list[str]:
        cleaned = [a.strip() for a in v if a.strip()]
        if len(cleaned) < 2:
            raise ValueError("at least 2 non-blank answers are required")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("answers must be unique")
        return cleaned


class PredictRequest(BaseModel):
    state: str = Field(min_length=1)
    questions: list[Question] = Field(min_length=1)


class RankedAnswer(BaseModel):
    answer: str
    probability: float


class QuestionResult(BaseModel):
    question: str
    prediction: str
    confidence: float
    answers: list[RankedAnswer]


class PredictResponse(BaseModel):
    results: list[QuestionResult]
    input_tokens: int | None = None


def build_upstream_payload(req: PredictRequest) -> dict:
    return {
        "state": req.state,
        "questions": {
            f"q{i}": {
                "type": "choice",
                "instructions": q.question,
                "criteria": {a: None for a in q.answers},
            }
            for i, q in enumerate(req.questions)
        },
    }


def rank_results(req: PredictRequest, upstream: dict) -> PredictResponse:
    answers = upstream.get("answers", {})
    results = []
    for i, q in enumerate(req.questions):
        a = answers.get(f"q{i}")
        if a is None:
            raise HTTPException(502, f"upstream response is missing question {i}")
        probs = a.get("probabilities", {})
        ranked = sorted(
            (RankedAnswer(answer=k, probability=v) for k, v in probs.items()),
            key=lambda r: r.probability,
            reverse=True,
        )
        results.append(
            QuestionResult(
                question=q.question,
                prediction=a["choice"],
                confidence=a["confidence"],
                answers=ranked,
            )
        )
    results.sort(key=lambda r: r.confidence, reverse=True)
    return PredictResponse(
        results=results,
        input_tokens=upstream.get("usage", {}).get("input_tokens"),
    )


def create_app(
    transport: httpx.AsyncBaseTransport | None = None,
    manage_server: bool = MANAGE_SERVER,
) -> FastAPI:
    client = httpx.AsyncClient(
        base_url=UPSTREAM_URL, timeout=UPSTREAM_TIMEOUT, transport=transport
    )
    llama = LlamaServer(UPSTREAM_URL) if manage_server else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # The app does not accept requests until the model server is ready.
        if llama is not None:
            await llama.start(client)
        try:
            yield
        finally:
            if llama is not None:
                await llama.stop()
            await client.aclose()

    app = FastAPI(title="Clef", lifespan=lifespan)

    @app.get("/api/health")
    async def health() -> dict:
        try:
            r = await client.get("/health", timeout=3)
            upstream_ok = r.status_code == 200
        except httpx.HTTPError:
            upstream_ok = False
        return {
            "ok": True,
            "upstream": UPSTREAM_URL,
            "upstream_ok": upstream_ok,
            "managed": llama is not None and llama.running,
        }

    @app.post("/api/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest) -> PredictResponse:
        try:
            r = await client.post("/v1/systemone", json=build_upstream_payload(req))
        except httpx.HTTPError as e:
            raise HTTPException(502, f"cannot reach model server at {UPSTREAM_URL}: {e}")
        if r.status_code != 200:
            raise HTTPException(502, f"model server returned {r.status_code}: {r.text}")
        return rank_results(req, r.json())

    async def evaluate_row(index: int, row: EvalRow) -> dict:
        result = {
            "index": index,
            "row_number": row.row_number,
            "context": row.context,
            "question": row.question,
            "options": row.options,
            "answer": row.answer,
            "prediction": None,
            "confidence": None,
            "answer_probability": None,
            "correct": None,
            "error": None,
        }
        payload = {
            "state": row.context,
            "questions": {
                "q0": {
                    "type": "choice",
                    "instructions": row.question,
                    "criteria": {o: None for o in row.options},
                }
            },
        }
        try:
            r = await client.post("/v1/systemone", json=payload)
            if r.status_code != 200:
                result["error"] = f"model server returned {r.status_code}: {r.text[:300]}"
                return result
            a = r.json()["answers"]["q0"]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            result["error"] = f"request failed: {e}"
            return result
        result["prediction"] = a["choice"]
        result["confidence"] = a["confidence"]
        result["answer_probability"] = a.get("probabilities", {}).get(row.answer)
        result["correct"] = a["choice"] == row.answer
        return result

    @app.post("/api/evaluate")
    async def evaluate(request: Request) -> StreamingResponse:
        """Takes a CSV body and streams NDJSON: a `start` line, one `row` line
        per row as it finishes, then a `summary` line."""
        body = await request.body()
        if len(body) > MAX_CSV_BYTES:
            raise HTTPException(413, f"the file is larger than {MAX_CSV_BYTES // 2**20} MB")
        try:
            rows = parse_csv(body.decode("utf-8"))
        except UnicodeDecodeError:
            raise HTTPException(400, "the file is not UTF-8 text")
        except CsvError as e:
            raise HTTPException(400, str(e))

        async def stream():
            yield json.dumps({"type": "start", "total": len(rows)}) + "\n"
            limit = asyncio.Semaphore(EVAL_CONCURRENCY)

            async def run(i: int, row: EvalRow) -> dict:
                async with limit:
                    return await evaluate_row(i, row)

            tasks = [asyncio.create_task(run(i, row)) for i, row in enumerate(rows)]
            results = []
            try:
                for done in asyncio.as_completed(tasks):
                    result = await done
                    results.append(result)
                    yield json.dumps({"type": "row", **result}) + "\n"
            finally:
                # Stops the remaining requests if the browser goes away.
                for t in tasks:
                    t.cancel()
            yield json.dumps({"type": "summary", **summarize(results)}) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
