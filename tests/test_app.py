import json

import httpx
from fastapi.testclient import TestClient

from clef.app import create_app


def fake_upstream(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/health":
        return httpx.Response(200, json={"status": "ok"})
    body = json.loads(request.content)
    assert body["questions"]["q0"]["type"] == "choice"
    assert body["questions"]["q0"]["criteria"] == {"billing": None, "technical": None}
    return httpx.Response(
        200,
        json={
            "answers": {
                "q0": {
                    "type": "choice",
                    "choice": "technical",
                    "probabilities": {"billing": 0.1, "technical": 0.9},
                    "confidence": 0.8,
                },
                "q1": {
                    "type": "choice",
                    "choice": "yes",
                    "probabilities": {"yes": 0.99, "no": 0.01},
                    "confidence": 0.98,
                },
            },
            "usage": {"input_tokens": 42, "output_tokens": 0},
        },
    )


client = TestClient(create_app(transport=httpx.MockTransport(fake_upstream), manage_server=False))

REQUEST = {
    "state": "Checkout has been failing for every customer.",
    "questions": [
        {"question": "Which team?", "answers": ["billing", " technical ", ""]},
        {"question": "Is a service down?", "answers": ["yes", "no"]},
    ],
}


def test_predict_ranks_questions_by_confidence_and_answers_by_probability():
    r = client.post("/api/predict", json=REQUEST)
    assert r.status_code == 200
    data = r.json()
    assert data["input_tokens"] == 42
    assert [x["question"] for x in data["results"]] == ["Is a service down?", "Which team?"]
    team = data["results"][1]
    assert team["prediction"] == "technical"
    assert [a["answer"] for a in team["answers"]] == ["technical", "billing"]


def test_rejects_fewer_than_two_answers():
    bad = {"state": "x", "questions": [{"question": "q", "answers": ["only", " "]}]}
    assert client.post("/api/predict", json=bad).status_code == 422


def test_rejects_duplicate_answers():
    bad = {"state": "x", "questions": [{"question": "q", "answers": ["a", "a "]}]}
    assert client.post("/api/predict", json=bad).status_code == 422


def test_upstream_down_returns_502():
    def down(request):
        raise httpx.ConnectError("refused")

    c = TestClient(create_app(transport=httpx.MockTransport(down), manage_server=False))
    r = c.post("/api/predict", json=REQUEST)
    assert r.status_code == 502
    assert c.get("/api/health").json()["upstream_ok"] is False


def test_serves_ui():
    r = client.get("/")
    assert r.status_code == 200
    assert "Clef" in r.text
    assert client.get("/static/app.js").status_code == 200
