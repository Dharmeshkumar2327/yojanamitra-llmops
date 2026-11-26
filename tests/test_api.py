"""API contract tests with a fake pipeline (skipped if fastapi is not installed)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic_settings")
pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient  # noqa: E402

from yojanamitra.api import main  # noqa: E402
from yojanamitra.monitoring.request_log import RequestLog  # noqa: E402


class FakePipeline:
    tracer = None
    store = SimpleNamespace(healthy=lambda: True)

    def __init__(self, log):
        self.log = log

    def answer(self, question, session_id=None, scheme_ids=None):
        rid = "req-1"
        self.log.write(
            {
                "request_id": rid,
                "question": question,
                "refused": False,
                "cache_hit": False,
                "latency_ms": 10.0,
                "cost_usd": 0.0,
            }
        )
        return SimpleNamespace(
            request_id=rid,
            answer="Rs 6,000 [S1]",
            language="English",
            schemes=["pm_kisan"],
            citations=[],
            refused=False,
            refusal_reason=None,
            cache_hit=False,
            confidence=0.8,
            prompt_version=3,
            latency_ms=10.0,
            cost_usd=0.0,
            pii_found=[],
        )


@pytest.fixture
def client(tmp_path):
    log = RequestLog(f"sqlite:///{tmp_path / 't.db'}")
    main.state.clear()
    main.state.update(pipeline=FakePipeline(log), log=log)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["qdrant"] is True


def test_ask_and_feedback(client):
    r = client.post("/ask", json={"question": "How much is PM-KISAN?"})
    assert r.status_code == 200 and r.json()["answer"].startswith("Rs 6,000")
    assert client.post("/feedback", json={"request_id": "req-1", "rating": -1}).status_code == 200
    assert client.post("/feedback", json={"request_id": "nope", "rating": 1}).status_code == 404
    assert client.get("/stats").json()["thumbs_down_rate"] == 1.0


def test_validation(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
