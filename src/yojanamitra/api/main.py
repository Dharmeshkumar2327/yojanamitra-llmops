"""FastAPI service.

uvicorn yojanamitra.api.main:app --reload --port 8000
open http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from yojanamitra import __version__
from yojanamitra.api.ratelimit import RateLimiter
from yojanamitra.api.schemas import AskRequest, AskResponse, FeedbackRequest, SchemeOut

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from yojanamitra.config import get_settings
    from yojanamitra.generation.factory import build_pipeline
    from yojanamitra.ingestion.registry import load_schemes
    from yojanamitra.monitoring.request_log import get_request_log

    s = get_settings()
    state["pipeline"] = state.get("pipeline") or build_pipeline()
    state["schemes"] = load_schemes()
    state["log"] = state.get("log") or get_request_log()
    state["limiter"] = RateLimiter(s.rate_limit_per_min)
    log.info("YojanaMitra API ready (prompt source loads lazily, rerank=%s)", s.rerank_enabled)
    yield
    if state["pipeline"].tracer:
        state["pipeline"].tracer.flush()


app = FastAPI(
    title="YojanaMitra API",
    description="LLMOps RAG support bot for Indian Government schemes (English + Hindi)",
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return (
        fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")
    )


@app.get("/health")
def health():
    pipeline = state.get("pipeline")
    qdrant_ok = bool(pipeline and pipeline.store.healthy())
    return {
        "status": "ok" if qdrant_ok else "degraded",
        "qdrant": qdrant_ok,
        "version": __version__,
    }


@app.get("/schemes", response_model=list[SchemeOut])
def schemes():
    return [
        SchemeOut(id=s.id, name=s.name, ministry=s.ministry, category=s.category)
        for s in state["schemes"]
    ]


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request):
    if not state["limiter"].allow(_client_ip(request)):
        raise HTTPException(status_code=429, detail="Too many requests - please wait a minute.")
    try:
        r = await run_in_threadpool(
            state["pipeline"].answer,
            body.question,
            session_id=body.session_id,
            scheme_ids=body.scheme_ids,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("ask failed")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable") from exc
    return AskResponse(
        request_id=r.request_id,
        answer=r.answer,
        language=r.language,
        schemes=r.schemes,
        citations=[c.__dict__ for c in r.citations],
        refused=r.refused,
        refusal_reason=r.refusal_reason,
        cache_hit=r.cache_hit,
        confidence=round(r.confidence, 4),
        prompt_version=r.prompt_version,
        latency_ms=round(r.latency_ms, 1),
        cost_usd=round(r.cost_usd, 6),
        pii_masked=r.pii_found,
    )


@app.post("/feedback")
def feedback(body: FeedbackRequest):
    found, trace_id = state["log"].set_feedback(body.request_id, body.rating, body.comment)
    if not found:
        raise HTTPException(status_code=404, detail="request_id not found")
    pipeline = state["pipeline"]
    if pipeline.tracer:
        pipeline.tracer.score(trace_id, "user_feedback", body.rating, body.comment)
    return {"status": "recorded"}


@app.get("/stats")
def stats():
    rows = state["log"].recent(1000)
    if not rows:
        return {"requests": 0}
    n = len(rows)
    fb = [r["feedback"] for r in rows if r["feedback"] is not None]
    return {
        "requests": n,
        "cache_hit_rate": round(sum(bool(r["cache_hit"]) for r in rows) / n, 3),
        "refusal_rate": round(sum(bool(r["refused"]) for r in rows) / n, 3),
        "avg_latency_ms": round(sum(r["latency_ms"] or 0 for r in rows) / n, 1),
        "total_cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 5),
        "thumbs_down_rate": round(sum(1 for f in fb if f == -1) / len(fb), 3) if fb else None,
    }
