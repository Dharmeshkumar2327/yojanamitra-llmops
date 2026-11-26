"""Request log store (SQLite locally, Postgres in production) - feeds the monitoring dashboard.

Langfuse holds full traces; this table holds compact per-request metrics so the
dashboard works even without Langfuse. Questions are stored AFTER PII masking.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
    select,
    update,
)

log = logging.getLogger(__name__)
metadata = MetaData()

requests_table = Table(
    "requests",
    metadata,
    Column("request_id", String(64), primary_key=True),
    Column("ts", DateTime(timezone=True), index=True),
    Column("session_id", String(64)),
    Column("trace_id", String(64), nullable=True),
    Column("question", Text),
    Column("answer", Text),
    Column("language", String(16)),
    Column("schemes", String(256)),
    Column("prompt_version", Integer),
    Column("model", String(64)),
    Column("latency_ms", Float),
    Column("retrieval_ms", Float),
    Column("llm_ms", Float),
    Column("input_tokens", Integer),
    Column("output_tokens", Integer),
    Column("cost_usd", Float),
    Column("confidence", Float),
    Column("cache_hit", Boolean),
    Column("refused", Boolean),
    Column("refusal_reason", String(32)),
    Column("pii_found", String(128)),
    Column("feedback", Integer, nullable=True),  # 1 = thumbs up, -1 = thumbs down
    Column("feedback_comment", Text, nullable=True),
)


class RequestLog:
    def __init__(self, url: str):
        if url.startswith("sqlite:///"):
            from pathlib import Path

            Path(url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, pool_pre_ping=True)
        metadata.create_all(self.engine)

    def write(self, row: dict) -> None:
        row = {**row, "ts": row.get("ts") or datetime.now(timezone.utc)}
        try:
            with self.engine.begin() as conn:
                conn.execute(insert(requests_table).values(**row))
        except Exception as exc:  # noqa: BLE001 - logging must not break requests
            log.warning("request log write failed: %s", exc)

    def set_feedback(
        self, request_id: str, rating: int, comment: str | None
    ) -> tuple[bool, str | None]:
        """Returns (found, trace_id) so the API can also score the Langfuse trace."""
        with self.engine.begin() as conn:
            res = conn.execute(
                update(requests_table)
                .where(requests_table.c.request_id == request_id)
                .values(feedback=rating, feedback_comment=comment)
            )
            if res.rowcount == 0:
                return False, None
            trace_id = conn.execute(
                select(requests_table.c.trace_id).where(requests_table.c.request_id == request_id)
            ).scalar()
            return True, trace_id

    def recent(self, limit: int = 5000) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(requests_table).order_by(requests_table.c.ts.desc()).limit(limit)
            )
            return [dict(r._mapping) for r in rows]


@lru_cache
def get_request_log() -> RequestLog:
    from yojanamitra.config import get_settings

    return RequestLog(get_settings().database_url)
