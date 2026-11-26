from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=2, max_length=1000, examples=["Who is eligible for PM-KISAN?"]
    )
    session_id: str | None = Field(None, max_length=64)
    scheme_ids: list[str] | None = Field(None, description="Optional filter, e.g. ['pm_kisan']")


class CitationOut(BaseModel):
    id: str
    scheme_id: str
    scheme_name: str
    section: str
    source_url: str
    snippet: str


class AskResponse(BaseModel):
    request_id: str
    answer: str
    language: str
    schemes: list[str]
    citations: list[CitationOut]
    refused: bool
    refusal_reason: str | None
    cache_hit: bool
    confidence: float
    prompt_version: int | None
    latency_ms: float
    cost_usd: float
    pii_masked: list[str]


class FeedbackRequest(BaseModel):
    request_id: str
    rating: Literal[1, -1]
    comment: str | None = Field(None, max_length=500)


class SchemeOut(BaseModel):
    id: str
    name: str
    ministry: str
    category: str
