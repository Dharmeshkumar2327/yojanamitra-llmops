"""Plain data types shared by retrieval components (no heavy dependencies)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    id: str
    score: float
    text: str
    scheme_id: str
    scheme_name: str
    section: str
    source_url: str
    source_id: str

    @classmethod
    def from_point(cls, p) -> RetrievedChunk:
        pl = p.payload or {}
        return cls(
            id=str(p.id),
            score=float(p.score or 0.0),
            text=pl.get("raw_text", ""),
            scheme_id=pl.get("scheme_id", ""),
            scheme_name=pl.get("scheme_name", ""),
            section=pl.get("section", ""),
            source_url=pl.get("source_url", ""),
            source_id=pl.get("source_id", ""),
        )
