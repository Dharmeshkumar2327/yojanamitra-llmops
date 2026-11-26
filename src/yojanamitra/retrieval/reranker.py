"""Cross-encoder reranker (multilingual). Optional - disable on 512 MB hosts."""

from __future__ import annotations

import math
from functools import lru_cache

from yojanamitra.retrieval.types import RetrievedChunk


class Reranker:
    def __init__(self, model_name: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        scores = list(self.model.rerank(query, [c.text for c in chunks]))
        for c, s in zip(chunks, scores, strict=True):
            c.score = 1 / (1 + math.exp(-float(s)))  # logit -> 0..1
        return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]


@lru_cache
def get_reranker() -> Reranker | None:
    from yojanamitra.config import get_settings

    s = get_settings()
    return Reranker(s.rerank_model) if s.rerank_enabled else None
