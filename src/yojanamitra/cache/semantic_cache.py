"""Semantic cache: if a new question is ~identical in meaning to a cached one
(cosine >= threshold) and asked under the same scheme filter + prompt version,
return the cached answer and skip retrieval + LLM entirely.

Backends: Redis (production, shared across workers) or in-memory (local/dev).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class CacheHit:
    answer: dict
    similarity: float


def _normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


class InMemoryBackend:
    def __init__(self, max_items: int):
        self.items: list[dict] = []
        self.max_items = max_items

    def all(self) -> list[dict]:
        return self.items

    def add(self, item: dict) -> None:
        self.items.append(item)
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items :]


class RedisBackend:
    KEY = "yojanamitra:semcache"

    def __init__(self, url: str, max_items: int):
        import redis

        self.r = redis.Redis.from_url(url, decode_responses=True)
        self.max_items = max_items

    def all(self) -> list[dict]:
        return [json.loads(x) for x in self.r.lrange(self.KEY, 0, -1)]

    def add(self, item: dict) -> None:
        pipe = self.r.pipeline()
        pipe.rpush(self.KEY, json.dumps(item, ensure_ascii=False))
        pipe.ltrim(self.KEY, -self.max_items, -1)
        pipe.execute()


class SemanticCache:
    def __init__(self, backend, threshold: float = 0.95, ttl_seconds: int = 86400):
        self.backend = backend
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(scheme_ids: list[str] | None, prompt_version: int) -> str:
        return f"{','.join(sorted(scheme_ids or []))}|v{prompt_version}"

    def lookup(
        self, embedding, scheme_ids: list[str] | None, prompt_version: int
    ) -> CacheHit | None:
        q = _normalize(embedding)
        key = self._key(scheme_ids, prompt_version)
        now = time.time()
        best, best_sim = None, -1.0
        for item in self.backend.all():
            if item["key"] != key or now - item["ts"] > self.ttl:
                continue
            sim = float(np.dot(q, _normalize(item["emb"])))
            if sim > best_sim:
                best, best_sim = item, sim
        if best is not None and best_sim >= self.threshold:
            self.hits += 1
            return CacheHit(best["answer"], best_sim)
        self.misses += 1
        return None

    def store(
        self, embedding, scheme_ids: list[str] | None, prompt_version: int, answer: dict
    ) -> None:
        self.backend.add(
            {
                "key": self._key(scheme_ids, prompt_version),
                "emb": _normalize(embedding).round(5).tolist(),
                "answer": answer,
                "ts": time.time(),
            }
        )

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def build_cache(redis_url: str, threshold: float, ttl: int, max_items: int = 2000) -> SemanticCache:
    if redis_url:
        try:
            backend = RedisBackend(redis_url, max_items)
            backend.r.ping()
            return SemanticCache(backend, threshold, ttl)
        except Exception:  # noqa: BLE001
            pass
    return SemanticCache(InMemoryBackend(max_items), threshold, ttl)
