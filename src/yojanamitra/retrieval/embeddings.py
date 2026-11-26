"""Dense + sparse embedders (fastembed, ONNX on CPU - no GPU / torch needed).

Dense : multilingual MiniLM (English + Hindi, 384-dim, ~220 MB) - fits free hosting.
        For best quality locally, set DENSE_MODEL=intfloat/multilingual-e5-large.
Sparse: Qdrant/bm25 - keyword matching for scheme names, amounts, form numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass
class SparseVec:
    indices: list[int]
    values: list[float]


class Embedder:
    def __init__(self, dense_model: str, sparse_model: str):
        from fastembed import SparseTextEmbedding, TextEmbedding

        self.dense = TextEmbedding(model_name=dense_model)
        self.sparse = SparseTextEmbedding(model_name=sparse_model)
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    # ---- dense
    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[np.ndarray]:
        return list(self.dense.embed(texts, batch_size=batch_size))

    def embed_query(self, text: str) -> np.ndarray:
        return next(iter(self.dense.query_embed(text)))

    # ---- sparse
    def sparse_documents(self, texts: list[str], batch_size: int = 32) -> list[SparseVec]:
        return [
            SparseVec(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self.sparse.embed(texts, batch_size=batch_size)
        ]

    def sparse_query(self, text: str) -> SparseVec:
        e = next(iter(self.sparse.query_embed(text)))
        return SparseVec(indices=e.indices.tolist(), values=e.values.tolist())


@lru_cache
def get_embedder() -> Embedder:
    from yojanamitra.config import get_settings

    s = get_settings()
    return Embedder(s.dense_model, s.sparse_model)
