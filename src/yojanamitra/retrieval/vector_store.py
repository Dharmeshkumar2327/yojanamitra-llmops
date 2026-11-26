"""Qdrant vector store with hybrid (dense + BM25 sparse) search fused by RRF."""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient, models

from yojanamitra.retrieval.embeddings import Embedder
from yojanamitra.retrieval.types import RetrievedChunk

log = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "sparse"


class VectorStore:
    def __init__(
        self,
        url: str,
        api_key: str | None,
        collection: str,
        embedder: Embedder,
        mode: str = "hybrid",
    ):
        self.client = QdrantClient(url=url, api_key=api_key or None, timeout=30)
        self.collection = collection
        self.embedder = embedder
        self.mode = mode

    # ------------------------------------------------------------ admin
    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if exists:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field in ("scheme_id", "source_id", "section"):
            self.client.create_payload_index(
                self.collection, field, models.PayloadSchemaType.KEYWORD
            )
        log.info("created collection %s (dim=%d)", self.collection, self.embedder.dim)

    def healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001
            return False

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    # ------------------------------------------------------------ write
    def upsert_chunks(self, chunks: list[dict], batch_size: int = 64) -> int:
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            dense = self.embedder.embed_documents(texts)
            sparse = self.embedder.sparse_documents(texts)
            points = [
                models.PointStruct(
                    id=c["id"],
                    vector={
                        DENSE: d.tolist(),
                        SPARSE: models.SparseVector(indices=s.indices, values=s.values),
                    },
                    payload=c,
                )
                for c, d, s in zip(batch, dense, sparse, strict=True)
            ]
            self.client.upsert(self.collection, points=points, wait=True)
            total += len(points)
        return total

    def delete_source(self, source_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=_match("source_id", [source_id])),
            wait=True,
        )

    def source_hashes(self) -> dict[str, str]:
        """source_id -> doc_hash currently indexed (used by the weekly refresh job)."""
        hashes: dict[str, str] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=256,
                offset=offset,
                with_payload=["source_id", "doc_hash"],
                with_vectors=False,
            )
            for p in points:
                pl = p.payload or {}
                hashes[pl.get("source_id", "")] = pl.get("doc_hash", "")
            if offset is None:
                break
        return hashes

    # ------------------------------------------------------------ read
    def hybrid_search(
        self, query: str, k: int = 5, prefetch_k: int = 20, scheme_ids: list[str] | None = None
    ):
        """Returns (chunks fused by RRF, best dense cosine score for confidence)."""
        flt = _match("scheme_id", scheme_ids) if scheme_ids else None
        dense_q = self.embedder.embed_query(query).tolist()

        if self.mode == "dense":  # baseline: dense vectors only
            pts = self.client.query_points(
                self.collection,
                query=dense_q,
                using=DENSE,
                limit=k,
                query_filter=flt,
                with_payload=True,
            ).points
            return [RetrievedChunk.from_point(p) for p in pts], (
                float(pts[0].score) if pts else 0.0
            )

        sparse_q = self.embedder.sparse_query(query)

        fused = self.client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=dense_q, using=DENSE, limit=prefetch_k, filter=flt),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_q.indices, values=sparse_q.values),
                    using=SPARSE,
                    limit=prefetch_k,
                    filter=flt,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True,
        ).points

        top_dense = self.client.query_points(
            self.collection,
            query=dense_q,
            using=DENSE,
            limit=1,
            query_filter=flt,
            with_payload=False,
        ).points
        best_dense = float(top_dense[0].score) if top_dense else 0.0
        return [RetrievedChunk.from_point(p) for p in fused], best_dense


def _match(field: str, values: list[str]) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key=field, match=models.MatchAny(any=values))])


def get_vector_store() -> VectorStore:
    from yojanamitra.config import get_settings
    from yojanamitra.retrieval.embeddings import get_embedder

    s = get_settings()
    return VectorStore(
        s.qdrant_url, s.qdrant_api_key, s.qdrant_collection, get_embedder(), s.retrieval_mode
    )
