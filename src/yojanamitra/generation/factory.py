"""Wires all components into a ready RAGPipeline (used by API, UI, evals)."""

from __future__ import annotations

from functools import lru_cache


@lru_cache
def build_pipeline(with_cache: bool = True, with_logging: bool = True):
    from yojanamitra.cache.semantic_cache import build_cache
    from yojanamitra.config import get_settings
    from yojanamitra.generation.llm import get_llm
    from yojanamitra.generation.rag import RAGPipeline
    from yojanamitra.ingestion.registry import load_schemes
    from yojanamitra.monitoring.request_log import get_request_log
    from yojanamitra.observability.tracing import get_tracer
    from yojanamitra.retrieval.reranker import get_reranker
    from yojanamitra.retrieval.router import SchemeRouter
    from yojanamitra.retrieval.vector_store import get_vector_store

    s = get_settings()
    return RAGPipeline(
        settings=s,
        store=get_vector_store(),
        llm=get_llm(),
        router=SchemeRouter(load_schemes()),
        reranker=get_reranker(),
        cache=build_cache(s.redis_url, s.cache_similarity, s.cache_ttl_seconds)
        if with_cache
        else None,
        tracer=get_tracer(),
        request_log=get_request_log() if with_logging else None,
    )
