"""The RAG pipeline - one place where every request flows through.

question
  -> PII masking -> language detection -> greeting / injection / off-topic guard
  -> scheme router (metadata filter)
  -> semantic cache lookup
  -> hybrid retrieval (dense + BM25, RRF) -> cross-encoder rerank
  -> confidence gate (low score => "not found" + helpline, no LLM call)
  -> grounded generation with versioned prompt + citations
  -> trace (Langfuse) + request log (DB) + cache store
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field

from yojanamitra.guardrails.pii import mask_pii
from yojanamitra.guardrails.topic import (
    GREETING_REPLY,
    NOT_FOUND_REPLY,
    OFF_TOPIC_REPLY,
    classify_with_llm,
    detect_language,
    heuristic_topic,
    is_greeting,
)

log = logging.getLogger(__name__)
_CITE = re.compile(r"\[S(\d+)\]")
_NOT_FOUND_MARKERS = ("could not find this in the official", "नहीं मिली")


@dataclass
class Citation:
    id: str
    scheme_id: str
    scheme_name: str
    section: str
    source_url: str
    snippet: str


@dataclass
class RAGResponse:
    request_id: str
    answer: str
    language: str
    schemes: list[str]
    question: str = ""  # PII-masked question
    session_id: str | None = None
    citations: list[Citation] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    retrieved_schemes: list[str] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    cache_hit: bool = False
    confidence: float = 0.0
    prompt_version: int | None = None
    model: str | None = None
    latency_ms: float = 0.0
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    pii_found: list[str] = field(default_factory=list)
    trace_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def format_context(chunks) -> str:
    return "\n\n".join(
        f"[S{i}] ({c.scheme_name} > {c.section})\n{c.text}" for i, c in enumerate(chunks, start=1)
    )


def extract_citations(answer: str, chunks) -> list[Citation]:
    used = sorted({int(n) for n in _CITE.findall(answer) if 1 <= int(n) <= len(chunks)})
    return [
        Citation(
            id=f"S{n}",
            scheme_id=chunks[n - 1].scheme_id,
            scheme_name=chunks[n - 1].scheme_name,
            section=chunks[n - 1].section,
            source_url=chunks[n - 1].source_url,
            snippet=chunks[n - 1].text[:240],
        )
        for n in used
    ]


class RAGPipeline:
    def __init__(
        self,
        *,
        settings,
        store,
        llm,
        router,
        reranker=None,
        cache=None,
        tracer=None,
        request_log=None,
    ):
        self.s = settings
        self.store = store
        self.llm = llm
        self.router = router
        self.reranker = reranker
        self.cache = cache
        self.tracer = tracer
        self.request_log = request_log

    # ------------------------------------------------------------------ public
    def answer(
        self,
        question: str,
        *,
        session_id: str | None = None,
        scheme_ids: list[str] | None = None,
        use_cache: bool = True,
        prompt_version: int | None = None,
        log_request: bool = True,
    ) -> RAGResponse:
        from yojanamitra.generation.prompts import load_prompt
        from yojanamitra.observability.tracing import Tracer

        tracer = self.tracer or Tracer(False)
        t0 = time.perf_counter()
        pii = mask_pii(question.strip())
        q = pii.text
        lang = detect_language(q)
        resp = RAGResponse(
            request_id=str(uuid.uuid4()),
            answer="",
            language=lang,
            schemes=[],
            question=q,
            session_id=session_id,
            pii_found=pii.found,
        )

        with tracer.span("rag-request", input={"question": q}) as root:
            root.update_trace(
                session_id=session_id, tags=[lang], metadata={"request_id": resp.request_id}
            )
            resp.trace_id = tracer.current_trace_id()

            # ---------------- guardrails
            with tracer.span("guardrails") as g:
                if is_greeting(q):
                    return self._finish(
                        resp,
                        GREETING_REPLY[lang],
                        t0,
                        refused=True,
                        reason="greeting",
                        log_request=log_request,
                    )
                routed = scheme_ids or self.router.route(q)
                resp.schemes = routed
                decision = heuristic_topic(q, routed) or classify_with_llm(
                    q, self.llm, self.s.llm_small_model
                )
                g.update(
                    output={
                        "pii": pii.found,
                        "routed": routed,
                        "topic": decision.reason,
                        "on_topic": decision.on_topic,
                    }
                )
                if not decision.on_topic:
                    return self._finish(
                        resp,
                        OFF_TOPIC_REPLY[lang],
                        t0,
                        refused=True,
                        reason=decision.reason,
                        log_request=log_request,
                    )

            prompt = load_prompt(prompt_version)
            resp.prompt_version = prompt.version

            # ---------------- semantic cache
            q_emb = None
            if self.cache is not None and use_cache:
                q_emb = self.store.embedder.embed_query(q)
                hit = self.cache.lookup(q_emb, routed, prompt.version)
                if hit:
                    cached = hit.answer
                    resp.cache_hit = True
                    resp.citations = [Citation(**c) for c in cached.get("citations", [])]
                    resp.confidence = cached.get("confidence", 0.0)
                    root.update(
                        output={"answer": cached["answer"], "cache_similarity": hit.similarity}
                    )
                    return self._finish(resp, cached["answer"], t0, log_request=log_request)

            # ---------------- retrieval
            tr = time.perf_counter()
            with tracer.span("retrieval", input={"query": q, "scheme_filter": routed}) as rs:
                k = self.s.top_k_prefetch if self.reranker else self.s.top_k_final
                chunks, best_dense = self.store.hybrid_search(
                    q, k=k, prefetch_k=self.s.top_k_prefetch, scheme_ids=routed or None
                )
                if routed and not chunks:  # filter too strict -> search everything
                    chunks, best_dense = self.store.hybrid_search(
                        q, k=k, prefetch_k=self.s.top_k_prefetch
                    )
                rs.update(
                    output={
                        "n": len(chunks),
                        "best_dense": best_dense,
                        "ids": [c.id for c in chunks[:10]],
                    }
                )
            if self.reranker and chunks:
                with tracer.span("rerank") as rr:
                    chunks = self.reranker.rerank(q, chunks, self.s.top_k_final)
                    rr.update(output={"scores": [round(c.score, 3) for c in chunks]})
            chunks = chunks[: self.s.top_k_final]
            resp.retrieval_ms = (time.perf_counter() - tr) * 1000
            resp.confidence = best_dense
            resp.contexts = [c.text for c in chunks]
            resp.retrieved_schemes = list(dict.fromkeys(c.scheme_id for c in chunks))

            # ---------------- confidence gate
            if not chunks or best_dense < self.s.min_dense_score:
                return self._finish(
                    resp,
                    NOT_FOUND_REPLY[lang],
                    t0,
                    refused=True,
                    reason="low_confidence",
                    log_request=log_request,
                )

            # ---------------- generation
            messages = prompt.compile(context=format_context(chunks), question=q, language=lang)
            gen_kwargs = {
                "model": self.s.llm_model,
                "input": messages,
                "metadata": {"prompt_version": prompt.version},
            }
            if prompt.langfuse_prompt is not None:
                gen_kwargs["prompt"] = prompt.langfuse_prompt
            with tracer.generation("answer", **gen_kwargs) as gen:
                result = self.llm.chat(
                    messages,
                    model=self.s.llm_model,
                    temperature=self.s.llm_temperature,
                    max_tokens=self.s.llm_max_tokens,
                )
                gen.update(
                    output=result.text,
                    usage_details={"input": result.input_tokens, "output": result.output_tokens},
                    cost_details={"total": result.cost_usd},
                )
            resp.model = result.model
            resp.llm_ms = result.latency_ms
            resp.input_tokens = result.input_tokens
            resp.output_tokens = result.output_tokens
            resp.cost_usd = result.cost_usd
            resp.citations = extract_citations(result.text, chunks)

            refused = any(m in result.text.lower() for m in _NOT_FOUND_MARKERS)
            if self.cache is not None and use_cache and not refused and q_emb is not None:
                self.cache.store(
                    q_emb,
                    routed,
                    prompt.version,
                    {
                        "answer": result.text,
                        "citations": [asdict(c) for c in resp.citations],
                        "confidence": best_dense,
                    },
                )
            root.update(output={"answer": result.text, "citations": [c.id for c in resp.citations]})
            return self._finish(
                resp,
                result.text,
                t0,
                refused=refused,
                reason="llm_not_found" if refused else None,
                log_request=log_request,
            )

    # ------------------------------------------------------------------ helpers
    def _finish(
        self,
        resp: RAGResponse,
        answer: str,
        t0: float,
        *,
        refused: bool = False,
        reason: str | None = None,
        log_request: bool = True,
    ) -> RAGResponse:
        resp.answer = answer
        resp.refused = refused
        resp.refusal_reason = reason
        resp.latency_ms = (time.perf_counter() - t0) * 1000
        if self.request_log is not None and log_request:
            self.request_log.write(
                {
                    "request_id": resp.request_id,
                    "session_id": resp.session_id,
                    "trace_id": resp.trace_id,
                    "question": resp.question,  # already PII-masked
                    "answer": answer,
                    "language": resp.language,
                    "schemes": ",".join(resp.schemes),
                    "prompt_version": resp.prompt_version,
                    "model": resp.model,
                    "latency_ms": resp.latency_ms,
                    "retrieval_ms": resp.retrieval_ms,
                    "llm_ms": resp.llm_ms,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_usd": resp.cost_usd,
                    "confidence": resp.confidence,
                    "cache_hit": resp.cache_hit,
                    "refused": refused,
                    "refusal_reason": reason,
                    "pii_found": ",".join(resp.pii_found),
                }
            )
        return resp
