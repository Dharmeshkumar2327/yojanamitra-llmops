"""Langfuse tracing wrapper.

Design rule: observability must NEVER break a user request. If Langfuse is not
configured or errors, every call becomes a no-op.

Trace layout per request:
    rag-request (span)
      ├── guardrails (span)
      ├── retrieval (span)       -> scheme filter, top chunks, best score
      ├── rerank (span)
      └── answer (generation)    -> model, prompt version, tokens, cost
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)


class _NoOp:
    def update(self, **_: Any) -> None: ...
    def update_trace(self, **_: Any) -> None: ...


@lru_cache
def get_langfuse():
    from langfuse import Langfuse

    from yojanamitra.config import get_settings

    s = get_settings()
    return Langfuse(
        public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, host=s.langfuse_host
    )


class Tracer:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._lf = None
        if enabled:
            try:
                self._lf = get_langfuse()
            except Exception as exc:  # noqa: BLE001
                log.warning("Langfuse disabled: %s", exc)
                self.enabled = False

    @contextmanager
    def span(self, name: str, **kwargs: Any):
        if not self.enabled:
            yield _NoOp()
            return
        try:
            cm = self._lf.start_as_current_span(name=name, **kwargs)
            obj = cm.__enter__()
        except Exception as exc:  # noqa: BLE001
            log.debug("span start failed: %s", exc)
            yield _NoOp()
            return
        try:
            yield obj
        finally:
            try:
                cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass

    @contextmanager
    def generation(self, name: str, **kwargs: Any):
        if not self.enabled:
            yield _NoOp()
            return
        try:
            cm = self._lf.start_as_current_generation(name=name, **kwargs)
            obj = cm.__enter__()
        except Exception as exc:  # noqa: BLE001
            log.debug("generation start failed: %s", exc)
            yield _NoOp()
            return
        try:
            yield obj
        finally:
            try:
                cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass

    def current_trace_id(self) -> str | None:
        if not self.enabled:
            return None
        try:
            return self._lf.get_current_trace_id()
        except Exception:  # noqa: BLE001
            return None

    def score(
        self, trace_id: str | None, name: str, value: float, comment: str | None = None
    ) -> None:
        if not (self.enabled and trace_id):
            return
        try:
            self._lf.create_score(trace_id=trace_id, name=name, value=value, comment=comment)
        except Exception as exc:  # noqa: BLE001
            log.debug("score failed: %s", exc)

    def flush(self) -> None:
        if self.enabled:
            try:
                self._lf.flush()
            except Exception:  # noqa: BLE001
                pass


@lru_cache
def get_tracer() -> Tracer:
    from yojanamitra.config import get_settings

    return Tracer(get_settings().langfuse_enabled)
