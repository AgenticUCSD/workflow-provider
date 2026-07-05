"""Safe, optional tracing helpers for retrieval spans.

Phase 5 (steps 1-2): make the pipeline's retrieval steps visible under the same
DeepEval / Confident AI thread as the LLM agent spans. The LLM agents are already
traced via LangChain's ``CallbackHandler`` (see ``agents/*.py``); this helper adds
spans around the *non-agent* retrieval calls (chroma queries, template search, the
outbound memory-unit hop) that the callback can't see.

Design contract — tracing must never break the request path:
- If deepeval is unavailable, ``traced`` is an identity decorator (no-op).
- Span emission is gated **at call time** on ``tracing_enabled()`` (a real
  ``CONFIDENT_API_KEY`` must be set), so offline/CI runs never emit or flush.
- We use a **custom span type**, never the built-in ``retriever`` type, which
  requires an ``embedder`` field and errors at flush when omitted.
- deepeval swallows trace-flush errors (logs, does not raise), so even when a span
  is emitted it never propagates an exception into the caller.
"""
from __future__ import annotations

import os
from functools import wraps
from typing import Callable

try:  # deepeval is already a provider dependency; keep tracing strictly optional
    from deepeval.tracing import observe as _observe  # type: ignore
    _HAS_DEEPEVAL = True
except Exception:  # pragma: no cover - import guard
    _observe = None
    _HAS_DEEPEVAL = False


def tracing_enabled() -> bool:
    """True only when deepeval is importable and a Confident key is set (and not
    explicitly disabled). Gating on key presence keeps offline/CI runs from
    emitting or flushing traces."""
    if not _HAS_DEEPEVAL:
        return False
    if not os.getenv("CONFIDENT_API_KEY"):
        return False
    return os.getenv("CONFIDENT_TRACING", "true").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


def traced(name: str) -> Callable:
    """Return a decorator emitting a custom-type span named ``name`` when tracing
    is enabled, and calling the plain function otherwise.

    The observed wrapper is built once at decoration time (no network/flush until
    the function is called); the enabled check is done per call so import ordering
    and env timing can't accidentally enable or disable it.
    """
    def decorator(fn: Callable) -> Callable:
        if not _HAS_DEEPEVAL:
            return fn
        try:
            observed = _observe(name=name, type="custom")(fn)
        except Exception:  # pragma: no cover - defensive: never fail on tracing
            return fn

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if tracing_enabled():
                return observed(*args, **kwargs)
            return fn(*args, **kwargs)

        return wrapper

    return decorator
