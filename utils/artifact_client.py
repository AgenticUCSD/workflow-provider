"""Thin, optional client for the executor's ``POST /artifacts`` endpoint.

Flag-gated on the ``EXECUTOR_ARTIFACTS_URL`` env var: if it is unset, writing is a
no-op and the pipeline behaves exactly as before. Uses only the standard library
(urllib) so it adds no dependency, and it never raises — any problem (feature
disabled, network error, non-200, bad payload) yields ``None`` so the caller can
treat template promotion as best-effort.

This is the transport half of P-LEARN1 (convergence.md): the pure field mapping
lives in :mod:`utils.artifact_envelope` (``to_envelope``); this module just
serializes that envelope and POSTs it, forwarding the caller's Google bearer +
``X-User-Id`` (the executor's ``/artifacts`` requires both).
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, Optional

from utils.artifact_envelope import GLOBAL_USER_ID, to_envelope
from utils.template import WorkflowTemplate
from utils.tracing import traced


def artifacts_enabled() -> bool:
    """True when an executor artifacts base URL is configured."""
    return bool(os.getenv("EXECUTOR_ARTIFACTS_URL"))


@traced(name="artifact.executor.create")
def _post_artifact(
    url: str, payload: bytes, headers: Dict[str, str], timeout: float
) -> Optional[Dict[str, Any]]:
    """Do the actual ``POST /artifacts``, traced as a span.

    Keeps the never-raises contract self-contained: returns the created (or
    deduplicated) artifact dict on success, and ``None`` on any error, so the span
    always completes cleanly regardless of network outcome."""
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        return data if isinstance(data, dict) else None
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def post_template(
    template: WorkflowTemplate,
    *,
    user_id: str = GLOBAL_USER_ID,
    x_user_id: Optional[str] = None,
    source_trace_ids: Optional[Iterable[str]] = None,
    authorization: Optional[str] = None,
    thread_id: Optional[str] = None,
    timeout: float = 15.0,
) -> Optional[Dict[str, Any]]:
    """Write ``template`` to the executor's ``/artifacts`` as a ``draft`` artifact.

    Returns the created (or, on content-hash dedup, the existing) artifact dict, or
    ``None`` on any problem — this function never raises, so callers can treat it as
    best-effort promotion. A no-op returning ``None`` when ``EXECUTOR_ARTIFACTS_URL``
    is unset.

    ``timeout`` defaults to 15s (higher than the interactive memory-unit client's 5s):
    template promotion is a background, latency-insensitive write, and the executor
    can cold-start — a too-short timeout would spuriously return ``None`` on a write
    that actually lands server-side.

    Two distinct identities (do not conflate):
    - ``user_id`` — the artifact's *ownership scope* in the body (``'*'`` == global,
      the MVP default; a real id scopes it to a user). Maps via ``to_envelope``.
    - ``x_user_id`` — the *caller's* Google ``sub``, sent as the ``X-User-Id`` header.
      The executor's ``get_user`` requires it to match the bearer's ``sub`` (auth), and
      it is what a non-``'*'`` body ``user_id`` gets forced to. ``authorization`` is the
      caller's ``"Bearer <token>"``. The executor's ``/artifacts`` requires both headers.
    """
    base_url = os.getenv("EXECUTOR_ARTIFACTS_URL")
    if not base_url:
        return None

    # The executor assigns its own artifact_id (uuid) and dedups on
    # (name, kind, content_hash), so drop our template-local artifact_id — the body
    # is exactly the executor's CreateArtifactRequest fields.
    envelope = to_envelope(
        template, user_id=user_id, source_trace_ids=source_trace_ids
    )
    body = {k: v for k, v in envelope.items() if k != "artifact_id"}

    url = base_url.rstrip("/") + "/artifacts"
    payload = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    # X-User-Id authenticates the caller (must match the bearer's sub); the bearer
    # authorizes the write — the executor's get_user dependency requires both.
    if x_user_id:
        headers["X-User-Id"] = x_user_id
    if thread_id:
        headers["X-Thread-Id"] = thread_id
    if authorization:
        headers["Authorization"] = authorization

    return _post_artifact(url, payload, headers, timeout)
