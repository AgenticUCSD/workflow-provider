"""Thin, optional client for the memory-unit ``/resolve`` and ``/learn`` endpoints.

Flag-gated on the ``MEMORY_URL`` env var: if it is unset, resolution is a no-op
and the pipeline behaves exactly as before. Uses only the standard library
(urllib) so it adds no dependency, and it never raises — any problem (feature
disabled, network error, non-200, bad payload) yields an empty result so the
planner falls back to its existing behavior (ask the human).
"""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from utils.tracing import traced

_DEFAULT_TIMEOUT = 5.0
_MIN_TIMEOUT = 1.0
_MAX_TIMEOUT = 60.0


def memory_enabled() -> bool:
    """True when a memory-unit base URL is configured."""
    return bool(os.getenv("MEMORY_URL"))


def memory_timeout() -> float:
    """Seconds to wait on a memory-unit call before giving up.

    The default of 5s is shorter than a memory-unit **cold start**, measured at
    20-22s (it hydrates a per-user index on first touch and Cloud Run runs it at
    min-instances=0). So the first call after an idle period times out, and both
    callers honour their never-raises contract by returning empty/0 — silently.
    The user sees "memory forgot", and on the write-back side the approval
    teaches nothing at all.

    Raising this trades a slow first call for a correct one, which is the right
    way round: the extension's ``fetchJson`` sets no client-side abort, so the
    user waits rather than losing the result. Tune with
    ``MEMORY_TIMEOUT_SECONDS``. Read per call, so a Cloud Run env flip takes
    effect on the next request without a redeploy.

    Unparseable values fall back to the default; out-of-range values are
    **clamped** rather than rejected (same shape as memory-unit's
    ``resolve_min_coverage``). Clamping matters here: reverting an over-large
    value to the 5s default would silently give the operator *less* time than
    they asked for, which is the exact failure this function exists to fix.
    """
    raw = os.getenv("MEMORY_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_TIMEOUT
    if value != value:  # NaN — float("nan") parses, so screen it explicitly
        return _DEFAULT_TIMEOUT
    return max(_MIN_TIMEOUT, min(_MAX_TIMEOUT, value))


@traced(name="retrieval.memory.resolve")
def _post_resolve(
    url: str, payload: bytes, headers: Dict[str, str], timeout: float
) -> List[Dict[str, Any]]:
    """Do the actual ``/resolve`` POST, traced as a span.

    Keeps the never-raises contract self-contained: always returns a list, never
    raises, so the span always completes cleanly regardless of network outcome."""
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        slots = data.get("slots", [])
        return slots if isinstance(slots, list) else []
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        # Degrade silently to the caller, but not to the operator: an empty list
        # is indistinguishable from "memory knew nothing", which is how a timeout
        # against a cold memory-unit stayed invisible for so long.
        logging.getLogger(__name__).warning(
            "memory-unit /resolve failed after %.1fs; degrading to no slots: %s: %s",
            timeout, type(exc).__name__, exc,
        )
        return []


def resolve_slots(
    fields: List[str],
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    scope: Optional[List[str]] = None,
    authorization: Optional[str] = None,
    timeout: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Call memory-unit ``/resolve`` for the given slot names.

    Returns a list of ``{field, value, source, confidence, status}`` dicts. On
    any problem returns ``[]`` — this function never raises, so callers can treat
    it as best-effort enrichment.

    ``authorization`` is the caller's incoming ``Authorization`` header
    (``"Bearer <token>"``), forwarded verbatim. memory-unit verifies it on
    ``/resolve`` when ``MEMORY_VALIDATE_TOKEN`` is on, so without it that call
    401s; harmless when memory-unit runs with validation off.

    ``timeout`` defaults to ``memory_timeout()`` (``MEMORY_TIMEOUT_SECONDS``,
    else 5s) — resolved here rather than in the signature so the env var is read
    per call, not once at import.
    """
    base_url = os.getenv("MEMORY_URL")
    if not base_url or not fields:
        return []
    if timeout is None:
        timeout = memory_timeout()

    url = base_url.rstrip("/") + "/resolve"
    payload = json.dumps({"fields": fields, "scope": scope}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    # memory-unit's tenancy guard requires X-User-Id; X-Thread-Id is optional;
    # the bearer is required only when memory-unit validates tokens.
    if user_id:
        headers["X-User-Id"] = user_id
    if thread_id:
        headers["X-Thread-Id"] = thread_id
    if authorization:
        headers["Authorization"] = authorization

    return _post_resolve(url, payload, headers, timeout)


@traced(name="retrieval.memory.learn")
def _post_learn(
    url: str, payload: bytes, headers: Dict[str, str], timeout: float
) -> int:
    """Do the actual ``/learn`` POST, traced as a span.

    Mirrors ``_post_resolve``'s never-raises contract: always returns an int
    (0 on any failure), so the span always completes cleanly regardless of
    network outcome."""
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        learned = data.get("learned", 0)
        return learned if isinstance(learned, int) else 0
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        # Same reasoning as _post_resolve, and it matters more here: 0 is also
        # what a successful "nothing new to learn" returns, so without this line
        # a write-back that never happened looks exactly like one that had
        # nothing to do.
        logging.getLogger(__name__).warning(
            "memory-unit /learn failed after %.1fs; nothing was learned: %s: %s",
            timeout, type(exc).__name__, exc,
        )
        return 0


def learn_facts(
    items: List[Dict[str, Any]],
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    authorization: Optional[str] = None,
    timeout: Optional[float] = None,
) -> int:
    """Call memory-unit ``/learn`` with pre-built ``{text, category, task_id, scope}``
    items (see ``utils/writeback.py``, which owns the sentence format and the
    provenance filtering — this function is a thin, best-effort transport).

    Returns the number of facts memory-unit reports as newly learned. On any
    problem returns ``0`` — this function never raises, exactly like
    ``resolve_slots``, so callers can treat it as best-effort write-back.

    ``authorization`` is the caller's incoming ``Authorization`` header
    (``"Bearer <token>"``), forwarded verbatim. memory-unit's ``/learn`` requires
    both a bearer and ``X-User-Id`` unconditionally (it authenticates writes even
    when ``MEMORY_VALIDATE_TOKEN`` is off), so without them the call 401s;
    harmless — it still just returns 0.

    ``timeout`` defaults to ``memory_timeout()``, exactly as in
    ``resolve_slots``. Write-back is the more damaging side of a timeout: a
    dropped ``/resolve`` costs one pre-fill, a dropped ``/learn`` costs the fact
    permanently, because nothing retries it.
    """
    base_url = os.getenv("MEMORY_URL")
    if not base_url or not items:
        return 0
    if timeout is None:
        timeout = memory_timeout()

    url = base_url.rstrip("/") + "/learn"
    payload = json.dumps({"items": items}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if user_id:
        headers["X-User-Id"] = user_id
    if thread_id:
        headers["X-Thread-Id"] = thread_id
    if authorization:
        headers["Authorization"] = authorization

    return _post_learn(url, payload, headers, timeout)
