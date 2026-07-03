"""Thin, optional client for the memory-unit ``/resolve`` endpoint.

Flag-gated on the ``MEMORY_URL`` env var: if it is unset, resolution is a no-op
and the pipeline behaves exactly as before. Uses only the standard library
(urllib) so it adds no dependency, and it never raises — any problem (feature
disabled, network error, non-200, bad payload) yields an empty result so the
planner falls back to its existing behavior (ask the human).
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


def memory_enabled() -> bool:
    """True when a memory-unit base URL is configured."""
    return bool(os.getenv("MEMORY_URL"))


def resolve_slots(
    fields: List[str],
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    scope: Optional[List[str]] = None,
    timeout: float = 5.0,
) -> List[Dict[str, Any]]:
    """Call memory-unit ``/resolve`` for the given slot names.

    Returns a list of ``{field, value, source, confidence, status}`` dicts. On
    any problem returns ``[]`` — this function never raises, so callers can treat
    it as best-effort enrichment.
    """
    base_url = os.getenv("MEMORY_URL")
    if not base_url or not fields:
        return []

    url = base_url.rstrip("/") + "/resolve"
    payload = json.dumps({"fields": fields, "scope": scope}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    # memory-unit's tenancy guard requires X-User-Id; X-Thread-Id is optional.
    if user_id:
        headers["X-User-Id"] = user_id
    if thread_id:
        headers["X-Thread-Id"] = thread_id

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        slots = data.get("slots", [])
        return slots if isinstance(slots, list) else []
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return []
