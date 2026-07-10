"""Deterministic slot typing (PIPELINE_REWORK Phase 2).

Fills the `type` attribute of task slots (`ContextItem`) with a small, fixed
vocabulary using a pure heuristic — no LLM, no network — so the typed signature is
stable and testable. Kept intentionally tiny (plan guardrail: don't build a type
theory): ``string | email | date | number | url | ref``.
"""

import re
from typing import List, Optional

from utils.task import ContextItem

# The small closed vocabulary. `string` is the fallback.
SLOT_TYPES = ("string", "email", "date", "number", "url", "ref")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")

# Field-name keyword hints (checked as substrings of the lowercased field name).
_DATE_HINTS = ("date", "deadline", "time_window", "timezone", "when", "day", "schedule_by")
_NUMBER_HINTS = ("duration", "count", "number", "num_", "quantity", "amount", "minutes", "hours")
_EMAIL_HINTS = ("email", "recipient", "sender", "cc", "delegatee", "to_address")
_URL_HINTS = ("link", "url", "artifact_link", "meeting_link")


def infer_slot_type(field: str, value: Optional[str] = None) -> str:
    """Infer a slot's type from its value shape first, then its field-name hints.

    Value shape wins when present (it's concrete evidence); otherwise fall back to
    keyword hints in the field name; else ``string``.
    """
    v = (value or "").strip()
    if v:
        if _EMAIL_RE.match(v):
            return "email"
        if _URL_RE.match(v):
            return "url"
        if _NUMBER_RE.match(v):
            return "number"

    name = (field or "").strip().lower()
    if any(h in name for h in _EMAIL_HINTS):
        return "email"
    if any(h in name for h in _URL_HINTS):
        return "url"
    if any(h in name for h in _DATE_HINTS):
        return "date"
    if any(h in name for h in _NUMBER_HINTS):
        return "number"
    return "string"


def normalize_slots(items: Optional[List[ContextItem]]) -> Optional[List[ContextItem]]:
    """Fill `type` on each slot **only when unset** (idempotent; never overwrites an
    explicit type). Leaves `required` alone. None-safe — returns the input as given.
    """
    if not items:
        return items
    for ci in items:
        if ci.type is None:
            ci.type = infer_slot_type(ci.field, ci.value)
    return items
