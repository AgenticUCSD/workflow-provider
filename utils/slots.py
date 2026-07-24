"""Deterministic slot typing (PIPELINE_REWORK Phase 2).

Fills the `type` attribute of task slots (`ContextItem`) with a small, fixed
vocabulary using a pure heuristic — no LLM, no network — so the typed signature is
stable and testable. Kept intentionally tiny (plan guardrail: don't build a type
theory): ``string | email | date | number | url | ref``.
"""

import os
import re
from typing import List, Optional

from utils.task import ContextItem
from utils.timezones import normalize_timezone
from utils.durations import normalize_duration
from utils.emails import normalize_email

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


def tz_normalize_enabled() -> bool:
    """Whether timezone slot *value* normalization is on. Default off."""
    return os.getenv("IDENTIFY_TZ_NORMALIZE", "").strip().lower() in ("1", "true", "yes", "on")


def normalize_slot_values(items: Optional[List[ContextItem]]) -> Optional[List[ContextItem]]:
    """Repair loose timezone slot *values* to canonical IANA form. Opt-in via
    IDENTIFY_TZ_NORMALIZE. Only touches slots whose field name looks like a timezone;
    all other slots pass through untouched. None-safe.
    """
    if not items:
        return items
    for ci in items:
        name = (ci.field or "").strip().lower()
        if "timezone" in name or "time_zone" in name or "tz" in name:
            ci.value = normalize_timezone(ci.value)
    return items


def duration_normalize_enabled() -> bool:
    """Whether duration slot *value* normalization is on. Default off."""
    return os.getenv("IDENTIFY_DURATION_NORMALIZE", "").strip().lower() in ("1", "true", "yes", "on")


def normalize_duration_slots(items: Optional[List[ContextItem]]) -> Optional[List[ContextItem]]:
    """Repair loose duration slot *values* to integer minutes. Opt-in via
    IDENTIFY_DURATION_NORMALIZE. Only touches slots whose field name looks like a
    duration/length; all other slots pass through untouched. None-safe.
    """
    if not items:
        return items
    for ci in items:
        name = (ci.field or "").strip().lower()
        if "duration" in name or "length" in name:
            ci.value = normalize_duration(ci.value)
    return items


def email_normalize_enabled() -> bool:
    """Whether email slot *value* normalization is on. Default off."""
    return os.getenv("IDENTIFY_EMAIL_NORMALIZE", "").strip().lower() in ("1", "true", "yes", "on")


_EMAIL_NORMALIZE_HINTS = (
    "email", "recipient", "sender", "cc", "delegatee", "to_address", "participant", "attendee",
)


def normalize_email_slots(items: Optional[List[ContextItem]]) -> Optional[List[ContextItem]]:
    """Repair loose email slot *values* to bare lowercased address(es). Opt-in via
    IDENTIFY_EMAIL_NORMALIZE. Targets slots typed "email" or whose field name looks
    like an email field; a no-op on values with no address. None-safe.
    """
    if not items:
        return items
    for ci in items:
        name = (ci.field or "").strip().lower()
        if ci.type == "email" or any(h in name for h in _EMAIL_NORMALIZE_HINTS):
            ci.value = normalize_email(ci.value)
    return items
