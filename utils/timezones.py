"""Deterministic, stdlib-only timezone value normalizer.

A conservative repair pass for loose timezone strings extracted by the LLM (e.g.
"Pacific Time", "PST") into canonical IANA identifiers (e.g. "America/Los_Angeles").
Only touches values that map unambiguously via a small, fixed alias table — it never
guesses, never blanks a value, and leaves anything it doesn't recognize (including
ambiguous abbreviations like "CST"/"IST"/"BST") exactly as extracted. No zoneinfo/
tzdata/pytz dependency; this is a pure string-normalization layer, not a real
timezone database.
"""

import re
from typing import Optional

# Mirrors the extension's canonical timezone list.
CANONICAL_TZS = frozenset(
    {
        "UTC",
        "America/Los_Angeles",
        "America/Denver",
        "America/Chicago",
        "America/New_York",
        "America/Sao_Paulo",
        "Europe/London",
        "Europe/Paris",
        "Europe/Berlin",
        "Africa/Cairo",
        "Asia/Dubai",
        "Asia/Kolkata",
        "Asia/Singapore",
        "Asia/Tokyo",
        "Australia/Sydney",
    }
)

# Unambiguous colloquial/abbreviation -> IANA. Keys are normalized (lowercased,
# trimmed, internal whitespace collapsed to single spaces). Deliberately excludes
# ambiguous bare abbreviations ("cst", "ist", "bst") that map to multiple real zones.
TZ_ALIASES = {
    "pacific": "America/Los_Angeles",
    "pacific time": "America/Los_Angeles",
    "pt": "America/Los_Angeles",
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "mountain": "America/Denver",
    "mountain time": "America/Denver",
    "mt": "America/Denver",
    "mst": "America/Denver",
    "mdt": "America/Denver",
    "eastern": "America/New_York",
    "eastern time": "America/New_York",
    "et": "America/New_York",
    "est": "America/New_York",
    "edt": "America/New_York",
    "central": "America/Chicago",
    "central time": "America/Chicago",
    "cdt": "America/Chicago",
    "gmt": "UTC",
    "utc": "UTC",
    "coordinated universal time": "UTC",
    "jst": "Asia/Tokyo",
    "tokyo": "Asia/Tokyo",
    "japan": "Asia/Tokyo",
    "india standard time": "Asia/Kolkata",
    "kolkata": "Asia/Kolkata",
    "india": "Asia/Kolkata",
    "sgt": "Asia/Singapore",
    "singapore": "Asia/Singapore",
    "dubai": "Asia/Dubai",
    "gst": "Asia/Dubai",
    "aest": "Australia/Sydney",
    "aedt": "Australia/Sydney",
    "sydney": "Australia/Sydney",
    "cet": "Europe/Paris",
    "cest": "Europe/Paris",
    "paris": "Europe/Paris",
    "cairo": "Africa/Cairo",
    "brt": "America/Sao_Paulo",
    "sao paulo": "America/Sao_Paulo",
    "berlin": "Europe/Berlin",
    "london": "Europe/London",
}

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_timezone(value: Optional[str]) -> Optional[str]:
    """Repair a loose timezone string to canonical IANA form when confidently known.

    Idempotent and conservative: canonical IANA values pass through unchanged,
    unambiguous aliases are mapped, and anything else (including None, blank
    strings, and ambiguous or unrecognized values) is returned unchanged.
    """
    if value is None:
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped in CANONICAL_TZS:
        return stripped
    key = _WHITESPACE_RE.sub(" ", stripped.lower()).strip()
    if key in TZ_ALIASES:
        return TZ_ALIASES[key]
    return value
