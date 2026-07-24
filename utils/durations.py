"""Deterministic, stdlib-only duration value normalizer.

A conservative repair pass for loose duration strings extracted by the LLM (e.g.
"30 minutes", "1 hour 30 min", "PT1H30M") into canonical integer minutes as a
string (e.g. "90"). Only converts values that map unambiguously via a small,
fixed set of patterns — it never guesses, never blanks a value, and leaves
anything it doesn't confidently recognize (including ranges like "30-45
minutes", days/weeks, and vague phrases like "a while") exactly as extracted.
No dateutil/isodate dependency; this is a pure string-normalization layer, not
a real duration parser.
"""

import re
from typing import Optional

_WHITESPACE_RE = re.compile(r"\s+")

_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")

# Strict, whole-string ISO-8601 duration: PT[nH][nM], at least one component.
_ISO_RE = re.compile(r"^pt(?:(\d+)h)?(?:(\d+)m)?$", re.IGNORECASE)

# "1:30" -> 1h30m.
_COLON_RE = re.compile(r"^(\d+):(\d{1,2})$")

# Combined hours+minutes: "1 hour 30 minutes", "1 hr 30 min", "1h30", "1h30m".
_COMBINED_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|hr|h)\s*(?:and\s*)?"
    r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|min|m)?$",
    re.IGNORECASE,
)

# Ranges ("30-45 minutes", "30 to 45 min", "30–45 min") — deliberately checked
# before the single-unit hour/minute patterns below so a range isn't greedily
# parsed as its first number.
_RANGE_RE = re.compile(
    r"^\d+(?:\.\d+)?\s*(?:-|–|to)\s*\d+(?:\.\d+)?\s*"
    r"(?:hours?|hrs?|hr|h|minutes?|mins?|min|m)$",
    re.IGNORECASE,
)

_HOURS_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|hr|h)$", re.IGNORECASE)
_MINUTES_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|min|m)$", re.IGNORECASE)

# Explicit small map for word forms. Keys are normalized (lowercased, trimmed,
# internal whitespace collapsed to single spaces) and matched against the
# whole string.
_WORD_MINUTES = {
    "half an hour": "30",
    "half hour": "30",
    "half-hour": "30",
    "a half hour": "30",
    "an hour": "60",
    "one hour": "60",
    "quarter hour": "15",
    "a quarter hour": "15",
    "a quarter of an hour": "15",
    "a minute": "1",
}


def normalize_duration(value: Optional[str]) -> Optional[str]:
    """Repair a loose duration string to canonical integer minutes when confidently known.

    Idempotent and conservative: bare numbers pass through unchanged (assumed
    already minutes), unambiguous unit/ISO/word forms are converted, and
    anything else (None, blank strings, ranges, days/weeks, vague phrases, or
    unrecognized text) is returned unchanged. Never blanks a value.
    """
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return value
    norm = _WHITESPACE_RE.sub(" ", stripped).lower()

    if _NUMBER_RE.match(norm):
        return value

    iso_match = _ISO_RE.match(norm)
    if iso_match and (iso_match.group(1) or iso_match.group(2)):
        hours = int(iso_match.group(1) or 0)
        minutes = int(iso_match.group(2) or 0)
        return _finalize(hours * 60 + minutes, value)

    colon_match = _COLON_RE.match(norm)
    if colon_match:
        hours = int(colon_match.group(1))
        minutes = int(colon_match.group(2))
        return _finalize(hours * 60 + minutes, value)

    combined_match = _COMBINED_RE.match(norm)
    if combined_match:
        hours = float(combined_match.group(1))
        minutes = float(combined_match.group(2))
        return _finalize(round(hours * 60 + minutes), value)

    if _RANGE_RE.match(norm):
        return value

    hours_match = _HOURS_RE.match(norm)
    if hours_match:
        hours = float(hours_match.group(1))
        return _finalize(round(hours * 60), value)

    minutes_match = _MINUTES_RE.match(norm)
    if minutes_match:
        minutes = float(minutes_match.group(1))
        return _finalize(round(minutes), value)

    if norm in _WORD_MINUTES:
        return _WORD_MINUTES[norm]

    return value


def _finalize(total_minutes: float, original: str) -> str:
    """Return the computed minutes as a string, or the original value unchanged
    when the computation is nonsensical (<= 0) — never guess, never blank."""
    total_minutes = int(total_minutes)
    if total_minutes <= 0:
        return original
    return str(total_minutes)
