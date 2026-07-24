"""Unit tests for the deterministic duration value normalizer (utils/durations.py)."""

from utils.durations import normalize_duration
from utils.slots import normalize_duration_slots
from utils.task import ContextItem


# ── normalize_duration: minutes ──────────────────────────────────

def test_minutes_variants():
    for value in ("30 minutes", "30 min", "30 mins", "30m"):
        assert normalize_duration(value) == "30"


# ── normalize_duration: hours ────────────────────────────────────

def test_hours_variants():
    assert normalize_duration("1 hour") == "60"
    assert normalize_duration("1 hr") == "60"
    assert normalize_duration("2 hours") == "120"
    assert normalize_duration("1.5 hours") == "90"


# ── normalize_duration: combined hours + minutes ─────────────────

def test_combined_hours_and_minutes():
    assert normalize_duration("1 hour 30 minutes") == "90"
    assert normalize_duration("1 hr 30 min") == "90"
    assert normalize_duration("1h30") == "90"
    assert normalize_duration("1h30m") == "90"
    assert normalize_duration("1:30") == "90"


# ── normalize_duration: ISO-8601 ─────────────────────────────────

def test_iso_8601():
    assert normalize_duration("PT30M") == "30"
    assert normalize_duration("PT1H") == "60"
    assert normalize_duration("PT1H30M") == "90"


# ── normalize_duration: word forms ───────────────────────────────

def test_word_forms():
    assert normalize_duration("half an hour") == "30"
    assert normalize_duration("half hour") == "30"
    assert normalize_duration("an hour") == "60"
    assert normalize_duration("quarter hour") == "15"


# ── normalize_duration: unchanged (ambiguous/unknown/blank) ──────

def test_bare_number_unchanged():
    assert normalize_duration("45") == "45"


def test_range_unchanged():
    assert normalize_duration("30-45 minutes") == "30-45 minutes"


def test_days_unchanged():
    assert normalize_duration("2 days") == "2 days"


def test_vague_unchanged():
    assert normalize_duration("a while") == "a while"


def test_none_and_blank():
    assert normalize_duration(None) is None
    assert normalize_duration("") == ""
    assert normalize_duration("   ") == "   "


def test_unknown_junk_unchanged():
    assert normalize_duration("banana") == "banana"


# ── normalize_duration: idempotence ───────────────────────────────

def test_idempotent():
    once = normalize_duration("30 minutes")
    twice = normalize_duration(once)
    assert once == twice == "30"


# ── normalize_duration_slots ──────────────────────────────────────

def test_normalize_duration_slots_none_safe():
    assert normalize_duration_slots(None) is None
    assert normalize_duration_slots([]) == []


def test_normalize_duration_slots_repairs_duration_field():
    items = [ContextItem(field="duration", status="present", value="30 minutes")]
    normalize_duration_slots(items)
    assert items[0].value == "30"


def test_normalize_duration_slots_ignores_non_duration_field():
    items = [ContextItem(field="notes", status="present", value="30 minutes")]
    normalize_duration_slots(items)
    assert items[0].value == "30 minutes"


def test_normalize_duration_slots_preserves_other_fields():
    items = [
        ContextItem(
            field="duration",
            status="present",
            value="30 minutes",
            source="email",
            confidence=0.9,
        )
    ]
    normalize_duration_slots(items)
    ci = items[0]
    assert ci.value == "30"
    assert ci.status == "present"
    assert ci.source == "email"
    assert ci.confidence == 0.9
