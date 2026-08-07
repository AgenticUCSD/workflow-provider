"""Unit tests for the deterministic timezone value normalizer (utils/timezones.py)."""

from utils.slots import normalize_slot_values
from utils.task import ContextItem
from utils.timezones import normalize_timezone


# ── normalize_timezone: canonical passthrough ──────────────────

def test_canonical_passthrough():
    assert normalize_timezone("America/Los_Angeles") == "America/Los_Angeles"


# ── normalize_timezone: unambiguous aliases ─────────────────────

def test_alias_pacific_variants():
    for value in ("Pacific Time", "pacific time", "PST", "PDT", "PT"):
        assert normalize_timezone(value) == "America/Los_Angeles"


def test_alias_eastern():
    assert normalize_timezone("Eastern Time") == "America/New_York"


def test_alias_gmt_utc():
    assert normalize_timezone("GMT") == "UTC"
    assert normalize_timezone("UTC") == "UTC"


def test_alias_jst():
    assert normalize_timezone("JST") == "Asia/Tokyo"


# ── normalize_timezone: ambiguous abbreviations left untouched ──

def test_ambiguous_untouched():
    assert normalize_timezone("CST") == "CST"
    assert normalize_timezone("IST") == "IST"
    assert normalize_timezone("BST") == "BST"


# ── normalize_timezone: None/blank ──────────────────────────────

def test_none_and_blank():
    assert normalize_timezone(None) is None
    assert normalize_timezone("") == ""
    assert normalize_timezone("   ") == "   "


# ── normalize_timezone: unknown junk left untouched ─────────────

def test_unknown_junk_untouched():
    assert normalize_timezone("Narnia Standard Time") == "Narnia Standard Time"


# ── normalize_timezone: idempotence ─────────────────────────────

def test_idempotent():
    once = normalize_timezone("Pacific Time")
    twice = normalize_timezone(once)
    assert once == twice == "America/Los_Angeles"


# ── normalize_slot_values ────────────────────────────────────────

def test_normalize_slot_values_none_safe():
    assert normalize_slot_values(None) is None
    assert normalize_slot_values([]) == []


def test_normalize_slot_values_repairs_timezone_field():
    items = [ContextItem(field="timezone", status="present", value="Pacific Time")]
    normalize_slot_values(items)
    assert items[0].value == "America/Los_Angeles"


def test_normalize_slot_values_ignores_non_timezone_field():
    items = [ContextItem(field="notes", status="present", value="Pacific Time")]
    normalize_slot_values(items)
    assert items[0].value == "Pacific Time"


def test_normalize_slot_values_preserves_other_fields():
    items = [
        ContextItem(
            field="timezone",
            status="present",
            value="Pacific Time",
            source="email",
            confidence=0.9,
        )
    ]
    normalize_slot_values(items)
    ci = items[0]
    assert ci.value == "America/Los_Angeles"
    assert ci.status == "present"
    assert ci.source == "email"
    assert ci.confidence == 0.9
