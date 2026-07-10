"""Unit tests for deterministic slot typing (utils/slots.py)."""

from utils.slots import infer_slot_type, normalize_slots
from utils.task import ContextItem


# ── infer_slot_type: value shape wins ──────────────────────────

def test_infer_from_value_email():
    assert infer_slot_type("participants", "alice@example.com") == "email"


def test_infer_from_value_url():
    assert infer_slot_type("notes", "https://example.com/doc") == "url"


def test_infer_from_value_number():
    assert infer_slot_type("whatever", "42") == "number"
    assert infer_slot_type("whatever", "-3.5") == "number"


# ── infer_slot_type: field-name hints when no value ────────────

def test_infer_from_field_email():
    assert infer_slot_type("recipient", None) == "email"


def test_infer_from_field_date():
    assert infer_slot_type("deadline", None) == "date"
    assert infer_slot_type("time_window", "") == "date"


def test_infer_from_field_number():
    assert infer_slot_type("duration", None) == "number"


def test_infer_from_field_url():
    assert infer_slot_type("artifact_link", None) == "url"


def test_infer_defaults_to_string():
    assert infer_slot_type("topic", None) == "string"
    assert infer_slot_type("", None) == "string"


def test_value_shape_beats_field_hint():
    # field name hints "date", but a concrete numeric value wins.
    assert infer_slot_type("deadline", "42") == "number"


# ── normalize_slots: fill-only-unset, idempotent, None-safe ────

def test_normalize_fills_only_unset_type():
    items = [
        ContextItem(field="recipient", status="missing"),          # -> email (hint)
        ContextItem(field="topic", status="present", value="x"),   # -> string
        ContextItem(field="anything", status="present", type="ref"),  # explicit kept
    ]
    normalize_slots(items)
    by = {c.field: c for c in items}
    assert by["recipient"].type == "email"
    assert by["topic"].type == "string"
    assert by["anything"].type == "ref"  # explicit type never overwritten


def test_normalize_is_idempotent():
    items = [ContextItem(field="duration", status="missing")]
    normalize_slots(items)
    first = items[0].type
    normalize_slots(items)
    assert items[0].type == first == "number"


def test_normalize_none_safe():
    assert normalize_slots(None) is None
    assert normalize_slots([]) == []
