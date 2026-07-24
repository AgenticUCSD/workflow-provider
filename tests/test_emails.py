"""Unit tests for the deterministic email value normalizer (utils/emails.py)."""

from utils.emails import normalize_email
from utils.slots import normalize_email_slots
from utils.task import ContextItem


# ── normalize_email: single address ──────────────────────────────

def test_display_name_and_bracket_address():
    assert normalize_email("Bob Smith <bob@x.com>") == "bob@x.com"


def test_uppercase_address_lowercased():
    assert normalize_email("BOB@X.COM") == "bob@x.com"


def test_bare_address_idempotent():
    once = normalize_email("bob@x.com")
    twice = normalize_email(once)
    assert once == twice == "bob@x.com"


def test_address_with_trailing_paren_name():
    assert normalize_email("bob@x.com (Bob)") == "bob@x.com"


# ── normalize_email: multiple addresses ──────────────────────────

def test_multiple_addresses_comma_joined():
    assert normalize_email("Alice <a@x.com>, Bob <b@y.com>") == "a@x.com, b@y.com"


def test_multiple_addresses_deduped():
    assert normalize_email("a@x.com, A@X.COM") == "a@x.com"


# ── normalize_email: unchanged (no address / blank) ──────────────

def test_name_only_unchanged():
    assert normalize_email("Bob Smith") == "Bob Smith"


def test_vague_reference_unchanged():
    assert normalize_email("the whole team") == "the whole team"


def test_none_unchanged():
    assert normalize_email(None) is None


def test_empty_string_unchanged():
    assert normalize_email("") == ""


def test_whitespace_only_unchanged():
    assert normalize_email("   ") == "   "


def test_malformed_address_unchanged():
    assert normalize_email("bob@") == "bob@"


# ── normalize_email_slots ─────────────────────────────────────────

def test_normalize_email_slots_none_safe():
    assert normalize_email_slots(None) is None
    assert normalize_email_slots([]) == []


def test_normalize_email_slots_repairs_cc_field():
    items = [ContextItem(field="cc", status="present", value="Bob <bob@x.com>")]
    normalize_email_slots(items)
    assert items[0].value == "bob@x.com"


def test_normalize_email_slots_repairs_participants_field():
    items = [ContextItem(field="participants", status="present", value="Alice <a@x.com>")]
    normalize_email_slots(items)
    assert items[0].value == "a@x.com"


def test_normalize_email_slots_ignores_non_email_field():
    items = [ContextItem(field="notes", status="present", value="Bob Smith", type="string")]
    normalize_email_slots(items)
    assert items[0].value == "Bob Smith"


def test_normalize_email_slots_targets_typed_email_regardless_of_field_name():
    items = [ContextItem(field="notes", status="present", value="Bob <bob@x.com>", type="email")]
    normalize_email_slots(items)
    assert items[0].value == "bob@x.com"


def test_normalize_email_slots_preserves_other_fields():
    items = [
        ContextItem(
            field="cc",
            status="present",
            value="Bob <bob@x.com>",
            source="email",
            confidence=0.9,
        )
    ]
    normalize_email_slots(items)
    ci = items[0]
    assert ci.value == "bob@x.com"
    assert ci.status == "present"
    assert ci.source == "email"
    assert ci.confidence == 0.9
