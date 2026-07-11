"""Unit tests for the provider↔executor Artifact-envelope adapter.

Fully offline (no DB / no HTTP / no LLM). Two contracts matter:
1. ``to_envelope`` emits exactly the kwargs the executor's ``insert_artifact``
   accepts, with the documented field mapping — and never the provider-local /
   executor-computed fields.
2. ``content_hash`` is byte-identical to the executor's ``_content_hash``
   (``sha256(content)[:16]``) over the *same* content string, so provider-side
   dedup checks match executor-side identity.

The executor's hash formula is replicated here (not imported — it lives in a
separate package/repo) so the parity is asserted against the real definition,
not against our own implementation.
"""

import hashlib
import json
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from utils.artifact_envelope import (
    ARTIFACT_KIND_TEMPLATE,
    GLOBAL_USER_ID,
    canonical_content,
    content_hash,
    to_envelope,
)
from utils.template import SlotSpec, Step, WorkflowTemplate


# --- the executor's canonical hash, replicated from
# workflow_executor/services/status_store.py:_content_hash ---
def _executor_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# The exact keyword parameters of the executor's insert_artifact (status_store.py).
# to_envelope() must be a subset of these and nothing else.
_INSERT_ARTIFACT_KWARGS = {
    "artifact_id",
    "kind",
    "name",
    "content",
    "status",
    "trust_tier",
    "source_trace_ids",
    "parent_artifact_id",
    "user_id",
}


def _template(**overrides) -> WorkflowTemplate:
    base = dict(
        template_id="tmpl_abc123",
        name="Schedule a meeting",
        description="Find a time and send an invite",
        required_slots=[SlotSpec(name="recipient", type="email", required=True)],
        steps=[Step(kind="llm", text="Find a free slot"), Step(text="Invite {recipient}")],
    )
    base.update(overrides)
    return WorkflowTemplate(**base)


def test_to_envelope_field_mapping():
    t = _template(parent_id="tmpl_parent", status="candidate")
    env = to_envelope(t, user_id="user_42", source_trace_ids=["thr_1", "thr_2"])

    assert env["artifact_id"] == "tmpl_abc123"          # template_id -> artifact_id
    assert env["kind"] == ARTIFACT_KIND_TEMPLATE == "template"
    assert env["name"] == "Schedule a meeting"
    assert env["parent_artifact_id"] == "tmpl_parent"   # parent_id -> parent_artifact_id
    assert env["status"] == "candidate"
    assert env["trust_tier"] == "T1"                    # candidate -> T1
    assert env["user_id"] == "user_42"                  # scope -> user_id
    assert env["source_trace_ids"] == ["thr_1", "thr_2"]
    assert env["content"] == canonical_content(t)


def test_to_envelope_defaults():
    env = to_envelope(_template())
    assert env["user_id"] == GLOBAL_USER_ID == "*"       # unscoped default
    assert env["source_trace_ids"] == []                 # None -> empty list
    assert env["parent_artifact_id"] is None
    assert env["status"] == "draft"                      # model default
    assert env["trust_tier"] == "T0"                     # draft -> T0


def test_to_envelope_emits_only_insert_artifact_kwargs():
    env = to_envelope(_template())
    # Exactly the executor's insert_artifact signature — no extra keys...
    assert set(env.keys()) == _INSERT_ARTIFACT_KWARGS
    # ...and specifically none of the executor-computed / provider-local fields.
    for forbidden in ("version", "content_hash", "eval_score", "source", "created_at"):
        assert forbidden not in env


def test_content_hash_matches_executor_formula():
    t = _template()
    # Parity with the executor: our hash == sha256(the content we send)[:16].
    assert content_hash(t) == _executor_content_hash(canonical_content(t))


def test_content_hash_ignores_id_version_status():
    # Identity is semantic: two templates differing only in id/version/status
    # (the envelope/lifecycle fields) must hash identically — so a re-created or
    # promoted template is recognized as the same content (no version churn).
    a = _template(template_id="tmpl_a", version=1, status="draft")
    b = _template(template_id="tmpl_b", version=7, status="candidate")
    assert content_hash(a) == content_hash(b)


def test_content_hash_changes_with_semantics():
    base = _template()
    assert content_hash(base) != content_hash(_template(name="Different name"))
    assert content_hash(base) != content_hash(
        _template(steps=[Step(text="Find a free slot")])  # dropped a step
    )
    # Step order is semantic — reordering is a different workflow.
    reordered = _template(
        steps=[Step(text="Invite {recipient}"), Step(kind="llm", text="Find a free slot")]
    )
    assert content_hash(base) != content_hash(reordered)


def test_canonical_content_is_deterministic_and_parseable():
    t = _template()
    # Byte-stable across calls (sort_keys + fixed separators).
    assert canonical_content(t) == canonical_content(t)
    # Valid, round-trippable JSON carrying the semantic fields.
    payload = json.loads(canonical_content(t))
    assert payload["name"] == "Schedule a meeting"
    assert payload["required_slots"] == [
        {"name": "recipient", "type": "email", "required": True}
    ]
    assert payload["steps"] == [
        {"kind": "llm", "text": "Find a free slot"},
        {"kind": "llm", "text": "Invite {recipient}"},  # ref omitted when unset
    ]


def test_ref_included_only_when_set():
    with_ref = _template(steps=[Step(kind="tool", text="do", ref="gmail.search")])
    payload = json.loads(canonical_content(with_ref))
    assert payload["steps"][0]["ref"] == "gmail.search"
    # And its presence changes identity vs. the same step without a ref.
    without_ref = _template(steps=[Step(kind="tool", text="do")])
    assert content_hash(with_ref) != content_hash(without_ref)


def test_envelope_kwargs_are_accepted_by_insert_artifact_signature():
    # Guard against drift: every key we emit must be a real insert_artifact param.
    # We can't import the executor here, so assert against the pinned kwarg set,
    # which mirrors status_store.insert_artifact (see _INSERT_ARTIFACT_KWARGS).
    env = to_envelope(_template(), source_trace_ids=["t"])
    assert set(env).issubset(_INSERT_ARTIFACT_KWARGS)
