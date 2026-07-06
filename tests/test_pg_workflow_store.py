"""Parity tests for PGWorkflowStore against a real Postgres + pgvector.

Exercises the pg backend for the flat workflow store (planner.workflows). GUARDED:
skipped unless TEST_PLANNER_DATABASE_URL points at a throwaway Postgres with the
`planner` schema applied — so the offline suite stays green and these never touch prod.

Local run: see the header of tests/test_pg_template_store.py (same ephemeral-cluster
recipe; the schema-only SQL now includes planner.workflows).

Embeddings are stubbed (deterministic, no OpenAI call) so vector search is exercised
with predictable distances.
"""

import os

import pytest

TEST_DB_URL = os.getenv("TEST_PLANNER_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="set TEST_PLANNER_DATABASE_URL (throwaway pg with the planner schema) to run",
)

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from utils.task import Workflow  # noqa: E402


def _vec(*head):
    v = [0.0] * 1536
    for i, x in enumerate(head):
        v[i] = float(x)
    return v


# Deterministic stub embeddings keyed on a marker word. Against query "one" (→ [1,0]):
# "one"→dist 0, "two"→~0.2, "three"→dist 1.
_EMB = {"one": _vec(1.0, 0.0), "two": _vec(0.8, 0.6), "three": _vec(0.0, 1.0)}


def _fake_embed(text: str):
    for marker, vec in _EMB.items():
        if marker in text:
            return vec
    return _vec(1.0, 0.0)


class _FakeTask:
    """Duck-typed stand-in — query_from_all_workflows_as_objects only calls to_string()."""

    def __init__(self, text):
        self._text = text

    def to_string(self):
        return self._text


def _wf(workflow_id, name="wf", steps=("do",), description="d"):
    return Workflow(
        workflow_id=workflow_id, name=name, description=description, steps=list(steps)
    )


@pytest.fixture
def store():
    from utils.pg_workflow_store import PGWorkflowStore, _TABLE

    s = PGWorkflowStore(database_url=TEST_DB_URL)
    s._embed = _fake_embed  # deterministic, no OpenAI
    with s._connect() as conn:
        conn.execute(f"TRUNCATE {_TABLE}")
    return s


def test_dedup_within_collection(store):
    id1 = store.add_workflow(_wf("a", name="alpha"), is_generated=True)
    id2 = store.add_workflow(_wf("b", name="alpha"), is_generated=True)  # same content
    assert id1 == id2  # dedup returns the existing doc id
    assert len(store.get_all_workflows()) == 1


def test_manual_and_generated_are_separate_collections(store):
    id_manual = store.add_workflow(_wf("a", name="alpha"), is_generated=False)
    id_gen = store.add_workflow(_wf("b", name="alpha"), is_generated=True)  # same content, other side
    assert id_manual != id_gen  # per-collection dedup, not global → two distinct rows


def test_get_all_roundtrip_and_workflow_id_dedup(store):
    store.add_workflow(_wf("w1", name="one", steps=["find", "invite"]), is_generated=False)
    store.add_workflow(_wf("w2", name="two"), is_generated=True)
    # Same workflow_id in both collections (different content) → deduped to one in get_all.
    store.add_workflow(_wf("dup", name="alpha"), is_generated=False)
    store.add_workflow(_wf("dup", name="beta"), is_generated=True)

    allw = store.get_all_workflows()
    assert sorted(w.workflow_id for w in allw) == ["dup", "w1", "w2"]
    w1 = next(w for w in allw if w.workflow_id == "w1")
    assert w1.name == "one" and w1.steps == ["find", "invite"]  # fields round-trip


def test_search_orders_by_distance_and_merges(store):
    store.add_workflow(_wf("wone", name="one"), is_generated=False)
    store.add_workflow(_wf("wtwo", name="two"), is_generated=False)
    store.add_workflow(_wf("wthree", name="three"), is_generated=True)

    results = store.query_from_all_workflows_as_objects(_FakeTask("one"), top_k=5)
    ids = [w.workflow_id for w in results]
    assert set(ids) == {"wone", "wtwo", "wthree"}  # both collections merged
    assert ids[0] == "wone"  # nearest to query "one" ([1,0]) is distance 0


def test_search_dedups_across_collections(store):
    # Same workflow_id present in both collections → appears once (manual first).
    store.add_workflow(_wf("shared", name="one"), is_generated=False)
    store.add_workflow(_wf("shared", name="two"), is_generated=True)
    results = store.query_from_all_workflows_as_objects(_FakeTask("one"), top_k=5)
    assert [w.workflow_id for w in results].count("shared") == 1
