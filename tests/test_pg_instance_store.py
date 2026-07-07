"""Parity tests for PGInstanceStore against a real Postgres.

Exercises persistence of EnrichedInstances to planner.enriched_instances. GUARDED:
skipped unless TEST_PLANNER_DATABASE_URL points at a throwaway Postgres with the
`planner` schema applied — so the offline suite stays green and these never touch prod.
"""

import os

import pytest

TEST_DB_URL = os.getenv("TEST_PLANNER_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="set TEST_PLANNER_DATABASE_URL (throwaway pg with the planner schema) to run",
)

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from utils.template import EnrichedInstance  # noqa: E402


def _instance(**over):
    kwargs = dict(
        template_id="tmpl_a",
        template_version=2,
        name="Schedule",
        bound_slots={"recipient": "dana@example.com", "when": "friday"},
        specialization_scope="user:u1",
        task_id="task-9",
    )
    kwargs.update(over)
    return EnrichedInstance(**kwargs)


@pytest.fixture
def store():
    from utils.pg_instance_store import PGInstanceStore, _TABLE

    s = PGInstanceStore(database_url=TEST_DB_URL)
    with s._connect() as conn:
        conn.execute(f"TRUNCATE {_TABLE}")
    return s


def test_add_and_get_roundtrips_lineage(store):
    inst = _instance()
    store.add_instance(inst, trace_id="thread-1")
    got = store.get_instance(inst.instance_id)
    assert got is not None
    assert got["template_id"] == "tmpl_a"
    assert got["template_version"] == 2
    assert got["bound_slots"] == {"recipient": "dana@example.com", "when": "friday"}
    assert got["specialization_scope"] == "user:u1"
    assert got["task_id"] == "task-9"
    assert got["trace_id"] == "thread-1"


def test_defaults_status_draft_outcome_null(store):
    inst = _instance()
    store.add_instance(inst)
    got = store.get_instance(inst.instance_id)
    assert got["status"] == "draft"  # table default
    assert got["outcome"] is None  # set later, post-execution
    assert got["trace_id"] is None  # none passed


def test_dedup_on_instance_id(store):
    inst = _instance()
    store.add_instance(inst, trace_id="a")
    store.add_instance(inst, trace_id="b")  # same instance_id → ON CONFLICT DO NOTHING
    got = store.get_instance(inst.instance_id)
    assert got is not None
    assert got["trace_id"] == "a"  # first write wins; no duplicate/exception


def test_get_missing_returns_none(store):
    assert store.get_instance("does-not-exist") is None


def test_distinct_instances_persist_separately(store):
    a, b = _instance(), _instance()  # distinct auto-generated instance_ids
    store.add_instance(a)
    store.add_instance(b)
    assert a.instance_id != b.instance_id
    assert store.get_instance(a.instance_id) is not None
    assert store.get_instance(b.instance_id) is not None
