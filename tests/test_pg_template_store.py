"""Parity tests for PGTemplateStore against a real Postgres + pgvector.

Mirrors tests/test_template_store.py (dedup, versioning, lineage, threshold search)
but exercises the actual pg backend. GUARDED: skipped unless TEST_PLANNER_DATABASE_URL
points at a throwaway Postgres with the `planner` schema applied — so the offline suite
stays green and these never touch the prod instance.

Local run (ephemeral cluster recipe in claude-context/phase1.md):
    PGBIN=/opt/homebrew/opt/postgresql@18/bin
    ... initdb + pg_ctl start on a /tmp socket ...
    $PGBIN/psql -f planner_bootstrap.sql        # apply the schema
    TEST_PLANNER_DATABASE_URL='postgresql://postgres@/postgres?host=/tmp/pgs.XXXX&port=5544' \
        ./venv/bin/python -m pytest tests/test_pg_template_store.py -q

Embeddings are stubbed (deterministic, no OpenAI call) so vector search is exercised
without a real key while keeping distances predictable.
"""

import os

import pytest

TEST_DB_URL = os.getenv("TEST_PLANNER_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="set TEST_PLANNER_DATABASE_URL (throwaway pg with the planner schema) to run",
)

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from utils.template import Step, WorkflowTemplate  # noqa: E402


def _vec(*head):
    """1536-dim vector with the given leading components (rest zero)."""
    v = [0.0] * 1536
    for i, x in enumerate(head):
        v[i] = float(x)
    return v


# Deterministic stub embeddings keyed on a marker word in the text. Chosen so that,
# against query "one" (→ [1,0]): "one"→dist 0, "two"→~0.2, "three"→dist 1.
_EMB = {
    "one": _vec(1.0, 0.0),
    "two": _vec(0.8, 0.6),
    "three": _vec(0.0, 1.0),
}


def _fake_embed(text: str):
    for marker, vec in _EMB.items():
        if marker in text:
            return vec
    return _vec(1.0, 0.0)


def _tmpl(template_id=None, name="Sched", steps=("Find time", "Invite"), version=1, parent_id=None):
    kwargs = dict(
        name=name,
        description="d",
        version=version,
        steps=[Step(text=s) for s in steps],
        parent_id=parent_id,
    )
    if template_id:
        kwargs["template_id"] = template_id
    return WorkflowTemplate(**kwargs)


@pytest.fixture
def store():
    from utils.pg_template_store import PGTemplateStore, _TABLE

    s = PGTemplateStore(database_url=TEST_DB_URL)
    s._embed = _fake_embed  # deterministic, no OpenAI
    with s._connect() as conn:
        conn.execute(f"TRUNCATE {_TABLE}")
    return s


def test_dedup_identical_content_not_reinserted(store):
    id1 = store.add_template(_tmpl(template_id="a", name="alpha"))
    id2 = store.add_template(_tmpl(template_id="b", name="alpha"))  # same content
    assert id1 == id2  # dedup returns the first row's id
    assert store.list_versions("a") == [1]
    assert store.get_template("b") is None  # second insert was skipped


def test_different_content_inserted(store):
    store.add_template(_tmpl(template_id="a", name="alpha"))
    store.add_template(_tmpl(template_id="b", name="beta"))
    assert store.get_template("a").name == "alpha"
    assert store.get_template("b").name == "beta"


def test_versioning_and_latest(store):
    store.add_template(_tmpl(template_id="t1", name="v1 content"))
    v2 = _tmpl(template_id="t1", name="v2 content")
    store.add_new_version(v2)

    assert v2.version == 2
    assert store.list_versions("t1") == [1, 2]
    assert store.get_template("t1").version == 2  # latest by default
    assert store.get_template("t1", version=1).version == 1
    assert store.get_template("t1", version=99) is None


def test_lineage_children(store):
    store.add_template(_tmpl(template_id="parent", name="base"))
    store.add_template(_tmpl(template_id="child", name="specialized", parent_id="parent"))

    children = store.children_of("parent")
    assert len(children) == 1
    assert children[0].template_id == "child"
    assert children[0].parent_id == "parent"


def test_roundtrip_fields_preserved(store):
    from utils.template import SlotSpec

    t = WorkflowTemplate(
        template_id="rt",
        name="round",
        description="desc",
        required_slots=[SlotSpec(name="recipient", type="email", required=True)],
        steps=[Step(kind="tool", text="Email {recipient}", ref="send_email")],
        tags=["a", "b"],
        status="trusted",
        source="human",
    )
    store.add_template(t)
    got = store.get_template("rt")
    assert got.name == "round" and got.description == "desc"
    assert [s.name for s in got.required_slots] == ["recipient"]
    assert got.required_slots[0].type == "email"
    assert got.steps[0].kind == "tool" and got.steps[0].ref == "send_email"
    assert got.tags == ["a", "b"]
    assert got.status == "trusted" and got.source == "human"


def test_threshold_search_filters_by_distance(store):
    store.add_template(_tmpl(template_id="1", name="one"))
    store.add_template(_tmpl(template_id="2", name="two"))
    store.add_template(_tmpl(template_id="3", name="three"))
    # Against query "one" (vec [1,0]): distances ~ one=0, two=0.2, three=1.
    matches = store.search_templates("one", top_k=3, max_distance=0.3)
    assert len(matches) == 2
    assert matches[0]["distance"] <= matches[1]["distance"]
    assert matches[0]["score"] >= matches[1]["score"]  # score monotonic with proximity


def test_get_template_missing_returns_none(store):
    assert store.get_template("nope") is None
