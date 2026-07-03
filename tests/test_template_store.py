"""Unit tests for TemplateStore: dedup, versioning, lineage, threshold search.

Offline: builds TemplateStore via __new__ and injects a fake collection (no
chromadb client / OpenAI key), mirroring tests/test_chroma_dedup.py.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from utils.template import Step, WorkflowTemplate
from utils.template_store import TemplateStore


class FakeCollection:
    """Stand-in chromadb collection supporting add(), get(where=), query()."""

    def __init__(self):
        self.ids = []
        self.documents = []
        self.metadatas = []
        self.add_calls = 0

    def add(self, documents, ids, metadatas):
        self.add_calls += 1
        self.documents.extend(documents)
        self.ids.extend(ids)
        self.metadatas.extend(metadatas)

    def get(self, where=None):
        if not where:
            return {"ids": list(self.ids), "metadatas": list(self.metadatas)}
        (key, value), = where.items()
        matched = [
            (i, m) for i, m in zip(self.ids, self.metadatas) if m.get(key) == value
        ]
        return {"ids": [i for i, _ in matched], "metadatas": [m for _, m in matched]}

    def query(self, query_texts, n_results=5):
        # Return stored items in insertion order with synthetic increasing
        # distances so max_distance filtering is deterministic.
        metas = self.metadatas[:n_results]
        dists = [round(0.2 * i, 3) for i in range(len(metas))]
        return {"metadatas": [metas], "distances": [dists]}


def make_store():
    store = TemplateStore.__new__(TemplateStore)
    store.templates = FakeCollection()
    return store


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


def test_dedup_identical_content_not_reinserted():
    store = make_store()
    id1 = store.add_template(_tmpl(template_id="a", name="alpha"))
    id2 = store.add_template(_tmpl(template_id="b", name="alpha"))  # same content
    assert store.templates.add_calls == 1
    assert id1 == id2


def test_different_content_inserted():
    store = make_store()
    store.add_template(_tmpl(name="alpha"))
    store.add_template(_tmpl(name="beta"))
    assert store.templates.add_calls == 2


def test_versioning_and_latest():
    store = make_store()
    store.add_template(_tmpl(template_id="t1", name="v1 content"))
    v2 = _tmpl(template_id="t1", name="v2 content")
    store.add_new_version(v2)

    assert v2.version == 2
    assert store.list_versions("t1") == [1, 2]
    assert store.get_template("t1").version == 2  # latest by default
    assert store.get_template("t1", version=1).version == 1
    assert store.get_template("t1", version=99) is None


def test_lineage_children():
    store = make_store()
    store.add_template(_tmpl(template_id="parent", name="base"))
    store.add_template(_tmpl(template_id="child", name="specialized", parent_id="parent"))

    children = store.children_of("parent")
    assert len(children) == 1
    assert children[0].template_id == "child"


def test_threshold_search_filters_by_distance():
    store = make_store()
    store.add_template(_tmpl(name="one"))
    store.add_template(_tmpl(name="two"))
    store.add_template(_tmpl(name="three"))
    # Synthetic distances are 0.0, 0.2, 0.4 → max_distance 0.3 keeps the first two.
    matches = store.search_templates("anything", top_k=3, max_distance=0.3)
    assert len(matches) == 2
    assert matches[0]["distance"] <= matches[1]["distance"]
    assert matches[0]["score"] >= matches[1]["score"]  # score monotonic with proximity


def test_get_template_missing_returns_none():
    store = make_store()
    assert store.get_template("nope") is None
