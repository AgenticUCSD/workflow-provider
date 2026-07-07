"""Offline test for PGWorkflowStore's fail-open read path (no real DB needed).

The live search path must NOT 500 when Postgres is unreachable/slow — it should
degrade to "no candidates" (→ the caller generates a fresh workflow, like a
cold-start Chroma). Constructing PGWorkflowStore doesn't open a connection, so we
can monkeypatch the connection to raise and assert the search degrades gracefully.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from utils.pg_workflow_store import PGWorkflowStore


class _FakeTask:
    def to_string(self):
        return "any task text"


def _store():
    # database_url is stored but never connected during __init__.
    store = PGWorkflowStore(database_url="postgresql://dummy/dummy")
    store._embed = lambda text: [0.0] * 1536  # valid vector, skip the OpenAI call
    return store


def test_search_degrades_to_empty_on_db_error():
    store = _store()

    def boom(*args, **kwargs):
        raise ConnectionError("db unreachable")

    store._connect = boom  # any DB access raises

    # Must NOT raise; degrades to no candidates.
    assert store.query_from_all_workflows_as_objects(_FakeTask(), top_k=5) == []


def test_search_returns_empty_when_embedding_unavailable():
    store = _store()
    store._embed = lambda text: None  # embeddings backend down → no query vector
    assert store.query_from_all_workflows_as_objects(_FakeTask(), top_k=5) == []
