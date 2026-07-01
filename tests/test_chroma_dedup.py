"""Unit tests for ChromaVectorStore's exact-content dedup gate.

Offline: builds a ChromaVectorStore via __new__ (bypassing the real chromadb client
and OpenAI embedding function) and injects fake collections, so no key/network is
required. Mirrors the stub style in tests/test_builder_agent.py.
"""

import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from utils.chroma import ChromaVectorStore
from utils.task import Workflow


def build_workflow(workflow_id: str, name: str = "wf", steps=None) -> Workflow:
    return Workflow(
        workflow_id=workflow_id,
        name=name,
        description="a workflow",
        steps=steps or ["step1", "step2"],
    )


class FakeCollection:
    """Minimal stand-in for a chromadb collection: supports add() and get(where=)."""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.metadatas: list[dict] = []
        self.add_calls = 0

    def add(self, documents, ids, metadatas):
        self.add_calls += 1
        self.ids.extend(ids)
        self.metadatas.extend(metadatas)

    def get(self, where=None):
        if not where:
            return {"ids": list(self.ids), "metadatas": list(self.metadatas)}
        # Only the single-key {"content_hash": h} form is used by add_workflow.
        (key, value), = where.items()
        matched = [
            (i, m) for i, m in zip(self.ids, self.metadatas) if m.get(key) == value
        ]
        return {
            "ids": [i for i, _ in matched],
            "metadatas": [m for _, m in matched],
        }


def make_store() -> ChromaVectorStore:
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.generated_workflows = FakeCollection()
    store.manual_workflows = FakeCollection()
    return store


class ContentHashTests(unittest.TestCase):
    def test_hash_is_deterministic_and_content_sensitive(self) -> None:
        a1 = build_workflow("id-A", name="alpha")
        a2 = build_workflow("id-B", name="alpha")  # different id, same content
        b = build_workflow("id-C", name="beta")

        # Same content (id is not part of to_string) -> same hash.
        self.assertEqual(
            ChromaVectorStore._content_hash(a1),
            ChromaVectorStore._content_hash(a2),
        )
        # Different content -> different hash.
        self.assertNotEqual(
            ChromaVectorStore._content_hash(a1),
            ChromaVectorStore._content_hash(b),
        )


class DedupTests(unittest.TestCase):
    def test_identical_workflow_is_not_reinserted(self) -> None:
        store = make_store()
        first_id = store.add_workflow(build_workflow("id-A", name="alpha"), is_generated=True)

        # Same content (even with a different workflow_id) must dedup.
        second_id = store.add_workflow(build_workflow("id-B", name="alpha"), is_generated=True)

        self.assertEqual(store.generated_workflows.add_calls, 1)
        self.assertEqual(first_id, second_id)
        self.assertEqual(len(store.generated_workflows.ids), 1)

    def test_different_workflow_is_inserted(self) -> None:
        store = make_store()
        store.add_workflow(build_workflow("id-A", name="alpha"), is_generated=True)
        store.add_workflow(build_workflow("id-C", name="beta"), is_generated=True)

        self.assertEqual(store.generated_workflows.add_calls, 2)
        self.assertEqual(len(store.generated_workflows.ids), 2)

    def test_content_hash_is_stored_in_metadata(self) -> None:
        store = make_store()
        wf = build_workflow("id-A", name="alpha")
        store.add_workflow(wf, is_generated=True)

        stored = store.generated_workflows.metadatas[0]
        self.assertEqual(stored["content_hash"], ChromaVectorStore._content_hash(wf))

    def test_dedup_is_per_collection(self) -> None:
        store = make_store()
        # Same content in the two different collections should both insert.
        store.add_workflow(build_workflow("id-A", name="alpha"), is_generated=True)
        store.add_workflow(build_workflow("id-A", name="alpha"), is_generated=False)

        self.assertEqual(store.generated_workflows.add_calls, 1)
        self.assertEqual(store.manual_workflows.add_calls, 1)


if __name__ == "__main__":
    unittest.main()
