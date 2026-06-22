"""Unit tests for BuilderAgent — focuses on persistence of generated workflows.

Offline: the LLM agent and workflow extraction are stubbed, so no OpenAI key or
network is required.
"""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agents.builder_agent import BuilderAgent
from utils.task import Objective, Status, Task, TaskTypes, Workflow


def build_task() -> Task:
    return Task(
        task_id="task_b_001",
        task_type=TaskTypes.EXECUTE,
        objective=Objective(
            objective_id="obj_b_001",
            name="test",
            description="test objective",
            inputs={},
            constraints={},
            success_criteria="done",
            expected_output={"status": "completed"},
            deadline=None,
        ),
        status=Status.PENDING,
    )


def build_workflow(workflow_id: str) -> Workflow:
    return Workflow(
        workflow_id=workflow_id,
        name=f"wf-{workflow_id}",
        description="generated workflow",
        steps=["step1", "step2"],
    )


class FakeVectorDB:
    def __init__(self) -> None:
        self.added = []

    def add_workflow(self, workflow, is_generated=True):
        self.added.append((workflow, is_generated))
        return "doc-id"


class _StubAgent:
    """Accepts whatever invoke args create_workflow_initial passes."""

    def invoke(self, *args, **kwargs):
        return {"stub": True}


def make_builder(vector_db) -> BuilderAgent:
    # Bypass create_agent (no LLM) and inject our stubs.
    agent = BuilderAgent.__new__(BuilderAgent)
    agent.vector_db = vector_db
    agent.agent = _StubAgent()
    return agent


class BuilderPersistenceTests(unittest.TestCase):
    def test_create_workflow_persists_generated(self) -> None:
        fake_db = FakeVectorDB()
        builder = make_builder(fake_db)
        generated = build_workflow("w-gen")

        with patch.object(BuilderAgent, "extract_workflow", return_value=generated):
            out = builder.create_workflow_initial(build_task())

        self.assertEqual(out.workflow_id, "w-gen")
        self.assertEqual(len(fake_db.added), 1)
        stored_workflow, is_generated = fake_db.added[0]
        self.assertEqual(stored_workflow.workflow_id, "w-gen")
        self.assertTrue(is_generated)

    def test_persist_failure_does_not_break_creation(self) -> None:
        class ExplodingDB:
            def add_workflow(self, workflow, is_generated=True):
                raise RuntimeError("embeddings unavailable")

        builder = make_builder(ExplodingDB())
        generated = build_workflow("w-gen")

        with patch.object(BuilderAgent, "extract_workflow", return_value=generated):
            out = builder.create_workflow_initial(build_task())

        # Creation still succeeds even though persistence raised.
        self.assertEqual(out.workflow_id, "w-gen")

    def test_no_vector_db_is_a_noop(self) -> None:
        builder = make_builder(None)
        generated = build_workflow("w-gen")

        with patch.object(BuilderAgent, "extract_workflow", return_value=generated):
            out = builder.create_workflow_initial(build_task())

        self.assertEqual(out.workflow_id, "w-gen")


if __name__ == "__main__":
    unittest.main()
