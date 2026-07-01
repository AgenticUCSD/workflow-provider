"""Endpoint tests for /create_workflow search-before-create enforcement.

Offline: the search and builder agents are patched, so no OpenAI key / network is
needed. Mirrors the /enrich hit/miss tests in tests/task_unit_test.py.
"""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi.testclient import TestClient

import app as app_module
from utils.task import Objective, Status, Task, TaskTypes, Workflow


CREATE_PATH = "/create_workflow"


def build_task() -> Task:
    return Task(
        task_id="task_create_001",
        task_type=TaskTypes.EXECUTE,
        objective=Objective(
            objective_id="obj_create_001",
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
        description="a workflow",
        steps=["step1", "step2"],
    )


class CreateWorkflowSearchFirstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app_module.app)

    def _payload(self, **extra):
        body = {"task": build_task().model_dump(mode="json")}
        body.update(extra)
        return body

    def test_fresh_create_search_hit_returns_existing_and_skips_create(self):
        existing = build_workflow("existing")
        with (
            patch.object(app_module.search_agent, "query_workflows_for_task", return_value=[existing]) as mock_search,
            patch.object(app_module.builder_agent, "create_workflow_initial") as mock_create,
        ):
            resp = self.client.post(CREATE_PATH, json=self._payload())

        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["workflow_id"], "existing")
        mock_search.assert_called_once()
        mock_create.assert_not_called()

    def test_fresh_create_search_miss_none_calls_create(self):
        created = build_workflow("created")
        with (
            patch.object(app_module.search_agent, "query_workflows_for_task", return_value=None) as mock_search,
            patch.object(app_module.builder_agent, "create_workflow_initial", return_value=created) as mock_create,
        ):
            resp = self.client.post(CREATE_PATH, json=self._payload())

        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["workflow_id"], "created")
        mock_search.assert_called_once()
        mock_create.assert_called_once()

    def test_fresh_create_search_empty_list_calls_create(self):
        created = build_workflow("created")
        with (
            patch.object(app_module.search_agent, "query_workflows_for_task", return_value=[]) as mock_search,
            patch.object(app_module.builder_agent, "create_workflow_initial", return_value=created) as mock_create,
        ):
            resp = self.client.post(CREATE_PATH, json=self._payload())

        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["workflow_id"], "created")
        mock_search.assert_called_once()
        mock_create.assert_called_once()

    def test_regeneration_with_rejected_workflows_skips_search(self):
        rejected = build_workflow("rejected")
        created = build_workflow("regenerated")
        with (
            patch.object(app_module.search_agent, "query_workflows_for_task") as mock_search,
            patch.object(app_module.builder_agent, "create_workflow_initial", return_value=created) as mock_create,
        ):
            resp = self.client.post(
                CREATE_PATH,
                json=self._payload(rejected_workflows=[rejected.model_dump(mode="json")]),
            )

        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["workflow_id"], "regenerated")
        mock_search.assert_not_called()
        mock_create.assert_called_once()

    def test_regeneration_with_user_feedback_skips_search(self):
        created = build_workflow("regenerated")
        with (
            patch.object(app_module.search_agent, "query_workflows_for_task") as mock_search,
            patch.object(app_module.builder_agent, "create_workflow_initial", return_value=created) as mock_create,
        ):
            resp = self.client.post(
                CREATE_PATH,
                json=self._payload(user_feedback="make it shorter"),
            )

        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["workflow_id"], "regenerated")
        mock_search.assert_not_called()
        mock_create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
