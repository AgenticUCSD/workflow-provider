import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

try:
    from fastapi.testclient import TestClient
    import app as app_module

    HAS_ENDPOINT_DEPS = True
except ModuleNotFoundError:
    HAS_ENDPOINT_DEPS = False

from utils.task import Objective, Status, Task, TaskTypes
from utils.task_identifier_agent import ContextItem, ContextPlan, IntentTag, TagResult


def build_task(task_type: TaskTypes) -> Task:
    return Task(
        task_id="task_test_001",
        task_type=task_type,
        objective=Objective(
            objective_id="obj_test_001",
            name="test",
            description="test objective",
            inputs={"processed_text": "input"},
            constraints={},
            success_criteria="done",
            expected_output={"status": "completed"},
            deadline=None,
        ),
        candidate_workflows=None,
        workflow=None,
        status=Status.PENDING,
        metadata={"source": "test"},
    )


@unittest.skipUnless(HAS_ENDPOINT_DEPS, "Endpoint dependencies are unavailable")
class IdentifyEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app_module.app)

    def test_identify_no_task_response_shape(self) -> None:
        with patch.object(app_module.task_identifier_agent, "preprocess_email", return_value="FYI only"), patch.object(
            app_module.task_identifier_agent,
            "detect_tags",
            return_value=TagResult(tags=[IntentTag(tag="no-task", short_description="no action required")]),
        ):
            response = self.client.post("/identify_task", json={"text": "FYI only"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "no_task")
        self.assertEqual(body["tasks"], [])
        self.assertEqual(body["detected_tags"], ["no-task"])
        self.assertEqual(body["context_items"], [])

    def test_identify_identified_response_shape(self) -> None:
        task = build_task(TaskTypes.ACTION_REQUIRED)
        with patch.object(app_module.task_identifier_agent, "preprocess_email", return_value="Please send status"), patch.object(
            app_module.task_identifier_agent,
            "detect_tags",
            return_value=TagResult(tags=[IntentTag(tag="action-request", short_description="send status update")]),
        ), patch.object(
            app_module.task_identifier_agent,
            "determine_context",
            return_value=ContextPlan(context_items=[ContextItem(field="participants", status="present", value="a@b.com")]),
        ), patch.object(
            app_module.task_identifier_agent, "tags_to_tasks", return_value=[task]
        ):
            response = self.client.post("/identify_task", json={"text": "Please send status update today"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "identified")
        self.assertEqual(len(body["tasks"]), 1)
        self.assertEqual(body["tasks"][0]["task_type"], "action_required")
        self.assertEqual(len(body["context_items"]), 1)
        self.assertEqual(body["context_items"][0]["field"], "participants")
        self.assertEqual(body["context_items"][0]["status"], "present")

    def test_identify_blocked_context_response_shape(self) -> None:
        task = build_task(TaskTypes.SCHEDULE)
        with patch.object(app_module.task_identifier_agent, "preprocess_email", return_value="Please schedule"), patch.object(
            app_module.task_identifier_agent,
            "detect_tags",
            return_value=TagResult(tags=[IntentTag(tag="schedule", short_description="schedule quick sync")]),
        ), patch.object(
            app_module.task_identifier_agent,
            "determine_context",
            return_value=ContextPlan(context_items=[ContextItem(field="participants", status="missing", value=None)]),
        ), patch.object(
            app_module.task_identifier_agent, "tags_to_tasks", return_value=[task]
        ):
            response = self.client.post("/identify_task", json={"text": "Please schedule this meeting"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "identified")
        self.assertEqual(len(body["context_items"]), 1)
        self.assertEqual(body["context_items"][0]["field"], "participants")
        self.assertEqual(body["context_items"][0]["status"], "missing")

    def test_identify_empty_text_returns_422(self) -> None:
        response = self.client.post("/identify_task", json={"text": ""})
        self.assertEqual(response.status_code, 422)

    def test_identify_malformed_metadata_returns_422(self) -> None:
        response = self.client.post("/identify_task", json={"text": "Please do this", "metadata": ["invalid"]})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
