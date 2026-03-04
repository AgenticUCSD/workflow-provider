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

from task_identification.task import Objective, Status, Task, TaskTypes, Workflow
from task_identification.task_identifier_agent import ContextItem, ContextPlan, IntentTag, TagResult


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


def build_workflow(workflow_id: str) -> Workflow:
    return Workflow(
        workflow_id=workflow_id,
        name=f"wf-{workflow_id}",
        description="test workflow",
        steps=["step1", "step2"],
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
        workflows = [build_workflow("w1")]
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
        ), patch.object(
            app_module.search_agent, "query_workflows_for_task", return_value=workflows
        ), patch.object(
            app_module.builder_agent, "create_workflow_initial"
        ):
            response = self.client.post("/identify_task", json={"text": "Please send status update today"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "identified")
        self.assertEqual(len(body["tasks"]), 1)
        self.assertEqual(body["tasks"][0]["task_type"], "action_required")
        self.assertEqual(len(body["tasks"][0]["candidate_workflows"]), 1)
        self.assertEqual(body["tasks"][0]["candidate_workflows"][0]["workflow_id"], "w1")
        self.assertEqual(len(body["context_items"]), 1)
        self.assertEqual(body["context_items"][0]["field"], "participants")
        self.assertEqual(body["context_items"][0]["status"], "present")

    def test_identify_blocked_context_response_shape(self) -> None:
        task = build_task(TaskTypes.SCHEDULE)
        workflows = [build_workflow("w2")]
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
        ), patch.object(
            app_module.search_agent, "query_workflows_for_task", return_value=workflows
        ), patch.object(
            app_module.builder_agent, "create_workflow_initial"
        ):
            response = self.client.post("/identify_task", json={"text": "Please schedule this meeting"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "identified")
        self.assertEqual(len(body["context_items"]), 1)
        self.assertEqual(body["context_items"][0]["field"], "participants")
        self.assertEqual(body["context_items"][0]["status"], "missing")
        self.assertEqual(body["tasks"][0]["candidate_workflows"][0]["workflow_id"], "w2")

    def test_identify_search_hit_populates_candidate_workflows(self) -> None:
        task = build_task(TaskTypes.REVIEW_FEEDBACK)
        workflows = [build_workflow("w-hit")]
        with patch.object(app_module.task_identifier_agent, "preprocess_email", return_value="Review this doc"), patch.object(
            app_module.task_identifier_agent,
            "detect_tags",
            return_value=TagResult(tags=[IntentTag(tag="review-feedback", short_description="review attached design")]),
        ), patch.object(
            app_module.task_identifier_agent,
            "determine_context",
            return_value=ContextPlan(context_items=[]),
        ), patch.object(
            app_module.task_identifier_agent, "tags_to_tasks", return_value=[task]
        ), patch.object(
            app_module.search_agent, "query_workflows_for_task", return_value=workflows
        ) as mock_search, patch.object(
            app_module.builder_agent, "create_workflow_initial"
        ) as mock_create:
            response = self.client.post("/identify_task", json={"text": "Review this design doc"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["tasks"][0]["candidate_workflows"][0]["workflow_id"], "w-hit")
        self.assertEqual(mock_search.call_count, 1)
        mock_create.assert_not_called()

    def test_identify_search_miss_calls_create_and_populates_candidate_workflows(self) -> None:
        task = build_task(TaskTypes.ACTION_REQUIRED)
        created = build_workflow("w-created")
        with patch.object(app_module.task_identifier_agent, "preprocess_email", return_value="Please do this"), patch.object(
            app_module.task_identifier_agent,
            "detect_tags",
            return_value=TagResult(tags=[IntentTag(tag="action-request", short_description="complete requested action")]),
        ), patch.object(
            app_module.task_identifier_agent,
            "determine_context",
            return_value=ContextPlan(context_items=[]),
        ), patch.object(
            app_module.task_identifier_agent, "tags_to_tasks", return_value=[task]
        ), patch.object(
            app_module.search_agent, "query_workflows_for_task", return_value=None
        ) as mock_search, patch.object(
            app_module.builder_agent, "create_workflow_initial", return_value=created
        ) as mock_create:
            response = self.client.post("/identify_task", json={"text": "Please do this today"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["tasks"][0]["candidate_workflows"][0]["workflow_id"], "w-created")
        self.assertEqual(mock_search.call_count, 1)
        self.assertEqual(mock_create.call_count, 1)

    def test_identify_multi_task_enriches_each_task_independently(self) -> None:
        first = build_task(TaskTypes.ACTION_REQUIRED)
        second = build_task(TaskTypes.SCHEDULE)
        created = build_workflow("w-created-2")
        searched = [build_workflow("w-search-1")]
        with patch.object(app_module.task_identifier_agent, "preprocess_email", return_value="Two tasks"), patch.object(
            app_module.task_identifier_agent,
            "detect_tags",
            return_value=TagResult(
                tags=[
                    IntentTag(tag="action-request", short_description="send requested summary"),
                    IntentTag(tag="schedule", short_description="schedule follow up call"),
                ]
            ),
        ), patch.object(
            app_module.task_identifier_agent,
            "determine_context",
            return_value=ContextPlan(context_items=[]),
        ), patch.object(
            app_module.task_identifier_agent, "tags_to_tasks", return_value=[first, second]
        ), patch.object(
            app_module.search_agent, "query_workflows_for_task", side_effect=[searched, None]
        ) as mock_search, patch.object(
            app_module.builder_agent, "create_workflow_initial", return_value=created
        ) as mock_create:
            response = self.client.post("/identify_task", json={"text": "Please send summary and schedule call"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["tasks"]), 2)
        self.assertEqual(body["tasks"][0]["candidate_workflows"][0]["workflow_id"], "w-search-1")
        self.assertEqual(body["tasks"][1]["candidate_workflows"][0]["workflow_id"], "w-created-2")
        self.assertEqual(mock_search.call_count, 2)
        self.assertEqual(mock_create.call_count, 1)

    def test_identify_no_task_does_not_call_search_or_create(self) -> None:
        with patch.object(app_module.task_identifier_agent, "preprocess_email", return_value="FYI only"), patch.object(
            app_module.task_identifier_agent,
            "detect_tags",
            return_value=TagResult(tags=[IntentTag(tag="no-task", short_description="no action required")]),
        ), patch.object(
            app_module.search_agent, "query_workflows_for_task"
        ) as mock_search, patch.object(
            app_module.builder_agent, "create_workflow_initial"
        ) as mock_create:
            response = self.client.post("/identify_task", json={"text": "FYI only"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "no_task")
        mock_search.assert_not_called()
        mock_create.assert_not_called()

    def test_identify_empty_text_returns_422(self) -> None:
        response = self.client.post("/identify_task", json={"text": ""})
        self.assertEqual(response.status_code, 422)

    def test_identify_malformed_metadata_returns_422(self) -> None:
        response = self.client.post("/identify_task", json={"text": "Please do this", "metadata": ["invalid"]})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
