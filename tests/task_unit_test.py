"""Consolidated unit tests for TaskIdentifierAgent, /identify_task, and /enrich_task_with_workflows endpoints."""

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

from agents.task_agent import (
    ContextItem,
    ContextPlan,
    DeadlineResult,
    IntentTag,
    TagResult,
    TaskIdentifierAgent,
)
from utils.task import Objective, Status, Task, TaskTypes, Workflow


IDENTIFY_PATH = "/identify_task"
ENRICH_PATH = "/enrich_task_with_workflows"
EDIT_TASK_PATH = "/edit_task"


class StubStructuredModel:
    """Minimal test double that returns a fixed structured response."""

    def __init__(self, output: object) -> None:
        self.output = output

    def invoke(self, _input: str) -> object:
        return self.output


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


class TaskIdentifierAgentTests(unittest.TestCase):
    """Focused unit tests for deterministic TaskIdentifierAgent behavior."""

    def make_agent(self) -> TaskIdentifierAgent:
        agent = TaskIdentifierAgent.__new__(TaskIdentifierAgent)
        agent.deadline_model = StubStructuredModel(
            DeadlineResult(has_deadline=False, deadline_iso=None, rationale="no deadline")
        )
        agent.context_planner_model = StubStructuredModel({"required_context": []})
        return agent

    def test_preprocess_email_removes_quote_and_signature(self) -> None:
        agent = self.make_agent()
        raw_text = (
            "Please review the proposal draft.\n\n"
            "Thanks,\n"
            "Aniket\n"
            "On Tue someone wrote:\n"
            "> older thread"
        )
        processed = agent.preprocess_email(raw_text, "Review request")
        self.assertNotIn("On Tue someone wrote", processed)
        self.assertNotIn("> older thread", processed)
        self.assertIn("Subject: Review request", processed)

    def test_normalize_tags_enforces_no_task_exclusivity(self) -> None:
        agent = self.make_agent()
        result = TagResult(
            tags=[
                IntentTag(tag="no-task", short_description="nothing required"),
                IntentTag(tag="action-request", short_description="send file now"),
            ]
        )
        normalized = agent.normalize_tags(result)
        self.assertEqual(len(normalized.tags), 1)
        self.assertEqual(normalized.tags[0].tag, "no-task")

    def test_tags_to_task_selects_highest_priority_tag(self) -> None:
        agent = self.make_agent()
        result = TagResult(
            tags=[
                IntentTag(tag="commitment-track", short_description="track promised delivery date"),
                IntentTag(tag="escalation-urgent", short_description="resolve production blocker immediately"),
            ]
        )
        task = agent.tags_to_task(
            tag_result=result,
            processed_text="Body text",
            metadata={"source": "email"},
        )
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task.task_type, TaskTypes.ESCALATION_URGENT)
        self.assertEqual(list(task.objective.inputs.keys()), ["processed_text"])

    def test_edit_task_returns_updated_task(self) -> None:
        agent = self.make_agent()
        original = build_task(TaskTypes.ACTION_REQUIRED)
        updated = build_task(TaskTypes.ACTION_REQUIRED)
        updated.objective.description = "Email the project team with today's progress, blockers, and ETA request."
        updated.objective.inputs["assignee"] = "alex@example.com"
        updated.metadata = {"source": "test", "feedback": "Add ETA request"}
        agent.task_editor_model = StubStructuredModel(updated)

        result = agent.edit_task(original, "Add a line asking for an ETA on staging access.")

        self.assertEqual(result.objective.description, updated.objective.description)
        self.assertEqual(result.objective.inputs["assignee"], "alex@example.com")
        self.assertEqual(result.metadata, updated.metadata)

    def test_schedule_task_has_deadline_guardrail(self) -> None:
        agent = self.make_agent()
        deadline_iso = "2026-02-24T17:00:00+00:00"
        agent.deadline_model = StubStructuredModel(
            DeadlineResult(
                has_deadline=True,
                deadline_iso=deadline_iso,
                rationale="explicit deadline",
            )
        )
        result = TagResult(tags=[IntentTag(tag="schedule", short_description="schedule roadmap meeting")])
        task = agent.tags_to_task(result, "Schedule a meeting by 5pm today.", {"source": "email"})
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task.objective.deadline, deadline_iso)
        self.assertEqual(
            task.objective.constraints.get("latest_scheduling_time"),
            deadline_iso,
        )


@unittest.skipUnless(HAS_ENDPOINT_DEPS, "Endpoint dependencies are unavailable")
class IdentifyEndpointTests(unittest.TestCase):
    """Endpoint-level tests with agent internals mocked for deterministic behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app_module.app)

    def test_identify_no_task_response_shape(self) -> None:
        with (
            patch.object(app_module.task_identifier_agent, "preprocess_email", return_value="FYI only"),
            patch.object(
                app_module.task_identifier_agent,
                "detect_tags",
                return_value=TagResult(tags=[IntentTag(tag="no-task", short_description="no action required")]),
            ),
        ):
            response = self.client.post(IDENTIFY_PATH, json={"text": "FYI only"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "no_task")
        self.assertIsNone(body["task"])
        self.assertEqual(body["detected_tag"], "no-task")
        self.assertEqual(body["context_items"], [])

    def test_identify_identified_response_shape(self) -> None:
        task = build_task(TaskTypes.ACTION_REQUIRED)
        with (
            patch.object(app_module.task_identifier_agent, "preprocess_email", return_value="Please send status"),
            patch.object(
                app_module.task_identifier_agent,
                "detect_tags",
                return_value=TagResult(tags=[IntentTag(tag="action-request", short_description="send status update")]),
            ),
            patch.object(
                app_module.task_identifier_agent,
                "determine_context",
                return_value=ContextPlan(
                    context_items=[ContextItem(field="participants", status="present", value="a@b.com")]
                ),
            ),
            patch.object(app_module.task_identifier_agent, "tags_to_task", return_value=task),
        ):
            response = self.client.post(IDENTIFY_PATH, json={"text": "Please send status update today"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "identified")
        self.assertEqual(body["task"]["task_type"], "action_required")
        self.assertIsNone(body["task"]["candidate_workflows"])
        self.assertEqual(len(body["context_items"]), 1)
        self.assertEqual(body["context_items"][0]["field"], "participants")
        self.assertEqual(body["context_items"][0]["status"], "present")

    def test_identify_blocked_context_response_shape(self) -> None:
        task = build_task(TaskTypes.SCHEDULE)
        with (
            patch.object(app_module.task_identifier_agent, "preprocess_email", return_value="Please schedule"),
            patch.object(
                app_module.task_identifier_agent,
                "detect_tags",
                return_value=TagResult(tags=[IntentTag(tag="schedule", short_description="schedule quick sync")]),
            ),
            patch.object(
                app_module.task_identifier_agent,
                "determine_context",
                return_value=ContextPlan(
                    context_items=[ContextItem(field="participants", status="missing", value=None)]
                ),
            ),
            patch.object(app_module.task_identifier_agent, "tags_to_task", return_value=task),
        ):
            response = self.client.post(IDENTIFY_PATH, json={"text": "Please schedule this meeting"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "identified")
        self.assertEqual(len(body["context_items"]), 1)
        self.assertEqual(body["context_items"][0]["field"], "participants")
        self.assertEqual(body["context_items"][0]["status"], "missing")
        self.assertIsNone(body["task"]["candidate_workflows"])

    def test_enrich_task_search_hit_populates_candidate_workflows(self) -> None:
        task = build_task(TaskTypes.REVIEW_FEEDBACK)
        workflows = [build_workflow("w-hit")]
        with (
            patch.object(app_module.search_agent, "query_workflows_for_task", return_value=workflows) as mock_search,
            patch.object(app_module.builder_agent, "create_workflow_initial") as mock_create,
        ):
            response = self.client.post(ENRICH_PATH, json=task.model_dump())
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["candidate_workflows"][0]["workflow_id"], "w-hit")
        self.assertEqual(mock_search.call_count, 1)
        mock_create.assert_not_called()

    def test_enrich_task_search_miss_calls_create_and_populates_candidate_workflows(self) -> None:
        task = build_task(TaskTypes.ACTION_REQUIRED)
        created = build_workflow("w-created")
        with (
            patch.object(app_module.search_agent, "query_workflows_for_task", return_value=None) as mock_search,
            patch.object(app_module.builder_agent, "create_workflow_initial", return_value=created) as mock_create,
        ):
            response = self.client.post(ENRICH_PATH, json=task.model_dump())
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["candidate_workflows"][0]["workflow_id"], "w-created")
        self.assertEqual(mock_search.call_count, 1)
        self.assertEqual(mock_create.call_count, 1)

    def test_identify_multi_tag_only_returns_highest_priority_task(self) -> None:
        selected = build_task(TaskTypes.ACTION_REQUIRED)
        with (
            patch.object(app_module.task_identifier_agent, "preprocess_email", return_value="Two tasks"),
            patch.object(
                app_module.task_identifier_agent,
                "detect_tags",
                return_value=TagResult(
                    tags=[
                        IntentTag(tag="action-request", short_description="send requested summary"),
                        IntentTag(tag="schedule", short_description="schedule follow up call"),
                    ]
                ),
            ),
            patch.object(
                app_module.task_identifier_agent,
                "determine_context",
                return_value=ContextPlan(context_items=[]),
            ),
            patch.object(app_module.task_identifier_agent, "tags_to_task", return_value=selected),
        ):
            response = self.client.post(IDENTIFY_PATH, json={"text": "Please send summary and schedule call"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "identified")
        self.assertEqual(body["detected_tag"], "action-request")
        self.assertIsNone(body["task"]["candidate_workflows"])

    def test_identify_no_task_does_not_call_search_or_create(self) -> None:
        with (
            patch.object(app_module.task_identifier_agent, "preprocess_email", return_value="FYI only"),
            patch.object(
                app_module.task_identifier_agent,
                "detect_tags",
                return_value=TagResult(tags=[IntentTag(tag="no-task", short_description="no action required")]),
            ),
            patch.object(app_module.search_agent, "query_workflows_for_task") as mock_search,
            patch.object(app_module.builder_agent, "create_workflow_initial") as mock_create,
        ):
            response = self.client.post(IDENTIFY_PATH, json={"text": "FYI only"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "no_task")
        mock_search.assert_not_called()
        mock_create.assert_not_called()

    def test_identify_empty_text_returns_422(self) -> None:
        response = self.client.post(IDENTIFY_PATH, json={"text": ""})
        self.assertEqual(response.status_code, 422)

    def test_identify_malformed_metadata_returns_422(self) -> None:
        response = self.client.post(IDENTIFY_PATH, json={"text": "Please do this", "metadata": ["invalid"]})
        self.assertEqual(response.status_code, 422)

    def test_edit_task_endpoint_returns_updated_task(self) -> None:
        original = build_task(TaskTypes.ACTION_REQUIRED)
        updated = build_task(TaskTypes.ACTION_REQUIRED)
        updated.objective.description = "Updated task description from feedback"

        with patch.object(app_module.task_identifier_agent, "edit_task", return_value=updated) as mock_edit:
            response = self.client.post(
                EDIT_TASK_PATH,
                json={
                    "task": original.model_dump(),
                    "user_feedback": "Please include the missing context item.",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "edited")
        self.assertEqual(body["task"]["task_id"], updated.task_id)
        self.assertEqual(body["task"]["objective"]["description"], updated.objective.description)
        self.assertEqual(body["detected_tag"], None)
        self.assertEqual(body["context_items"], [])
        mock_edit.assert_called_once()


if __name__ == "__main__":
    unittest.main()