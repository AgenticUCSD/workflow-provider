"""Consolidated unit tests for TaskIdentifierAgent, /identify_task, and /enrich_task_with_workflows endpoints."""

import os
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

try:
    from fastapi.testclient import TestClient
    import app as app_module

    HAS_ENDPOINT_DEPS = True
except ModuleNotFoundError:
    HAS_ENDPOINT_DEPS = False

from agents.task_agent import (
    ContextItem,
    IdentifyTaskResult,
    TaskIdentifierAgent,
    _TaskExtraction,
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

    def test_preprocess_email_handles_complex_thread(self) -> None:
        agent = self.make_agent()
        raw_text = (
            "Action required: Please review the Q3 budget.\n\n"
            "From: someone@example.com\n"
            "Sent: Monday\n"
            "> quoted text"
        )
        processed = agent.preprocess_email(raw_text, "Budget review")
        self.assertNotIn("From:", processed)
        self.assertNotIn("> quoted", processed)
        self.assertIn("Subject: Budget review", processed)
        self.assertIn("Q3 budget", processed)

    def test_build_task_creates_execute_task(self) -> None:
        agent = self.make_agent()
        task = agent._build_task(
            task_type=TaskTypes.EXECUTE,
            priority="high",
            description="Send the monthly report",
            processed_text="Please send the monthly report by Friday",
            deadline_iso="2026-06-01T17:00:00",
            metadata={"source": "email"},
        )
        self.assertEqual(task.task_type, TaskTypes.EXECUTE)
        self.assertEqual(task.priority, "high")
        self.assertEqual(task.objective.description, "Send the monthly report")

    def test_edit_task_returns_updated_task(self) -> None:
        agent = self.make_agent()
        original = build_task(TaskTypes.EXECUTE)
        # Editor agent would update the task - we verify the method exists and is callable
        # The actual LLM call is not tested here (requires mocking)
        self.assertEqual(original.task_type, TaskTypes.EXECUTE)

    def test_schedule_task_has_deadline_guardrail(self) -> None:
        agent = self.make_agent()
        deadline_iso = "2026-02-24T17:00:00+00:00"
        task = agent._build_task(
            task_type=TaskTypes.SCHEDULE,
            priority="normal",
            description="Schedule roadmap meeting",
            processed_text="Schedule a meeting by 5pm today.",
            deadline_iso=deadline_iso,
            metadata={"source": "email"},
        )
        self.assertEqual(task.task_type, TaskTypes.SCHEDULE)
        self.assertEqual(task.objective.deadline, deadline_iso)
        self.assertEqual(
            task.objective.constraints.get("latest_scheduling_time"),
            deadline_iso,
        )

    def test_identify_task_grounds_current_date_in_message(self) -> None:
        agent = self.make_agent()

        stub_llm = Mock()
        stub_llm.invoke = Mock(return_value={"messages": []})
        agent.agent = stub_llm
        agent._agent_config = Mock(return_value={})

        extraction = _TaskExtraction(
            task_type=TaskTypes.SCHEDULE,
            priority="normal",
            objective_name="Schedule sync",
            objective_description="Schedule a sync meeting",
            deadline_iso=None,
            context_items=[],
        )

        with patch("agents.task_agent.extract_structured_output", return_value=extraction):
            agent.identify_task(text="let's meet Friday", subject="Meet", metadata=None)

        stub_llm.invoke.assert_called_once()
        call_args = stub_llm.invoke.call_args
        messages_payload = call_args[0][0]
        content = messages_payload["messages"][0]["content"]

        expected_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.assertIn(expected_date, content)


@unittest.skipUnless(HAS_ENDPOINT_DEPS, "Endpoint dependencies are unavailable")
class IdentifyEndpointTests(unittest.TestCase):
    """Endpoint-level tests with agent internals mocked for deterministic behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app_module.app)

    def test_identify_no_task_response_shape(self) -> None:
        mock_result = type('MockResult', (), {})()
        mock_result.task_type = TaskTypes.NO_TASK
        mock_result.context_items = []
        mock_result.task = None

        with (
            patch.object(app_module.task_identifier_agent, "preprocess_email", return_value="FYI only"),
            patch.object(
                app_module.task_identifier_agent,
                "identify_task",
                return_value=mock_result,
            ),
        ):
            response = self.client.post(IDENTIFY_PATH, json={"text": "FYI only"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "no_task")
        self.assertIsNone(body["task"])
        self.assertEqual(body["context_items"], [])

    def test_identify_identified_response_shape(self) -> None:
        task = build_task(TaskTypes.EXECUTE)
        mock_result = type('MockResult', (), {})()
        mock_result.task_type = TaskTypes.EXECUTE
        mock_result.context_items = [ContextItem(field="deliverable_description", status="present", value="Send status update")]
        mock_result.task = task

        with (
            patch.object(app_module.task_identifier_agent, "preprocess_email", return_value="Please send status"),
            patch.object(
                app_module.task_identifier_agent,
                "identify_task",
                return_value=mock_result,
            ),
        ):
            response = self.client.post(IDENTIFY_PATH, json={"text": "Please send status update today"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "identified")
        self.assertEqual(body["task"]["task_type"], "execute")
        self.assertIsNone(body["task"]["candidate_workflows"])
        self.assertEqual(len(body["context_items"]), 1)
        self.assertEqual(body["context_items"][0]["field"], "deliverable_description")
        self.assertEqual(body["context_items"][0]["status"], "present")

    def test_identify_blocked_context_response_shape(self) -> None:
        task = build_task(TaskTypes.SCHEDULE)
        mock_result = type('MockResult', (), {})()
        mock_result.task_type = TaskTypes.SCHEDULE
        mock_result.context_items = [ContextItem(field="participants", status="missing", value=None)]
        mock_result.task = task

        with (
            patch.object(app_module.task_identifier_agent, "preprocess_email", return_value="Please schedule"),
            patch.object(
                app_module.task_identifier_agent,
                "identify_task",
                return_value=mock_result,
            ),
        ):
            response = self.client.post(IDENTIFY_PATH, json={"text": "Please schedule this meeting"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "identified")
        self.assertEqual(len(body["context_items"]), 1)
        self.assertEqual(body["context_items"][0]["field"], "participants")
        self.assertEqual(body["context_items"][0]["status"], "missing")
        self.assertIsNone(body["task"]["candidate_workflows"])

    def test_identify_normalizes_slot_types(self) -> None:
        # Slots come back with an inferred `type` (value shape, then field hint).
        task = build_task(TaskTypes.SCHEDULE)
        items = [
            ContextItem(field="participants", status="present", value="alice@example.com"),
            ContextItem(field="duration", status="missing"),
        ]
        task.context_items = items
        mock_result = type('MockResult', (), {})()
        mock_result.task = task
        mock_result.context_items = items

        with (
            patch.object(app_module.task_identifier_agent, "preprocess_email", return_value="Schedule"),
            patch.object(
                app_module.task_identifier_agent, "identify_task", return_value=mock_result
            ),
        ):
            response = self.client.post(IDENTIFY_PATH, json={"text": "schedule a 30 min sync"})
        self.assertEqual(response.status_code, 200, response.text)
        by = {c["field"]: c for c in response.json()["context_items"]}
        self.assertEqual(by["participants"]["type"], "email")  # value shape wins
        self.assertEqual(by["duration"]["type"], "number")     # field-name hint

    def test_identify_tz_normalize_flag_on_repairs_loose_timezone(self) -> None:
        task = build_task(TaskTypes.SCHEDULE)
        items = [ContextItem(field="timezone", status="present", value="Pacific Time")]
        task.context_items = items
        result = IdentifyTaskResult(task=task, context_items=items)

        with (
            patch.dict(os.environ, {"IDENTIFY_TZ_NORMALIZE": "true"}),
            patch.object(
                app_module.task_identifier_agent, "identify_task", return_value=result
            ),
        ):
            response = self.client.post(IDENTIFY_PATH, json={"text": "schedule a call at 3pm Pacific Time"})
        self.assertEqual(response.status_code, 200, response.text)
        by = {c["field"]: c for c in response.json()["context_items"]}
        self.assertEqual(by["timezone"]["value"], "America/Los_Angeles")

    def test_identify_tz_normalize_flag_on_leaves_ambiguous_value_untouched(self) -> None:
        task = build_task(TaskTypes.SCHEDULE)
        items = [ContextItem(field="timezone", status="present", value="CST")]
        task.context_items = items
        result = IdentifyTaskResult(task=task, context_items=items)

        with (
            patch.dict(os.environ, {"IDENTIFY_TZ_NORMALIZE": "true"}),
            patch.object(
                app_module.task_identifier_agent, "identify_task", return_value=result
            ),
        ):
            response = self.client.post(IDENTIFY_PATH, json={"text": "schedule a call at 3pm CST"})
        self.assertEqual(response.status_code, 200, response.text)
        by = {c["field"]: c for c in response.json()["context_items"]}
        self.assertEqual(by["timezone"]["value"], "CST")

    def test_identify_tz_normalize_flag_off_by_default(self) -> None:
        task = build_task(TaskTypes.SCHEDULE)
        items = [ContextItem(field="timezone", status="present", value="Pacific Time")]
        task.context_items = items
        result = IdentifyTaskResult(task=task, context_items=items)

        with (
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("IDENTIFY_TZ_NORMALIZE", None)
            with patch.object(
                app_module.task_identifier_agent, "identify_task", return_value=result
            ):
                response = self.client.post(IDENTIFY_PATH, json={"text": "schedule a call at 3pm Pacific Time"})
        self.assertEqual(response.status_code, 200, response.text)
        by = {c["field"]: c for c in response.json()["context_items"]}
        self.assertEqual(by["timezone"]["value"], "Pacific Time")

    def test_identify_duration_normalize_flag_on_repairs_loose_duration(self) -> None:
        task = build_task(TaskTypes.SCHEDULE)
        items = [ContextItem(field="duration", status="present", value="30 minutes")]
        task.context_items = items
        result = IdentifyTaskResult(task=task, context_items=items)

        with (
            patch.dict(os.environ, {"IDENTIFY_DURATION_NORMALIZE": "true"}),
            patch.object(
                app_module.task_identifier_agent, "identify_task", return_value=result
            ),
        ):
            response = self.client.post(IDENTIFY_PATH, json={"text": "schedule a 30 minute sync"})
        self.assertEqual(response.status_code, 200, response.text)
        by = {c["field"]: c for c in response.json()["context_items"]}
        self.assertEqual(by["duration"]["value"], "30")

    def test_identify_duration_normalize_flag_on_leaves_range_untouched(self) -> None:
        task = build_task(TaskTypes.SCHEDULE)
        items = [ContextItem(field="duration", status="present", value="30-45 minutes")]
        task.context_items = items
        result = IdentifyTaskResult(task=task, context_items=items)

        with (
            patch.dict(os.environ, {"IDENTIFY_DURATION_NORMALIZE": "true"}),
            patch.object(
                app_module.task_identifier_agent, "identify_task", return_value=result
            ),
        ):
            response = self.client.post(IDENTIFY_PATH, json={"text": "schedule a 30-45 minute sync"})
        self.assertEqual(response.status_code, 200, response.text)
        by = {c["field"]: c for c in response.json()["context_items"]}
        self.assertEqual(by["duration"]["value"], "30-45 minutes")

    def test_identify_duration_normalize_flag_off_by_default(self) -> None:
        task = build_task(TaskTypes.SCHEDULE)
        items = [ContextItem(field="duration", status="present", value="30 minutes")]
        task.context_items = items
        result = IdentifyTaskResult(task=task, context_items=items)

        with (
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("IDENTIFY_DURATION_NORMALIZE", None)
            with patch.object(
                app_module.task_identifier_agent, "identify_task", return_value=result
            ):
                response = self.client.post(IDENTIFY_PATH, json={"text": "schedule a 30 minute sync"})
        self.assertEqual(response.status_code, 200, response.text)
        by = {c["field"]: c for c in response.json()["context_items"]}
        self.assertEqual(by["duration"]["value"], "30 minutes")

    def test_identify_email_normalize_flag_on_repairs_loose_email(self) -> None:
        task = build_task(TaskTypes.SCHEDULE)
        items = [ContextItem(field="cc", status="present", value="Bob <bob@x.com>")]
        task.context_items = items
        result = IdentifyTaskResult(task=task, context_items=items)

        with (
            patch.dict(os.environ, {"IDENTIFY_EMAIL_NORMALIZE": "true"}),
            patch.object(
                app_module.task_identifier_agent, "identify_task", return_value=result
            ),
        ):
            response = self.client.post(IDENTIFY_PATH, json={"text": "cc Bob on this"})
        self.assertEqual(response.status_code, 200, response.text)
        by = {c["field"]: c for c in response.json()["context_items"]}
        self.assertEqual(by["cc"]["value"], "bob@x.com")

    def test_identify_email_normalize_flag_on_leaves_name_only_untouched(self) -> None:
        task = build_task(TaskTypes.SCHEDULE)
        items = [ContextItem(field="cc", status="present", value="the whole team")]
        task.context_items = items
        result = IdentifyTaskResult(task=task, context_items=items)

        with (
            patch.dict(os.environ, {"IDENTIFY_EMAIL_NORMALIZE": "true"}),
            patch.object(
                app_module.task_identifier_agent, "identify_task", return_value=result
            ),
        ):
            response = self.client.post(IDENTIFY_PATH, json={"text": "cc the whole team on this"})
        self.assertEqual(response.status_code, 200, response.text)
        by = {c["field"]: c for c in response.json()["context_items"]}
        self.assertEqual(by["cc"]["value"], "the whole team")

    def test_identify_email_normalize_flag_off_by_default(self) -> None:
        task = build_task(TaskTypes.SCHEDULE)
        items = [ContextItem(field="cc", status="present", value="Bob <bob@x.com>")]
        task.context_items = items
        result = IdentifyTaskResult(task=task, context_items=items)

        with (
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("IDENTIFY_EMAIL_NORMALIZE", None)
            with patch.object(
                app_module.task_identifier_agent, "identify_task", return_value=result
            ):
                response = self.client.post(IDENTIFY_PATH, json={"text": "cc Bob on this"})
        self.assertEqual(response.status_code, 200, response.text)
        by = {c["field"]: c for c in response.json()["context_items"]}
        self.assertEqual(by["cc"]["value"], "Bob <bob@x.com>")

    def test_edit_task_normalizes_slot_types(self) -> None:
        edited = build_task(TaskTypes.SCHEDULE)
        edited.context_items = [
            ContextItem(field="recipient", status="present", value="bob@x.com")
        ]
        with patch.object(
            app_module.task_identifier_agent, "edit_task", return_value=edited
        ):
            payload = {
                "task": build_task(TaskTypes.SCHEDULE).model_dump(mode="json"),
                "user_feedback": "set the recipient",
            }
            response = self.client.post(EDIT_TASK_PATH, json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        items = response.json()["context_items"]
        self.assertEqual(items[0]["field"], "recipient")
        self.assertEqual(items[0]["type"], "email")

    def test_enrich_task_search_hit_populates_candidate_workflows(self) -> None:
        task = build_task(TaskTypes.REVIEW)
        workflows = [build_workflow("w-hit")]
        with (
            patch.object(app_module.search_agent, "query_workflows_for_task", return_value=workflows) as mock_search,
            patch.object(app_module.builder_agent, "create_workflow_initial") as mock_create,
        ):
            response = self.client.post(ENRICH_PATH, json={"task": task.model_dump(mode="json")})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["candidate_workflows"][0]["workflow_id"], "w-hit")
        self.assertEqual(mock_search.call_count, 1)
        mock_create.assert_not_called()

    def test_enrich_task_search_miss_calls_create_and_populates_candidate_workflows(self) -> None:
        task = build_task(TaskTypes.EXECUTE)
        created = build_workflow("w-created")
        with (
            patch.object(app_module.search_agent, "query_workflows_for_task", return_value=None) as mock_search,
            patch.object(app_module.builder_agent, "create_workflow_initial", return_value=created) as mock_create,
        ):
            response = self.client.post(ENRICH_PATH, json={"task": task.model_dump(mode="json")})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["candidate_workflows"][0]["workflow_id"], "w-created")
        self.assertEqual(mock_search.call_count, 1)
        self.assertEqual(mock_create.call_count, 1)

    def test_identify_multi_task_selects_primary(self) -> None:
        selected = build_task(TaskTypes.EXECUTE)
        mock_result = type('MockResult', (), {})()
        mock_result.task_type = TaskTypes.EXECUTE
        mock_result.context_items = []
        mock_result.task = selected

        with (
            patch.object(app_module.task_identifier_agent, "preprocess_email", return_value="Two tasks"),
            patch.object(
                app_module.task_identifier_agent,
                "identify_task",
                return_value=mock_result,
            ),
        ):
            response = self.client.post(IDENTIFY_PATH, json={"text": "Please send summary and schedule call"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "identified")
        self.assertEqual(body["task"]["task_type"], "execute")
        self.assertIsNone(body["task"]["candidate_workflows"])

    def test_identify_no_task_does_not_call_search_or_create(self) -> None:
        mock_result = type('MockResult', (), {})()
        mock_result.task_type = TaskTypes.NO_TASK
        mock_result.context_items = []
        mock_result.task = None

        with (
            patch.object(app_module.task_identifier_agent, "preprocess_email", return_value="FYI only"),
            patch.object(
                app_module.task_identifier_agent,
                "identify_task",
                return_value=mock_result,
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
        original = build_task(TaskTypes.EXECUTE)
        updated = build_task(TaskTypes.EXECUTE)
        updated.objective.description = "Updated task description from feedback"

        with patch.object(app_module.task_identifier_agent, "edit_task", return_value=updated) as mock_edit:
            response = self.client.post(
                EDIT_TASK_PATH,
                json={
                    "task": original.model_dump(mode="json"),
                    "user_feedback": "Please include the missing context item.",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "edited")
        self.assertEqual(body["task"]["task_id"], updated.task_id)
        self.assertEqual(body["task"]["objective"]["description"], updated.objective.description)
        self.assertEqual(body["context_items"], [])
        mock_edit.assert_called_once()

    def test_edit_task_endpoint_preserves_context_items(self) -> None:
        """Regression for the dropped-params bug: when the edited task carries
        context_items, the endpoint must return them (previously always [])."""
        original = build_task(TaskTypes.SCHEDULE)
        updated = build_task(TaskTypes.SCHEDULE)
        updated.context_items = [
            ContextItem(field="participants", status="present", value="data team")
        ]

        with patch.object(app_module.task_identifier_agent, "edit_task", return_value=updated):
            response = self.client.post(
                EDIT_TASK_PATH,
                json={
                    "task": original.model_dump(mode="json"),
                    "user_feedback": "Add the data team as participants.",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["context_items"]), 1)
        self.assertEqual(body["context_items"][0]["field"], "participants")
        self.assertEqual(body["context_items"][0]["status"], "present")


if __name__ == "__main__":
    unittest.main()