"""Unit tests for AnalyzerAgent and /analyze_traces endpoint.

Tests include realistic trace data representing full email->workflow flows
including feedback steps.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

try:
    from fastapi.testclient import TestClient
    import app as app_module

    HAS_ENDPOINT_DEPS = True
except ModuleNotFoundError:
    HAS_ENDPOINT_DEPS = False

from agents.analyzer_agent import (
    ANALYZER_PATH,
    AnalysisResult,
    AnalyzerAgent,
    TraceData,
)


def build_email_to_workflow_traces() -> list[TraceData]:
    """Build realistic traces representing a full email->workflow flow.

    Simulates:
    1. Email received -> task identified
    2. Workflow searched -> created
    3. Workflow edited based on feedback
    4. Task enriched
    """
    return [
        # Trace 1: Initial task identification from email
        TraceData(
            trace_id="trace_001",
            input="Please send the weekly status report by end of day",
            output=json.dumps({
                "task_type": "execute",
                "priority": "normal",
                "objective_name": "Send weekly status",
                "objective_description": "Email weekly status report",
            }),
            metadata={
                "agent": "task_identifier",
                "deadline_detected": True,
                "deadline": "2024-01-15T17:00:00",
            },
            spans=[
                {"name": "preprocess_email", "input": "email", "output": "cleaned"},
                {"name": "extract_task", "input": "cleaned", "output": "task_data"},
            ],
        ),
        # Trace 2: Workflow search (no matches found)
        TraceData(
            trace_id="trace_002",
            input="Task: Send weekly status",
            output=json.dumps({"workflows_found": 0, "query": "status report workflow"}),
            metadata={
                "agent": "search_agent",
                "collection": "manual_workflows",
                "results": 0,
            },
            spans=[
                {"name": "embed_query", "input": "task", "output": "embedding"},
                {"name": "query_chroma", "input": "embedding", "output": []},
            ],
        ),
        # Trace 3: Workflow created
        TraceData(
            trace_id="trace_003",
            input="Create workflow for: Send weekly status",
            output=json.dumps({
                "workflow_id": "wf_status_001",
                "name": "Send status report",
                "steps": [
                    "Gather weekly data",
                    "Draft status email",
                    "Send to team"
                ],
            }),
            metadata={
                "agent": "builder_agent",
                "workflow_created": True,
                "steps_count": 3,
            },
            spans=[
                {"name": "build_prompt", "input": "task", "output": "prompt"},
                {"name": "generate_workflow", "input": "prompt", "output": "workflow"},
            ],
        ),
        # Trace 4: User feedback on workflow
        TraceData(
            trace_id="trace_004",
            input="Feedback: Keep it under 200 words, mention blockers",
            output=json.dumps({
                "workflow_id": "wf_status_001_edited",
                "name": "Send concise status report",
                "steps": [
                    "Summarize weekly data (max 100 words)",
                    "List blockers briefly",
                    "Draft email",
                    "Verify <200 words",
                    "Send to team"
                ],
            }),
            metadata={
                "agent": "builder_agent",
                "workflow_edited": True,
                "feedback_applied": True,
                "steps_count": 5,
            },
            spans=[
                {"name": "parse_feedback", "input": "feedback", "output": "changes"},
                {"name": "edit_workflow", "input": "workflow+changes", "output": "updated"},
            ],
        ),
        # Trace 5: Task enrichment
        TraceData(
            trace_id="trace_005",
            input="Enrich task with workflows",
            output=json.dumps({
                "task_id": "task_001",
                "candidate_workflows": ["wf_status_001_edited"],
                "enrichment_complete": True,
            }),
            metadata={
                "agent": "enrich_task",
                "candidate_count": 1,
            },
            spans=[
                {"name": "search_workflows", "input": "task", "output": "candidates"},
                {"name": "attach_workflow", "input": "task+candidates", "output": "enriched_task"},
            ],
        ),
    ]


def build_schedule_meeting_traces() -> list[TraceData]:
    """Build traces for a scheduling task flow."""
    return [
        TraceData(
            trace_id="trace_006",
            input="Schedule 30min sync with team for tomorrow afternoon",
            output=json.dumps({
                "task_type": "schedule",
                "priority": "high",
            }),
            metadata={
                "agent": "task_identifier",
                "task_complexity": "multi_participant",
            },
            spans=[
                {"name": "extract_task", "input": "email", "output": "schedule_task"},
            ],
        ),
        TraceData(
            trace_id="trace_007",
            input="Task: 30min team sync",
            output=json.dumps({"workflows_found": 2, "top_match_score": 0.94}),
            metadata={
                "agent": "search_agent",
                "collection": "manual_workflows",
                "results": 2,
            },
            spans=[
                {"name": "semantic_search", "input": "query", "output": "results"},
            ],
        ),
    ]


class AnalyzerAgentUnitTests(unittest.TestCase):
    """Unit tests for AnalyzerAgent in isolation."""

    def make_agent(self) -> AnalyzerAgent:
        agent = AnalyzerAgent.__new__(AnalyzerAgent)
        return agent

    @patch.object(AnalyzerAgent, "fetch_traces_by_thread")
    def test_analyze_traces_empty_thread(self, mock_fetch) -> None:
        """Test that thread with no traces returns no_insights."""
        mock_fetch.return_value = []
        agent = self.make_agent()
        result = agent.analyze_traces("empty_thread_id")
        self.assertEqual(result.status, "no_insights")
        self.assertIn("No traces found", result.summary)

    @patch.object(AnalyzerAgent, "__init__", lambda x: None)
    @patch.object(AnalyzerAgent, "fetch_traces_by_thread")
    @patch("agents.analyzer_agent.extract_structured_output")
    def test_analyze_traces_success(self, mock_extract, mock_fetch) -> None:
        """Test successful analysis with mock LLM response."""
        from agents.analyzer_agent import KnowledgeUpdate

        traces = build_email_to_workflow_traces()
        mock_fetch.return_value = traces

        mock_extract.return_value = KnowledgeUpdate(
            user_preferences="[2024-01-15 14:32] Prefers concise communications",
            task_patterns="[2024-01-15 14:32] Common: status reports with blockers",
            workflow_trends="[2024-01-15 14:32] 5-step workflows most successful",
        )

        agent = AnalyzerAgent.__new__(AnalyzerAgent)

        # Create a mock agent that returns a valid result
        mock_langchain_agent = MagicMock()
        mock_langchain_agent.invoke.return_value = MagicMock()
        agent.agent = mock_langchain_agent

        result = agent.analyze_traces("test_thread_id")

        self.assertEqual(result.status, "success")
        self.assertTrue(len(result.user_preferences_added) > 0)
        self.assertTrue(len(result.task_patterns_added) > 0)
        self.assertTrue(len(result.workflow_trends_added) > 0)


@unittest.skipUnless(HAS_ENDPOINT_DEPS, "Endpoint dependencies are unavailable")
class AnalyzerEndpointTests(unittest.TestCase):
    """Endpoint tests for /analyze_traces."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app_module.app)

    @patch.object(AnalyzerAgent, "fetch_traces_by_thread")
    def test_analyze_traces_endpoint_empty_thread(self, mock_fetch) -> None:
        """Test endpoint with thread that has no traces returns no_insights."""
        mock_fetch.return_value = []

        response = self.client.post(
            ANALYZER_PATH,
            json={"thread_id": "empty_thread_001"}
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "no_insights")

    @patch.object(AnalyzerAgent, "fetch_traces_by_thread")
    def test_analyze_traces_endpoint_full_flow(self, mock_fetch) -> None:
        """Test endpoint fetches traces from thread and analyzes them."""
        traces = build_email_to_workflow_traces()
        mock_fetch.return_value = traces

        response = self.client.post(
            ANALYZER_PATH,
            json={"thread_id": "test_thread_001"}
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("status", body)
        self.assertIn("summary", body)
        self.assertIn("files_updated", body)
        mock_fetch.assert_called_once_with("test_thread_001")


if __name__ == "__main__":
    unittest.main()
