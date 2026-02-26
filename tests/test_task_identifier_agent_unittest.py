import unittest

from utils.task import TaskTypes
from utils.task_identifier_agent import DeadlineResult, IntentTag, TagResult, TaskIdentifierAgent


class StubStructuredModel:
    def __init__(self, output: object) -> None:
        self.output = output

    def invoke(self, _input: str) -> object:
        return self.output


class TaskIdentifierAgentTests(unittest.TestCase):
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

    def test_tags_to_tasks_maps_commitment_and_escalation(self) -> None:
        agent = self.make_agent()
        result = TagResult(
            tags=[
                IntentTag(tag="commitment-track", short_description="track promised delivery date"),
                IntentTag(tag="escalation-urgent", short_description="resolve production blocker immediately"),
            ]
        )
        tasks = agent.tags_to_tasks(
            tag_result=result,
            processed_text="Body text",
            metadata={"source": "email"},
        )
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].task_type, TaskTypes.COMMITMENT_TRACK)
        self.assertEqual(tasks[1].task_type, TaskTypes.ESCALATION_URGENT)
        self.assertEqual(list(tasks[0].objective.inputs.keys()), ["processed_text"])

    def test_schedule_task_has_deadline_guardrail(self) -> None:
        agent = self.make_agent()
        agent.deadline_model = StubStructuredModel(
            DeadlineResult(
                has_deadline=True,
                deadline_iso="2026-02-24T17:00:00+00:00",
                rationale="explicit deadline",
            )
        )
        result = TagResult(tags=[IntentTag(tag="schedule", short_description="schedule roadmap meeting")])
        tasks = agent.tags_to_tasks(result, "Schedule a meeting by 5pm today.", {"source": "email"})
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].objective.deadline, "2026-02-24T17:00:00+00:00")
        self.assertEqual(
            tasks[0].objective.constraints.get("latest_scheduling_time"),
            "2026-02-24T17:00:00+00:00",
        )


if __name__ == "__main__":
    unittest.main()
