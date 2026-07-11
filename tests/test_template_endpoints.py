"""Endpoint tests for the Phase 3 template endpoints.

Offline: the builder and template store are patched, so no OpenAI key / network is
needed. Mirrors tests/test_create_workflow_search_first.py.
"""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi.testclient import TestClient

import app as app_module
from utils.task import Objective, Status, Task, TaskTypes, Workflow
from utils.template import SlotSpec, Step, WorkflowTemplate


def build_task() -> Task:
    return Task(
        task_id="task_t1",
        task_type=TaskTypes.EXECUTE,
        objective=Objective(
            objective_id="obj_1",
            name="test",
            description="test objective",
            inputs={},
            success_criteria="done",
            expected_output={"status": "completed"},
        ),
        status=Status.PENDING,
    )


def build_template(template_id="tmpl_1", version=1) -> WorkflowTemplate:
    return WorkflowTemplate(
        template_id=template_id,
        name="Sched",
        description="d",
        version=version,
        required_slots=[SlotSpec(name="recipient")],
        steps=[Step(text="Find time"), Step(text="Invite {recipient}")],
    )


class CreateTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app_module.app)

    def test_create_generates_and_persists(self):
        wf = Workflow(workflow_id="w1", name="Sched", description="d", steps=["Find time", "Invite {recipient}"])
        with (
            patch.object(app_module.builder_agent, "create_workflow_initial", return_value=wf) as mock_build,
            patch.object(app_module.template_store, "add_template", return_value="doc1") as mock_add,
        ):
            resp = self.client.post("/create_template", json={"task": build_task().model_dump(mode="json")})

        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual([s["text"] for s in body["steps"]], ["Find time", "Invite {recipient}"])
        mock_build.assert_called_once()
        mock_add.assert_called_once()

    def test_create_threads_scope_to_template(self):
        wf = Workflow(workflow_id="w1", name="Sched", description="d", steps=["Find time"])
        with (
            patch.object(app_module.builder_agent, "create_workflow_initial", return_value=wf),
            patch.object(app_module.template_store, "add_template", return_value="doc1"),
        ):
            resp = self.client.post(
                "/create_template",
                json={"task": build_task().model_dump(mode="json"), "scope": "user:U1"},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["scope"], "user:U1")

    def test_create_default_scope_is_global(self):
        wf = Workflow(workflow_id="w1", name="Sched", description="d", steps=["Find time"])
        with (
            patch.object(app_module.builder_agent, "create_workflow_initial", return_value=wf),
            patch.object(app_module.template_store, "add_template", return_value="doc1"),
        ):
            resp = self.client.post("/create_template", json={"task": build_task().model_dump(mode="json")})
        self.assertEqual(resp.json()["scope"], "global")

    def test_create_reuses_within_threshold_and_skips_build(self):
        existing = build_template("reused")
        match = {"template": existing, "distance": 0.1, "score": 0.9}
        with (
            patch.object(app_module.template_store, "search_templates", return_value=[match]) as mock_search,
            patch.object(app_module.builder_agent, "create_workflow_initial") as mock_build,
        ):
            resp = self.client.post(
                "/create_template",
                json={"task": build_task().model_dump(mode="json"), "max_distance": 0.5},
            )

        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["template_id"], "reused")
        mock_search.assert_called_once()
        mock_build.assert_not_called()


class SearchTemplatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app_module.app)

    def test_search_returns_matches(self):
        match = {"template": build_template("m1"), "distance": 0.2, "score": 0.83}
        with patch.object(app_module.template_store, "search_templates", return_value=[match]):
            resp = self.client.post("/search_templates", json={"query": "schedule a meeting"})

        self.assertEqual(resp.status_code, 200, resp.text)
        matches = resp.json()["matches"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["template"]["template_id"], "m1")
        self.assertEqual(matches[0]["score"], 0.83)

    def test_search_forwards_scope_preference(self):
        match = {"template": build_template("m1"), "distance": 0.2, "score": 0.83}
        with patch.object(
            app_module.template_store, "search_templates", return_value=[match]
        ) as mock_search:
            resp = self.client.post(
                "/search_templates",
                json={"query": "schedule", "scope": ["user:U1", "global"]},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        # The ordered scope preference reached the store.
        self.assertEqual(mock_search.call_args.kwargs["scope"], ["user:U1", "global"])

    def test_search_without_query_or_task_is_400(self):
        resp = self.client.post("/search_templates", json={})
        self.assertEqual(resp.status_code, 400)


class EnrichTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app_module.app)

    def test_enrich_binds_slots_and_returns_flat_workflow(self):
        with patch.object(app_module.template_store, "get_template", return_value=build_template()):
            resp = self.client.post(
                "/enrich_template",
                json={"template_id": "tmpl_1", "bound_slots": {"recipient": "a@b.com"}, "task_id": "task_9"},
            )

        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["instance"]["template_id"], "tmpl_1")
        self.assertEqual(body["instance"]["task_id"], "task_9")
        self.assertEqual(body["instance"]["missing_slots"], [])
        # The flat workflow the executor runs — plain strings, slot substituted.
        self.assertEqual(body["workflow"]["steps"], ["Find time", "Invite a@b.com"])

    def test_enrich_reports_missing_slots(self):
        with patch.object(app_module.template_store, "get_template", return_value=build_template()):
            resp = self.client.post(
                "/enrich_template", json={"template_id": "tmpl_1", "bound_slots": {}}
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["instance"]["missing_slots"], ["recipient"])

    def test_enrich_unknown_template_is_404(self):
        with patch.object(app_module.template_store, "get_template", return_value=None):
            resp = self.client.post("/enrich_template", json={"template_id": "nope"})
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
