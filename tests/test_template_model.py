"""Unit tests for Phase 3 template models + the executor-boundary adapter.

Fully offline (no chroma / no LLM). The key contract test is that a template or
instance always materializes to a flat Workflow with `steps: List[str]`.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from utils.task import ContextItem, Objective, Status, Task, TaskTypes, Workflow
from utils.template import EnrichedInstance, SlotSpec, Step, WorkflowTemplate


def _task(context_items=None) -> Task:
    return Task(
        task_id="t1",
        task_type=TaskTypes.SCHEDULE,
        objective=Objective(
            objective_id="o1",
            name="n",
            description="d",
            inputs={},
            success_criteria="s",
            expected_output={},
        ),
        status=Status.PENDING,
        context_items=context_items,
    )


def test_step_render_substitutes_and_preserves_unbound():
    s = Step(text="Email {recipient} about {topic}")
    assert s.render({"recipient": "a@b.com"}) == "Email a@b.com about {topic}"


def test_to_workflow_is_flat_list_of_str():
    t = WorkflowTemplate(
        name="Schedule",
        description="d",
        required_slots=[SlotSpec(name="recipient")],
        steps=[Step(text="Find a time"), Step(text="Invite {recipient}")],
    )
    wf = t.to_workflow({"recipient": "a@b.com"})

    assert isinstance(wf, Workflow)
    assert wf.steps == ["Find a time", "Invite a@b.com"]
    # Executor contract: every step must be a plain string.
    assert all(isinstance(s, str) for s in wf.steps)


def test_from_workflow_bridges_and_infers_slots_from_task():
    task = _task(
        [
            ContextItem(field="recipient", status="missing"),
            ContextItem(field="topic", status="present", value="Q3"),
        ]
    )
    wf = Workflow(workflow_id="w1", name="n", description="d", steps=["a", "b"])
    t = WorkflowTemplate.from_workflow(wf, task=task)

    assert [s.text for s in t.steps] == ["a", "b"]
    slots = {s.name: s.required for s in t.required_slots}
    # A value the email already supplied ("present") is not a required slot.
    assert slots == {"recipient": True, "topic": False}
    assert t.status == "candidate"


def test_missing_slots():
    t = WorkflowTemplate(
        name="n",
        required_slots=[SlotSpec(name="a"), SlotSpec(name="b", required=False)],
        steps=[],
    )
    assert t.missing_slots({}) == ["a"]
    assert t.missing_slots({"a": "x"}) == []


def test_enriched_instance_records_lineage_and_materializes():
    t = WorkflowTemplate(
        template_id="tmpl_1",
        version=3,
        name="n",
        description="d",
        required_slots=[SlotSpec(name="recipient")],
        steps=[Step(text="Invite {recipient}")],
    )
    inst = EnrichedInstance.from_template(
        t, bound_slots={"recipient": "a@b.com"}, task_id="task_9"
    )

    assert inst.template_id == "tmpl_1"
    assert inst.template_version == 3  # exact version recorded (lineage)
    assert inst.task_id == "task_9"
    assert inst.missing_slots == []
    assert inst.steps == ["Invite a@b.com"]
    assert inst.to_workflow().steps == ["Invite a@b.com"]


def test_enriched_instance_reports_missing_slots():
    t = WorkflowTemplate(
        name="n",
        required_slots=[SlotSpec(name="recipient")],
        steps=[Step(text="Invite {recipient}")],
    )
    inst = EnrichedInstance.from_template(t, bound_slots={})

    assert inst.missing_slots == ["recipient"]
    # Unbound placeholder is preserved, not blanked.
    assert inst.steps == ["Invite {recipient}"]
