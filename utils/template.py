"""Phase 3 — workflow templates and enriched instances.

A **WorkflowTemplate** is the generic, versioned, parameterized form of a
workflow: typed `Step`s plus `required_slots`. An **EnrichedInstance** is a
template with its slots bound to concrete values for a specific task.

This layer is *additive*: the flat ``Workflow`` (``utils.task.Workflow``, with
``steps: List[str]``) remains the wire format the executor consumes. Templates and
instances **materialize down** to that flat form via ``to_workflow()`` — the
executor never sees typed steps (its ``/workflow/execute`` takes ``List[str]`` and
rejects unknown fields). This is the contract-safe adapter from PIPELINE_REWORK
Phase 3.
"""

import re
import uuid
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from utils.task import Task, Workflow

StepKind = Literal["tool", "skill", "llm", "hitl", "subtemplate"]
TemplateStatus = Literal["draft", "candidate", "trusted", "deprecated"]
TemplateSource = Literal["human", "generated"]

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_]+)\}")


def _slot_present(value) -> bool:
    """A slot counts as bound iff it has a non-empty value.

    The single source of truth shared by ``Step.render`` and
    ``WorkflowTemplate.missing_slots`` so the two never disagree (e.g. a value of
    ``0`` must be treated the same by both — present, not missing)."""
    return value not in (None, "")


class Step(BaseModel):
    """A single typed workflow step.

    ``text`` is the human-readable instruction (what the flat ``List[str]`` steps
    were). It may contain ``{slot}`` placeholders that get substituted at
    materialization time. ``kind``/``ref``/``input_bindings``/``output_name`` are
    the richer metadata; they are provider-internal and dropped at the executor
    boundary.
    """

    kind: StepKind = "llm"
    text: str
    ref: Optional[str] = None
    input_bindings: Dict[str, str] = Field(default_factory=dict)
    output_name: Optional[str] = None

    def render(self, bound: Dict[str, str]) -> str:
        """Return ``text`` with ``{slot}`` placeholders substituted from ``bound``.

        Unbound placeholders are left intact so nothing is silently blanked.
        """

        def _sub(m: "re.Match") -> str:
            key = m.group(1)
            value = bound.get(key)
            return str(value) if _slot_present(value) else m.group(0)

        return _PLACEHOLDER_RE.sub(_sub, self.text)


class SlotSpec(BaseModel):
    """A parameter the template needs bound before it can run."""

    name: str
    type: str = "string"
    required: bool = True


class WorkflowTemplate(BaseModel):
    """A generic, versioned, parameterized workflow (Artifact ``kind=template``).

    Envelope fields conform to the executor's canonical ``artifacts`` table
    (``workflow_executor/services/status_store.py``), the shared source of truth:
    a freshly created/generated template is a **draft** (``trust_tier`` T0) and is
    only promoted ``draft → candidate (T1) → trusted`` through the executor's
    eval gate. ``template_id`` maps to the envelope's ``artifact_id`` and
    ``parent_id`` to ``parent_artifact_id``. ``source`` here is provider-local
    provenance; the envelope's provenance is ``source_trace_ids`` (trace lineage).
    """

    template_id: str = Field(default_factory=lambda: f"tmpl_{uuid.uuid4().hex[:8]}")
    name: str
    description: str = ""
    version: int = 1
    required_slots: List[SlotSpec] = Field(default_factory=list)
    steps: List[Step] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    parent_id: Optional[str] = None  # lineage: specialization of another template
    source: TemplateSource = "generated"
    status: TemplateStatus = "draft"  # envelope initial state; promoted via the gate

    def to_string(self) -> str:
        """Stable content representation (for embedding + content-hash dedup).

        Excludes id/version/status so two templates with identical content hash
        identically, mirroring ``Workflow.to_string()``.
        """
        lines = [f"Template: {self.name}", f"Description: {self.description}"]
        if self.required_slots:
            lines.append("Slots: " + ", ".join(s.name for s in self.required_slots))
        lines.append("Steps:")
        lines.extend(f"- {s.text}" for s in self.steps)
        return "\n".join(lines)

    def missing_slots(self, bound: Dict[str, str]) -> List[str]:
        """Required slot names not satisfied by ``bound`` (same rule as render)."""
        return [
            s.name
            for s in self.required_slots
            if s.required and not _slot_present(bound.get(s.name))
        ]

    def to_workflow(
        self, bound_slots: Optional[Dict[str, str]] = None, workflow_id: Optional[str] = None
    ) -> Workflow:
        """Materialize into a flat ``Workflow`` (``steps: List[str]``) for the executor.

        Renders each typed step to a string with slot substitution. This is the
        only form that crosses the executor boundary.
        """
        bound = bound_slots or {}
        return Workflow(
            workflow_id=workflow_id or f"wf_{uuid.uuid4().hex[:8]}",
            name=self.name,
            description=self.description,
            steps=[s.render(bound) for s in self.steps],
        )

    @classmethod
    def from_workflow(
        cls,
        workflow: Workflow,
        task: Optional[Task] = None,
        required_slots: Optional[List[SlotSpec]] = None,
        source: TemplateSource = "generated",
        status: TemplateStatus = "draft",
    ) -> "WorkflowTemplate":
        """Bridge a flat ``Workflow`` into a template (no LLM).

        Each flat step becomes an ``llm`` ``Step``. When ``required_slots`` is not
        given, they are inferred from the task's ``context_items`` (a slot per
        field, required when the email did not already supply it).
        """
        slots = required_slots
        if slots is None and task is not None and task.context_items:
            slots = [
                SlotSpec(name=ci.field, required=(ci.status != "present"))
                for ci in task.context_items
            ]
        return cls(
            name=workflow.name,
            description=workflow.description,
            required_slots=slots or [],
            steps=[Step(kind="llm", text=s) for s in workflow.steps],
            source=source,
            status=status,
        )


class EnrichedInstance(BaseModel):
    """A template with its slots bound for a specific task (Artifact kind=instance)."""

    instance_id: str = Field(default_factory=lambda: f"inst_{uuid.uuid4().hex[:8]}")
    template_id: str
    template_version: int = 1
    name: str
    description: str = ""
    bound_slots: Dict[str, str] = Field(default_factory=dict)
    specialization_scope: Optional[str] = None
    task_id: Optional[str] = None
    missing_slots: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)  # materialized flat steps

    @classmethod
    def from_template(
        cls,
        template: WorkflowTemplate,
        bound_slots: Optional[Dict[str, str]] = None,
        task_id: Optional[str] = None,
        specialization_scope: Optional[str] = None,
    ) -> "EnrichedInstance":
        """Bind slots to a template and materialize its steps.

        Records the exact ``template_id`` + ``version`` it came from (lineage) and
        any still-missing required slots (so the caller can fall back to HITL).
        """
        bound = bound_slots or {}
        materialized = template.to_workflow(bound)
        return cls(
            template_id=template.template_id,
            template_version=template.version,
            name=template.name,
            description=template.description,
            bound_slots=bound,
            specialization_scope=specialization_scope,
            task_id=task_id,
            missing_slots=template.missing_slots(bound),
            steps=materialized.steps,
        )

    def to_workflow(self, workflow_id: Optional[str] = None) -> Workflow:
        """The flat ``Workflow`` (``List[str]`` steps) to hand to the executor."""
        return Workflow(
            workflow_id=workflow_id or f"wf_{uuid.uuid4().hex[:8]}",
            name=self.name,
            description=self.description,
            steps=list(self.steps),
        )
