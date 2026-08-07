from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from agents.analyzer_agent import AnalysisResult, AnalyzerAgent, TraceData
from agents.builder_agent import BuilderAgent
from agents.search_agent import SearchAgent
from utils.task import Task, TaskTypes, Workflow
from agents.task_agent import ContextItem, Metadata, TaskIdentifierAgent
from agents.intent_agent import IntentClassifierAgent, IntentLabel, intent_router_enabled
from utils.population import auto_populate_enabled, populate_context_items
from utils.slots import (
    normalize_slots,
    normalize_slot_values,
    tz_normalize_enabled,
    normalize_duration_slots,
    duration_normalize_enabled,
    normalize_email_slots,
    email_normalize_enabled,
)
from utils.template import EnrichedInstance, WorkflowTemplate
from utils.config import make_template_store, make_workflow_store, make_instance_store
from utils.artifact_client import artifacts_enabled, auto_promote_enabled, post_template
from utils.artifact_envelope import GLOBAL_USER_ID
from utils.safety import SafetyFinding, SafetyReport, scan_template

app = FastAPI(title="Agent Infrastructure API")

workflow_store = make_workflow_store()
builder_agent = BuilderAgent(vector_db=workflow_store)
search_agent = SearchAgent(vector_db=workflow_store)
task_identifier_agent = TaskIdentifierAgent()
intent_classifier_agent = IntentClassifierAgent()
template_store = make_template_store()
instance_store = make_instance_store()


class CreateWorkflowRequest(BaseModel):
    task: Task
    rejected_workflows: Optional[List[Workflow]] = None
    user_feedback: Optional[str] = None
    thread_id: Optional[str] = None


class EditWorkflowRequest(BaseModel):
    task: Task
    proposed_workflow: Workflow
    feedback: str
    thread_id: Optional[str] = None


class EditTaskRequest(BaseModel):
    task: Task
    user_feedback: str
    thread_id: Optional[str] = None


class IdentifyTaskRequest(BaseModel):
    text: str = Field(..., min_length=1)
    subject: Optional[str] = None
    metadata: Optional[Metadata] = None
    thread_id: Optional[str] = None


class IdentifyTaskResponse(BaseModel):
    status: Literal["identified", "no_task"]
    task: Optional[Task] = None
    context_items: List[ContextItem] = Field(default_factory=list)


class ClassifyIntentRequest(BaseModel):
    text: str = Field(..., min_length=1)
    phase: str = "task"
    thread_id: Optional[str] = None


class ClassifyIntentResponse(BaseModel):
    intent: Optional[IntentLabel] = None
    status: Literal["classified", "disabled", "error"] = "classified"


class EditTaskResponse(BaseModel):
    status: Literal["edited"]
    task: Optional[Task] = None
    context_items: List[ContextItem] = Field(default_factory=list)


class PopulateTaskContextRequest(BaseModel):
    task: Task
    thread_id: Optional[str] = None


class PopulateWorkflowsRequest(BaseModel):
    workflows: List[Workflow] = Field(default_factory=list)


class PopulateWorkflowsResponse(BaseModel):
    inserted_count: int
    document_ids: List[str]


class AddWorkflowRequest(BaseModel):
    workflow: Workflow
    is_generated: bool = False




class ListWorkflowsResponse(BaseModel):
    workflows: List[Workflow]


@app.get("/health")
def health_check():
    return {"status": "ok"}


class SearchWorkflowsRequest(BaseModel):
    task: Task
    thread_id: Optional[str] = None


@app.post("/search_workflows", response_model=List[Workflow] | None)
def search_workflows_endpoint(
    request: SearchWorkflowsRequest,
    x_thread_id: Optional[str] = Header(None),
):
    thread_id = request.thread_id or x_thread_id
    try:
        return search_agent.query_workflows_for_task(request.task, thread_id=thread_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/create_workflow", response_model=Workflow)
def create_workflow_endpoint(
    request: CreateWorkflowRequest,
    x_thread_id: Optional[str] = Header(None),
):
    thread_id = request.thread_id or x_thread_id
    try:
        # Search-before-create: on a fresh create (no rejected workflows and no
        # feedback), reuse an existing strict match instead of generating a near-dup.
        # A regeneration request — the user already saw candidates and rejected them,
        # or gave feedback — skips the search and always generates a new workflow.
        is_regeneration = bool(request.rejected_workflows) or bool(request.user_feedback)
        if not is_regeneration:
            matches = search_agent.query_workflows_for_task(
                request.task, thread_id=thread_id
            )
            if matches:  # truthy => a 95%+ match exists (best match first)
                return matches[0]
        return builder_agent.create_workflow_initial(
            request.task,
            request.rejected_workflows,
            request.user_feedback,
            thread_id=thread_id
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Phase 3: workflow templates → enriched instances (additive) ──────────────
# Templates are the versioned/parameterized form; they materialize down to the
# flat Workflow the executor consumes via WorkflowTemplate/EnrichedInstance
# .to_workflow(). The flat endpoints above are unchanged.

class CreateTemplateRequest(BaseModel):
    task: Task
    user_feedback: Optional[str] = None
    thread_id: Optional[str] = None
    # When provided, reuse an existing template within this embedding distance
    # instead of generating (threshold search-before-create). Omit to always
    # generate (no false reuse until the threshold is calibrated).
    max_distance: Optional[float] = None
    # Retrieval-preference scope for the created template: a bare level
    # ("global"|"org"|"role"|"user") or an owner-qualified label ("user:<id>").
    scope: str = "global"


class SearchTemplatesRequest(BaseModel):
    task: Optional[Task] = None
    query: Optional[str] = None
    top_k: int = 5
    max_distance: Optional[float] = None
    # Ordered scope preference, most-specific first (e.g. ["user:U1","role:R","global"]).
    # When set, more-specific-scoped templates rank ahead of closer-but-less-specific
    # ones; unscoped templates still match as a fallback. Omit for pure proximity.
    scope: Optional[List[str]] = None


class TemplateMatch(BaseModel):
    template: WorkflowTemplate
    distance: float
    score: float


class SearchTemplatesResponse(BaseModel):
    matches: List[TemplateMatch]


class EnrichTemplateRequest(BaseModel):
    template_id: str
    version: Optional[int] = None
    bound_slots: Dict[str, str] = Field(default_factory=dict)
    task_id: Optional[str] = None
    specialization_scope: Optional[str] = None


class EnrichTemplateResponse(BaseModel):
    instance: EnrichedInstance
    workflow: Workflow  # flat, ready for the executor's /workflow/execute


@app.post("/create_template", response_model=WorkflowTemplate)
def create_template_endpoint(
    request: CreateTemplateRequest,
    x_thread_id: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Generate a versioned template for a task (threshold search-before-create).

    Reuses the builder to produce steps, then wraps them as a typed template with
    slots inferred from the task. Persists the new template as a `draft` (the
    Artifact-envelope initial state; promoted to candidate/trusted via the gate).

    Best-effort auto-promotion (``ARTIFACT_AUTO_PROMOTE``, off by default): when
    enabled alongside ``EXECUTOR_ARTIFACTS_URL`` and the caller's identity headers,
    a *newly generated* template is also written to the executor's artifact store
    as a draft, so recurring patterns reach the eval gate without a manual
    ``/promote_template`` call. Never affects this endpoint's response or status.
    """
    thread_id = request.thread_id or x_thread_id
    try:
        is_regeneration = bool(request.user_feedback)
        if not is_regeneration and request.max_distance is not None:
            matches = template_store.search_templates(
                request.task.to_string(), top_k=1, max_distance=request.max_distance
            )
            if matches:
                # Reused an existing template — nothing new was created, so there
                # is nothing to auto-promote.
                return matches[0]["template"]

        workflow = builder_agent.create_workflow_initial(
            request.task, None, request.user_feedback, thread_id=thread_id
        )
        template = WorkflowTemplate.from_workflow(
            workflow, task=request.task, scope=request.scope
        )
        template_store.add_template(template)
        if artifacts_enabled() and auto_promote_enabled():
            if x_user_id and authorization:
                try:
                    post_template(
                        template,
                        user_id=GLOBAL_USER_ID,
                        x_user_id=x_user_id,
                        source_trace_ids=[thread_id] if thread_id else [],
                        authorization=authorization,
                        thread_id=thread_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[create_template] auto-promote failed: {exc}")
            else:
                print("[create_template] auto-promote skipped: missing X-User-Id/Authorization")
        return template
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/search_templates", response_model=SearchTemplatesResponse)
def search_templates_endpoint(request: SearchTemplatesRequest):
    """Score-based template search (returns distance + monotonic score per match)."""
    query = request.query or (request.task.to_string() if request.task else "")
    if not query.strip():
        raise HTTPException(status_code=400, detail="Provide a task or a query")
    try:
        matches = template_store.search_templates(
            query,
            top_k=request.top_k,
            max_distance=request.max_distance,
            scope=request.scope,
        )
        return SearchTemplatesResponse(matches=[TemplateMatch(**m) for m in matches])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/enrich_template", response_model=EnrichTemplateResponse)
def enrich_template_endpoint(
    request: EnrichTemplateRequest,
    x_thread_id: Optional[str] = Header(None),
):
    """Bind slots to a template → an EnrichedInstance + the flat Workflow to run.

    Records the exact template_id@version (lineage) and any still-missing required
    slots so the caller can fall back to HITL.
    """
    template = template_store.get_template(request.template_id, version=request.version)
    if template is None:
        raise HTTPException(status_code=404, detail="Template not found")
    instance = EnrichedInstance.from_template(
        template,
        bound_slots=request.bound_slots,
        task_id=request.task_id,
        specialization_scope=request.specialization_scope,
    )
    # Best-effort lineage persistence (no-op unless STORE_BACKEND=pg). A storage
    # failure must never fail enrichment — mirrors builder_agent._persist_generated.
    try:
        instance_store.add_instance(instance, trace_id=x_thread_id)
    except Exception as exc:  # noqa: BLE001
        print(f"[enrich_template] failed to persist instance: {exc}")
    return EnrichTemplateResponse(instance=instance, workflow=instance.to_workflow())


class RefineTemplateRequest(BaseModel):
    template_id: str
    version: Optional[int] = None
    user_feedback: str = Field(..., min_length=1)
    task: Task
    source_trace_ids: List[str] = Field(default_factory=list)
    thread_id: Optional[str] = None


class RefineTemplateResponse(BaseModel):
    template: WorkflowTemplate
    safety: SafetyReport
    promoted: Optional[Dict] = None


@app.post("/refine_template", response_model=RefineTemplateResponse)
def refine_template_endpoint(
    request: RefineTemplateRequest,
    x_thread_id: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Distill a parent template into a refined child with lineage (P-LEARN1).

    The child is a **new** lineage (its own ``template_id``, ``version=1``) with
    ``parent_id`` set to the parent's ``template_id`` — ``to_envelope`` maps that to
    the executor's ``parent_artifact_id``, so the executor can trace a candidate
    back to what it was distilled from.

    Safety gate (P-SEC3): the child is only ever persisted as ``candidate`` when
    ``scan_template`` finds no ``block``-severity content; otherwise it is persisted
    as ``draft`` and never auto-promoted, so a hostile email cannot ride user
    feedback into a higher-trust shared artifact.
    """
    thread_id = request.thread_id or x_thread_id
    parent = template_store.get_template(request.template_id, version=request.version)
    if parent is None:
        raise HTTPException(status_code=404, detail="Template not found")

    try:
        refined = builder_agent.edit_proposed_workflow(
            request.task, parent.to_workflow(), request.user_feedback, thread_id=thread_id
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    child = WorkflowTemplate.from_workflow(refined, task=request.task, scope=parent.scope)
    child.parent_id = parent.template_id  # lineage: this is a distillation, not a fresh template

    safety = scan_template(child)
    if safety.safe:
        child.status = "candidate"
    # else: leave the default "draft" — unsafe content never reaches candidate.

    template_store.add_template(child)

    promoted = None
    if safety.safe and artifacts_enabled() and auto_promote_enabled():
        if x_user_id and authorization:
            try:
                promoted = post_template(
                    child,
                    user_id=GLOBAL_USER_ID,
                    x_user_id=x_user_id,
                    source_trace_ids=request.source_trace_ids or ([thread_id] if thread_id else []),
                    authorization=authorization,
                    thread_id=thread_id,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[refine_template] auto-promote failed: {exc}")
        else:
            print("[refine_template] auto-promote skipped: missing X-User-Id/Authorization")

    return RefineTemplateResponse(template=child, safety=safety, promoted=promoted)


class PromoteTemplateRequest(BaseModel):
    template_id: str
    version: Optional[int] = None
    source_trace_ids: List[str] = Field(default_factory=list)


class PromoteTemplateResponse(BaseModel):
    status: Literal["written", "disabled", "error", "blocked"]
    artifact: Optional[Dict] = None
    detail: Optional[str] = None
    findings: List[SafetyFinding] = Field(default_factory=list)


@app.post("/promote_template", response_model=PromoteTemplateResponse)
def promote_template_endpoint(
    request: PromoteTemplateRequest,
    x_user_id: Optional[str] = Header(None),
    x_thread_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Publish a stored template to the executor's artifact store as a **draft**.

    This writes the template (via ``to_envelope``) to the executor's ``POST
    /artifacts`` so it can be promoted through the executor's eval gate and read
    back at execute time (P-LEARN1). It does **not** promote to ``trusted`` — that
    is the executor's gate, not ours. (Named ``/promote_template`` per convergence.md.)

    Flag-gated: a no-op returning ``status="disabled"`` unless ``EXECUTOR_ARTIFACTS_URL``
    is set. The template is written as a **global** (``user_id="*"``) artifact so the
    executor's read-back serves it to any user. Requires the caller's Google bearer +
    ``X-User-Id`` (both forwarded to the executor, which mandates them).

    Safety-gated (P-SEC3): ``scan_template`` runs before the write. A template with
    any ``block``-severity finding is never written to the executor — the endpoint
    returns ``status="blocked"`` with the findings instead.
    """
    if not artifacts_enabled():
        return PromoteTemplateResponse(status="disabled")
    if not x_user_id or not authorization:
        raise HTTPException(
            status_code=400,
            detail="X-User-Id and Authorization headers are required",
        )
    template = template_store.get_template(request.template_id, version=request.version)
    if template is None:
        raise HTTPException(status_code=404, detail="Template not found")

    safety = scan_template(template)
    if not safety.safe:
        return PromoteTemplateResponse(
            status="blocked",
            detail="Template failed the safety scan",
            findings=safety.findings,
        )

    artifact = post_template(
        template,
        user_id=GLOBAL_USER_ID,
        x_user_id=x_user_id,
        source_trace_ids=request.source_trace_ids,
        authorization=authorization,
        thread_id=x_thread_id,
    )
    if artifact is None:
        return PromoteTemplateResponse(
            status="error", detail="Executor artifact write failed"
        )
    return PromoteTemplateResponse(status="written", artifact=artifact)


@app.post("/edit_workflow", response_model=Workflow)
def edit_workflow_endpoint(
    request: EditWorkflowRequest,
    x_thread_id: Optional[str] = Header(None),
):
    thread_id = request.thread_id or x_thread_id
    try:
        return builder_agent.edit_proposed_workflow(
            request.task,
            request.proposed_workflow,
            request.feedback,
            thread_id=thread_id
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/edit_task", response_model=EditTaskResponse)
def edit_task_endpoint(
    request: EditTaskRequest,
    x_thread_id: Optional[str] = Header(None),
):
    thread_id = request.thread_id or x_thread_id
    try:
        edited_task = task_identifier_agent.edit_task(request.task, request.user_feedback, thread_id=thread_id)
        # A conversation edit round-trips the full typed slot signature: re-fill `type`
        # on any slot the editor left untyped (idempotent; explicit types preserved).
        edited_task.context_items = normalize_slots(edited_task.context_items)
        if tz_normalize_enabled():
            edited_task.context_items = normalize_slot_values(edited_task.context_items)
        if duration_normalize_enabled():
            edited_task.context_items = normalize_duration_slots(edited_task.context_items)
        if email_normalize_enabled():
            edited_task.context_items = normalize_email_slots(edited_task.context_items)
        context_items = edited_task.context_items or []
        return EditTaskResponse(
            status="edited",
            task=edited_task,
            context_items=context_items,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/populate_task_context", response_model=Task)
def populate_task_context_endpoint(
    request: PopulateTaskContextRequest,
    x_user_id: Optional[str] = Header(None),
    x_thread_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Fill a task's *missing* parameters from user context (memory-unit) before HITL.

    Additive + flag-gated: when ``MEMORY_URL`` is unset (or memory-unit is
    unreachable) the task is returned unchanged. Only ``missing`` slots are
    touched — email-provided values are preserved — and resolved values are
    marked ``guessed`` with a ``source``/``confidence`` so the UI can confirm them.

    The caller's ``Authorization`` bearer is forwarded to memory-unit so
    ``/resolve`` passes when memory-unit validates tokens.
    """
    thread_id = request.thread_id or x_thread_id
    try:
        return populate_context_items(
            request.task,
            user_id=x_user_id,
            thread_id=thread_id,
            authorization=authorization,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# identify task and then return candidate workflows
@app.post("/identify_task", response_model=IdentifyTaskResponse)
def identify_task_endpoint(
    request: IdentifyTaskRequest,
    x_user_id: Optional[str] = Header(None),
    x_thread_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    thread_id = request.thread_id or x_thread_id
    try:
        identification = task_identifier_agent.identify_task(
            text=request.text,
            subject=request.subject,
            metadata=request.metadata,
            thread_id=thread_id,
        )

        task = identification.task
        if task is None or task.task_type == TaskTypes.NO_TASK:
            return IdentifyTaskResponse(
                status="no_task",
                task=None,
                context_items=identification.context_items,
            )

        # Optionally fill missing slots from user context before HITL. Off by
        # default (MEMORY_AUTO_POPULATE); a no-op unless MEMORY_URL is also set.
        if auto_populate_enabled():
            task = populate_context_items(
                task,
                user_id=x_user_id,
                thread_id=thread_id,
                authorization=authorization,
            )

        # Fill the typed signature (slot `type`) deterministically before returning,
        # so every emitted slot is fully typed. Idempotent; never overwrites.
        task.context_items = normalize_slots(task.context_items)
        if tz_normalize_enabled():
            task.context_items = normalize_slot_values(task.context_items)
        if duration_normalize_enabled():
            task.context_items = normalize_duration_slots(task.context_items)
        if email_normalize_enabled():
            task.context_items = normalize_email_slots(task.context_items)

        return IdentifyTaskResponse(
            status="identified",
            task=task,
            context_items=task.context_items or identification.context_items,
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Task identification failed")


@app.post("/classify_intent", response_model=ClassifyIntentResponse)
def classify_intent_endpoint(
    request: ClassifyIntentRequest,
    x_user_id: Optional[str] = Header(None),
    x_thread_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    if not intent_router_enabled():
        return ClassifyIntentResponse(intent=None, status="disabled")
    thread_id = request.thread_id or x_thread_id
    try:
        result = intent_classifier_agent.classify(
            text=request.text, phase=request.phase, thread_id=thread_id
        )
        if result is None:
            return ClassifyIntentResponse(intent=None, status="error")
        return ClassifyIntentResponse(intent=result.intent, status="classified")
    except Exception:
        return ClassifyIntentResponse(intent=None, status="error")


class EnrichTaskRequest(BaseModel):
    task: Task
    thread_id: Optional[str] = None


@app.post("/enrich_task_with_workflows", response_model=Task)
def enrich_task_with_workflows_endpoint(
    request: EnrichTaskRequest,
    x_thread_id: Optional[str] = Header(None),
):
    thread_id = request.thread_id or x_thread_id
    candidates = search_agent.query_workflows_for_task(request.task, thread_id=thread_id)
    if candidates is None:
        created = builder_agent.create_workflow_initial(request.task, rejected_workflows=None, thread_id=thread_id)
        candidates = [created]
    request.task.candidate_workflows = candidates
    return request.task

@app.post("/populate_workflows", response_model=PopulateWorkflowsResponse)
def populate_workflows_endpoint(request: PopulateWorkflowsRequest):
    try:
        document_ids = search_agent.populate_manual_workflows(request.workflows)
        return PopulateWorkflowsResponse(
            inserted_count=len(document_ids),
            document_ids=document_ids,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/add_workflow")
def add_workflow_endpoint(request: AddWorkflowRequest):
    try:
        workflow_store.add_single_workflow(
            request.workflow,
            is_generated=request.is_generated
        )
        return {"status": "success"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/workflows", response_model=ListWorkflowsResponse)
def list_workflows_endpoint():
    try:
        workflows = workflow_store.get_all_workflows()
        return ListWorkflowsResponse(workflows=workflows)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# Analyzer Agent
analyzer_agent = AnalyzerAgent()


class AnalyzeTracesRequest(BaseModel):
    """Request to analyze traces from a thread."""
    thread_id: str


class AnalyzeTracesResponse(BaseModel):
    """Response from trace analysis."""
    status: str
    summary: str
    files_updated: List[str] = Field(default_factory=list)
    user_preferences_added: List[str] = Field(default_factory=list)
    task_patterns_added: List[str] = Field(default_factory=list)
    workflow_trends_added: List[str] = Field(default_factory=list)


@app.post("/analyze_traces", response_model=AnalyzeTracesResponse)
def analyze_traces_endpoint(request: AnalyzeTracesRequest):
    """Analyze all traces in a thread and update knowledge files.

    Fetches traces from Confident AI using the provided thread_id,
    analyzes them for patterns, and updates knowledge files with new
    insights. Existing trends are folded/strengthened rather than duplicated.
    """
    try:
        result = analyzer_agent.analyze_traces(thread_id=request.thread_id)

        return AnalyzeTracesResponse(
            status=result.status,
            summary=result.summary,
            files_updated=[
                fname for fname in [
                    "user_preferences.txt" if result.user_preferences_added else None,
                    "task_patterns.txt" if result.task_patterns_added else None,
                    "workflow_trends.txt" if result.workflow_trends_added else None,
                ] if fname is not None
            ],
            user_preferences_added=result.user_preferences_added,
            task_patterns_added=result.task_patterns_added,
            workflow_trends_added=result.workflow_trends_added,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
