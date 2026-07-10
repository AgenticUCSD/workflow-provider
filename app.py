from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from agents.analyzer_agent import AnalysisResult, AnalyzerAgent, TraceData
from agents.builder_agent import BuilderAgent
from agents.search_agent import SearchAgent
from utils.task import Task, TaskTypes, Workflow
from agents.task_agent import ContextItem, Metadata, TaskIdentifierAgent
from utils.population import auto_populate_enabled, populate_context_items
from utils.template import EnrichedInstance, WorkflowTemplate
from utils.config import make_template_store, make_workflow_store, make_instance_store

app = FastAPI(title="Agent Infrastructure API")

workflow_store = make_workflow_store()
builder_agent = BuilderAgent(vector_db=workflow_store)
search_agent = SearchAgent(vector_db=workflow_store)
task_identifier_agent = TaskIdentifierAgent()
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


class SearchTemplatesRequest(BaseModel):
    task: Optional[Task] = None
    query: Optional[str] = None
    top_k: int = 5
    max_distance: Optional[float] = None


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
):
    """Generate a versioned template for a task (threshold search-before-create).

    Reuses the builder to produce steps, then wraps them as a typed template with
    slots inferred from the task. Persists the new template as a `draft` (the
    Artifact-envelope initial state; promoted to candidate/trusted via the gate).
    """
    thread_id = request.thread_id or x_thread_id
    try:
        is_regeneration = bool(request.user_feedback)
        if not is_regeneration and request.max_distance is not None:
            matches = template_store.search_templates(
                request.task.to_string(), top_k=1, max_distance=request.max_distance
            )
            if matches:
                return matches[0]["template"]

        workflow = builder_agent.create_workflow_initial(
            request.task, None, request.user_feedback, thread_id=thread_id
        )
        template = WorkflowTemplate.from_workflow(workflow, task=request.task)
        template_store.add_template(template)
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
            query, top_k=request.top_k, max_distance=request.max_distance
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
):
    """Fill a task's *missing* parameters from user context (memory-unit) before HITL.

    Additive + flag-gated: when ``MEMORY_URL`` is unset (or memory-unit is
    unreachable) the task is returned unchanged. Only ``missing`` slots are
    touched — email-provided values are preserved — and resolved values are
    marked ``guessed`` with a ``source``/``confidence`` so the UI can confirm them.
    """
    thread_id = request.thread_id or x_thread_id
    try:
        return populate_context_items(
            request.task, user_id=x_user_id, thread_id=thread_id
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# identify task and then return candidate workflows
@app.post("/identify_task", response_model=IdentifyTaskResponse)
def identify_task_endpoint(
    request: IdentifyTaskRequest,
    x_user_id: Optional[str] = Header(None),
    x_thread_id: Optional[str] = Header(None),
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
                task, user_id=x_user_id, thread_id=thread_id
            )

        return IdentifyTaskResponse(
            status="identified",
            task=task,
            context_items=task.context_items or identification.context_items,
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Task identification failed")

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
