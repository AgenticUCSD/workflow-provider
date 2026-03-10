from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.builder_agent import BuilderAgent
from agents.search_agent import SearchAgent
from utils.task import Task, Workflow
from agents.task_agent import ContextItem, Metadata, TaskIdentifierAgent

app = FastAPI(title="Agent Infrastructure API")

builder_agent = BuilderAgent()
search_agent = SearchAgent()
task_identifier_agent = TaskIdentifierAgent()


def enrich_task_with_workflows(task: Task) -> Task:
    candidates = search_agent.query_workflows_for_task(task)
    if candidates is None:
        created = builder_agent.create_workflow_initial(task, rejected_workflows=None)
        candidates = [created]
    task.candidate_workflows = candidates
    return task


class CreateWorkflowRequest(BaseModel):
    task: Task
    rejected_workflows: Optional[List[Workflow]] = None
    user_feedback: Optional[str] = None


class EditWorkflowRequest(BaseModel):
    task: Task
    proposed_workflow: Workflow
    feedback: str


class IdentifyTaskRequest(BaseModel):
    text: str = Field(..., min_length=1)
    subject: Optional[str] = None
    metadata: Optional[Metadata] = None


class IdentifyTaskResponse(BaseModel):
    status: Literal["identified", "no_task"]
    task: Optional[Task] = None
    detected_tag: Optional[str] = None
    context_items: List[ContextItem] = Field(default_factory=list)


class PopulateWorkflowsRequest(BaseModel):
    workflows: List[Workflow] = Field(default_factory=list)


class PopulateWorkflowsResponse(BaseModel):
    inserted_count: int
    document_ids: List[str]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/search_workflows", response_model=List[Workflow] | None)
def search_workflows_endpoint(task: Task):
    try:
        return search_agent.query_workflows_for_task(task)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/create_workflow", response_model=Workflow)
def create_workflow_endpoint(request: CreateWorkflowRequest):
    try:
        return builder_agent.create_workflow_initial(request.task, request.rejected_workflows)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/edit_workflow", response_model=Workflow)
def edit_workflow_endpoint(request: EditWorkflowRequest):
    try:
        return builder_agent.edit_proposed_workflow(request.task, request.proposed_workflow, request.feedback)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# identify task and then return candidate workflows
@app.post("/identify_task", response_model=IdentifyTaskResponse)
def identify_task_endpoint(request: IdentifyTaskRequest):
    try:
        identification = task_identifier_agent.identify_task(
            text=request.text,
            subject=request.subject,
            metadata=request.metadata,
        )

        if identification.status == "no_task" or identification.task is None:
            return IdentifyTaskResponse(
                status="no_task",
                task=None,
                detected_tag=identification.detected_tag or "no-task",
                context_items=[],
            )

        enriched_task = enrich_task_with_workflows(identification.task)
        return IdentifyTaskResponse(
            status="identified",
            task=enriched_task,
            detected_tag=identification.detected_tag,
            context_items=identification.context_items,
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Task identification failed")


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
