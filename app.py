from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils.builder_agent import BuilderAgent
from utils.search_agent import SearchAgent
from task_identification.task import Task, Workflow
from task_identification.task_identifier_agent import ContextItem, Metadata, TaskIdentifierAgent

app = FastAPI(title="Agent Infrastructure API")

builder_agent = BuilderAgent()
search_agent = SearchAgent()
task_identifier_agent = TaskIdentifierAgent()


def enrich_tasks_with_candidates(tasks: List[Task]) -> List[Task]:
    for task in tasks:
        candidates = search_agent.query_workflows_for_task(task)
        if candidates is None:
            created = builder_agent.create_workflow_initial(task, rejected_workflows=None)
            candidates = [created]
        task.candidate_workflows = candidates
    return tasks


class CreateWorkflowRequest(BaseModel):
    task: Task
    rejected_workflows: Optional[List[Workflow]] = None


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
    tasks: Optional[List[Task]] = None
    detected_tags: Optional[List[str]] = None
    context_items: Optional[List[ContextItem]] = None


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


@app.post("/identify_task", response_model=IdentifyTaskResponse)
def identify_task_endpoint(request: IdentifyTaskRequest):
    try:
        processed = task_identifier_agent.preprocess_email(request.text, request.subject)
        tag_result = task_identifier_agent.detect_tags(processed, request.metadata)
        detected_tags = [item.tag for item in tag_result.tags]

        if any(tag == "no-task" for tag in detected_tags):
            return IdentifyTaskResponse(
                status="no_task",
                tasks=[],
                detected_tags=["no-task"],
                context_items=[],
            )

        context_plan = task_identifier_agent.determine_context(
            tag_result=tag_result,
            processed_text=processed,
            metadata=request.metadata,
        )
        tasks = task_identifier_agent.tags_to_tasks(
            tag_result=tag_result,
            processed_text=processed,
            metadata=request.metadata,
        )
        enriched_tasks = enrich_tasks_with_candidates(tasks)
        return IdentifyTaskResponse(
            status="identified",
            tasks=enriched_tasks,
            detected_tags=detected_tags,
            context_items=context_plan.context_items,
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
