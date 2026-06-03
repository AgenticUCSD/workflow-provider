from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.builder_agent import BuilderAgent
from agents.search_agent import SearchAgent
from utils.task import Task, TaskTypes, Workflow
from agents.task_agent import ContextItem, Metadata, TaskIdentifierAgent
from utils.chroma import ChromaVectorStore

app = FastAPI(title="Agent Infrastructure API")

chroma_store = ChromaVectorStore()
builder_agent = BuilderAgent()
search_agent = SearchAgent(vector_db=chroma_store)
task_identifier_agent = TaskIdentifierAgent()


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
def search_workflows_endpoint(request: SearchWorkflowsRequest):
    try:
        return search_agent.query_workflows_for_task(request.task, thread_id=request.thread_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/create_workflow", response_model=Workflow)
def create_workflow_endpoint(request: CreateWorkflowRequest):
    try:
        return builder_agent.create_workflow_initial(
            request.task,
            request.rejected_workflows,
            request.user_feedback,
            thread_id=request.thread_id
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/edit_workflow", response_model=Workflow)
def edit_workflow_endpoint(request: EditWorkflowRequest):
    try:
        return builder_agent.edit_proposed_workflow(
            request.task,
            request.proposed_workflow,
            request.feedback,
            thread_id=request.thread_id
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/edit_task", response_model=EditTaskResponse)
def edit_task_endpoint(request: EditTaskRequest):
    try:
        edited_task = task_identifier_agent.edit_task(request.task, request.user_feedback, thread_id=request.thread_id)
        context_items = getattr(edited_task, "context_items", [])
        return EditTaskResponse(
            status="edited",
            task=edited_task,
            context_items=context_items,
        )
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
            thread_id=request.thread_id,
        )

        task = identification.task
        if task is None or task.task_type == TaskTypes.NO_TASK:
            return IdentifyTaskResponse(
                status="no_task",
                task=None,
                context_items=identification.context_items,
            )
        return IdentifyTaskResponse(
            status="identified",
            task=task,
            context_items=identification.context_items,
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Task identification failed")

class EnrichTaskRequest(BaseModel):
    task: Task
    thread_id: Optional[str] = None


@app.post("/enrich_task_with_workflows", response_model=Task)
def enrich_task_with_workflows_endpoint(request: EnrichTaskRequest):
    candidates = search_agent.query_workflows_for_task(request.task, thread_id=request.thread_id)
    if candidates is None:
        created = builder_agent.create_workflow_initial(request.task, rejected_workflows=None, thread_id=request.thread_id)
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
        chroma_store.add_single_workflow(
            request.workflow,
            is_generated=request.is_generated
        )
        return {"status": "success"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/workflows", response_model=ListWorkflowsResponse)
def list_workflows_endpoint():
    try:
        workflows = chroma_store.get_all_workflows()
        return ListWorkflowsResponse(workflows=workflows)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
