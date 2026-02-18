from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from utils.builder_agent import BuilderAgent
from utils.search_agent import SearchAgent
from utils.task import Task, Workflow

app = FastAPI(title="Agent Infrastructure API")

builder_agent = BuilderAgent()
search_agent = SearchAgent()


class CreateWorkflowRequest(BaseModel):
    task: Task
    rejected_workflows: Optional[List[Workflow]] = None


class EditWorkflowRequest(BaseModel):
    task: Task
    proposed_workflow: Workflow
    feedback: str


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
