from fastapi import FastAPI, HTTPException
from typing import List, Optional

from utils.agent import create_workflow_initial, edit_proposed_workflow
from utils.task import Task, Workflow

app = FastAPI(title="Agent Infrastructure API")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/create_workflow", response_model=Workflow)
def create_workflow_endpoint(task: Task, rejected_workflows: Optional[List[Workflow]] = None):
    try:
        return create_workflow_initial(task, rejected_workflows)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/edit_workflow", response_model=Workflow)
def edit_workflow_endpoint(task: Task, proposed_workflow: Workflow, feedback: str):
    try:
        return edit_proposed_workflow(task, proposed_workflow, feedback)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
