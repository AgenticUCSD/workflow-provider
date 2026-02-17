from fastapi import FastAPI, HTTPException

from utils.agent import task_to_workflow
from utils.task import Task, Workflow

app = FastAPI(title="Agent Infrastructure API")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/task_to_workflow", response_model=Workflow)
def task_to_workflow_endpoint(task: Task):
    try:
        return task_to_workflow(task)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
