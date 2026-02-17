# Workflow Planner

## Architecture
FastAPI + Uvicorn webserver that just queries the Langchain agent on every call to its endpoint

- Input `Task (Objective, TaskTypes, Status)` is sent to `POST /task_to_workflow`.
- The agent returns a structured `Workflow (id, name, description, steps)` due to it being initialized with `response_format=ToolStrategy(Workflow)`

No storage/memory in the server, and the skills and custom middleware has beeen removed as well so that this agent can function as clean and minimal as possible.

## Setup
Conda environment for clean local dev environments

```
conda create -n "agents_ucsd" python==3.11
conda activate agents_ucsd
pip install -r requirements.txt
uvicorn app:app --reload
python ./utils/tester.py
```




