# Workflow Planner

## Architecture
FastAPI + Uvicorn webserver that calls a Langchain agent on each request.

- `POST /create_workflow` accepts a `task` payload plus optional `rejected_workflows` and returns a structured `Workflow`.
- `POST /edit_workflow` accepts a `task`, `proposed_workflow`, and `feedback`, then returns an updated `Workflow`.
- The agent uses `response_format=ToolStrategy(Workflow)` to enforce structured output.

No storage/memory in the server, and the skills and custom middleware has beeen removed as well so that this agent can be as clean and minimal as possible.

## Setup
Conda environment for clean local dev environments.

```
conda create -n "agents_ucsd" python==3.11
conda activate agents_ucsd
pip install -r requirements.txt
uvicorn app:app --reload
python ./utils/tester.py
```

### Testing
The tester reads mock prompts from `./prompts` and calls the API for both initial
workflow creation and optional feedback edits.

- Set `WORKFLOW_API_URL` to point at a running server (defaults to `http://127.0.0.1:8000`).
- Each prompt can include `rejected_workflows`, `proposed_workflow`, and `feedback` to exercise both endpoints.




