# Workflow Provider

## Architecture
FastAPI + Uvicorn webserver with a three-agent system (BuilderAgent, SearchAgent, and TaskIdentifierAgent) backed by ChromaDB vector storage for RAG-based workflow retrieval.

### Components
- **BuilderAgent**: Creates and edits workflows using structured LLM output via `ToolStrategy(Workflow)`
- **SearchAgent**: Retrieves relevant workflows from vector store using semantic similarity
- **TaskIdentifierAgent**: Uses agent structured-output calls for intent classification, deadline extraction, context detection, and task construction
- **ChromaVectorStore**: Manages two ChromaDB collections (manual_workflows, generated_workflows) with OpenAI embeddings


### API Endpoints
- `POST /create_workflow` accepts a `CreateWorkflowRequest` (`task`, optional `rejected_workflows`, optional `user_feedback`) and returns a structured `Workflow`
- `POST /edit_workflow` accepts an `EditWorkflowRequest` (task, proposed_workflow, feedback) and returns an updated `Workflow`
- `POST /edit_task` accepts an `EditTaskRequest` (`task`, `user_feedback`) and returns an `EditTaskResponse` with `status: "edited"`, the edited `Task`, and any detected tag/context items
- `POST /search_workflows` accepts a `Task` and returns relevant workflows from the vector database using RAG
- `POST /identify_task` accepts raw text/email input and returns one of:
  - `identified` with `task: Task`, `detected_tag: str`, and `context_items: List[ContextItem]`
  - `no_task` with `task: null`, `detected_tag: "no-task"`, and empty `context_items`
- `POST /enrich_task_with_workflows` accepts a `Task`, attaches candidate workflows, and returns the enriched task
- `POST /populate_workflows` accepts `{ workflows: List[Workflow] }` and returns inserted IDs/count for the manual workflow collection
- `GET /health` for health checks


## Setup
Conda environment for clean local dev environments.

```
conda create -n "agents_ucsd" python==3.11
conda activate agents_ucsd
pip install -r requirements.txt
uvicorn app:app --reload --port 8080
python .\tests\test_suite.py
```

### Testing
The integration test suite (`tests/test_suite.py`) initializes the vector database and tests workflow, task-identification, and task-enrichment endpoints:

1. **Vector DB Initialization**: Loads workflows from `prompts/random_workflows.json` into ChromaDB (manual_workflows collection) on first run
2. **Manual Workflow Population**: Calls `/populate_workflows` to seed the manual workflow collection
3. **Workflow Search**: Tests `/search_workflows` to retrieve semantically similar workflows for each task
4. **Workflow Creation**: Tests `/create_workflow` with task and optional rejected workflows
5. **Workflow Editing**: Tests `/edit_workflow` if `proposed_workflow` and `feedback` are provided in the mock task file
6. **Task Identification**: Tests `/identify_task` using prompt fixtures for `no_task`, single-intent, multi-intent, commitment tracking, urgent escalation, and ambiguous inputs
7. **Task Enrichment**: Tests `/enrich_task_with_workflows` separately so identification no longer depends on workflow lookup

**Task identification specifics:**
- Deadline extraction is performed during identification; if a mail says `by 5pm today`, the returned task deadline reflects that constraint.
- For schedule tasks, detected deadlines are enforced as scheduling guardrails (`latest_scheduling_time` constraint).
- Context resolution is returned in `context_items` with per-field `present`/`missing` status.
- `/identify_task` now returns the classified task only; workflow lookup is handled by `/enrich_task_with_workflows` after identification when candidate workflows are needed.
- `/enrich_task_with_workflows` searches for matching workflows first and falls back to `/create_workflow` if no relevant workflows are found.


Run unit tests:

```bash
python -m unittest tests.task_unit_test -v
```

**Configuration:**
- Set `WORKFLOW_API_URL` environment variable to point at a running server (defaults to `http://127.0.0.1:8080`) if you want to locally test cloud endpoints. Otherwise the default works perfectly.
- Mock tasks in `./prompts/*.txt` are JSON files that can include `rejected_workflows`, `proposed_workflow`, and `feedback` fields

## Docker (Cloud Run-ready)

Build locally:

```bash
docker build -t workflow-planner .
docker run --rm -p 8080:8080 -e OPENAI_API_KEY=your_key workflow-planner
```

The container starts with:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Notes:
- `OPENAI_API_KEY` must be provided as an environment variable (prefer Secret Manager on GCP).
- `CHROMA_PERSIST_DIR` defaults to `/tmp/chroma_db` in the container. Cloud Run filesystem is ephemeral, so vector data does not persist across instance restarts unless you externalize storage.
