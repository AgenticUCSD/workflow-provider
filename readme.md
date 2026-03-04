# Workflow Provider

## Architecture
FastAPI + Uvicorn webserver with a three-agent system (BuilderAgent, SearchAgent, and TaskIdentifierAgent) backed by ChromaDB vector storage for RAG-based workflow retrieval.

### Components
- **BuilderAgent**: Creates and edits workflows using structured LLM output via `ToolStrategy(Workflow)`
- **SearchAgent**: Retrieves relevant workflows from vector store using semantic similarity
- **TaskIdentifierAgent**: Uses direct LLM structured-output calls for intent classification, deadline extraction, context detection, and task construction
- **ChromaVectorStore**: Manages two ChromaDB collections (manual_workflows, generated_workflows) with OpenAI embeddings


### API Endpoints
- `POST /create_workflow` accepts a `CreateWorkflowRequest` (task, optional rejected_workflows) and returns a structured `Workflow`
- `POST /edit_workflow` accepts an `EditWorkflowRequest` (task, proposed_workflow, feedback) and returns an updated `Workflow`
- `POST /search_workflows` accepts a `Task` and returns relevant workflows from the vector database using RAG
- `POST /identify_task` accepts raw text/email input and returns one of:
  - `identified` with `tasks: List[Task]`
  - `no_task` with empty `tasks`
- `GET /health` for health checks


## Setup
Conda environment for clean local dev environments.

```
conda create -n "agents_ucsd" python==3.11
conda activate agents_ucsd
pip install -r requirements.txt
uvicorn app:app --reload --port 8080
python ./utils/tester.py
```

### Testing
The tester (`utils/tester.py`) initializes the vector database and tests all four API endpoints:

1. **Vector DB Initialization**: Loads workflows from `prompts/random_workflows.json` into ChromaDB (manual_workflows collection) on first run
2. **Workflow Search**: Tests `/search_workflows` to retrieve semantically similar workflows for each task
3. **Workflow Creation**: Tests `/create_workflow` with task and optional rejected workflows
4. **Workflow Editing**: Tests `/edit_workflow` if `proposed_workflow` and `feedback` are provided in the mock task file
5. **Task Identification**: Tests `/identify_task` using prompt fixtures for `no_task`, single-intent, multi-intent, commitment tracking, urgent escalation, and ambiguous inputs

Task identification specifics:
- Deadline extraction is performed during identification; if a mail says `by 5pm today`, the returned task deadline reflects that constraint.
- For schedule tasks, detected deadlines are enforced as scheduling guardrails (`latest_scheduling_time` constraint).
- Context resolution is returned in `context_items` with per-field `present`/`missing` status.
- Clarification is handled by the workflow planner module, not this endpoint.

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
