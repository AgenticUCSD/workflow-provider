# Workflow Provider

## Architecture
FastAPI + Uvicorn webserver with a three-agent system (BuilderAgent, SearchAgent, and TaskIdentifierAgent) backed by ChromaDB vector storage for RAG-based workflow retrieval.

### Components
- **BuilderAgent**: Creates and edits workflows using structured LLM output via `ToolStrategy(Workflow)`
- **SearchAgent**: Retrieves relevant workflows from vector store using semantic similarity
- **TaskIdentifierAgent**: Uses agent structured-output calls for intent classification, deadline extraction, context detection, and task construction
- **AnalyzerAgent**: Analyzes traces from Confident AI to extract patterns and update knowledge files (user preferences, task patterns, workflow trends)
- **ChromaVectorStore**: Manages two ChromaDB collections (manual_workflows, generated_workflows) with OpenAI embeddings


### API Endpoints
- `POST /create_workflow` accepts a `CreateWorkflowRequest` (`task`, optional `rejected_workflows`, optional `user_feedback`, optional `thread_id`) and returns a structured `Workflow`
- `POST /edit_workflow` accepts an `EditWorkflowRequest` (`task`, `proposed_workflow`, `feedback`, optional `thread_id`) and returns an updated `Workflow`
- `POST /edit_task` accepts an `EditTaskRequest` (`task`, `user_feedback`, optional `thread_id`) and returns an `EditTaskResponse` with `status: "edited"`, the edited `Task`, and context items
- `POST /search_workflows` accepts a `SearchWorkflowsRequest` (`task`, optional `thread_id`) and returns relevant workflows from the vector database using RAG
- `POST /identify_task` accepts an `IdentifyTaskRequest` (`text`, optional `subject`, optional `metadata`, optional `thread_id`) and returns one of:
  - `identified` with `task: Task`, and `context_items: List[ContextItem]`
  - `no_task` with `task: null`, and empty `context_items`
- `POST /enrich_task_with_workflows` accepts an `EnrichTaskRequest` (`task`, optional `thread_id`), attaches candidate workflows, and returns the enriched task
- `POST /analyze_traces` accepts an `AnalyzeTracesRequest` (`thread_id`) and analyzes all traces from the thread to extract patterns and update knowledge files
- `POST /populate_workflows` accepts `{ workflows: List[Workflow] }` and returns inserted IDs/count for the manual workflow collection
- `GET /health` for health checks

**Note:** All endpoints that invoke LLM agents accept an optional `thread_id` parameter. When provided, it is used for DeepEval logging to enable request tracing and observability. If not provided, a new UUID is generated automatically.

**Knowledge Files:** The analyzer agent maintains three runtime-generated knowledge files in the `knowledge/` directory: `user_preferences.txt`, `task_patterns.txt`, and `workflow_trends.txt`. These are populated by analyzing traces from Confident AI and should not be committed to version control.

**Environment Variables:**
- `OPENAI_API_KEY` (required): For LLM calls and embeddings
- `CONFIDENT_API_KEY` (required for trace analysis): For fetching traces from Confident AI API
- `CHROMA_PERSIST_DIR` (optional): Vector DB persistence directory (default: `./chroma_db`)


## Setup
Conda environment for clean local dev environments.

```
conda create -n "agents_ucsd" python==3.11
conda activate agents_ucsd
pip install -r requirements.txt
uvicorn app:app --reload --port 8080
```


**If you want to use docker:**

Build locally:

```bash
docker build -t workflow-planner .
docker run --rm -p 8080:8080 -e OPENAI_API_KEY=your_key workflow-planner
```

The container starts with:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```


## Testing

Run integration tests:
```bash
python tests/test_suite.py
```

Run unit tests:
```bash
python -m unittest tests.task_unit_test -v
```

**Configuration:**
- Set `WORKFLOW_API_URL` environment variable to point at a running server (defaults to `http://127.0.0.1:8080`)
- Mock tasks in `./prompts/*.txt` can include `rejected_workflows`, `proposed_workflow`, and `feedback` fields
