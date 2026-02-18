# Workflow Provider

## Architecture
FastAPI + Uvicorn webserver with a dual-agent system (BuilderAgent and SearchAgent) backed by ChromaDB vector storage for RAG-based workflow retrieval.

### Components
- **BuilderAgent**: Creates and edits workflows using structured LLM output via `ToolStrategy(Workflow)`
- **SearchAgent**: Retrieves relevant workflows from vector store using semantic similarity
- **ChromaVectorStore**: Manages two ChromaDB collections (manual_workflows, generated_workflows) with OpenAI embeddings


### API Endpoints
- `POST /create_workflow` accepts a `CreateWorkflowRequest` (task, optional rejected_workflows) and returns a structured `Workflow`
- `POST /edit_workflow` accepts an `EditWorkflowRequest` (task, proposed_workflow, feedback) and returns an updated `Workflow`
- `POST /search_workflows` accepts a `Task` and returns relevant workflows from the vector database using RAG
- `GET /health` for health checks


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
The tester (`utils/tester.py`) initializes the vector database and tests all three API endpoints:

1. **Vector DB Initialization**: Loads workflows from `prompts/random_workflows.json` into ChromaDB (manual_workflows collection) on first run
2. **Workflow Search**: Tests `/search_workflows` to retrieve semantically similar workflows for each task
3. **Workflow Creation**: Tests `/create_workflow` with task and optional rejected workflows
4. **Workflow Editing**: Tests `/edit_workflow` if `proposed_workflow` and `feedback` are provided in the mock task file

**Configuration:**
- Set `WORKFLOW_API_URL` environment variable to point at a running server (defaults to `http://127.0.0.1:8000`)
- Mock tasks in `./prompts/*.txt` are JSON files that can include `rejected_workflows`, `proposed_workflow`, and `feedback` fields




