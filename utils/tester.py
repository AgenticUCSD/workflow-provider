import json
import os
import sys
from typing import Any, Dict, List, Optional
import requests

# Add project root to path BEFORE importing local modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.task import Task, Workflow
from utils.chroma import ChromaVectorStore

PROMPTS_DIR = os.path.join(PROJECT_ROOT, "prompts")
PROMPT_FILES = [
    os.path.join(PROMPTS_DIR, "mock_task_send_email.txt"),
    os.path.join(PROMPTS_DIR, "mock_task_schedule_meeting.txt"),
    os.path.join(PROMPTS_DIR, "mock_task_check_status.txt"),
]
WORKFLOWS_FILE = os.path.join(PROMPTS_DIR, "random_workflows.json")
BASE_URL = os.environ.get("WORKFLOW_API_URL", "http://127.0.0.1:8000")

def load_task(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_workflows(workflows_data: List[Dict[str, Any]]) -> List[Workflow]:
    return [Workflow.model_validate(item) for item in workflows_data]


def parse_workflow(workflow_data: Optional[Dict[str, Any]]) -> Optional[Workflow]:
    if not workflow_data:
        return None
    return Workflow.model_validate(workflow_data)


def post_json(path: str, payload: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{BASE_URL.rstrip('/')}{path}"
    response = requests.post(url, params=params, json=payload, timeout=60)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        details = response.text.strip()
        message = f"{exc} - {details}" if details else str(exc)
        raise RuntimeError(message) from exc
    return response.json()


def initialize_vector_db() -> None:
    """Initialize vector database with random workflows if not already populated."""
    vector_db = ChromaVectorStore()
    
    # Check if the database is already populated
    try:
        # Query with a dummy workflow to see if any workflows exist
        existing = vector_db.manual_workflows.get()
        if existing and existing.get("ids") and len(existing["ids"]) > 0:
            print("Vector DB already populated with manual workflows, skipping initialization.")
            return
    except Exception:
        pass
    
    # Load and add workflows from the JSON file
    if not os.path.exists(WORKFLOWS_FILE):
        print(f"Warning: Workflows file not found at {WORKFLOWS_FILE}")
        return
    
    try:
        with open(WORKFLOWS_FILE, "r", encoding="utf-8") as f:
            workflows_data = json.load(f)
        
        for workflow_data in workflows_data:
            workflow = Workflow.model_validate(workflow_data)
            vector_db.add_workflow(workflow, is_generated=False)
            print(f"Added workflow: {workflow.name}")
        
        print(f"\nSuccessfully initialized vector DB with {len(workflows_data)} workflows.")
    except Exception as e:
        print(f"Error initializing vector DB: {e}")
        raise


def search_workflows_for_task(task: Task) -> List[Workflow] | None:
    """Search for relevant workflows using the search endpoint."""
    try:
        search_payload = task.model_dump()
        result = post_json("/search_workflows", search_payload)
        
        if result is None:
            return None
        
        if isinstance(result, list):
            return parse_workflows(result)
        
        if isinstance(result, dict) and "workflows" in result:
            return parse_workflows(result["workflows"])
        
        return None
    except Exception as e:
        print(f"Error searching workflows: {e}")
        return None


def main() -> int:
    initialize_vector_db()
    
    for path in PROMPT_FILES:
        print(f"\n=== {path} ===")
        payload = load_task(path)
        rejected_workflows_data = payload.pop("rejected_workflows", [])
        proposed_workflow_data = payload.pop("proposed_workflow", None)
        feedback = payload.pop("feedback", None)

        task = Task(**payload)
        rejected_workflows = parse_workflows(rejected_workflows_data)
        proposed_workflow = parse_workflow(proposed_workflow_data)

        # Search for relevant workflows
        print("\nSearching for relevant workflows...")
        search_results = search_workflows_for_task(task)
        if search_results:
            print(f"Found {len(search_results)} relevant workflows:")
            for workflow in search_results:
                print(f"  - {workflow.name}: {workflow.description}")
        else:
            print("No relevant workflows found.")

        # Create workflow - use nested structure
        create_payload = {
            "task": task.model_dump(),
            "rejected_workflows": [w.model_dump() for w in rejected_workflows],
        }
        initial_workflow = Workflow.model_validate(post_json("/create_workflow", create_payload))
        print("\nInitial workflow:")
        print(json.dumps(initial_workflow.model_dump(), indent=2))

        if proposed_workflow and feedback:
            # Edit workflow - use nested structure
            edit_payload = {
                "task": task.model_dump(),
                "proposed_workflow": proposed_workflow.model_dump(),
                "feedback": feedback,
            }
            updated_workflow = Workflow.model_validate(
                post_json("/edit_workflow", edit_payload)
            )
            print("Edited workflow:")
            print(json.dumps(updated_workflow.model_dump(), indent=2))
        else:
            print("Edited workflow: skipped (missing proposed workflow or feedback)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
