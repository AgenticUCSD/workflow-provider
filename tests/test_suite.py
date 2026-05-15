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

PROMPTS_DIR = os.path.join(PROJECT_ROOT, "prompts")
PROMPT_FILES = [
    os.path.join(PROMPTS_DIR, "mock_task_send_email.txt"),
    os.path.join(PROMPTS_DIR, "mock_task_schedule_meeting.txt"),
    os.path.join(PROMPTS_DIR, "mock_task_check_status.txt"),
]
IDENTIFY_PROMPT_FILES = [
    os.path.join(PROMPTS_DIR, "mock_identify_no_task.txt"),
    os.path.join(PROMPTS_DIR, "mock_identify_action_request.txt"),
    os.path.join(PROMPTS_DIR, "mock_identify_multi_intent.txt"),
    os.path.join(PROMPTS_DIR, "mock_identify_commitment_track.txt"),
    os.path.join(PROMPTS_DIR, "mock_identify_escalation_urgent.txt"),
    os.path.join(PROMPTS_DIR, "mock_identify_ambiguous.txt"),
]
WORKFLOWS_FILE = os.path.join(PROMPTS_DIR, "random_workflows.json")
BASE_URL = os.environ.get("WORKFLOW_API_URL", "http://127.0.0.1:8080")



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
    """Initialize manual workflows through the API route."""
    if not os.path.exists(WORKFLOWS_FILE):
        print(f"Warning: Workflows file not found at {WORKFLOWS_FILE}")
        return
    
    try:
        with open(WORKFLOWS_FILE, "r", encoding="utf-8") as f:
            workflows_data = json.load(f)

        workflows = parse_workflows(workflows_data)
        result = populate_workflows(workflows)
        if result is None:
            print("Failed to initialize manual workflows through API.")
            return

        inserted_count = result.get("inserted_count")
        print(f"\nInitialized manual workflows through API. inserted_count={inserted_count}")
    except Exception as e:
        print(f"Error initializing vector DB: {e}")
        raise


def search_workflows_for_task(task: Task, thread_id: str | None = None) -> List[Workflow] | None:
    """Search for relevant workflows using the search endpoint."""
    try:
        search_payload = {"task": task.model_dump(), "thread_id": thread_id}
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


def populate_workflows(workflows: List[Workflow]) -> Dict[str, Any] | None:
    """Populate manual workflow collection through the API route."""
    try:
        payload = {"workflows": [workflow.model_dump() for workflow in workflows]}
        return post_json("/populate_workflows", payload)
    except Exception as e:
        print(f"Error populating workflows: {e}")
        return None


def add_single_workflow(workflow: Workflow, is_generated: bool = False) -> bool:
    """Add a single workflow to the vector store through the API route."""
    try:
        payload = {
            "workflow": workflow.model_dump(),
            "is_generated": is_generated
        }
        result = post_json("/add_workflow", payload)
        return result.get("status") == "success"
    except Exception as e:
        print(f"Error adding workflow: {e}")
        return False


def list_all_workflows() -> List[Workflow] | None:
    """Get all workflows from the vector store through the API route."""
    try:
        url = f"{BASE_URL.rstrip('/')}/workflows"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        result = response.json()
        workflows_data = result.get("workflows", [])
        return parse_workflows(workflows_data)
    except Exception as e:
        print(f"Error listing workflows: {e}")
        return None


def identify_task_payload(path: str) -> Dict[str, Any]:
    return load_task(path)


def enrich_task_with_workflows(task: Task, thread_id: str | None = None) -> Optional[Task]:
    try:
        result = post_json("/enrich_task_with_workflows", {"task": task.model_dump(), "thread_id": thread_id})
        return Task.model_validate(result)
    except Exception as e:
        print(f"Error enriching task with workflows: {e}")
        return None


def run_identify_task(path: str) -> None:
    payload = identify_task_payload(path)
    result = post_json("/identify_task", payload)
    print(f"\n=== {path} ===")
    print(f"status: {result.get('status')}")
    print(f"detected_tag: {result.get('detected_tag')}")
    print(f"context_items: {result.get('context_items')}")

    task = result.get("task")
    if isinstance(task, dict):
        print(f"task_type: {task.get('task_type')}")
        candidate_workflows = task.get("candidate_workflows") or []
        print(f"candidate_workflow_count: {len(candidate_workflows)}")

        enriched_task = enrich_task_with_workflows(Task.model_validate(task))
        if enriched_task is not None:
            enriched_candidates = enriched_task.candidate_workflows or []
            print(f"enriched_candidate_workflow_count: {len(enriched_candidates)}")
    else:
        print("task_type: None")
        print("candidate_workflow_count: 0")



def main() -> int:
    global BASE_URL
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

    print("\n=== identify_task endpoint tests ===")
    for path in IDENTIFY_PROMPT_FILES:
        run_identify_task(path)

    print("\n=== add_workflow and list_workflows endpoint tests ===")
    test_workflow = Workflow(
        workflow_id="test_add_workflow_001",
        name="Test Workflow via Add Endpoint",
        description="A test workflow added through the add_workflow endpoint.",
        steps=["Step 1: Test step one", "Step 2: Test step two"]
    )

    print("\nAdding single workflow via /add_workflow...")
    add_success = add_single_workflow(test_workflow, is_generated=False)
    if add_success:
        print("Successfully added workflow via /add_workflow")
    else:
        print("Failed to add workflow via /add_workflow")

    print("\nListing all workflows via /workflows...")
    all_workflows = list_all_workflows()
    if all_workflows is not None:
        print(f"Found {len(all_workflows)} total workflows")
        test_found = any(w.workflow_id == "test_add_workflow_001" for w in all_workflows)
        if test_found:
            print("Test workflow confirmed in list")
        else:
            print("Test workflow NOT found in list")
    else:
        print("Failed to list workflows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
