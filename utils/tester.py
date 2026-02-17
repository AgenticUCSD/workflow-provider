import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests

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


def main() -> int:
    for path in PROMPT_FILES:
        print(f"\n=== {path} ===")
        payload = load_task(path)
        rejected_workflows_data = payload.pop("rejected_workflows", [])
        proposed_workflow_data = payload.pop("proposed_workflow", None)
        feedback = payload.pop("feedback", None)

        task = Task(**payload)
        rejected_workflows = parse_workflows(rejected_workflows_data)
        proposed_workflow = parse_workflow(proposed_workflow_data)

        create_payload = {
            "task": task.model_dump(),
            "rejected_workflows": [w.model_dump() for w in rejected_workflows],
        }
        initial_workflow = Workflow.model_validate(post_json("/create_workflow", create_payload))
        print("Initial workflow:")
        print(json.dumps(initial_workflow.model_dump(), indent=2))

        if proposed_workflow and feedback:
            edit_payload = {
                "task": task.model_dump(),
                "proposed_workflow": proposed_workflow.model_dump(),
                "feedback": feedback,
            }
            updated_workflow = Workflow.model_validate(
                post_json("/edit_workflow", edit_payload, params={"feedback": feedback})
            )
            print("Edited workflow:")
            print(json.dumps(updated_workflow.model_dump(), indent=2))
        else:
            print("Edited workflow: skipped (missing proposed workflow or feedback)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
