import json
import os
import sys
from typing import Dict, Any

import requests

BASE_URL = os.getenv("AGENT_API_URL", "http://127.0.0.1:8000")
ENDPOINT = f"{BASE_URL}/task_to_workflow"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "prompts")
PROMPT_FILES = [
    os.path.join(PROMPTS_DIR, "mock_task_send_email.txt"),
    os.path.join(PROMPTS_DIR, "mock_task_schedule_meeting.txt"),
    os.path.join(PROMPTS_DIR, "mock_task_check_status.txt"),
]

def load_task(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def call_api(task_payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(ENDPOINT, json=task_payload, timeout=120)
    response.raise_for_status()
    return response.json()


def main() -> int:
    for path in PROMPT_FILES:
        print(f"\n=== {path} ===")
        payload = load_task(path)
        try:
            workflow = call_api(payload)
        except requests.RequestException as exc:
            print(f"Request failed: {exc}")
            return 1

        print(json.dumps(workflow, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
