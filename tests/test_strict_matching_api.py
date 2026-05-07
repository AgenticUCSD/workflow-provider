"""
API-based integration test for strict workflow matching (95%+ threshold).

This test validates that the SearchAgent only returns workflows with 95%+ match via API calls.
"""

import json
import os
import sys
import requests
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.task import Task, Workflow, Objective, TaskTypes, Status

BASE_URL = os.environ.get("WORKFLOW_API_URL", "http://127.0.0.1:8080")


def post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to make POST requests to API"""
    url = f"{BASE_URL.rstrip('/')}{path}"
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def create_test_task(description: str, task_type: TaskTypes = TaskTypes.ACTION_REQUIRED) -> Task:
    """Helper to create a task"""
    return Task(
        task_id="test_strict_001",
        task_type=task_type,
        objective=Objective(
            objective_id="test_obj_strict_001",
            name=f"{task_type.value} task",
            description=description,
            inputs={"description": description},
            success_criteria="Task completed successfully",
            expected_output={"status": "completed"}
        ),
        status=Status.PENDING
    )


def search_workflows(task: Task) -> List[Workflow] | None:
    """Search for workflows via API"""
    try:
        result = post_json("/search_workflows", task.model_dump())
        if result is None or result == []:
            return None
        if isinstance(result, list):
            return [Workflow.model_validate(w) for w in result]
        return None
    except Exception as e:
        print(f"Error searching workflows: {e}")
        return None


def run_test(test_name: str, task_description: str, should_return_workflows: bool, expected_workflow_name: str = None):
    """Run a single test case"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"Task: {task_description}")
    print(f"Expected: {'Should return workflow' if should_return_workflows else 'Should return NOTHING (null/empty)'}")

    task = create_test_task(task_description)
    results = search_workflows(task)

    if should_return_workflows:
        if results is None or len(results) == 0:
            print(f"[FAIL] Expected workflows but got None/empty")
            return False
        else:
            print(f"[PASS] Returned {len(results)} workflow(s)")
            for w in results:
                print(f"  - {w.name}: {w.description}")
            if expected_workflow_name:
                workflow_names = [w.name for w in results]
                if expected_workflow_name in workflow_names:
                    print(f"[PASS] Found expected workflow '{expected_workflow_name}'")
                else:
                    print(f"[FAIL] Expected workflow '{expected_workflow_name}' not found. Got: {workflow_names}")
                    return False
            return True
    else:
        if results is None or len(results) == 0:
            print(f"[PASS] Correctly returned None/empty")
            return True
        else:
            print(f"[FAIL] Should return nothing but got {len(results)} workflows:")
            for w in results:
                print(f"  - {w.name}: {w.description}")
            return False


def main():
    """Run all strict matching tests"""
    print("="*80)
    print("STRICT WORKFLOW MATCHING TESTS (95%+ threshold)")
    print("="*80)

    # Make sure server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        response.raise_for_status()
        print(f"[OK] Server is running at {BASE_URL}")
    except Exception as e:
        print(f"[ERROR] Server not running at {BASE_URL}: {e}")
        print("Please start the server with: uvicorn app:app --reload --port 8080")
        return 1

    test_results = []

    # Test 1: Exact match - should return workflow
    test_results.append(run_test(
        "Test 1: Exact match for team standup",
        "Schedule and run our daily team standup meeting to sync on today's progress and blockers",
        should_return_workflows=True,
        expected_workflow_name="Organize team standup"
    ))

    # Test 2: Exact match - production bug
    test_results.append(run_test(
        "Test 2: Exact match for production bug fix",
        "Critical bug in production causing user login failures - need to investigate and fix immediately",
        should_return_workflows=True,
        expected_workflow_name="Fix critical bug in production"
    ))

    # Test 3: Exact match - deploy to staging
    test_results.append(run_test(
        "Test 3: Exact match for staging deployment",
        "Deploy the new authentication feature to staging environment for QA testing",
        should_return_workflows=True,
        expected_workflow_name="Deploy feature to staging"
    ))

    # Test 4: Close but not 95% - different type of meeting
    test_results.append(run_test(
        "Test 4: Close match (~80%) - 1:1 meeting not team standup",
        "Schedule a 1-on-1 meeting with my manager to discuss my career growth",
        should_return_workflows=False  # Should NOT return team standup workflow
    ))

    # Test 5: Tangentially related - same category but different domain
    test_results.append(run_test(
        "Test 5: Tangential match - doctor appointment vs work meeting",
        "Book a doctor's appointment for next Tuesday at 2pm",
        should_return_workflows=False  # Should NOT return any work meeting workflows
    ))

    # Test 6: Completely unrelated
    test_results.append(run_test(
        "Test 6: Completely unrelated - vacation planning",
        "Research vacation destinations in Europe for summer trip",
        should_return_workflows=False  # Should NOT return any workflows
    ))

    # Test 7: Similar domain but different task - personal vs work
    test_results.append(run_test(
        "Test 7: Similar but different domain - personal laptop setup",
        "Set up my personal laptop for software development with VS Code and Git",
        should_return_workflows=False  # Should NOT return onboarding workflow
    ))

    # Test 8: Partial overlap - just PR, not full deployment
    test_results.append(run_test(
        "Test 8: Partial overlap - PR only, not deployment",
        "Create a pull request for my code changes and get it reviewed by the team",
        should_return_workflows=False  # Should NOT return full deployment workflow
    ))

    # Test 9: Exact match - onboarding
    test_results.append(run_test(
        "Test 9: Exact match for new hire onboarding",
        "Set up the new software engineer with development environment, credentials, and orientation meetings",
        should_return_workflows=True,
        expected_workflow_name="Onboard new team member"
    ))

    # Test 10: Different specific task in same category
    test_results.append(run_test(
        "Test 10: Same category but different task - customer refund vs technical work",
        "Send a promotional email to all customers about our new product features",
        should_return_workflows=False  # Should NOT match any technical workflows
    ))

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    passed = sum(test_results)
    total = len(test_results)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")

    if passed == total:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
