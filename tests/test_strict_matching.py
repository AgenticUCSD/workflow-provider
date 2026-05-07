"""
Test suite to verify SearchAgent only returns workflows with 95%+ match to tasks.

This test suite validates that the SearchAgent is strict about workflow matching:
- Returns workflows ONLY when they are 95%+ match
- Returns null/empty when workflows are close but not exact enough
- Returns null/empty when workflows are tangentially related
- Returns null/empty when workflows are completely unrelated
"""

import json
import os
import sys
import unittest
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.task import Task, Workflow, Objective, TaskTypes, Status
from agents.search_agent import SearchAgent


class TestStrictWorkflowMatching(unittest.TestCase):
    """Test cases for strict workflow matching (95%+ threshold)"""

    @classmethod
    def setUpClass(cls):
        """Initialize SearchAgent and populate test workflows once for all tests"""
        cls.search_agent = SearchAgent()

        # Clear any existing workflows
        cls.search_agent.vector_db.clear_collections()

        # Define test workflows
        cls.test_workflows = [
            Workflow(
                workflow_id="test_wf_001",
                name="Organize team standup",
                description="Schedule and conduct a daily team standup meeting to sync on progress.",
                steps=[
                    "Check calendar for team availability",
                    "Send invite for 15-minute standup meeting",
                    "Prepare agenda with focus areas",
                    "Conduct meeting and capture action items",
                    "Send meeting notes to team"
                ]
            ),
            Workflow(
                workflow_id="test_wf_002",
                name="Fix critical production bug",
                description="Rapidly identify and resolve a critical production issue.",
                steps=[
                    "Check error logs and monitoring alerts",
                    "Reproduce the issue in staging environment",
                    "Identify root cause and affected systems",
                    "Implement emergency fix or rollback",
                    "Deploy hotfix to production",
                    "Monitor system stability and performance"
                ]
            ),
            Workflow(
                workflow_id="test_wf_003",
                name="Deploy feature to staging",
                description="Move a completed feature to the staging environment for testing.",
                steps=[
                    "Run all unit and integration tests locally",
                    "Create pull request with detailed description",
                    "Get code review from team members",
                    "Address review feedback",
                    "Merge to develop branch",
                    "Build and deploy to staging"
                ]
            ),
            Workflow(
                workflow_id="test_wf_004",
                name="Onboard new team member",
                description="Set up a new engineer with all necessary tools and knowledge.",
                steps=[
                    "Send welcome email with team info",
                    "Set up development environment and credentials",
                    "Schedule orientation meetings with department leads",
                    "Assign initial training tasks"
                ]
            ),
            Workflow(
                workflow_id="test_wf_005",
                name="Conduct performance review",
                description="Complete annual performance review for direct report.",
                steps=[
                    "Gather 360-degree feedback from colleagues",
                    "Review performance metrics and achievements",
                    "Prepare observations on strengths and growth areas",
                    "Schedule private review meeting",
                    "Conduct meeting and discuss career goals"
                ]
            )
        ]

        # Populate workflows into vector DB
        for workflow in cls.test_workflows:
            cls.search_agent.vector_db.add_workflow(workflow, is_generated=False)

    def _create_task(self, description: str, task_type: TaskTypes = TaskTypes.ACTION_REQUIRED) -> Task:
        """Helper to create a task with given description"""
        return Task(
            task_id="test_task_001",
            task_type=task_type,
            objective=Objective(
                objective_id="test_obj_001",
                name=f"{task_type.value} task",
                description=description,
                inputs={"description": description},
                success_criteria="Task completed successfully",
                expected_output={"status": "completed"}
            ),
            status=Status.PENDING
        )

    def test_exact_match_returns_workflow(self):
        """Test case 1: Exact match (95%+) should return the workflow"""
        task = self._create_task(
            "Schedule and run our daily team standup meeting to sync on today's progress and blockers"
        )

        results = self.search_agent.query_workflows_for_task(task)

        # Should return the "Organize team standup" workflow
        self.assertIsNotNone(results, "Exact match should return workflows, not None")
        self.assertGreater(len(results), 0, "Exact match should return at least one workflow")

        # Check that the returned workflow is the standup one
        workflow_names = [w.name for w in results]
        self.assertIn("Organize team standup", workflow_names,
                     "Should return the 'Organize team standup' workflow for exact match")

    def test_close_but_not_95_match_returns_nothing(self):
        """Test case 2: Close match (~80-90%) should NOT return workflows"""
        task = self._create_task(
            "Schedule a 1-on-1 meeting with my manager to discuss my career goals"
        )

        results = self.search_agent.query_workflows_for_task(task)

        # This is about scheduling a meeting, but it's a 1-on-1, not a team standup
        # It's related but not 95%+ match
        # Should return None or empty list
        if results is not None:
            # Filter out any performance review workflow (which might match "discuss career goals")
            non_perf_review = [w for w in results if "performance review" not in w.name.lower()]
            self.assertEqual(len(non_perf_review), 0,
                           f"Close but not 95% match should not return workflows. Got: {[w.name for w in non_perf_review]}")

    def test_tangential_match_returns_nothing(self):
        """Test case 3: Tangentially related should NOT return workflows"""
        task = self._create_task(
            "Book a doctor's appointment for next Tuesday at 2pm"
        )

        results = self.search_agent.query_workflows_for_task(task)

        # This involves scheduling but is completely different domain (healthcare vs work meetings)
        # Should return None or empty
        if results is not None:
            self.assertEqual(len(results), 0,
                           f"Tangentially related task should not return workflows. Got: {[w.name for w in results]}")

    def test_completely_unrelated_returns_nothing(self):
        """Test case 4: Completely unrelated should NOT return workflows"""
        task = self._create_task(
            "Research vacation destinations in Europe for summer trip"
        )

        results = self.search_agent.query_workflows_for_task(task)

        # This is completely unrelated to any workflow
        # Should return None or empty
        if results is not None:
            self.assertEqual(len(results), 0,
                           f"Unrelated task should not return workflows. Got: {[w.name for w in results]}")

    def test_exact_match_production_bug_fix(self):
        """Test case 5: Exact match for production bug should return bug fix workflow"""
        task = self._create_task(
            "There's a critical bug in production causing user login failures - need to investigate and fix immediately"
        )

        results = self.search_agent.query_workflows_for_task(task)

        # Should return the "Fix critical production bug" workflow
        self.assertIsNotNone(results, "Production bug fix task should return workflows")
        self.assertGreater(len(results), 0, "Should return at least one workflow")

        workflow_names = [w.name for w in results]
        self.assertIn("Fix critical production bug", workflow_names,
                     "Should return the bug fix workflow for critical production issue")

    def test_similar_but_different_domain_returns_nothing(self):
        """Test case 6: Similar task but different domain should NOT return workflows"""
        task = self._create_task(
            "Deploy my personal website to Netlify for hosting"
        )

        results = self.search_agent.query_workflows_for_task(task)

        # This is about deployment, but to Netlify (personal project), not staging (work project)
        # The "Deploy feature to staging" workflow shouldn't match at 95%+
        if results is not None:
            deploy_workflows = [w for w in results if "deploy" in w.name.lower() and "staging" in w.description.lower()]
            self.assertEqual(len(deploy_workflows), 0,
                           f"Similar but different domain should not return staging deployment workflow. Got: {[w.name for w in deploy_workflows]}")

    def test_partial_overlap_returns_nothing(self):
        """Test case 7: Partial overlap (some steps match but not complete) should NOT return"""
        task = self._create_task(
            "Create a pull request for my code changes and get it reviewed"
        )

        results = self.search_agent.query_workflows_for_task(task)

        # This overlaps with "Deploy feature to staging" workflow (PR + code review)
        # But it doesn't include the deployment, testing, etc.
        # Not a 95%+ match - should not return or should be filtered
        if results is not None:
            # It's acceptable if nothing is returned
            # If something is returned, it better be a perfect match workflow
            for workflow in results:
                # The deploy workflow has more steps than just PR + review
                if workflow.name == "Deploy feature to staging":
                    self.fail(f"Partial overlap should not return 'Deploy feature to staging' workflow as it includes many more steps than just PR and review")

    def test_onboarding_exact_match(self):
        """Test case 8: Exact match for onboarding should return onboarding workflow"""
        task = self._create_task(
            "Set up the new software engineer with development environment, credentials, and schedule orientation meetings"
        )

        results = self.search_agent.query_workflows_for_task(task)

        self.assertIsNotNone(results, "Onboarding task should return workflows")
        self.assertGreater(len(results), 0, "Should return at least one workflow")

        workflow_names = [w.name for w in results]
        self.assertIn("Onboard new team member", workflow_names,
                     "Should return the onboarding workflow")

    def test_general_category_match_returns_nothing(self):
        """Test case 9: Same general category but different specific task should NOT return"""
        task = self._create_task(
            "Set up my personal laptop for software development"
        )

        results = self.search_agent.query_workflows_for_task(task)

        # This is about setting up development environment, similar to onboarding
        # But it's for personal use, not onboarding a new team member
        # Should not return the onboarding workflow
        if results is not None:
            onboarding_workflows = [w for w in results if "onboard" in w.name.lower()]
            self.assertEqual(len(onboarding_workflows), 0,
                           f"Personal laptop setup should not match team onboarding workflow. Got: {[w.name for w in onboarding_workflows]}")


def run_strict_matching_tests():
    """Run the strict matching test suite"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStrictWorkflowMatching)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_strict_matching_tests())
