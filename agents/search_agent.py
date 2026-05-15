from typing import List, Optional
from pydantic import BaseModel
from utils.model import extract_structured_output, model
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from deepeval.integrations.langchain import CallbackHandler

from utils.task import Task, Workflow
from utils.chroma import ChromaVectorStore

import uuid


class WorkflowSearchResult(BaseModel):
    """Result from workflow search - can contain workflows or be empty if no good matches found"""
    workflows: Optional[List[Workflow]] = None



class SearchAgent:
    def __init__(self, vector_db: ChromaVectorStore | None = None):
        self.agent = create_agent(
            model=model,
            response_format=ToolStrategy(WorkflowSearchResult),
            system_prompt=(
                """You are a RAG workflow search agent with STRICT matching criteria. You take in a task object with additional context and a list of workflows.

                MATCHING RULE: Only return a workflow if it is a 95%+ semantic match to the task.

                A 95%+ match means:
                - The workflow directly solves the task described (not a similar but different task)
                - The workflow's name and description align with the task's description
                - The workflow addresses the same domain and primary action as the task
                - The workflow would NOT require significant modification to complete the task

                Examples of 95%+ matches:
                - Task: "Schedule and run our daily team standup meeting" → Workflow: "Organize team standup" ✓
                - Task: "Critical bug in production causing login failures - fix immediately" → Workflow: "Fix critical production bug" ✓

                Examples that are NOT 95%+ matches (DO NOT return):
                - Task: "Schedule a 1-on-1 with my manager" → Workflow: "Organize team standup" (different meeting type) ✗
                - Task: "Book a doctor's appointment" → Workflow: "Organize team standup" (different domain entirely) ✗
                - Task: "Create a pull request" → Workflow: "Deploy feature to staging" (only partial overlap) ✗

                Return formats:
                - If you find workflow(s) that are 95%+ matches: return them in order of relevance (best match first)
                - If NO workflows meet the 95%+ threshold: return empty result (workflows: null or [])

                You can only return workflows that are in the list of provided workflows.

                When in doubt between two workflows where one is clearly a better match, return only the better match.
                When workflows are only somewhat related, return nothing.
                """
            ),
        )
        self.vector_db = vector_db if vector_db is not None else ChromaVectorStore()

    def populate_manual_workflows(self, workflows: List[Workflow]) -> List[str]:
        return [
            self.vector_db.add_workflow(workflow=workflow, is_generated=False)
            for workflow in workflows
        ]

    def query_workflows_for_task(self, task: Task, thread_id: str | None = None) -> List[Workflow] | None:
        # Use provided thread_id or generate a unique one for this task
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}, "callbacks": [CallbackHandler()]}
        
        candidate_workflows = self.vector_db.query_from_all_workflows_as_objects(task=task, top_k=5)
        if not candidate_workflows:
            return None
        
        content = (
            "Given the following task, return a list of relevant workflows that would solve that task from the ones given to you. "
            f"Task: {task.model_dump()}\n\n"
            f"Candidate workflows: {[w.model_dump() for w in candidate_workflows]}"
        )

        chat = [
            {
                "role": "user",
                "content": content,
            }
        ]

        result = self.agent.invoke({"messages": chat}, config=config)
        parsed_workflows = self.extract_workflows(result)
        return parsed_workflows

    def extract_workflows(self, result) -> List[Workflow] | None:
        if isinstance(result, list) and all(isinstance(w, Workflow) for w in result):
            return result

        parsed = extract_structured_output(result, WorkflowSearchResult, raise_on_error=False)
        if parsed is not None:
            return parsed.workflows

        return None  # Return None instead of raising error if no workflows found

    def process_result(self, result):
        for message in result["messages"]:
            if hasattr(message, 'pretty_print'):
                message.pretty_print()
            else:
                print(f"{message.type}: {message.content}")


