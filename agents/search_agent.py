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
    def __init__(self):
        self.agent = create_agent(
            model=model,
            response_format=ToolStrategy(WorkflowSearchResult),
            system_prompt=(
                """You are a RAG workflow search agent. You take in a task object with additional context and a list of workflows. You will return a WorkflowSearchResult containing a list of relevant workflows that would solve that task. 
                The workflows should be returned in order of relevance, with the most relevant workflow first. Return only workflow objects that conform to the required schema.
                You can only return workflows that are in the list of provided workflows. Do not generate new workflows and do not edit existing workflows.
                
                IMPORTANT: If none of the provided workflows are a good match for the task, return an empty result (workflows: null or workflows: []) 
                Only return workflows that are truly relevant and would fully solve the given task. Any workflows that are tangentionally related should not be returned.\n\n
                """
            ),
        )
        self.vector_db = ChromaVectorStore()

    def populate_manual_workflows(self, workflows: List[Workflow]) -> List[str]:
        return [
            self.vector_db.add_workflow(workflow=workflow, is_generated=False)
            for workflow in workflows
        ]

    def query_workflows_for_task(self, task: Task) -> List[Workflow] | None:
        # Generate a unique thread ID for this task
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
        if parsed_workflows is None:
            return candidate_workflows
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


