from typing import List, Optional
from pydantic import BaseModel
from utils.model import model
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

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
                Only return workflows that are truly relevant and would help solve the given task.\n\n
                """
            ),
        )
        self.vector_db = ChromaVectorStore()

    def query_workflows_for_task(self, task: Task) -> List[Workflow] | None:
        # Generate a unique thread ID for this task
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        vector_db_results = self.vector_db.query_from_all_workflows(task=task, top_k=5)
        
        content = f"Given the following task, return a list of relevant workflows that would solve that task from the ones given to you. Task: {task.model_dump()}\n\n. Here are the set of workflows from the vector database: {vector_db_results}"

        chat = [
            {
                "role": "user",
                "content": content,
            }
        ]

        result = self.agent.invoke({"messages": chat}, config=config)
        return self.extract_workflows(result)

    def extract_workflows(self, result) -> List[Workflow] | None:
        # Handle WorkflowSearchResult
        if isinstance(result, WorkflowSearchResult):
            return result.workflows
        
        if isinstance(result, list) and all(isinstance(w, Workflow) for w in result):
            return result

        if isinstance(result, dict):
            for key in ("output", "structured_output"):
                if key in result:
                    search_result = WorkflowSearchResult.model_validate(result[key])
                    return search_result.workflows

            messages = result.get("messages")
            if isinstance(messages, list):
                for message in reversed(messages):
                    content = getattr(message, "content", None)
                    if isinstance(content, dict):
                        try:
                            search_result = WorkflowSearchResult.model_validate(content)
                            return search_result.workflows
                        except Exception:
                            pass

                    additional = getattr(message, "additional_kwargs", None)
                    if isinstance(additional, dict):
                        for key in ("tool_calls", "parsed", "structured_output", "output"):
                            payload = additional.get(key)
                            if isinstance(payload, dict):
                                try:
                                    search_result = WorkflowSearchResult.model_validate(payload)
                                    return search_result.workflows
                                except Exception:
                                    pass

                    tool_calls = getattr(message, "tool_calls", None)
                    if isinstance(tool_calls, list):
                        for call in tool_calls:
                            args = None
                            if isinstance(call, dict):
                                args = call.get("args")
                            else:
                                args = getattr(call, "args", None)
                            if isinstance(args, dict):
                                try:
                                    search_result = WorkflowSearchResult.model_validate(args)
                                    return search_result.workflows
                                except Exception:
                                    pass

            try:
                search_result = WorkflowSearchResult.model_validate(result)
                return search_result.workflows
            except Exception:
                pass

        return None  # Return None instead of raising error if no workflows found

    def process_result(self, result):
        for message in result["messages"]:
            if hasattr(message, 'pretty_print'):
                message.pretty_print()
            else:
                print(f"{message.type}: {message.content}")


