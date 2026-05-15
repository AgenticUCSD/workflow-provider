from typing import List
from utils.model import extract_structured_output, model
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from utils.task import Task, Workflow
from deepeval.integrations.langchain import CallbackHandler

import uuid


class BuilderAgent:
    def __init__(self):
        self.agent = create_agent(
            model=model,
            response_format=ToolStrategy(Workflow),
            system_prompt=(
                "You are a workflow planner. You take in a task object with additional context and then produce a workflow for it. "
                "Return only a workflow object that conforms to the required schema.\n\n"
                "The workflow should be broken down into steps that are as atomic as possible. "
                "Each step should be a task that can be completed in one turn (tool call, llm call, etc)."
            ),
        )

    def create_workflow_initial(self, task: Task, rejected_workflows: List[Workflow] = None, user_feedback: str = None, thread_id: str | None = None) -> Workflow:
        # Use provided thread_id or generate a unique one for this task
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}, "callbacks": [CallbackHandler()]}
        
        rejected_workflows = [w.model_dump() for w in rejected_workflows] if rejected_workflows else None

        content = f"Generate a workflow for this task given the workflows the user rejected and potential user feedback. Task: {task.model_dump()}\n\n. Rejected workflows: { rejected_workflows }\n\n User feedback on the rejected workflows: {user_feedback}"

        chat = [
            {
                "role": "user",
                "content": content,
            }
        ]

        result = self.agent.invoke({"messages": chat}, config=config)
        return self.extract_workflow(result)

    def edit_proposed_workflow(self, task: Task, proposed_workflow: Workflow, feedback: str, thread_id: str | None = None):
        if thread_id is None:
            config_payload = getattr(proposed_workflow, "config", None)
            if isinstance(config_payload, dict):
                thread_id = config_payload.get("thread_id")
            else:
                thread_id = getattr(config_payload, "thread_id", None)

            if not thread_id:
                thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}, "callbacks": [CallbackHandler()]}

        content = f"Here is the workflow you proposed for the task. Task: {task.model_dump()}\n\n Proposed workflow: {proposed_workflow.model_dump()}\n\nThe user provided the following feedback on how to improve the workflow: {feedback}\n\nEdit the workflow to address the user's feedback. Only return the updated workflow, do not include any explanations."

        chat = [
            {
                "role": "user",
                "content": content,
            }
        ]

        result = self.agent.invoke({"messages": chat}, config=config)
        return self.extract_workflow(result)

    def extract_workflow(self, result) -> Workflow:
        parsed = extract_structured_output(result, Workflow)
        if parsed is None:
            raise ValueError("Could not parse workflow from agent result")
        return parsed

    def process_result(self, result):
        for message in result["messages"]:
            if hasattr(message, 'pretty_print'):
                message.pretty_print()
            else:
                print(f"{message.type}: {message.content}")


