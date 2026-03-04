from typing import List
from utils.model import model
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from task_identification.task import Task, Workflow
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

    def create_workflow_initial(self, task: Task, rejected_workflows: List[Workflow] = None):
        # Generate a unique thread ID for this task
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}, "callbacks": [CallbackHandler()]}
        
        content = f"Generate a workflow for this task given the workflows the user rejected. Task: {task.model_dump()}\n\n. Rejected workflows: { [w.model_dump() for w in rejected_workflows] if rejected_workflows else 'None'}"

        chat = [
            {
                "role": "user",
                "content": content,
            }
        ]

        result = self.agent.invoke({"messages": chat}, config=config)
        return self.extract_workflow(result)

    def edit_proposed_workflow(self, task: Task, proposed_workflow: Workflow, feedback: str):
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
        if isinstance(result, Workflow):
            return result

        if isinstance(result, dict):
            for key in ("output", "structured_output"):
                if key in result:
                    return Workflow.model_validate(result[key])

            messages = result.get("messages")
            if isinstance(messages, list):
                for message in reversed(messages):
                    content = getattr(message, "content", None)
                    if isinstance(content, dict):
                        try:
                            return Workflow.model_validate(content)
                        except Exception:
                            pass

                    additional = getattr(message, "additional_kwargs", None)
                    if isinstance(additional, dict):
                        for key in ("tool_calls", "parsed", "structured_output", "output"):
                            payload = additional.get(key)
                            if isinstance(payload, dict):
                                try:
                                    return Workflow.model_validate(payload)
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
                                    return Workflow.model_validate(args)
                                except Exception:
                                    pass

            try:
                return Workflow.model_validate(result)
            except Exception:
                pass

        raise ValueError("Could not parse workflow from agent result")

    def process_result(self, result):
        for message in result["messages"]:
            if hasattr(message, 'pretty_print'):
                message.pretty_print()
            else:
                print(f"{message.type}: {message.content}")


