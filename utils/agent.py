from utils.model import model
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from utils.task import Task, Workflow

import json
import uuid


agent = create_agent(
    model=model,
    response_format=ToolStrategy(Workflow),
    system_prompt=(
        "You are a workflow planner. You take in a task object and produce a workflow for it. "
        "Return only a workflow object that conforms to the required schema.\n\n"
        "The workflow should be broken down into steps that are as atomic as possible. "
        "Each step should be a task that can be completed in one turn (tool call, llm call, etc)."
    ),
)

def run_agent(task: Task):

    # Generate a unique thread ID for this task
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    content = f"Task: {task.model_dump()}\n\nGenerate a workflow for this task."

    chat = [
        {
            "role": "user",
            "content": content,
        }
    ]

    result = agent.invoke({"messages": chat}, config=config)
    return result


def extract_workflow(result) -> Workflow:
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


def task_to_workflow(task: Task) -> Workflow:
    result = run_agent(task)
    return extract_workflow(result)


def process_result(result):
    for message in result["messages"]:
        if hasattr(message, 'pretty_print'):
            message.pretty_print()
        else:
            print(f"{message.type}: {message.content}")


if __name__ == "__main__":
    with open("./prompts/mock_task_send_email.txt", encoding="utf-8") as f:
        task_data = json.load(f)

    task = Task(**task_data)
    result = run_agent(task)
    process_result(result)