import os
import uuid
from enum import Enum

from deepeval.integrations.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel

from utils.model import extract_structured_output, model


class IntentLabel(str, Enum):
    POPULATE_CONTEXT = "populate_context"
    PROVENANCE = "provenance"
    STATUS = "status"
    HELP = "help"
    ACTION = "action"


class IntentResult(BaseModel):
    intent: IntentLabel


INTENT_PROMPT = """You are a chat-intent classifier for an agentic email assistant's side panel.
Classify the user's chat message into exactly one of these five intents:

- status: The user is asking about progress — "where are we", what's currently running,
  or asking for an update on the task/workflow.
- provenance: The user is asking where a value came from — "how did you know that",
  "what's the source of this", questioning the origin of a field or fact.
- populate_context: The user is asking to fill in task fields from their saved
  context/memory (e.g. "use my usual info", "fill this in from what you know about me").
  Only meaningful during the "task" or "refining" phase; in any other phase, prefer
  "action" instead.
- help: The user is asking what they can do or say here — asking for options or
  guidance on how to use the assistant.
- action: Anything else — a concrete command, instruction, correction, or refinement
  the user wants applied.

You are given the current Phase along with the user's message; use it to disambiguate
(especially for populate_context, which only applies in the "task"/"refining" phase).

When uncertain, choose "action".

Return exactly one of: populate_context, provenance, status, help, action.""".strip()


def intent_router_enabled() -> bool:
    return os.getenv("INTENT_ROUTER_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


class IntentClassifierAgent:
    def __init__(self) -> None:
        self.agent = create_agent(
            model=model,
            response_format=ToolStrategy(IntentResult),
            system_prompt=INTENT_PROMPT,
        )

    def _agent_config(self, thread_id: str | None = None) -> dict:
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        return {
            "configurable": {"thread_id": thread_id},
            "callbacks": [CallbackHandler(thread_id=thread_id)],
        }

    def classify(self, text: str, phase: str, thread_id: str | None = None) -> IntentResult | None:
        content = f"Phase: {phase}\n\nUser message:\n{text}"
        chat = [{"role": "user", "content": content}]
        result = self.agent.invoke({"messages": chat}, config=self._agent_config(thread_id))
        return extract_structured_output(result, IntentResult, raise_on_error=False)
