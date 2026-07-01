import uuid
from typing import Optional

from deepeval.integrations.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field

from utils.model import extract_structured_output, model
from utils.task import ContextItem, Objective, Status, Task, TaskTypes

MetadataValue = str | int | float | bool | None
Metadata = dict[str, MetadataValue]


class _TaskExtraction(BaseModel):
    """Internal model for LLM extraction output."""
    task_type: TaskTypes
    priority: Optional[str] = None
    objective_name: str
    objective_description: str
    deadline_iso: Optional[str] = None
    context_items: list[ContextItem] = Field(default_factory=list)


class IdentifyTaskResult(BaseModel):
    """Task identification result containing the constructed Task and context."""
    task: Optional[Task] = None
    context_items: list[ContextItem] = Field(default_factory=list)


SYSTEM_PROMPT = """You are an email task identification agent. Analyze email content and extract structured task information.

Task Types:
- draft: Create content (email, document, message, etc.)
- review: Review or approve something
- schedule: Find time or book a meeting
- respond: Reply to a communication
- execute: Do a concrete task or action
- decision: Make or record a decision
- delegate: Hand off to someone else
- no_task: Nothing actionable

Priority levels (if discernible):
- urgent: Production issues, deadlines today, blocking others
- high: Important but not immediate crisis
- normal: Standard work
- low: FYI, whenever

Context fields by task type (extract or infer what you can):
- draft: topic, format, audience, key_points
- review: artifact_link, review_criteria, deadline
- schedule: participants, duration, time_window, timezone, meeting link
- respond: question_context, sender_priority
- execute: deliverable_description, resources_needed
- decision: stakeholders, decision_criteria, default_action
- delegate: delegatee, handoff_context, follow_up_needed

For each context item, set status to:
- "present": explicitly found in the text
- "guessed": inferred or reasonably assumed
- "missing": needed but not found or guessable

Deadline extraction rules:
- If email mentions a real deadline (e.g., "by 5pm today", "before Friday", "EOD"), extract as ISO8601
- Schedule tasks may have deadlines for completing the scheduling itself
- If no deadline, leave deadline_iso null
""".strip()

TASK_EDITOR_PROMPT = """You are a task editor. Revise the provided task using the user's feedback.
Preserve the original task identity unless feedback clearly requires a change.
Merge any filled-in context into the task appropriately.
Update context_items status as needed (present/missing/guessed).
Return the complete updated task with all fields.""".strip()


class TaskIdentifierAgent:
    def __init__(self) -> None:
        self.agent = create_agent(
            model=model,
            response_format=ToolStrategy(_TaskExtraction),
            system_prompt=SYSTEM_PROMPT,
        )
        self.task_editor_agent = create_agent(
            model=model,
            response_format=ToolStrategy(Task),
            system_prompt=TASK_EDITOR_PROMPT,
        )

    def _agent_config(self, thread_id: str | None = None) -> dict:
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        return {
            "configurable": {"thread_id": thread_id},
            "callbacks": [CallbackHandler()],
        }

    def preprocess_email(self, text: str, subject: str | None) -> str:
        normalized_lines = text.replace("\r\n", "\n").split("\n")
        core_lines: list[str] = []
        for line in normalized_lines:
            stripped = line.strip()
            lowered = stripped.lower()
            if lowered.startswith("from:") or lowered.startswith("sent:"):
                break
            if lowered.startswith("on ") and lowered.endswith(" wrote:"):
                break
            if lowered.startswith("forwarded message"):
                break
            if stripped.startswith(">"):
                continue
            core_lines.append(line)
        body = self._trim_signature_and_footer(core_lines)
        return f"Subject: {subject or ''}\n\nBody:\n{body}".strip()

    def _trim_signature_and_footer(self, lines: list[str]) -> str:
        markers = {"--", "thanks,", "best,", "regards,", "sent from my", "confidentiality notice"}
        result: list[str] = []
        for line in lines:
            if line.strip().lower() in markers:
                break
            result.append(line)
        return "\n".join(result).strip()

    def identify_task(
        self, text: str, subject: str | None, metadata: Metadata | None, thread_id: str | None = None
    ) -> IdentifyTaskResult:
        processed = self.preprocess_email(text, subject)

        content = f"Analyze this email and extract task information.\n\n{processed}"
        if metadata:
            content += f"\n\nAdditional metadata: {metadata}"

        chat = [{"role": "user", "content": content}]
        result = self.agent.invoke({"messages": chat}, config=self._agent_config(thread_id))
        parsed = extract_structured_output(result, _TaskExtraction)

        if parsed is None:
            raise ValueError("Could not parse task identification result")

        task = self._build_task_from_extraction(parsed, processed, metadata)

        return IdentifyTaskResult(
            task=task,
            context_items=parsed.context_items,
        )

    def _build_task(
        self,
        task_type: TaskTypes,
        priority: Optional[str],
        description: str,
        processed_text: str,
        deadline_iso: Optional[str],
        metadata: Metadata | None,
        context_items: list[ContextItem] | None = None,
    ) -> Task:
        constraints: dict[str, MetadataValue] = {}
        if task_type == TaskTypes.SCHEDULE and deadline_iso:
            constraints["latest_scheduling_time"] = deadline_iso

        objective = Objective(
            objective_id=f"obj_{uuid.uuid4().hex[:8]}",
            name=f"{task_type.value} task",
            description=description,
            inputs={"processed_text": processed_text},
            constraints=constraints,
            success_criteria=f"Task completed before {deadline_iso}" if deadline_iso else "Task completed successfully",
            expected_output={"status": "completed"},
            deadline=deadline_iso,
        )

        task_metadata: Metadata = dict(metadata or {})
        task_metadata["detected_task_type"] = task_type.value
        task_metadata["source_channel"] = "email_text"
        task_metadata["deadline_detected"] = deadline_iso is not None

        return Task(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            task_type=task_type,
            priority=priority,
            objective=objective,
            candidate_workflows=None,
            workflow=None,
            status=Status.PENDING,
            context_items=context_items,
            metadata=task_metadata,
        )

    def _build_task_from_extraction(
        self, extraction: _TaskExtraction, processed_text: str, metadata: Metadata | None
    ) -> Task:
        return self._build_task(
            task_type=extraction.task_type,
            priority=extraction.priority,
            description=extraction.objective_description,
            processed_text=processed_text,
            deadline_iso=extraction.deadline_iso,
            metadata=metadata,
            context_items=extraction.context_items,
        )

    def edit_task(self, task: Task, user_feedback: str, thread_id: str | None = None) -> Task:
        content = f"Original task: {task.model_dump()}\n\nUser feedback: {user_feedback}\n\nReturn the updated task."
        chat = [{"role": "user", "content": content}]
        result = self.task_editor_agent.invoke({"messages": chat}, config=self._agent_config(thread_id))
        parsed = extract_structured_output(result, Task)
        if parsed is None:
            raise ValueError("Could not parse edited task result")
        return parsed
