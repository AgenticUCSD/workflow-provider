import json
import re
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Literal

from deepeval.integrations.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field

from utils.model import extract_structured_output
from utils.task import Objective, Status, Task, TaskTypes

MetadataValue = str | int | float | bool | None
Metadata = dict[str, MetadataValue]

TagName = Literal[
    "no-task",
    "action-request",
    "reply-needed",
    "review-feedback",
    "schedule",
    "commitment-track",
    "escalation-urgent",
    "forward-delegate",
]


class IntentTag(BaseModel):
    tag: TagName
    short_description: str


class TagResult(BaseModel):
    tags: list[IntentTag]


class DeadlineResult(BaseModel):
    has_deadline: bool
    deadline_iso: str | None
    rationale: str


class ContextItemType(str, Enum):
    PARTICIPANTS = "participants"
    TIME_WINDOW = "time_window"
    TIMEZONE = "timezone"
    ARTIFACT_LINK = "artifact_link"
    QUESTION_CONTEXT = "question_context"
    DELIVERABLE_DEFINITION = "deliverable_definition"
    OWNER = "owner"
    INCIDENT_STATE = "incident_state"
    IMPACT = "impact"


class ContextItem(BaseModel):
    field: str
    status: Literal["present", "missing"]
    value: str | None = None


class ContextPlan(BaseModel):
    context_items: list[ContextItem]


class ContextRequirementResult(BaseModel):
    required_context: list[str]


class IdentifyTaskResult(BaseModel):
    status: Literal["identified", "no_task"]
    task: Task | None = None
    detected_tag: str | None = None
    context_items: list[ContextItem] = Field(default_factory=list)


TAG_TO_TASK_TYPE: dict[str, TaskTypes] = {
    "no-task": TaskTypes.NO_TASK,
    "action-request": TaskTypes.ACTION_REQUIRED,
    "reply-needed": TaskTypes.REPLY_NEEDED,
    "review-feedback": TaskTypes.REVIEW_FEEDBACK,
    "schedule": TaskTypes.SCHEDULE,
    "forward-delegate": TaskTypes.FORWARD_DELEGATE,
    "commitment-track": TaskTypes.COMMITMENT_TRACK,
    "escalation-urgent": TaskTypes.ESCALATION_URGENT,
}

TAG_PRIORITY: dict[str, int] = {
    "escalation-urgent": 0,
    "action-request": 1,
    "review-feedback": 2,
    "schedule": 3,
    "reply-needed": 4,
    "forward-delegate": 5,
    "commitment-track": 6,
    "no-task": 7,
}

CLASSIFICATION_PROMPT = """
You are an email triage classifier.
Identify ALL task-related intents from this text.
Allowed tags: no-task, action-request, reply-needed, review-feedback, schedule, commitment-track, escalation-urgent, forward-delegate.
Rules:
- If no-task is selected, it must be the only tag.
- Return all applicable actionable tags otherwise.
- Keep tags in priority order.
- short_description must be 3 to 5 words.
Return only JSON matching TagResult.
""".strip()

DEADLINE_PROMPT = """
Extract task deadline from email text.
If text includes a real deadline (for example: by 5pm today, before Friday, EOD), set has_deadline=true and provide deadline_iso in ISO8601.
If no deadline exists, set has_deadline=false and deadline_iso=null.
For schedule tasks, treat phrases like schedule by 5pm today as deadline for completing scheduling.
Return only JSON matching DeadlineResult.
""".strip()

CONTEXT_PLANNER_PROMPT = """
Determine required context fields for executing detected task types.
Return only fields from allowed_fields.
Return only JSON matching ContextRequirementResult.
""".strip()

TASK_EDITOR_PROMPT = """
Revise the provided task using the user's feedback.
The feedback may include explicit missing context items and/or general guidance.
Preserve the original task identity and status unless the feedback clearly requires a change.
Merge any filled-in context into the task objective inputs, constraints, or metadata as appropriate.
Return only JSON matching Task.
""".strip()

CONTEXT_REQUIRED_BY_TASK_TYPE: dict[TaskTypes, list[ContextItemType]] = {
    TaskTypes.SCHEDULE: [
        ContextItemType.PARTICIPANTS,
        ContextItemType.TIME_WINDOW,
        ContextItemType.TIMEZONE,
    ],
    TaskTypes.REVIEW_FEEDBACK: [ContextItemType.ARTIFACT_LINK],
    TaskTypes.REPLY_NEEDED: [ContextItemType.QUESTION_CONTEXT],
    TaskTypes.ACTION_REQUIRED: [ContextItemType.DELIVERABLE_DEFINITION],
    TaskTypes.ESCALATION_URGENT: [
        ContextItemType.INCIDENT_STATE,
        ContextItemType.IMPACT,
        ContextItemType.OWNER,
    ],
}


class TaskIdentifierAgent:
    def __init__(self) -> None:
        from utils.model import model

        self.tag_agent = create_agent(
            model=model,
            response_format=ToolStrategy(TagResult),
            system_prompt="You classify email intents and return only structured output.",
        )
        self.deadline_agent = create_agent(
            model=model,
            response_format=ToolStrategy(DeadlineResult),
            system_prompt="You extract deadlines and return only structured output.",
        )
        self.context_planner_agent = create_agent(
            model=model,
            response_format=ToolStrategy(ContextRequirementResult),
            system_prompt="You identify required execution context fields and return only structured output.",
        )
        self.task_editor_agent = create_agent(
            model=model,
            response_format=ToolStrategy(Task),
            system_prompt="You revise tasks using feedback and return only structured output.",
        )

        # Backward-compatible aliases used by existing tests and callers.
        self.tag_model = self.tag_agent
        self.deadline_model = self.deadline_agent
        self.context_planner_model = self.context_planner_agent
        self.task_editor_model = self.task_editor_agent

    def _get_structured_runner(self, *attribute_names: str):
        for name in attribute_names:
            candidate = getattr(self, name, None)
            if candidate is not None:
                return candidate
        raise AttributeError(f"TaskIdentifierAgent is missing structured runner: {attribute_names}")

    def _agent_config(self) -> dict[str, object]:
        thread_id = str(uuid.uuid4())
        return {
            "configurable": {"thread_id": thread_id},
            "callbacks": [CallbackHandler()],
        }

    def _invoke_structured_agent(
        self,
        agent,
        payload: dict[str, object],
        model_type: type[BaseModel],
    ) -> BaseModel:
        chat = [{"role": "user", "content": json.dumps(payload, ensure_ascii=True)}]
        try:
            result = agent.invoke({"messages": chat}, config=self._agent_config())
        except TypeError:
            # Some tests inject lightweight stubs that only accept one positional input.
            try:
                result = agent.invoke(json.dumps(payload, ensure_ascii=True))
            except TypeError:
                result = agent.invoke({"messages": chat})
        parsed = extract_structured_output(result, model_type)
        if parsed is None:
            raise ValueError(f"Could not parse {model_type.__name__} from agent result")
        return parsed

    def preprocess_email(self, text: str, subject: str | None) -> str:
        normalized_lines = text.replace("\r\n", "\n").split("\n")
        core_lines: list[str] = []
        for line in normalized_lines:
            stripped = line.strip()
            lowered = stripped.lower()
            if lowered.startswith("from:") or lowered.startswith("sent:"):
                break
            if re.match(r"^on .+ wrote:$", lowered):
                break
            if lowered.startswith("forwarded message"):
                break
            if stripped.startswith(">"):
                continue
            core_lines.append(line)
        body = self._trim_signature_and_footer(core_lines)
        return f"Subject: {subject or ''}\n\nBody:\n{body}".strip()

    def identify_task(self, text: str, subject: str | None, metadata: Metadata | None) -> IdentifyTaskResult:
        processed = self.preprocess_email(text, subject)
        tag_result = self.detect_tags(processed, metadata)
        detected_tags = [item.tag for item in tag_result.tags]

        if any(tag == "no-task" for tag in detected_tags):
            return IdentifyTaskResult(
                status="no_task",
                task=None,
                detected_tag="no-task",
                context_items=[],
            )

        prioritized_tag = self.prioritized_actionable_tag(tag_result)
        if prioritized_tag is None:
            return IdentifyTaskResult(
                status="no_task",
                task=None,
                detected_tag="no-task",
                context_items=[],
            )

        prioritized_tag_result = TagResult(
            tags=[IntentTag(tag=prioritized_tag.tag, short_description=prioritized_tag.short_description)]
        )

        context_plan = self.determine_context(
            tag_result=prioritized_tag_result,
            processed_text=processed,
            metadata=metadata,
        )
        task = self.tags_to_task(
            tag_result=prioritized_tag_result,
            processed_text=processed,
            metadata=metadata,
        )
        if task is None:
            return IdentifyTaskResult(
                status="no_task",
                task=None,
                detected_tag="no-task",
                context_items=context_plan.context_items,
            )

        return IdentifyTaskResult(
            status="identified",
            task=task,
            detected_tag=prioritized_tag.tag,
            context_items=context_plan.context_items,
        )

    def edit_task(self, task: Task, user_feedback: str) -> Task:
        payload = {
            "instructions": TASK_EDITOR_PROMPT,
            "task": task.model_dump(),
            "user_feedback": user_feedback,
        }
        editor_runner = self._get_structured_runner("task_editor_agent", "task_editor_model")
        response = self._invoke_structured_agent(editor_runner, payload, Task)
        parsed = response if isinstance(response, Task) else Task.model_validate(response)
        return parsed

    def _trim_signature_and_footer(self, lines: list[str]) -> str:
        markers = {"--", "thanks,", "best,", "regards,", "sent from my", "confidentiality notice"}
        result: list[str] = []
        for line in lines:
            if line.strip().lower() in markers:
                break
            result.append(line)
        return "\n".join(result).strip()

    def detect_tags(self, processed_text: str, metadata: Metadata | None) -> TagResult:
        payload = {
            "instructions": CLASSIFICATION_PROMPT,
            "processed_text": processed_text,
            "metadata": metadata or {},
        }
        tag_runner = self._get_structured_runner("tag_agent", "tag_model")
        response = self._invoke_structured_agent(tag_runner, payload, TagResult)
        parsed = response if isinstance(response, TagResult) else TagResult.model_validate(response)
        return self.normalize_tags(parsed)

    def normalize_tags(self, tag_result: TagResult) -> TagResult:
        normalized: list[IntentTag] = []
        seen: set[str] = set()
        for item in tag_result.tags:
            canonical = self._canonicalize_tag(item.tag)
            if canonical is None or canonical in seen:
                continue
            normalized.append(
                IntentTag(
                    tag=canonical,
                    short_description=self._normalize_short_description(item.short_description),
                )
            )
            seen.add(canonical)

        if not normalized or any(item.tag == "no-task" for item in normalized):
            return TagResult(tags=[IntentTag(tag="no-task", short_description="no action required")])

        normalized.sort(key=lambda item: TAG_PRIORITY[item.tag])
        return TagResult(tags=normalized)

    def _canonicalize_tag(self, tag: str) -> TagName | None:
        normalized = tag.strip().lower().replace("_", "-")
        normalized = normalized.replace("action required", "action-request")
        normalized = normalized.replace("reply needed", "reply-needed")
        normalized = normalized.replace("review feedback", "review-feedback")
        normalized = normalized.replace("forward delegate", "forward-delegate")
        normalized = normalized.replace("commitment track", "commitment-track")
        normalized = normalized.replace("escalation urgent", "escalation-urgent")
        valid: set[str] = set(TAG_TO_TASK_TYPE.keys())
        if normalized in valid:
            return normalized  # type: ignore[return-value]
        return None

    def _normalize_short_description(self, description: str) -> str:
        words = re.findall(r"[A-Za-z0-9]+", description.lower())
        if not words:
            return "follow up task required"
        if len(words) < 3:
            words.extend(["task"] * (3 - len(words)))
        if len(words) > 5:
            words = words[:5]
        return " ".join(words)

    def plan_required_context(self, tag_result: TagResult, processed_text: str) -> list[str]:
        task_types = [TAG_TO_TASK_TYPE[item.tag] for item in tag_result.tags if item.tag != "no-task"]
        required: set[str] = set()
        for task_type in task_types:
            for field in CONTEXT_REQUIRED_BY_TASK_TYPE.get(task_type, []):
                required.add(field.value)

        allowed_fields = [item.value for item in ContextItemType]
        payload = {
            "instructions": CONTEXT_PLANNER_PROMPT,
            "processed_text": processed_text,
            "task_types": [item.value for item in task_types],
            "allowed_fields": allowed_fields,
        }
        context_runner = self._get_structured_runner("context_planner_agent", "context_planner_model")
        response = self._invoke_structured_agent(context_runner, payload, ContextRequirementResult)
        parsed = (
            response
            if isinstance(response, ContextRequirementResult)
            else ContextRequirementResult.model_validate(response)
        )
        for field in parsed.required_context:
            if field in allowed_fields:
                required.add(field)
        return sorted(required)

    def determine_context(self, tag_result: TagResult, processed_text: str, metadata: Metadata | None) -> ContextPlan:
        required_fields = self.plan_required_context(tag_result, processed_text)
        resolved = self._auto_resolve_context(processed_text, metadata)
        items: list[ContextItem] = []

        for field in required_fields:
            value = resolved.get(field)
            if value:
                items.append(ContextItem(field=field, status="present", value=value))
            else:
                items.append(ContextItem(field=field, status="missing", value=None))
        return ContextPlan(context_items=items)

    def _auto_resolve_context(self, processed_text: str, metadata: Metadata | None) -> dict[str, str]:
        lowered = processed_text.lower()
        resolved: dict[str, str] = {}

        emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}", processed_text)
        if emails:
            resolved[ContextItemType.PARTICIPANTS.value] = ", ".join(emails)
        if re.search(r"\\b(today|tomorrow|next week|next|monday|tuesday|wednesday|thursday|friday)\\b", lowered):
            resolved[ContextItemType.TIME_WINDOW.value] = "time window detected"
        timezone_match = re.search(r"\\b(utc|pt|pst|pdt|est|edt|cst|cdt)\\b", lowered)
        if timezone_match:
            resolved[ContextItemType.TIMEZONE.value] = timezone_match.group(1).upper()
        link_match = re.search(r"https?://\\S+", processed_text)
        if link_match:
            resolved[ContextItemType.ARTIFACT_LINK.value] = link_match.group(0)
        if "?" in processed_text:
            resolved[ContextItemType.QUESTION_CONTEXT.value] = "question detected"
        if re.search(r"\\b(send|submit|create|fix|complete|deliver)\\b", lowered):
            resolved[ContextItemType.DELIVERABLE_DEFINITION.value] = "deliverable detected"
        if re.search(r"\\bowner|assigned|i will|we will\\b", lowered):
            resolved[ContextItemType.OWNER.value] = "owner detected"
        if re.search(r"\\bincident|outage|degraded|down|failing\\b", lowered):
            resolved[ContextItemType.INCIDENT_STATE.value] = "incident state detected"
        if re.search(r"\\bimpact|blocked|revenue|customers\\b", lowered):
            resolved[ContextItemType.IMPACT.value] = "impact detected"

        for field, value in (metadata or {}).items():
            if isinstance(value, str) and field in [item.value for item in ContextItemType]:
                resolved[field] = value
        return resolved

    def detect_deadline(self, tag_result: TagResult, processed_text: str) -> DeadlineResult:
        payload = {
            "instructions": DEADLINE_PROMPT,
            "processed_text": processed_text,
            "detected_tags": [item.tag for item in tag_result.tags],
        }
        deadline_runner = self._get_structured_runner("deadline_agent", "deadline_model")
        response = self._invoke_structured_agent(deadline_runner, payload, DeadlineResult)
        parsed = response if isinstance(response, DeadlineResult) else DeadlineResult.model_validate(response)
        if parsed.has_deadline and parsed.deadline_iso:
            return parsed
        fallback = self._fallback_deadline_iso(processed_text)
        if fallback:
            return DeadlineResult(
                has_deadline=True,
                deadline_iso=fallback,
                rationale="fallback parsed from explicit by-time phrase",
            )
        return parsed

    def _fallback_deadline_iso(self, processed_text: str) -> str | None:
        lowered = processed_text.lower()
        now = datetime.now().astimezone()
        if re.search(r"\b(?:by\s+)?eod\s+today\b", lowered):
            return now.replace(hour=23, minute=59, second=0, microsecond=0).isoformat()

        time_match = re.search(
            r"\bby\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s+(today|tomorrow)\b",
            lowered,
        )
        if not time_match:
            return None

        hour = int(time_match.group(1))
        minute = int(time_match.group(2) or "0")
        meridian = time_match.group(3)
        day_token = time_match.group(4)
        if hour == 12:
            hour_24 = 0 if meridian == "am" else 12
        else:
            hour_24 = hour if meridian == "am" else hour + 12
        base_day = now.date() if day_token == "today" else (now + timedelta(days=1)).date()
        deadline = now.replace(
            year=base_day.year,
            month=base_day.month,
            day=base_day.day,
            hour=hour_24,
            minute=minute,
            second=0,
            microsecond=0,
        )
        return deadline.isoformat()

    def prioritized_actionable_tag(self, tag_result: TagResult) -> IntentTag | None:
        actionable_tags = [item for item in tag_result.tags if item.tag != "no-task"]
        if not actionable_tags:
            return None
        return min(actionable_tags, key=lambda item: TAG_PRIORITY.get(item.tag, 999))

    def tags_to_task(self, tag_result: TagResult, processed_text: str, metadata: Metadata | None) -> Task | None:
        deadline = self.detect_deadline(tag_result, processed_text)
        selected = self.prioritized_actionable_tag(tag_result)
        if selected is None:
            return None

        return self.build_task(
            task_type=TAG_TO_TASK_TYPE[selected.tag],
            description=selected.short_description,
            processed_text=processed_text,
            deadline=deadline,
            metadata=metadata,
        )

    def tags_to_tasks(self, tag_result: TagResult, processed_text: str, metadata: Metadata | None) -> list[Task]:
        """Backward-compatible wrapper that now returns at most one task."""
        task = self.tags_to_task(tag_result, processed_text, metadata)
        return [task] if task is not None else []

    def build_task(
        self,
        task_type: TaskTypes,
        description: str,
        processed_text: str,
        deadline: DeadlineResult,
        metadata: Metadata | None,
    ) -> Task:
        constraints: dict[str, MetadataValue] = {}
        if task_type == TaskTypes.SCHEDULE and deadline.has_deadline and deadline.deadline_iso is not None:
            constraints["latest_scheduling_time"] = deadline.deadline_iso

        objective = Objective(
            objective_id=f"obj_{uuid.uuid4().hex[:8]}",
            name=f"{task_type.value} task",
            description=description,
            inputs={"processed_text": processed_text},
            constraints=constraints,
            success_criteria=(
                f"Task completed before {deadline.deadline_iso}"
                if deadline.has_deadline and deadline.deadline_iso is not None
                else "Task completed successfully"
            ),
            expected_output={"status": "completed"},
            deadline=deadline.deadline_iso if deadline.has_deadline else None,
        )

        task_metadata: Metadata = dict(metadata or {})
        task_metadata["detected_task_type"] = task_type.value
        task_metadata["source_channel"] = "email_text"
        task_metadata["deadline_detected"] = deadline.has_deadline

        return Task(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            task_type=task_type,
            objective=objective,
            candidate_workflows=None,
            workflow=None,
            status=Status.PENDING,
            metadata=task_metadata,
        )
