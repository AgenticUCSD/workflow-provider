"""Slot population: fill missing task parameters from memory-unit context.

Before falling back to a human (HITL), we try to resolve the parameters the
email did *not* supply from the user's own context via memory-unit. Only
``missing`` slots are touched; values the email provided (``present``) are never
overwritten. Everything is best-effort: if memory-unit is disabled or
unreachable the task is returned unchanged.
"""

from typing import Optional

from utils.memory_client import resolve_slots
from utils.task import Task


def populate_context_items(
    task: Task,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    min_confidence: float = 0.0,
) -> Task:
    """Fill ``missing`` context items on ``task`` from memory-unit.

    A slot is only filled when memory-unit returns a value whose confidence is
    ``>= min_confidence``. Filled slots are marked ``status="guessed"`` (not
    ``"present"``) and carry ``source`` + ``confidence`` so the UI can still
    surface them for the user to confirm — a resolved-from-context value is a
    suggestion, not ground truth. Returns the same ``task`` (mutated in place).
    """
    items = task.context_items or []
    missing = [ci for ci in items if ci.status == "missing" or ci.value in (None, "")]
    if not missing:
        return task

    fields = [ci.field for ci in missing]
    resolved = resolve_slots(fields, user_id=user_id, thread_id=thread_id)
    by_field = {r.get("field"): r for r in resolved if isinstance(r, dict)}

    for ci in missing:
        r = by_field.get(ci.field)
        if not r:
            continue
        value = r.get("value")
        confidence = r.get("confidence") or 0.0
        if value and confidence >= min_confidence:
            ci.value = value
            ci.source = r.get("source") or "context"
            ci.confidence = confidence
            ci.status = "guessed"

    return task
