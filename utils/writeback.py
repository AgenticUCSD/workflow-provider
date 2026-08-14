"""Memory write-back: turn a confirmed task's parameters into memory-unit ``/learn``
items, so a later task's ``resolve()`` can pre-fill the same slots.

This is the mirror image of ``utils/population.py`` (which reads *from* memory into
a task's ``missing`` slots): here we read a task's *filled* slots and write them
*into* memory. Pure, no I/O — the network hop lives in ``utils/memory_client.py``.

Two hard constraints (see PIPELINE_REWORK / the write-back task spec):

1. The learned sentence must literally contain every token of the slot name, because
   memory-unit's ``resolve()`` applies a term-coverage floor
   (``MEMORY_RESOLVE_MIN_COVERAGE``, default 1.0) over those exact tokens
   (``field.replace("_", " ")``, BM25-tokenized — see ``memory_unit/core.py``
   ``_best_evidence_for``). So ``meeting_duration`` must appear as "meeting
   duration" in the sentence, not be dropped or abbreviated.
2. Never write back a value memory itself produced. ``populate_context_items``
   (utils/population.py) marks anything it fills as ``status="guessed"``, so a
   ``guessed`` slot is presumed to have come from memory (or from the LLM merely
   *inferring* it — see agents/task_agent.py SYSTEM_PROMPT, where "present" means
   "explicitly found in the text"). Learning those back would let a guess harden
   into a permanent "fact" nothing ever contradicts.

   The one exception is a slot the **user themselves typed**, which is the single
   most valuable signal there is: it is an explicit correction. The extension sets
   ``source="user"`` only when the value actually *changes*
   (chrome_extension/background.js, `Only flip source to "user" when the value
   actually changes`), and memory-unit's ``resolve()`` only ever reports
   ``source`` as ``"context"`` or ``None`` (memory_unit/core.py) — never
   ``"user"``. So ``guessed + source=="user"`` unambiguously means "memory or the
   LLM proposed something and the human overrode it", which is exactly what we
   most want to remember. Without this, a wrong guess is re-offered forever:
   memory suggests 30 minutes, the user corrects it to 45 every single time, and
   memory never learns.
"""

import os
import re
from typing import Any, Dict, List, Optional

from utils.task import Task

# The whole email body is not a task parameter; never learn it as one.
_EXCLUDED_FIELDS = {"body"}

# Only slots that are plausibly durable facts *about the user* are written back.
# The first cut learned every confirmed parameter, which meant one meeting's
# time window and Meet link were stored as if they were preferences — a later,
# unrelated task then pre-filled `time_window` and `meeting_link` from a meeting
# that had already happened. A stale suggestion is worse than no suggestion: it
# looks authoritative and the user has to notice it is wrong.
#
# Allowlist rather than denylist, deliberately. Getting this wrong in the
# "learn too much" direction damages trust in every future pre-fill, while
# getting it wrong in the "learn too little" direction just means no suggestion.
# Widen it once real usage shows what is actually stable.
_DURABLE_FIELD_HINTS = (
    "timezone", "time_zone",
    "duration", "length",
    "location", "venue", "room",
    "recipient", "sender", "cc", "participant", "attendee", "address",
    "language", "signature", "tone",
    "preferred", "default", "usual",
)

# Second guard, on the value rather than the field name: even an allowlisted slot
# must not carry something that is obviously a single occurrence. An absolute
# datetime or a URL is a fact about one event, never a standing preference.
# (These are also the two shapes memory-unit's _extract_value mangles on the way
# back out — `2026-07-16T14:00:00-07:00` resolves to "2026" because the field
# name contains "time" and the number branch wins; a URL gets clause-split at its
# first dot. Not learning them sidesteps that entirely.)
_EPHEMERAL_VALUE_RE = re.compile(
    r"""(
        \d{4}-\d{2}-\d{2}      # ISO date / datetime, incl. ranges
      | https?://              # any URL
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def writeback_enabled() -> bool:
    """Whether ``/learn_task_context`` is allowed to call memory-unit.

    Default OFF, independent of ``MEMORY_URL`` (the caller must also check that —
    see ``memory_enabled()`` in utils/memory_client.py — same two-flag shape as
    ``MEMORY_AUTO_POPULATE``/``auto_populate_enabled()`` in utils/population.py).
    Parsed with the same truthy set used by every other opt-in flag in this
    codebase (IDENTIFY_TZ_NORMALIZE et al., utils/slots.py; INTENT_ROUTER_ENABLED,
    agents/intent_agent.py) — kept as a same-shape local check rather than a
    shared import since none of those flag functions are themselves reusable
    (each just wraps its own env var name).
    """
    return os.getenv("MEMORY_WRITEBACK", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _is_durable(field: str, value: str) -> bool:
    """Whether this slot is a standing fact about the user, not about one event.

    Both guards must pass: the field has to look like a preference, and the value
    must not be obviously single-occurrence. See the constants above for why this
    is an allowlist.
    """
    name = (field or "").strip().lower()
    if not any(h in name for h in _DURABLE_FIELD_HINTS):
        return False
    return not _EPHEMERAL_VALUE_RE.search(value or "")


def _sentence_for(field: str, value: str) -> str:
    """Render "<Field words>: <value>".

    Underscores become spaces so every token of the slot name is literally
    present — memory-unit's resolve() enforces a term-coverage floor over exactly
    those tokens. A colon is used rather than "is" because ``_extract_value``
    accepts ``is``/``are``/``:``/``=`` equally, and the copula forced a choice of
    number the field name cannot supply: the first cut emitted "The participants
    is Anvay Patil, ..." for every plural slot.
    """
    words = field.replace("_", " ").strip()
    label = words[:1].upper() + words[1:] if words else words
    return f"{label}: {value}"


def _is_learnable(ci) -> bool:
    """Whether this slot's value may be written back to memory.

    ``present`` is the email/user-explicit case. ``guessed`` is only eligible when
    the user overrode it (``source="user"``), which is a correction rather than a
    guess — see the module docstring. ``missing`` never qualifies.
    """
    if ci.status == "present":
        return True
    return ci.status == "guessed" and ci.source == "user"


def build_learn_items(
    task: Task,
    category: Optional[str] = "task_patterns",
    scope: Optional[str] = "user",
) -> List[Dict[str, Any]]:
    """Turn ``task``'s user/email-sourced, filled slots into ``/learn`` items.

    Returns ``[]`` when the task has no context items, or none qualify. A slot
    qualifies only when:
      - its field is not ``body`` (the whole email, not a parameter),
      - its value is a non-empty string once stripped, and
      - it is either ``status="present"`` (explicitly found in the email/user
        text) or a user override (``source="user"``) — never a bare ``guessed``
        and never ``missing``. See the module docstring for why the user-override
        case is included: it is the correction signal, and it cannot be memory's
        own output.

    Each returned dict is ``{text, category, task_id, scope}``, matching
    memory-unit's ``LearnItem`` shape exactly (utils/memory_client.learn_facts
    posts these verbatim as ``{"items": [...]}"``).
    """
    items = task.context_items or []
    out: List[Dict[str, Any]] = []
    for ci in items:
        if ci.field in _EXCLUDED_FIELDS:
            continue
        if not _is_learnable(ci):
            continue
        value = (ci.value or "").strip()
        if not value:
            continue
        if not _is_durable(ci.field, value):
            continue
        out.append(
            {
                "text": _sentence_for(ci.field, value),
                "category": category,
                "task_id": task.task_id,
                "scope": scope,
            }
        )
    return out
