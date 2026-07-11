"""Provider ↔ executor Artifact-envelope adapter (Phase 5 step 3 / P-LEARN1).

Pure, I/O-free mapping from a provider :class:`~utils.template.WorkflowTemplate`
to the executor's ``public.artifacts`` envelope
(``workflow_executor/services/status_store.py``). **No DB, no HTTP** — the
transport layer (an HTTP client if the executor exposes ``POST /artifacts``, or a
``PGArtifactStore`` if we get a scoped DB grant) is deliberately deferred until the
write-mechanism decision. See ``claude-context/convergence.md``.

This module defines exactly the three things that are identical regardless of the
transport Saanvi picks:

1. :func:`canonical_content` — the deterministic ``content`` string we store.
2. :func:`content_hash` — ``sha256(content)[:16]``, byte-identical to the
   executor's ``services.status_store._content_hash``, so provider-side
   check-before-write dedup matches executor-side identity.
3. :func:`to_envelope` — the field mapping, returned as the exact kwargs the
   executor's ``insert_artifact`` accepts.

Envelope field map (canonical in convergence.md):
``template_id → artifact_id``, ``name → name``, ``parent_id → parent_artifact_id``,
``status → status``, :func:`canonical_content` ``→ content``, ``scope → user_id``
(``'*'`` == global). ``trust_tier`` follows ``status`` (draft→T0, candidate→T1).
``eval_score``/``source`` are **provider-local** (no envelope column) and are
intentionally NOT emitted. ``version``/``content_hash`` are computed by the
executor at insert time and are NOT sent.
"""

import hashlib
import json
from typing import Iterable, Optional

from utils.template import WorkflowTemplate

# The artifact `kind` for a promotable workflow template. Matches the provider's
# own model docstring convention (WorkflowTemplate = "kind=template",
# EnrichedInstance = "kind=instance") and distinguishes a parameterized template
# from a flat executor skill (kind="skill"). This is a contract value the
# executor's read-back keys on — proposed to Saanvi in convergence.md.
ARTIFACT_KIND_TEMPLATE = "template"

# Envelope default for an unscoped/global artifact (executor's `user_id` default).
GLOBAL_USER_ID = "*"

# status → trust_tier. We only ever *write* draft (T0) and candidate (T1);
# `trusted` is reached via the executor's `promote_artifact`, not by us. Kept as a
# total map (default T0) so an unexpected status never emits an invalid tier.
_STATUS_TRUST_TIER = {"draft": "T0", "candidate": "T1"}


def _trust_tier_for(status: str) -> str:
    return _STATUS_TRUST_TIER.get(status, "T0")


def canonical_content(template: WorkflowTemplate) -> str:
    """Deterministic JSON ``content`` string for the artifact.

    Encodes the template's *semantic* fields only — name, description,
    required_slots, and steps (with ``{slot}`` placeholders intact, i.e. the
    template form, not a bound instance). It excludes ``template_id``/``version``/
    ``status`` so two templates with identical semantics produce identical content
    (and thus an identical :func:`content_hash`), mirroring
    :meth:`WorkflowTemplate.to_string`'s dedup intent.

    ``sort_keys`` + fixed separators make the byte string stable across runs and
    processes, so the hash is reproducible on both sides of the boundary. List
    order (steps, slots) is preserved — it is semantic (reordered steps are a
    different workflow).
    """
    payload = {
        "name": template.name,
        "description": template.description,
        "required_slots": [
            {"name": s.name, "type": s.type, "required": s.required}
            for s in template.required_slots
        ],
        "steps": [
            # `ref` only when set, so an all-None-ref template hashes stably
            # whether or not the field was ever touched.
            {"kind": s.kind, "text": s.text, **({"ref": s.ref} if s.ref else {})}
            for s in template.steps
        ],
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def content_hash(template: WorkflowTemplate) -> str:
    """``sha256(canonical_content)[:16]`` — identical to the executor's
    ``_content_hash``.

    Used by the (deferred) transport layer for check-before-write dedup: because
    the executor computes the same hash over the same ``content`` we send, a match
    here means the artifact already exists and we should skip the insert (avoiding
    version churn — ``insert_artifact`` has no content dedup of its own).
    """
    return hashlib.sha256(
        canonical_content(template).encode("utf-8")
    ).hexdigest()[:16]


def to_envelope(
    template: WorkflowTemplate,
    *,
    user_id: str = GLOBAL_USER_ID,
    source_trace_ids: Optional[Iterable[str]] = None,
) -> dict:
    """Map a :class:`WorkflowTemplate` to the executor's ``insert_artifact`` kwargs.

    Returns a dict of exactly the keyword arguments
    ``services.status_store.insert_artifact`` accepts — the transport layer
    forwards these unchanged (as an HTTP JSON body or as bound SQL params). It
    deliberately omits ``version`` and ``content_hash`` (the executor computes
    both) and ``eval_score``/``source`` (provider-local, no envelope column).

    Args:
        user_id: envelope scope. ``'*'`` (default) == global; pass the owner's
            id to scope the artifact (the provider ``scope`` field maps here).
        source_trace_ids: trace lineage for provenance — seed from the
            ``planner.enriched_instances.trace_id`` of runs that produced/earned
            this template. Empty list when unknown.
    """
    return {
        "artifact_id": template.template_id,
        "kind": ARTIFACT_KIND_TEMPLATE,
        "name": template.name,
        "content": canonical_content(template),
        "status": template.status,
        "trust_tier": _trust_tier_for(template.status),
        "source_trace_ids": list(source_trace_ids or []),
        "parent_artifact_id": template.parent_id,
        "user_id": user_id,
    }
