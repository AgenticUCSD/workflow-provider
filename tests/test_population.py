"""Tests for slot population (memory-unit wiring) + the /populate_task_context endpoint.

All offline: the memory-unit call (`resolve_slots`) is monkeypatched, so no
network / MEMORY_URL is needed. Verifies that missing slots get filled from
context while email-provided values are preserved.
"""

from fastapi.testclient import TestClient

import app as app_module
import utils.memory_client as memory_client
import utils.population as population
from agents.task_agent import IdentifyTaskResult
from utils.task import ContextItem, Objective, Status, Task, TaskTypes


def _task_with_items(items):
    return Task(
        task_id="t1",
        task_type=TaskTypes.SCHEDULE,
        objective=Objective(
            objective_id="o1",
            name="n",
            description="d",
            inputs={},
            success_criteria="s",
            expected_output={},
        ),
        status=Status.PENDING,
        context_items=items,
    )


# ── populate_context_items ─────────────────────────────────────

def test_populate_fills_missing(monkeypatch):
    monkeypatch.setattr(
        population,
        "resolve_slots",
        lambda fields, **kw: [
            {
                "field": "recipient",
                "value": "alice@example.com",
                "source": "context",
                "confidence": 0.8,
                "status": "present",
            }
        ],
    )
    task = _task_with_items(
        [
            ContextItem(field="recipient", status="missing"),
            ContextItem(field="topic", status="present", value="Q3 planning"),
        ]
    )

    out = population.populate_context_items(task, user_id="user-1")
    by = {c.field: c for c in out.context_items}

    # Missing slot filled from context, marked "guessed" with provenance.
    assert by["recipient"].value == "alice@example.com"
    assert by["recipient"].status == "guessed"
    assert by["recipient"].source == "context"
    assert by["recipient"].confidence == 0.8
    # Email-provided value untouched.
    assert by["topic"].value == "Q3 planning"
    assert by["topic"].status == "present"


def test_populate_noop_when_nothing_missing(monkeypatch):
    calls = {"n": 0}

    def fake(fields, **kw):
        calls["n"] += 1
        return []

    monkeypatch.setattr(population, "resolve_slots", fake)
    task = _task_with_items([ContextItem(field="topic", status="present", value="x")])

    population.populate_context_items(task)
    assert calls["n"] == 0  # short-circuits before calling memory-unit


def test_populate_skips_unresolved(monkeypatch):
    monkeypatch.setattr(population, "resolve_slots", lambda fields, **kw: [])
    task = _task_with_items([ContextItem(field="recipient", status="missing")])

    out = population.populate_context_items(task)
    assert out.context_items[0].status == "missing"
    assert out.context_items[0].value is None


def test_populate_never_touches_present_or_guessed(monkeypatch):
    # Regression: only status=="missing" is filled. A present-but-empty value and
    # an already-guessed value must be left alone (and memory-unit not consulted).
    calls = {"n": 0}

    def fake(fields, **kw):
        calls["n"] += 1
        return []

    monkeypatch.setattr(population, "resolve_slots", fake)
    task = _task_with_items(
        [
            ContextItem(field="cc", status="present", value=""),
            ContextItem(field="topic", status="guessed", value="g"),
        ]
    )

    out = population.populate_context_items(task, user_id="u")
    by = {c.field: c for c in out.context_items}
    assert calls["n"] == 0  # nothing missing -> memory-unit not called
    assert by["cc"].status == "present" and by["cc"].value == ""
    assert by["topic"].status == "guessed" and by["topic"].value == "g"


# ── bearer forwarding (provider -> memory-unit) ────────────────

def test_resolve_slots_forwards_authorization(monkeypatch):
    seen = {}

    def capture(url, payload, headers, timeout):
        seen["headers"] = headers
        return []

    monkeypatch.setenv("MEMORY_URL", "http://localhost:9")
    monkeypatch.setattr(memory_client, "_post_resolve", capture)
    memory_client.resolve_slots(
        ["recipient"], user_id="u1", authorization="Bearer ya29.tok"
    )
    assert seen["headers"]["Authorization"] == "Bearer ya29.tok"
    assert seen["headers"]["X-User-Id"] == "u1"


def test_resolve_slots_omits_authorization_when_absent(monkeypatch):
    seen = {}
    monkeypatch.setenv("MEMORY_URL", "http://localhost:9")
    monkeypatch.setattr(
        memory_client, "_post_resolve",
        lambda url, payload, headers, timeout: seen.setdefault("headers", headers) or [],
    )
    memory_client.resolve_slots(["recipient"], user_id="u1")
    assert "Authorization" not in seen["headers"]


# ── call timeout (MEMORY_TIMEOUT_SECONDS) ──────────────────────

def test_memory_timeout_defaults_to_five(monkeypatch):
    monkeypatch.delenv("MEMORY_TIMEOUT_SECONDS", raising=False)
    assert memory_client.memory_timeout() == 5.0


def test_memory_timeout_honors_the_env_var(monkeypatch):
    monkeypatch.setenv("MEMORY_TIMEOUT_SECONDS", "30")
    assert memory_client.memory_timeout() == 30.0


def test_memory_timeout_clamps_out_of_range(monkeypatch):
    # Clamped, not rejected: reverting an over-large value to the 5s default
    # would hand the operator LESS time than they asked for.
    monkeypatch.setenv("MEMORY_TIMEOUT_SECONDS", "600")
    assert memory_client.memory_timeout() == 60.0
    monkeypatch.setenv("MEMORY_TIMEOUT_SECONDS", "0.1")
    assert memory_client.memory_timeout() == 1.0


def test_memory_timeout_falls_back_on_garbage(monkeypatch):
    for bad in ("", "  ", "abc", "30s", "nan"):
        monkeypatch.setenv("MEMORY_TIMEOUT_SECONDS", bad)
        assert memory_client.memory_timeout() == 5.0, bad


def test_resolve_slots_uses_the_configured_timeout(monkeypatch):
    """Read per call, not at import: a Cloud Run env flip must take effect on the
    next request without a redeploy."""
    seen = {}
    monkeypatch.setenv("MEMORY_URL", "http://localhost:9")
    monkeypatch.setattr(
        memory_client, "_post_resolve",
        lambda url, payload, headers, timeout: seen.update(timeout=timeout) or [],
    )

    monkeypatch.delenv("MEMORY_TIMEOUT_SECONDS", raising=False)
    memory_client.resolve_slots(["recipient"], user_id="u1")
    assert seen["timeout"] == 5.0

    monkeypatch.setenv("MEMORY_TIMEOUT_SECONDS", "30")
    memory_client.resolve_slots(["recipient"], user_id="u1")
    assert seen["timeout"] == 30.0


def test_learn_facts_uses_the_configured_timeout(monkeypatch):
    seen = {}
    monkeypatch.setenv("MEMORY_URL", "http://localhost:9")
    monkeypatch.setenv("MEMORY_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(
        memory_client, "_post_learn",
        lambda url, payload, headers, timeout: seen.update(timeout=timeout) or 0,
    )
    memory_client.learn_facts([{"text": "The recipient is a@b.com."}], user_id="u1")
    assert seen["timeout"] == 30.0


def test_explicit_timeout_still_wins(monkeypatch):
    seen = {}
    monkeypatch.setenv("MEMORY_URL", "http://localhost:9")
    monkeypatch.setenv("MEMORY_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(
        memory_client, "_post_resolve",
        lambda url, payload, headers, timeout: seen.update(timeout=timeout) or [],
    )
    memory_client.resolve_slots(["recipient"], user_id="u1", timeout=2.0)
    assert seen["timeout"] == 2.0


def test_populate_forwards_authorization(monkeypatch):
    seen = {}

    def fake(fields, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(population, "resolve_slots", fake)
    task = _task_with_items([ContextItem(field="recipient", status="missing")])
    population.populate_context_items(task, user_id="u1", authorization="Bearer t")
    assert seen["authorization"] == "Bearer t"


def test_populate_endpoint_forwards_incoming_bearer(monkeypatch):
    seen = {}

    def fake(fields, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(population, "resolve_slots", fake)
    task = _task_with_items([ContextItem(field="recipient", status="missing")])
    client = TestClient(app_module.app)
    resp = client.post(
        "/populate_task_context",
        json={"task": task.model_dump(mode="json")},
        headers={"X-User-Id": "user-1", "Authorization": "Bearer ya29.incoming"},
    )
    assert resp.status_code == 200, resp.text
    assert seen["authorization"] == "Bearer ya29.incoming"


# ── memory_client (flag-gated) ─────────────────────────────────

def test_resolve_slots_disabled_returns_empty(monkeypatch):
    monkeypatch.delenv("MEMORY_URL", raising=False)
    assert memory_client.resolve_slots(["a", "b"]) == []


def test_resolve_slots_empty_fields(monkeypatch):
    monkeypatch.setenv("MEMORY_URL", "http://localhost:9")
    assert memory_client.resolve_slots([]) == []


# ── /populate_task_context endpoint ────────────────────────────

def test_populate_endpoint(monkeypatch):
    monkeypatch.setattr(
        population,
        "resolve_slots",
        lambda fields, **kw: [
            {
                "field": "recipient",
                "value": "bob@example.com",
                "source": "context",
                "confidence": 0.6,
                "status": "present",
            }
        ],
    )
    task = _task_with_items([ContextItem(field="recipient", status="missing")])
    client = TestClient(app_module.app)

    resp = client.post(
        "/populate_task_context",
        json={"task": task.model_dump(mode="json")},
        headers={"X-User-Id": "user-1"},
    )
    assert resp.status_code == 200, resp.text
    items = resp.json()["context_items"]
    assert items[0]["value"] == "bob@example.com"
    assert items[0]["status"] == "guessed"
    assert items[0]["source"] == "context"


# ── /identify_task auto-population (flag-gated) ────────────────

def _identify_returns(monkeypatch, task):
    result = IdentifyTaskResult(task=task, context_items=task.context_items)
    monkeypatch.setattr(
        app_module.task_identifier_agent, "identify_task", lambda **kw: result
    )


def test_identify_auto_populates_when_enabled(monkeypatch):
    task = _task_with_items([ContextItem(field="recipient", status="missing")])
    _identify_returns(monkeypatch, task)
    monkeypatch.setenv("MEMORY_AUTO_POPULATE", "true")
    monkeypatch.setattr(
        population,
        "resolve_slots",
        lambda fields, **kw: [
            {
                "field": "recipient",
                "value": "a@b.com",
                "source": "context",
                "confidence": 0.7,
                "status": "present",
            }
        ],
    )
    client = TestClient(app_module.app)

    resp = client.post("/identify_task", json={"text": "hi"}, headers={"X-User-Id": "u1"})
    assert resp.status_code == 200, resp.text
    item = resp.json()["context_items"][0]
    assert item["value"] == "a@b.com"
    assert item["status"] == "guessed"


def test_identify_does_not_populate_when_disabled(monkeypatch):
    task = _task_with_items([ContextItem(field="recipient", status="missing")])
    _identify_returns(monkeypatch, task)
    monkeypatch.delenv("MEMORY_AUTO_POPULATE", raising=False)
    calls = {"n": 0}

    def fake(fields, **kw):
        calls["n"] += 1
        return []

    monkeypatch.setattr(population, "resolve_slots", fake)
    client = TestClient(app_module.app)

    resp = client.post("/identify_task", json={"text": "hi"}, headers={"X-User-Id": "u1"})
    assert resp.status_code == 200, resp.text
    assert calls["n"] == 0  # memory-unit not consulted when the flag is off
    assert resp.json()["context_items"][0]["status"] == "missing"


def test_populate_ignores_slots_memory_reports_missing(monkeypatch):
    # memory-unit's relevance floor reports an unresolved slot as status="missing".
    # Even if a value/confidence rides along (an older unit, or the floor relaxed),
    # the provider must not pre-fill it -- a fabricated guess is worse than asking.
    monkeypatch.setattr(
        population,
        "resolve_slots",
        lambda fields, **kw: [
            {
                "field": "recipient",
                "value": "the-nearest-unrelated-snippet",
                "source": "context",
                "confidence": 0.93,
                "status": "missing",
            }
        ],
    )
    task = _task_with_items([ContextItem(field="recipient", status="missing")])

    out = population.populate_context_items(task, user_id="user-1")
    ci = out.context_items[0]

    assert ci.status == "missing"
    assert ci.value is None


def test_populate_still_fills_when_status_absent(monkeypatch):
    # Backward compatibility: a payload without `status` at all is not treated as
    # unresolved, so older memory-unit builds keep working.
    monkeypatch.setattr(
        population,
        "resolve_slots",
        lambda fields, **kw: [
            {"field": "recipient", "value": "alice@example.com", "confidence": 0.8}
        ],
    )
    task = _task_with_items([ContextItem(field="recipient", status="missing")])

    out = population.populate_context_items(task, user_id="user-1")
    assert out.context_items[0].status == "guessed"
    assert out.context_items[0].value == "alice@example.com"
