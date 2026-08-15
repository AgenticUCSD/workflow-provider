"""Tests for memory write-back (utils/writeback.py, memory_client.learn_facts,
the /learn_task_context endpoint).

All offline: the memory-unit call (`learn_facts`) is monkeypatched, so no
network / MEMORY_URL is needed for endpoint tests. The round-trip test
reimplements memory-unit's BM25 tokenizer (a plain
``re.findall(r'\\b[a-zA-Z]+\\b', text.lower())`` — see
memory-unit/memory_unit/storage/bm25_search.py) to prove, without importing that
separate service, that every token of a slot name survives into the generated
sentence: this is the exact coverage rule memory-unit's resolve() enforces.
"""

import re

import pytest
from fastapi.testclient import TestClient

import app as app_module
import utils.memory_client as memory_client
import utils.writeback as writeback
from utils.task import ContextItem, Objective, Status, Task, TaskTypes


def _tokenize(text: str):
    # Mirrors memory_unit/storage/bm25_search.py BM25Searcher.tokenize exactly.
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def _task_with_items(items, task_id="t1"):
    return Task(
        task_id=task_id,
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


# ── build_learn_items ───────────────────────────────────────────

def test_multi_token_slot_name_covered():
    task = _task_with_items(
        [ContextItem(field="meeting_duration", status="present", value="45 minutes")]
    )
    items = writeback.build_learn_items(task)
    assert len(items) == 1
    text = items[0]["text"]
    assert "meeting" in text.lower()
    assert "duration" in text.lower()


def test_guessed_slot_not_written_back():
    # This is the loop-prevention guarantee: anything memory itself resolved is
    # marked "guessed" by populate_context_items and must never be re-learned.
    task = _task_with_items(
        [ContextItem(field="recipient", status="guessed", value="alice@example.com")]
    )
    assert writeback.build_learn_items(task) == []


def test_missing_slot_skipped():
    task = _task_with_items([ContextItem(field="recipient", status="missing")])
    assert writeback.build_learn_items(task) == []


def test_empty_value_skipped():
    task = _task_with_items(
        [ContextItem(field="recipient", status="present", value="")]
    )
    assert writeback.build_learn_items(task) == []


def test_none_value_skipped():
    task = _task_with_items([ContextItem(field="recipient", status="present", value=None)])
    assert writeback.build_learn_items(task) == []


def test_body_field_skipped_even_when_present():
    task = _task_with_items(
        [ContextItem(field="body", status="present", value="the whole email text")]
    )
    assert writeback.build_learn_items(task) == []


def test_present_value_is_written_back():
    task = _task_with_items(
        [ContextItem(field="recipient", status="present", value="grace@example.com")],
        task_id="task-42",
    )
    items = writeback.build_learn_items(task)
    assert len(items) == 1
    item = items[0]
    assert "grace@example.com" in item["text"]
    assert item["task_id"] == "task-42"
    assert item["category"]
    assert item["scope"]


def test_no_context_items_returns_empty():
    task = _task_with_items(None)
    assert writeback.build_learn_items(task) == []


def test_mixed_statuses_only_present_written():
    task = _task_with_items(
        [
            ContextItem(field="recipient", status="present", value="a@b.com"),
            ContextItem(field="topic", status="guessed", value="g"),
            ContextItem(field="duration", status="missing"),
        ]
    )
    items = writeback.build_learn_items(task)
    assert len(items) == 1
    assert "a@b.com" in items[0]["text"]


# ── the allowlist matches word starts, not substrings ──────────────────────
#
# Plain `hint in name` quietly widened the allowlist far past what it lists,
# because two hints are short and common: "cc" is inside "occasion",
# "success_criteria" and "accuracy_target"; "tone" is inside "milestone". Each of
# those is exactly the single-event value this filter exists to keep out of the
# user's permanent memory, and the value guard does not catch them (it only
# rejects ISO dates and URLs).

@pytest.mark.parametrize(
    "field",
    ["occasion", "milestone", "success_criteria", "accuracy_target", "topic"],
)
def test_substring_lookalike_fields_are_not_durable(field):
    task = _task_with_items(
        [ContextItem(field=field, status="present", value="Q3 planning offsite")]
    )
    assert writeback.build_learn_items(task) == []


@pytest.mark.parametrize(
    "field,value",
    [
        # Plural slots must keep matching singular hints -- "participants" is one
        # of the facts already learned in production, so whole-word equality
        # would have been a regression, not a fix.
        ("participants", "Anvay, Sam"),
        ("attendees", "Anvay, Sam"),
        ("meeting_duration", "30"),
        # Spelling variants of the same hint.
        ("timezone", "America/Los_Angeles"),
        ("time_zone", "America/Los_Angeles"),
        # A hint that is itself a whole short field.
        ("cc", "a@b.com"),
        ("preferred_location", "SF"),
        ("signature_block", "Best, A"),
    ],
)
def test_real_durable_fields_still_match(field, value):
    task = _task_with_items([ContextItem(field=field, status="present", value=value)])
    items = writeback.build_learn_items(task)
    assert len(items) == 1, f"{field} stopped being durable"
    assert value in items[0]["text"]


# ── round-trip: generated text satisfies memory-unit's coverage rule ────────

@pytest.mark.parametrize(
    "field,value",
    [
        ("recipient", "grace@example.com"),
        ("meeting_duration", "45 minutes"),
        ("timezone", "America/Los_Angeles"),
        ("to_address", "bob@example.com"),
    ],
)
def test_sentence_satisfies_coverage_rule(field, value):
    task = _task_with_items([ContextItem(field=field, status="present", value=value)])
    items = writeback.build_learn_items(task)
    text = items[0]["text"]

    # memory-unit queries with field.replace("_", " ") then BM25-tokenizes it.
    query_tokens = _tokenize(field.replace("_", " "))
    doc_tokens = set(_tokenize(text))
    coverage = sum(1 for t in query_tokens if t in doc_tokens) / len(query_tokens)

    assert coverage == 1.0, f"{field!r} tokens {query_tokens} not fully covered by {text!r}"


# ── memory_client.learn_facts (flag-gated, never-raises) ───────────────────

def test_learn_facts_disabled_returns_zero(monkeypatch):
    monkeypatch.delenv("MEMORY_URL", raising=False)
    assert memory_client.learn_facts([{"text": "The recipient is a@b.com."}]) == 0


def test_learn_facts_empty_items(monkeypatch):
    monkeypatch.setenv("MEMORY_URL", "http://localhost:9")
    assert memory_client.learn_facts([]) == 0


def test_learn_facts_forwards_headers(monkeypatch):
    seen = {}

    def capture(url, payload, headers, timeout):
        seen["url"] = url
        seen["headers"] = headers
        return 1

    monkeypatch.setenv("MEMORY_URL", "http://localhost:9")
    monkeypatch.setattr(memory_client, "_post_learn", capture)
    n = memory_client.learn_facts(
        [{"text": "The recipient is a@b.com."}],
        user_id="u1",
        thread_id="th-1",
        authorization="Bearer ya29.tok",
    )
    assert n == 1
    assert seen["url"] == "http://localhost:9/learn"
    assert seen["headers"]["Authorization"] == "Bearer ya29.tok"
    assert seen["headers"]["X-User-Id"] == "u1"
    assert seen["headers"]["X-Thread-Id"] == "th-1"


def test_learn_facts_network_error_returns_zero(monkeypatch):
    import urllib.error

    def boom(url, payload, headers, timeout):
        raise urllib.error.URLError("no route")

    # _post_learn itself is what swallows the error (mirrors _post_resolve); call
    # it directly to prove the never-raises contract at that layer too.
    monkeypatch.setenv("MEMORY_URL", "http://localhost:9")
    req_mod = memory_client.urllib.request

    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(req_mod, "urlopen", fake_urlopen)
    assert memory_client.learn_facts([{"text": "The recipient is a@b.com."}]) == 0


# ── /learn_task_context endpoint ────────────────────────────────

def _task_present(field="recipient", value="a@b.com"):
    return _task_with_items([ContextItem(field=field, status="present", value=value)])


def test_endpoint_disabled_by_default_no_network_call(monkeypatch):
    monkeypatch.delenv("MEMORY_WRITEBACK", raising=False)
    monkeypatch.setenv("MEMORY_URL", "http://localhost:9")
    calls = {"n": 0}

    def fake(*a, **kw):
        calls["n"] += 1
        return 1

    monkeypatch.setattr(app_module, "learn_facts", fake)
    client = TestClient(app_module.app)
    task = _task_present()

    resp = client.post(
        "/learn_task_context",
        json={"task": task.model_dump(mode="json")},
        headers={"X-User-Id": "user-1"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["learned"] == 0
    assert body["status"] == "disabled"
    assert calls["n"] == 0


def test_endpoint_no_memory_url_no_network_call(monkeypatch):
    monkeypatch.setenv("MEMORY_WRITEBACK", "true")
    monkeypatch.delenv("MEMORY_URL", raising=False)
    calls = {"n": 0}

    def fake(*a, **kw):
        calls["n"] += 1
        return 1

    monkeypatch.setattr(app_module, "learn_facts", fake)
    client = TestClient(app_module.app)
    task = _task_present()

    resp = client.post(
        "/learn_task_context",
        json={"task": task.model_dump(mode="json")},
        headers={"X-User-Id": "user-1"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["learned"] == 0
    assert body["status"] == "disabled"
    assert calls["n"] == 0


def test_endpoint_memory_unit_error_still_200(monkeypatch):
    monkeypatch.setenv("MEMORY_WRITEBACK", "true")
    monkeypatch.setenv("MEMORY_URL", "http://localhost:9")

    def fake(*a, **kw):
        return 0  # learn_facts's own never-raises contract: 0 on any problem

    monkeypatch.setattr(app_module, "learn_facts", fake)
    client = TestClient(app_module.app)
    task = _task_present()

    resp = client.post(
        "/learn_task_context",
        json={"task": task.model_dump(mode="json")},
        headers={"X-User-Id": "user-1"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["learned"] == 0


def test_endpoint_memory_unit_raises_still_200(monkeypatch):
    # Belt-and-suspenders: even if learn_facts's own contract were ever broken,
    # the endpoint itself must not propagate a 5xx.
    monkeypatch.setenv("MEMORY_WRITEBACK", "true")
    monkeypatch.setenv("MEMORY_URL", "http://localhost:9")

    def fake(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module, "learn_facts", fake)
    client = TestClient(app_module.app)
    task = _task_present()

    resp = client.post(
        "/learn_task_context",
        json={"task": task.model_dump(mode="json")},
        headers={"X-User-Id": "user-1"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["learned"] == 0
    assert resp.json()["status"] == "error"


def test_endpoint_happy_path_forwards_bearer_and_user(monkeypatch):
    monkeypatch.setenv("MEMORY_WRITEBACK", "true")
    monkeypatch.setenv("MEMORY_URL", "http://localhost:9")
    seen = {}

    def fake(items, **kw):
        seen["items"] = items
        seen.update(kw)
        return len(items)

    monkeypatch.setattr(app_module, "learn_facts", fake)
    client = TestClient(app_module.app)
    task = _task_present()

    resp = client.post(
        "/learn_task_context",
        json={"task": task.model_dump(mode="json")},
        headers={"X-User-Id": "user-1", "Authorization": "Bearer ya29.incoming"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["learned"] == 1
    assert body["status"] == "learned"
    assert seen["user_id"] == "user-1"
    assert seen["authorization"] == "Bearer ya29.incoming"
    assert len(seen["items"]) == 1
    assert "a@b.com" in seen["items"][0]["text"]


def test_endpoint_no_qualifying_items_short_circuits(monkeypatch):
    # All slots guessed/missing -> build_learn_items() is empty -> no network call.
    monkeypatch.setenv("MEMORY_WRITEBACK", "true")
    monkeypatch.setenv("MEMORY_URL", "http://localhost:9")
    calls = {"n": 0}

    def fake(*a, **kw):
        calls["n"] += 1
        return 1

    monkeypatch.setattr(app_module, "learn_facts", fake)
    client = TestClient(app_module.app)
    task = _task_with_items([ContextItem(field="recipient", status="guessed", value="a@b.com")])

    resp = client.post(
        "/learn_task_context",
        json={"task": task.model_dump(mode="json")},
        headers={"X-User-Id": "user-1"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["learned"] == 0
    assert calls["n"] == 0


# ── the user-override exception (the correction signal) ───────────────

def test_user_override_of_a_guess_is_learned():
    """A guess the human corrected is the most valuable thing to remember.

    Without this, a wrong guess is re-offered forever: memory suggests 30
    minutes, the user fixes it to 45 on every single task, and memory never
    learns. The extension sets source="user" only when the value actually
    changed, so this cannot fire for a guess the user merely left alone.
    """
    task = _task_with_items(
        [
            ContextItem(
                field="meeting_duration",
                status="guessed",
                value="45 minutes",
                source="user",
            )
        ]
    )
    items = writeback.build_learn_items(task)
    assert len(items) == 1
    assert "45 minutes" in items[0]["text"]


def test_guess_left_untouched_by_the_user_is_not_learned():
    """The loop-prevention guarantee, restated against the widened filter.

    memory-unit's resolve() reports source as "context" or None, never "user"
    (memory_unit/core.py), so a memory-filled slot the user never edited can
    never satisfy the override exception.
    """
    for src in ("context", None):
        task = _task_with_items(
            [
                ContextItem(
                    field="recipient",
                    status="guessed",
                    value="alice@example.com",
                    source=src,
                )
            ]
        )
        assert writeback.build_learn_items(task) == [], f"source={src!r} must not be learned"


def test_missing_slot_is_not_learned_even_if_user_sourced():
    task = _task_with_items(
        [ContextItem(field="recipient", status="missing", value=None, source="user")]
    )
    assert writeback.build_learn_items(task) == []


# ── durable-only filter (the real-run regression) ─────────────────────

def _present(field, value):
    return ContextItem(field=field, status="present", value=value, source="user")


def test_real_run_learns_only_the_durable_slots():
    """The exact five slots a real UI approval produced on 2026-08-13.

    The first cut learned all five, so an unrelated standup a week later
    pre-filled time_window and meeting_link from a meeting that had already
    happened. Only timezone and duration are facts about the *user*.
    """
    task = _task_with_items([
        _present("participants", "Anvay Patil, Pranav Soma, Saanvi Rao"),
        _present("duration", "30"),
        _present("time_window", "2026-07-16T14:00:00-07:00 to 2026-07-16T14:30:00-07:00"),
        _present("timezone", "America/Los_Angeles"),
        _present("meeting_link", "https://meet.google.com/xyz-abcd-efg"),
    ])
    learned = {i["text"].split(":")[0] for i in writeback.build_learn_items(task)}
    assert learned == {"Participants", "Duration", "Timezone"}, learned


def test_ephemeral_values_are_refused_even_on_an_allowlisted_field():
    # "duration" is allowlisted, but an ISO datetime is never a standing
    # preference — the value guard has to override the field guard.
    task = _task_with_items([_present("duration", "2026-07-16T14:00:00-07:00")])
    assert writeback.build_learn_items(task) == []
    task = _task_with_items([_present("location", "https://meet.google.com/abc")])
    assert writeback.build_learn_items(task) == []


def test_unknown_field_is_not_learned():
    # Allowlist, not denylist: anything unrecognised is skipped by default.
    task = _task_with_items([_present("invoice_number", "INV-42")])
    assert writeback.build_learn_items(task) == []


def test_plural_field_does_not_produce_broken_grammar():
    # Was: "The participants is Anvay Patil, ..."
    task = _task_with_items([_present("participants", "Anvay Patil, Pranav Soma")])
    text = writeback.build_learn_items(task)[0]["text"]
    assert " is " not in text
    assert text == "Participants: Anvay Patil, Pranav Soma"


def test_multi_token_field_still_satisfies_the_coverage_floor():
    # The property that must survive any change to the sentence template.
    task = _task_with_items([_present("meeting_duration", "45 minutes")])
    text = writeback.build_learn_items(task)[0]["text"]
    for tok in _tokenize("meeting duration"):
        assert tok in _tokenize(text), f"{tok!r} missing from {text!r}"
