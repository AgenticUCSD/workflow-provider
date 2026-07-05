"""Phase 5 (steps 1-2): the provider must correlate one thread_id per request.

Covers:
- the X-Thread-Id header is used as the request's thread_id when the body omits it,
- an explicit body thread_id still wins over the header,
- no id anywhere leaves thread_id None (agents keep their own uuid4 fallback),
- the retrieval tracing helper is a safe no-op when tracing is disabled.
"""
import os
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi.testclient import TestClient

import app as app_module
from utils import tracing


def _client_capturing_thread_id(monkeypatch):
    captured = {}

    def fake_identify(text, subject=None, metadata=None, thread_id=None):
        captured["thread_id"] = thread_id
        return SimpleNamespace(task=None, context_items=[])

    monkeypatch.setattr(app_module.task_identifier_agent, "identify_task", fake_identify)
    return TestClient(app_module.app), captured


def test_x_thread_id_header_used_when_body_omits(monkeypatch):
    client, captured = _client_capturing_thread_id(monkeypatch)
    resp = client.post("/identify_task", json={"text": "hi"}, headers={"X-Thread-Id": "T-123"})
    assert resp.status_code == 200
    assert captured["thread_id"] == "T-123"


def test_body_thread_id_wins_over_header(monkeypatch):
    client, captured = _client_capturing_thread_id(monkeypatch)
    resp = client.post(
        "/identify_task",
        json={"text": "hi", "thread_id": "BODY"},
        headers={"X-Thread-Id": "HDR"},
    )
    assert resp.status_code == 200
    assert captured["thread_id"] == "BODY"


def test_no_thread_id_anywhere_is_none(monkeypatch):
    client, captured = _client_capturing_thread_id(monkeypatch)
    resp = client.post("/identify_task", json={"text": "hi"})
    assert resp.status_code == 200
    assert captured["thread_id"] is None


def test_traced_is_identity_when_disabled(monkeypatch):
    monkeypatch.setenv("CONFIDENT_TRACING", "false")
    calls = []

    @tracing.traced(name="test.span")
    def double(x):
        calls.append(x)
        return x * 2

    assert double(3) == 6
    assert calls == [3]


def test_tracing_disabled_without_key(monkeypatch):
    monkeypatch.delenv("CONFIDENT_API_KEY", raising=False)
    monkeypatch.delenv("CONFIDENT_TRACING", raising=False)
    assert tracing.tracing_enabled() is False
