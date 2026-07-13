"""Tests for the executor artifact transport (utils/artifact_client.py) and the
/promote_template endpoint.

All offline: the HTTP worker (`_post_artifact`) or the `post_template` function is
monkeypatched, so no network / EXECUTOR_ARTIFACTS_URL server is needed. Mirrors the
memory_client tests in test_population.py.
"""

import json
import os
import urllib.error
import urllib.request

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi.testclient import TestClient

import app as app_module
import utils.artifact_client as artifact_client
from utils.template import SlotSpec, Step, WorkflowTemplate


def _template(**overrides) -> WorkflowTemplate:
    base = dict(
        template_id="tmpl_abc123",
        name="Schedule a meeting",
        description="Find a time and send an invite",
        required_slots=[SlotSpec(name="recipient", type="email", required=True)],
        steps=[Step(kind="llm", text="Find a free slot"), Step(text="Invite {recipient}")],
    )
    base.update(overrides)
    return WorkflowTemplate(**base)


# The executor's CreateArtifactRequest fields = to_envelope() minus artifact_id
# (the executor assigns its own uuid). Replicated here to guard against drift.
_EXPECTED_BODY_KEYS = {
    "kind",
    "name",
    "content",
    "status",
    "trust_tier",
    "source_trace_ids",
    "parent_artifact_id",
    "user_id",
}


# ── post_template: header forwarding + body shape ──────────────

def test_post_template_forwards_auth_headers(monkeypatch):
    seen = {}

    def capture(url, payload, headers, timeout):
        seen["url"] = url
        seen["headers"] = headers
        return {"artifact_id": "exec-uuid", "status": "draft"}

    monkeypatch.setenv("EXECUTOR_ARTIFACTS_URL", "http://localhost:9")
    monkeypatch.setattr(artifact_client, "_post_artifact", capture)
    out = artifact_client.post_template(
        _template(),
        x_user_id="sub-1",
        authorization="Bearer ya29.tok",
        thread_id="th-1",
    )
    assert seen["headers"]["Authorization"] == "Bearer ya29.tok"
    assert seen["headers"]["X-User-Id"] == "sub-1"
    assert seen["headers"]["X-Thread-Id"] == "th-1"
    assert seen["url"].endswith("/artifacts")
    assert out == {"artifact_id": "exec-uuid", "status": "draft"}


def test_post_template_omits_headers_when_absent(monkeypatch):
    seen = {}
    monkeypatch.setenv("EXECUTOR_ARTIFACTS_URL", "http://localhost:9")
    monkeypatch.setattr(
        artifact_client, "_post_artifact",
        lambda url, payload, headers, timeout: seen.setdefault("headers", headers),
    )
    artifact_client.post_template(_template())
    assert "Authorization" not in seen["headers"]
    assert "X-User-Id" not in seen["headers"]
    assert "X-Thread-Id" not in seen["headers"]


def test_post_template_body_shape(monkeypatch):
    seen = {}
    monkeypatch.setenv("EXECUTOR_ARTIFACTS_URL", "http://localhost:9")
    monkeypatch.setattr(
        artifact_client, "_post_artifact",
        lambda url, payload, headers, timeout: seen.setdefault("body", json.loads(payload)),
    )
    artifact_client.post_template(
        _template(), user_id="*", x_user_id="sub-1", authorization="Bearer t"
    )
    body = seen["body"]
    # Exactly the executor's CreateArtifactRequest fields — no artifact_id.
    assert set(body.keys()) == _EXPECTED_BODY_KEYS
    assert "artifact_id" not in body
    assert body["kind"] == "template"
    assert body["status"] == "draft"
    assert body["user_id"] == "*"


# ── flag-gating (disabled = no-op) ─────────────────────────────

def test_post_template_disabled_returns_none(monkeypatch):
    monkeypatch.delenv("EXECUTOR_ARTIFACTS_URL", raising=False)
    called = {"n": 0}
    monkeypatch.setattr(
        artifact_client, "_post_artifact",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1),
    )
    assert artifact_client.post_template(_template()) is None
    assert called["n"] == 0  # worker never invoked when the flag is unset


# ── never-raises boundary ──────────────────────────────────────

def test_post_artifact_never_raises(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert artifact_client._post_artifact("http://x/artifacts", b"{}", {}, 1.0) is None


# ── /promote_template endpoint ─────────────────────────────────

def test_promote_endpoint_disabled(monkeypatch):
    monkeypatch.delenv("EXECUTOR_ARTIFACTS_URL", raising=False)
    client = TestClient(app_module.app)
    resp = client.post(
        "/promote_template",
        json={"template_id": "x"},
        headers={"X-User-Id": "u", "Authorization": "Bearer t"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "disabled"


def test_promote_endpoint_requires_headers(monkeypatch):
    monkeypatch.setenv("EXECUTOR_ARTIFACTS_URL", "http://localhost:9")
    client = TestClient(app_module.app)
    resp = client.post("/promote_template", json={"template_id": "x"})
    assert resp.status_code == 400


def test_promote_endpoint_unknown_template(monkeypatch):
    monkeypatch.setenv("EXECUTOR_ARTIFACTS_URL", "http://localhost:9")
    monkeypatch.setattr(
        app_module.template_store, "get_template",
        lambda template_id, version=None: None,
    )
    client = TestClient(app_module.app)
    resp = client.post(
        "/promote_template",
        json={"template_id": "nope"},
        headers={"X-User-Id": "u", "Authorization": "Bearer t"},
    )
    assert resp.status_code == 404


def test_promote_endpoint_happy_path(monkeypatch):
    monkeypatch.setenv("EXECUTOR_ARTIFACTS_URL", "http://localhost:9")
    monkeypatch.setattr(
        app_module.template_store, "get_template",
        lambda template_id, version=None: _template(),
    )
    captured = {}

    def fake_post(template, **kw):
        captured.update(kw)
        return {"artifact_id": "exec-uuid", "kind": "template", "status": "draft"}

    monkeypatch.setattr(app_module, "post_template", fake_post)
    client = TestClient(app_module.app)
    resp = client.post(
        "/promote_template",
        json={"template_id": "tmpl_abc123", "source_trace_ids": ["tr1"]},
        headers={
            "X-User-Id": "sub-1",
            "Authorization": "Bearer ya29.x",
            "X-Thread-Id": "th-9",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "written"
    assert body["artifact"]["artifact_id"] == "exec-uuid"
    # The endpoint forwarded the right identities to the transport:
    assert captured["x_user_id"] == "sub-1"          # caller's sub → X-User-Id
    assert captured["authorization"] == "Bearer ya29.x"
    assert captured["user_id"] == "*"                # global scope in the body
    assert captured["source_trace_ids"] == ["tr1"]
    assert captured["thread_id"] == "th-9"


def test_promote_endpoint_transport_error(monkeypatch):
    monkeypatch.setenv("EXECUTOR_ARTIFACTS_URL", "http://localhost:9")
    monkeypatch.setattr(
        app_module.template_store, "get_template",
        lambda template_id, version=None: _template(),
    )
    monkeypatch.setattr(app_module, "post_template", lambda template, **kw: None)
    client = TestClient(app_module.app)
    resp = client.post(
        "/promote_template",
        json={"template_id": "tmpl_abc123"},
        headers={"X-User-Id": "u", "Authorization": "Bearer t"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "error"
