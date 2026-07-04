"""Regression: a missing OPENAI_API_KEY must not crash startup.

The container previously crashed at import — `ChatOpenAI(api_key=None)` raised
`openai.OpenAIError: Missing credentials`, so the Cloud Run container never listened
on PORT and the deploy failed. `openai_api_key_or_placeholder()` prevents that.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import utils.config as config


def test_returns_real_key_when_set(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-real-123")
    assert config.openai_api_key_or_placeholder() == "sk-real-123"


def test_returns_placeholder_when_unset(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    val = config.openai_api_key_or_placeholder()
    assert val == config._OPENAI_KEY_PLACEHOLDER
    assert val  # non-empty, so downstream constructors won't reject it


def test_chatopenai_constructs_with_missing_key(monkeypatch):
    # The exact previously-crashing path: build the LLM client with no key set.
    # With the placeholder it must construct without raising (a real API call would
    # still fail, but the app boots and can serve /health).
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    from langchain_openai import ChatOpenAI

    m = ChatOpenAI(model="gpt-4.1", api_key=config.openai_api_key_or_placeholder())
    assert m is not None
