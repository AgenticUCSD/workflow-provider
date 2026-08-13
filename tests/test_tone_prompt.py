"""Tests for the IDENTIFY_TONE_PROMPT flag (user-facing description tone).

All offline: these assert how the prompts are *assembled*, never what an LLM does
with them. Whether the tone guidance actually preserves extraction quality is an
empirical question a unit test cannot answer -- that is measured against a real
model by claude-context/calibration/ab_tone_prompt.py.
"""

import agents.task_agent as task_agent
from agents.task_agent import (
    SYSTEM_PROMPT,
    TASK_EDITOR_PROMPT,
    TONE_GUIDANCE,
    identify_system_prompt,
    task_editor_prompt,
    tone_prompt_enabled,
)


# ── flag parsing ─────────────────────────────────────────────────────────────

def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("IDENTIFY_TONE_PROMPT", raising=False)
    assert tone_prompt_enabled() is False


def test_flag_truthy_forms(monkeypatch):
    for raw in ("1", "true", "TRUE", "yes", "on", "  true  "):
        monkeypatch.setenv("IDENTIFY_TONE_PROMPT", raw)
        assert tone_prompt_enabled() is True, raw


def test_flag_falsy_forms(monkeypatch):
    for raw in ("", "0", "false", "no", "off", "banana"):
        monkeypatch.setenv("IDENTIFY_TONE_PROMPT", raw)
        assert tone_prompt_enabled() is False, raw


# ── prompt assembly ──────────────────────────────────────────────────────────

def test_prompts_are_byte_identical_when_off(monkeypatch):
    # The flag-off path must be indistinguishable from the pre-flag code, or the
    # change is not additive and every live extraction is in scope.
    monkeypatch.delenv("IDENTIFY_TONE_PROMPT", raising=False)
    assert identify_system_prompt() == SYSTEM_PROMPT
    assert task_editor_prompt() == TASK_EDITOR_PROMPT


def test_tone_is_appended_when_on(monkeypatch):
    monkeypatch.setenv("IDENTIFY_TONE_PROMPT", "true")
    got = identify_system_prompt()
    assert got.startswith(SYSTEM_PROMPT)   # the tuned prompt is preserved verbatim
    assert TONE_GUIDANCE in got
    edit = task_editor_prompt()
    assert edit.startswith(TASK_EDITOR_PROMPT)
    assert TONE_GUIDANCE in edit


def test_editor_gets_tone_too(monkeypatch):
    # Gating only the identify path would let the description revert to the old
    # voice the first time a user refined a task.
    monkeypatch.setenv("IDENTIFY_TONE_PROMPT", "on")
    assert TONE_GUIDANCE in task_editor_prompt()


def test_pr19_date_grounding_survives_both_ways(monkeypatch):
    # Regression guard: PR #19's date/weekday grounding is why this change had to be
    # flag-gated. It must be present whether the tone flag is on or off.
    for raw, present in (("true", True), ("", False)):
        monkeypatch.setenv("IDENTIFY_TONE_PROMPT", raw)
        prompt = identify_system_prompt()
        assert "current date given in the message" in prompt
        assert "correct year in ISO8601" in prompt
        assert (TONE_GUIDANCE in prompt) is present


# ── the guidance itself ──────────────────────────────────────────────────────

def test_guidance_practises_what_it_preaches():
    # A tone instruction that itself contains em/en dashes is self-undermining.
    assert "—" not in TONE_GUIDANCE
    assert "–" not in TONE_GUIDANCE


def test_guidance_firewalls_extraction():
    # The load-bearing sentence: tone must not change what is extracted. If this
    # wording is ever dropped, the A/B result no longer transfers.
    lowered = TONE_GUIDANCE.lower()
    assert "context_items" in lowered
    assert "never what you extract" in lowered


# ── the flag reaches the constructed agent ───────────────────────────────────

def test_constructed_agent_receives_the_toned_prompt(monkeypatch):
    # The prompt is chosen when the agent is built. If it were captured at import
    # time instead, the flag would silently do nothing and an A/B would measure
    # noise. Capture what create_agent actually receives.
    seen = []

    def _fake_create_agent(*, model, response_format, system_prompt):
        seen.append(system_prompt)
        return object()

    monkeypatch.setattr(task_agent, "create_agent", _fake_create_agent)

    monkeypatch.delenv("IDENTIFY_TONE_PROMPT", raising=False)
    task_agent.TaskIdentifierAgent()
    assert seen == [SYSTEM_PROMPT, TASK_EDITOR_PROMPT]

    seen.clear()
    monkeypatch.setenv("IDENTIFY_TONE_PROMPT", "true")
    task_agent.TaskIdentifierAgent()
    assert len(seen) == 2
    assert all(TONE_GUIDANCE in p for p in seen)
