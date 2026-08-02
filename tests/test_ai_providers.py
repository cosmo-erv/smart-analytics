"""Provider selection and the OpenAI coaching backend.

The coaching layer can be served by either Anthropic or OpenAI. The prompts,
schema and briefing are shared; these tests cover the parts that aren't — which
provider gets chosen, and whether the OpenAI request is shaped the way its
structured-output API requires.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from smart_analytics.ai import insights, openai_backend
from smart_analytics.config import Settings

VALID_PAYLOAD = {
    "headline": "Rear delts are the gap",
    "assessment": "Some markdown.",
    "priorities": [{"title": "Train rear delts", "area": "strength",
                    "evidence": "1.5 sets/week vs a 10-set target",
                    "why_it_matters": "Shoulder balance.", "action": "Add 3 sets twice a week.",
                    "timeframe": "next 4 weeks"}],
    "quick_wins": ["Add face pulls to pull day."],
    "ignore_for_now": [],
    "four_week_plan": [{"week": 1, "focus": "Rebalance", "key_changes": ["+3 sets rear delts"]}],
    "data_gaps": [],
}


# --- which provider gets used -----------------------------------------------

@pytest.mark.parametrize("anthropic_key, openai_key, forced, expected", [
    ("sk-ant", "", "auto", "anthropic"),
    ("", "sk-oai", "auto", "openai"),
    # Both set: Anthropic wins by default, but the setting overrides it.
    ("sk-ant", "sk-oai", "auto", "anthropic"),
    ("sk-ant", "sk-oai", "openai", "openai"),
    ("sk-ant", "sk-oai", "anthropic", "anthropic"),
    # Forcing a provider whose key is missing selects nothing rather than
    # silently falling through to the other one.
    ("sk-ant", "", "openai", ""),
    ("", "", "auto", ""),
])
def test_provider_selection(anthropic_key, openai_key, forced, expected):
    config = Settings(anthropic_api_key=anthropic_key, openai_api_key=openai_key,
                      ai_provider=forced)
    assert config.provider == expected
    assert config.has_ai_key is bool(expected)


def test_the_active_model_follows_the_active_provider():
    both = dict(anthropic_api_key="sk-ant", openai_api_key="sk-oai",
                model="claude-opus-5", openai_model="gpt-4.1")
    assert Settings(**both, ai_provider="anthropic").active_model == "claude-opus-5"
    assert Settings(**both, ai_provider="openai").active_model == "gpt-4.1"


# --- the OpenAI request shape ------------------------------------------------

class FakeCompletions:
    def __init__(self, message: Any, usage: Any = None) -> None:
        self.message = message
        self.usage = usage
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        choice = type("Choice", (), {"message": self.message})
        return type("Response", (), {"choices": [choice], "usage": self.usage,
                                     "model": kwargs["model"]})


def _fake_openai(monkeypatch, message: Any, usage: Any = None) -> FakeCompletions:
    completions = FakeCompletions(message, usage)
    client = type("Client", (), {
        "chat": type("Chat", (), {"completions": completions})})
    monkeypatch.setattr(openai_backend, "client", lambda config: client)
    return completions


def _message(content: str | None, refusal: str | None = None):
    return type("Message", (), {"content": content, "refusal": refusal})


@pytest.fixture
def openai_config():
    return Settings(openai_api_key="sk-oai", ai_provider="openai", openai_model="gpt-4.1")


def test_the_schema_is_sent_in_strict_mode(monkeypatch, openai_config):
    """Strict mode is what guarantees the JSON matches the schema."""
    completions = _fake_openai(monkeypatch, _message(json.dumps(VALID_PAYLOAD)))
    insights.coach_report({"any": "briefing"}, openai_config)

    fmt = completions.kwargs["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] is insights.COACH_SCHEMA


def test_the_coach_schema_satisfies_strict_mode_requirements():
    """Every object must forbid extra properties and require all of its keys.

    OpenAI rejects the request otherwise, so this is worth asserting rather than
    discovering at runtime.
    """
    def check(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node.get("required", [])) == set(node.get("properties", {}))
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for item in node:
                check(item)

    check(insights.COACH_SCHEMA)


def test_the_briefing_precedes_the_instruction_so_the_cache_can_hit(monkeypatch,
                                                                    openai_config):
    """OpenAI caches on a stable prefix, so anything varying must come last."""
    completions = _fake_openai(monkeypatch, _message(json.dumps(VALID_PAYLOAD)))
    insights.coach_report({"marker": "briefing-content"}, openai_config)

    roles = [m["role"] for m in completions.kwargs["messages"]]
    assert roles == ["system", "user", "user"]
    briefing_msg, instruction_msg = completions.kwargs["messages"][1:]
    assert "briefing-content" in briefing_msg["content"]
    assert instruction_msg["content"] == insights.COACH_INSTRUCTION


def test_a_successful_report_is_tagged_with_its_provider(monkeypatch, openai_config):
    usage = type("Usage", (), {
        "prompt_tokens": 7500, "completion_tokens": 900,
        "prompt_tokens_details": type("Details", (), {"cached_tokens": 7000})})
    _fake_openai(monkeypatch, _message(json.dumps(VALID_PAYLOAD)), usage)

    report = insights.coach_report({"any": "briefing"}, openai_config)

    assert report.ok
    assert report.source == "openai"
    assert report.model == "gpt-4.1"
    assert report.headline == "Rear delts are the gap"
    assert report.usage["cache_read_input_tokens"] == 7000


def test_a_refusal_is_reported_rather_than_parsed(monkeypatch, openai_config):
    _fake_openai(monkeypatch, _message(None, refusal="I can't help with that."))
    report = insights.coach_report({"any": "briefing"}, openai_config)

    assert not report.ok
    assert "declined" in report.error
    assert report.source == "openai"


def test_unparseable_json_fails_cleanly(monkeypatch, openai_config):
    _fake_openai(monkeypatch, _message("not json at all"))
    report = insights.coach_report({"any": "briefing"}, openai_config)

    assert not report.ok
    assert "parse" in report.error.lower()


def test_a_transport_error_becomes_a_report_error_not_an_exception(monkeypatch,
                                                                   openai_config):
    """A failed call must not take the whole page down."""
    def explode(config):
        raise RuntimeError("connection reset")
    monkeypatch.setattr(openai_backend, "client", explode)

    report = insights.coach_report({"any": "briefing"}, openai_config)
    assert not report.ok
    assert "connection reset" in report.error


def test_a_missing_key_is_reported_before_any_request(openai_config):
    with pytest.raises(openai_backend.OpenAIUnavailable, match="OPENAI_API_KEY"):
        openai_backend.client(Settings())


# --- follow-up questions -----------------------------------------------------

def test_streaming_answers_yield_text_and_skip_empty_chunks(monkeypatch, openai_config):
    def chunk(text: str | None, with_choice: bool = True):
        delta = type("Delta", (), {"content": text})
        choices = [type("Choice", (), {"delta": delta})] if with_choice else []
        return type("Chunk", (), {"choices": choices})

    class StreamingCompletions:
        kwargs: dict[str, Any] = {}

        def create(self, **kwargs):
            type(self).kwargs = kwargs
            return [chunk("Rear "), chunk(None), chunk(None, with_choice=False),
                    chunk("delts.")]

    client = type("Client", (), {"chat": type("Chat", (), {
        "completions": StreamingCompletions()})})
    monkeypatch.setattr(openai_backend, "client", lambda config: client)

    out = "".join(insights.ask("What should I fix?", {"any": "briefing"},
                               history=[{"role": "user", "content": "earlier"},
                                        {"role": "assistant", "content": "reply"}],
                               config=openai_config))

    assert out == "Rear delts."
    assert StreamingCompletions.kwargs["stream"] is True
    # History is preserved between the briefing prefix and the new question.
    assert [m["content"] for m in StreamingCompletions.kwargs["messages"]][-3:] == [
        "earlier", "reply", "What should I fix?"]


# --- no key at all -----------------------------------------------------------

def test_the_fallback_summary_names_both_providers():
    report = insights.fallback_report([])
    assert report.source == "rules"
    assert "ANTHROPIC_API_KEY" in report.assessment
    assert "OPENAI_API_KEY" in report.assessment


# --- running a model locally, for free --------------------------------------

def test_a_base_url_alone_is_enough_to_be_configured():
    """Ollama and LM Studio ignore the key, so requiring one would block them."""
    config = Settings(openai_base_url="http://localhost:11434/v1")
    assert config.has_openai_key
    assert config.provider == "openai"
    assert config.is_local_ai


def test_a_plain_key_is_not_treated_as_local():
    config = Settings(openai_api_key="sk-oai")
    assert config.provider == "openai"
    assert not config.is_local_ai


def test_the_local_server_gets_the_base_url_and_a_placeholder_key(monkeypatch):
    captured: dict[str, Any] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "openai",
                        type("Mod", (), {"OpenAI": FakeOpenAI}))
    openai_backend.client(Settings(openai_base_url="http://localhost:11434/v1",
                                   openai_model="llama3.1"))

    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["api_key"]          # sent, but any value will do
    assert not captured["api_key"].startswith("sk-")


def test_no_base_url_means_no_base_url_argument(monkeypatch):
    captured: dict[str, Any] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "openai",
                        type("Mod", (), {"OpenAI": FakeOpenAI}))
    openai_backend.client(Settings(openai_api_key="sk-oai"))

    assert "base_url" not in captured
    assert captured["api_key"] == "sk-oai"


class PickyCompletions:
    """A server that rejects json_schema, as some local ones do."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["response_format"]["type"] == "json_schema":
            raise self.error
        message = type("Message", (), {"content": json.dumps(VALID_PAYLOAD),
                                       "refusal": None})
        choice = type("Choice", (), {"message": message})
        return type("Response", (), {"choices": [choice], "usage": None,
                                     "model": kwargs["model"]})


def test_a_server_that_cannot_do_json_schema_falls_back_to_plain_json(monkeypatch):
    completions = PickyCompletions(
        RuntimeError("response_format json_schema is not supported by this model"))
    client_obj = type("Client", (), {"chat": type("Chat", (), {
        "completions": completions})})
    monkeypatch.setattr(openai_backend, "client", lambda config: client_obj)

    config = Settings(openai_base_url="http://localhost:11434/v1",
                      openai_model="llama3.1", ai_provider="openai")
    report = insights.coach_report({"any": "briefing"}, config)

    assert report.ok
    assert [c["response_format"]["type"] for c in completions.calls] == [
        "json_schema", "json_object"]
    # The shape has to be asked for in words once the schema can't be enforced.
    assert "conforming exactly to this schema" in completions.calls[1]["messages"][-1]["content"]


def test_an_unrelated_failure_is_not_retried(monkeypatch):
    """Retrying a rate limit or a bad key just doubles the failure."""
    completions = PickyCompletions(RuntimeError("429 rate limit exceeded"))
    client_obj = type("Client", (), {"chat": type("Chat", (), {
        "completions": completions})})
    monkeypatch.setattr(openai_backend, "client", lambda config: client_obj)

    report = insights.coach_report(
        {"any": "briefing"}, Settings(openai_api_key="sk-oai", ai_provider="openai"))

    assert not report.ok
    assert "rate limit" in report.error
    assert len(completions.calls) == 1
