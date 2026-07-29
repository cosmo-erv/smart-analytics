"""OpenAI backend for the coaching layer.

Same contract as the Anthropic path: the model receives the computed briefing and
never does arithmetic. Three differences the code has to account for:

* **Structured output** is ``response_format={"type": "json_schema", ...}`` with
  ``strict: true``, which requires every object in the schema to list all of its
  properties as required and set ``additionalProperties: false``. The coach
  schema already satisfies that, so it's passed through unchanged.
* **Caching is automatic** for long prompts — there are no cache breakpoints to
  place. What matters is that the *prefix* stays byte-identical between calls, so
  the briefing goes in its own message ahead of the varying instruction, exactly
  as it does for Anthropic.
* **Refusals** come back on ``message.refusal`` rather than a stop reason.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from ..config import Settings

log = logging.getLogger(__name__)


class OpenAIUnavailable(RuntimeError):
    """No key, or the SDK isn't installed."""


def client(config: Settings):
    """An OpenAI client, or one pointed at a local OpenAI-compatible server.

    ``OPENAI_BASE_URL`` covers Ollama, LM Studio and llama.cpp, all of which speak
    this API — which means the coaching layer can run locally, for free, with no
    account. Those servers ignore the key, so a placeholder is sent when none is
    configured rather than refusing the request for a credential nothing checks.
    """
    if not config.has_openai_key:
        raise OpenAIUnavailable(
            "No OPENAI_API_KEY or OPENAI_BASE_URL set. The analytics still work — add "
            "a key, or point OPENAI_BASE_URL at a local model, for the coaching narrative."
        )
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise OpenAIUnavailable(
            "The `openai` package isn't installed. Run `pip install -e \".[dev]\"` "
            "again, or `pip install openai`."
        ) from exc

    kwargs: dict[str, Any] = {"api_key": config.openai_api_key or "not-needed-locally"}
    if config.openai_base_url:
        kwargs["base_url"] = config.openai_base_url
    return openai.OpenAI(**kwargs)


def _messages(system: str, briefing: dict[str, Any], instruction: str) -> list[dict[str, str]]:
    """System, then the briefing, then the instruction.

    Keeping the briefing in its own message ahead of anything that varies is what
    lets the automatic prompt cache hit across follow-up questions.
    """
    return [
        {"role": "system", "content": system},
        {"role": "user",
         "content": ("Here is the athlete's computed training briefing.\n\n```json\n"
                     + json.dumps(briefing, indent=2, sort_keys=True, default=str)
                     + "\n```")},
        {"role": "user", "content": instruction},
    ]


def _usage_of(response) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    details = getattr(usage, "prompt_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "cache_read_input_tokens": getattr(details, "cached_tokens", None),
    }


def _unsupported_format(exc: Exception) -> bool:
    """Whether a failure looks like "this server can't do json_schema".

    Local servers implement varying slices of the API, and a rejected
    ``response_format`` is worth retrying differently rather than surfacing — but
    a rate limit or a bad key is not, so the match stays narrow.
    """
    text = str(exc).lower()
    return (("response_format" in text or "json_schema" in text or "strict" in text)
            and ("support" in text or "invalid" in text or "unknown" in text
                 or "unexpected" in text))


def coach_json(config: Settings, *, system: str, briefing: dict[str, Any],
               instruction: str, schema: dict[str, Any],
               max_tokens: int) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return ``(payload, model, usage)`` for a schema-constrained coaching call."""
    api = client(config)
    messages = _messages(system, briefing, instruction)

    def send(response_format: dict[str, Any], extra: list[dict[str, str]] | None = None):
        return api.chat.completions.create(
            model=config.openai_model,
            max_completion_tokens=max_tokens,
            messages=messages + (extra or []),
            response_format=response_format,
        )

    try:
        response = send({
            "type": "json_schema",
            "json_schema": {"name": "coach_report", "strict": True, "schema": schema},
        })
    except Exception as exc:
        if not _unsupported_format(exc):
            raise
        # No schema enforcement available, so the shape has to be asked for in
        # words. Less reliable than strict mode, which is why it's the fallback.
        log.info("Server rejected json_schema (%s); retrying as plain JSON", exc)
        response = send(
            {"type": "json_object"},
            [{"role": "user",
              "content": ("Reply with JSON only, conforming exactly to this schema — "
                          "every listed property is required:\n\n"
                          + json.dumps(schema, indent=2))}],
        )

    message = response.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"The model declined to answer: {message.refusal}")

    payload = json.loads(message.content or "{}")
    return payload, getattr(response, "model", config.openai_model), _usage_of(response)


def stream_answer(config: Settings, *, system: str, briefing: dict[str, Any],
                  instruction: str, history: list[dict[str, str]] | None,
                  question: str, max_tokens: int) -> Iterator[str]:
    api = client(config)
    messages = _messages(system, briefing, instruction)
    for turn in history or []:
        if turn.get("role") in {"user", "assistant"} and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})

    stream = api.chat.completions.create(
        model=config.openai_model,
        max_completion_tokens=max_tokens,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        text = getattr(chunk.choices[0].delta, "content", None)
        if text:
            yield text
