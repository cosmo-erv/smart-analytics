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
    if not config.has_openai_key:
        raise OpenAIUnavailable(
            "No OPENAI_API_KEY set. The analytics still work — add a key to .env "
            "to enable the coaching narrative."
        )
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise OpenAIUnavailable(
            "The `openai` package isn't installed. Run `pip install -e \".[dev]\"` "
            "again, or `pip install openai`."
        ) from exc
    return openai.OpenAI(api_key=config.openai_api_key)


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


def coach_json(config: Settings, *, system: str, briefing: dict[str, Any],
               instruction: str, schema: dict[str, Any],
               max_tokens: int) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return ``(payload, model, usage)`` for a schema-constrained coaching call."""
    api = client(config)
    response = api.chat.completions.create(
        model=config.openai_model,
        max_completion_tokens=max_tokens,
        messages=_messages(system, briefing, instruction),
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "coach_report", "strict": True, "schema": schema},
        },
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
