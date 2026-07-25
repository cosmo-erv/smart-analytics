"""The AI coaching layer.

Design rule: **the model never sees raw activity data and never computes
numbers.** The deterministic engines in ``analytics/`` do all the arithmetic and
hand over a compact briefing (:meth:`TrainingReport.briefing`); the model's job is
to interpret it — rank what matters, explain the mechanism, and turn findings into
a plan. That keeps the numbers reproducible and auditable, and stops the model
from inventing statistics.

Either Anthropic or OpenAI can serve it, chosen by whichever key is configured
(``AI_PROVIDER`` forces one). The prompts, schema and briefing are shared; only
the transport differs — see :mod:`.openai_backend`.

Two entry points:

* :func:`coach_report` — structured output (schema-constrained JSON) driving the
  AI Coach tab: assessment, ranked priorities, four-week plan.
* :func:`ask` — free-form Q&A against the same briefing, for follow-ups.

Without an API key the module degrades cleanly: :func:`fallback_report` builds
the same shape from the rule-based findings, so the GUI is never empty.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..analytics.findings import SEVERITY_LABEL, Finding
from ..config import Settings, settings as default_settings

log = logging.getLogger(__name__)

MAX_TOKENS = 8000
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """\
You are an experienced endurance and strength coach reviewing an athlete's \
training data. You are analytical, specific and honest — you tell them what the \
numbers actually say, including when something is going badly.

Ground rules:

1. Every claim you make must trace to a number in the briefing JSON. Quote the \
number. If the briefing doesn't cover something, say you can't tell from this \
data rather than guessing.
2. Never invent a metric, a date or a trend. If a field is null or a finding \
says "insufficient data", treat that as a real limitation worth stating.
3. Explain the mechanism, not just the observation. "Rear delts are at 2.4 \
sets/week against a 10-set target, and front delts get 3x that — that ratio is \
what drives shoulders forward over time" beats "train rear delts more".
4. Prioritise ruthlessly. The athlete can change two or three things, not \
fifteen. Rank by expected impact, and be explicit about what to ignore for now.
5. Respect the units: effective sets are weekly and already weight secondary \
muscle involvement; pace is seconds per kilometre; m/beat is metres per \
heartbeat where higher is better; ACWR is 7-day load over the 28-day weekly \
average.
6. Be concrete about actions — sets, sessions, paces, weeks. "Add 4 effective \
sets of rowing per week, split across two sessions" not "increase back volume".
7. You are not a doctor. If the data suggests a medical issue (sustained \
elevated resting HR, pain patterns), say so plainly and recommend they see a \
professional, without diagnosing.

Keep prose tight. No preamble, no restating the question, no filler."""

COACH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headline": {
            "type": "string",
            "description": "One sentence, under 120 chars: the single most important thing.",
        },
        "assessment": {
            "type": "string",
            "description": ("2-4 short paragraphs of markdown assessing the current training "
                            "state, citing specific numbers from the briefing."),
        },
        "priorities": {
            "type": "array",
            "description": "2-4 ranked priorities, most important first.",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "area": {"type": "string", "enum": ["strength", "running", "load", "recovery"]},
                    "evidence": {"type": "string",
                                 "description": "The numbers that justify this, quoted."},
                    "why_it_matters": {"type": "string"},
                    "action": {"type": "string",
                               "description": "Specific, countable change to make."},
                    "timeframe": {"type": "string",
                                  "description": "e.g. 'next 4 weeks', 'this week'."},
                },
                "required": ["title", "area", "evidence", "why_it_matters", "action", "timeframe"],
                "additionalProperties": False,
            },
        },
        "quick_wins": {
            "type": "array",
            "description": "0-3 low-effort changes worth making immediately.",
            "items": {"type": "string"},
        },
        "ignore_for_now": {
            "type": "array",
            "description": ("0-3 things the data flags that are NOT worth acting on yet, with "
                            "the reason."),
            "items": {"type": "string"},
        },
        "four_week_plan": {
            "type": "array",
            "description": "Four entries, one per week.",
            "items": {
                "type": "object",
                "properties": {
                    "week": {"type": "integer"},
                    "focus": {"type": "string"},
                    "key_changes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["week", "focus", "key_changes"],
                "additionalProperties": False,
            },
        },
        "data_gaps": {
            "type": "array",
            "description": "What extra data would sharpen this analysis.",
            "items": {"type": "string"},
        },
    },
    "required": ["headline", "assessment", "priorities", "quick_wins", "ignore_for_now",
                 "four_week_plan", "data_gaps"],
    "additionalProperties": False,
}


@dataclass
class CoachReport:
    headline: str = ""
    assessment: str = ""
    priorities: list[dict[str, Any]] = field(default_factory=list)
    quick_wins: list[str] = field(default_factory=list)
    ignore_for_now: list[str] = field(default_factory=list)
    four_week_plan: list[dict[str, Any]] = field(default_factory=list)
    data_gaps: list[str] = field(default_factory=list)
    model: str = ""
    source: str = "anthropic"  # anthropic | openai | rules
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.headline)


class CoachUnavailable(RuntimeError):
    """No API key, or the SDK isn't installed."""


def _client(config: Settings):
    if not config.has_anthropic_key:
        raise CoachUnavailable(
            "No ANTHROPIC_API_KEY set. The analytics still work — add a key to .env "
            "to enable the coaching narrative."
        )
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise CoachUnavailable("The `anthropic` package isn't installed.") from exc
    return anthropic.Anthropic(api_key=config.anthropic_api_key)


def _request(client, config: Settings, *, system, messages, max_tokens: int,
             output_config: dict[str, Any] | None = None):
    """Send a request, preferring server-side refusal fallbacks when available.

    ``fallbacks="default"`` lets the API re-run a declined request on a suitable
    model server-side. It's a beta, so if this account can't use it we retry the
    identical request on the stable endpoint rather than failing the user's click.
    """
    import anthropic

    kwargs: dict[str, Any] = {
        "model": config.model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    if output_config:
        kwargs["output_config"] = output_config

    try:
        return client.beta.messages.create(
            betas=[FALLBACK_BETA], fallbacks="default", **kwargs
        )
    except anthropic.BadRequestError as exc:
        message = str(exc).lower()
        if "fallback" not in message and "beta" not in message:
            raise
        log.info("Server-side fallbacks unavailable (%s); using the standard endpoint", exc)
        return client.messages.create(**kwargs)


def _text_of(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


def _usage_of(response) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
    }


def _briefing_message(briefing: dict[str, Any], instruction: str) -> list[dict[str, Any]]:
    """Briefing first with a cache breakpoint, varying instruction after it.

    The briefing is identical across every question in a session, so caching it
    makes follow-up questions cheap; the instruction must sit *after* the
    breakpoint or each question would write its own cache entry.
    """
    return [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": ("Here is the athlete's computed training briefing.\n\n```json\n"
                         + json.dumps(briefing, indent=2, sort_keys=True, default=str)
                         + "\n```"),
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": instruction},
        ],
    }]


COACH_INSTRUCTION = """\
Review this briefing and produce your coaching assessment.

Work through it in this order:
1. What is the athlete doing well? (Don't skip this — it tells them what to keep.)
2. What is the highest-impact problem, and what's the evidence?
3. Which flagged findings are noise or not worth acting on yet?
4. What should the next four weeks look like?

Rank priorities by expected impact on the athlete's progress, not by how alarming \
the metric looks. A muscle at zero volume matters more than a 2% pace regression."""

ASK_INSTRUCTION = (
    "Answer questions about this briefing. Cite the specific numbers you rely on, and say "
    "so plainly when the data can't answer the question."
)


def coach_report(briefing: dict[str, Any], config: Settings | None = None) -> CoachReport:
    """Ask the configured model for a structured coaching assessment."""
    config = config or default_settings
    if config.provider == "openai":
        return _openai_coach_report(briefing, config)

    client = _client(config)

    try:
        response = _request(
            client, config,
            system=[{"type": "text", "text": SYSTEM_PROMPT}],
            messages=_briefing_message(briefing, COACH_INSTRUCTION),
            max_tokens=MAX_TOKENS,
            output_config={"format": {"type": "json_schema", "schema": COACH_SCHEMA}},
        )
    except Exception as exc:
        log.exception("Coaching request failed")
        return CoachReport(error=f"{type(exc).__name__}: {exc}", model=config.model)

    if getattr(response, "stop_reason", None) == "refusal":
        return CoachReport(
            error="The model declined to answer this request.", model=config.model)

    try:
        payload = json.loads(_text_of(response))
    except json.JSONDecodeError as exc:
        return CoachReport(error=f"Could not parse the model's response: {exc}",
                           model=config.model)

    return _report_from(payload, getattr(response, "model", config.model),
                        _usage_of(response), "anthropic")


def _report_from(payload: dict[str, Any], model: str, usage: dict[str, Any],
                 source: str) -> CoachReport:
    return CoachReport(
        headline=payload.get("headline", ""),
        assessment=payload.get("assessment", ""),
        priorities=payload.get("priorities", []),
        quick_wins=payload.get("quick_wins", []),
        ignore_for_now=payload.get("ignore_for_now", []),
        four_week_plan=payload.get("four_week_plan", []),
        data_gaps=payload.get("data_gaps", []),
        model=model,
        usage=usage,
        source=source,
    )


def _openai_coach_report(briefing: dict[str, Any], config: Settings) -> CoachReport:
    from . import openai_backend

    try:
        payload, model, usage = openai_backend.coach_json(
            config,
            system=SYSTEM_PROMPT,
            briefing=briefing,
            instruction=COACH_INSTRUCTION,
            schema=COACH_SCHEMA,
            max_tokens=MAX_TOKENS,
        )
    except json.JSONDecodeError as exc:
        return CoachReport(error=f"Could not parse the model's response: {exc}",
                           model=config.openai_model, source="openai")
    except Exception as exc:
        log.exception("Coaching request failed")
        return CoachReport(error=f"{type(exc).__name__}: {exc}",
                           model=config.openai_model, source="openai")

    return _report_from(payload, model, usage, "openai")


def ask(question: str, briefing: dict[str, Any], history: list[dict[str, str]] | None = None,
        config: Settings | None = None) -> Iterator[str]:
    """Stream an answer to a follow-up question about the briefing."""
    config = config or default_settings
    if config.provider == "openai":
        from . import openai_backend

        yield from openai_backend.stream_answer(
            config, system=SYSTEM_PROMPT, briefing=briefing,
            instruction=ASK_INSTRUCTION, history=history, question=question,
            max_tokens=4000)
        return

    client = _client(config)
    messages = _briefing_message(briefing, ASK_INSTRUCTION)
    for turn in history or []:
        if turn.get("role") in {"user", "assistant"} and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})

    with client.messages.stream(
        model=config.model,
        max_tokens=4000,
        system=[{"type": "text", "text": SYSTEM_PROMPT}],
        messages=messages,
    ) as stream:
        for chunk in stream.text_stream:
            yield chunk


# --- no-API-key path -------------------------------------------------------

def fallback_report(findings: list[Finding]) -> CoachReport:
    """Build the same report shape from the deterministic findings alone.

    Not as good as the narrative version — it can't weigh a stalled row against a
    grey-zone running problem — but it keeps the app fully functional and honest
    about what it is.
    """
    ranked = [f for f in findings if f.severity in {"act", "watch"}]
    good = [f for f in findings if f.severity == "good"]

    priorities = []
    for finding in ranked[:4]:
        priorities.append({
            "title": finding.title,
            "area": finding.area if finding.area in
                    {"strength", "running", "load", "recovery"} else "load",
            "evidence": finding.metric or finding.detail,
            "why_it_matters": finding.detail,
            "action": finding.recommendation or "Address this before adding training load.",
            "timeframe": "next 4 weeks",
        })

    act_count = sum(1 for f in findings if f.severity == "act")
    headline = (f"{act_count} issue{'s' if act_count != 1 else ''} need attention"
                if act_count else "Nothing urgent — keep building")

    lines = [
        f"The analytics engine produced {len(findings)} findings: "
        f"{act_count} needing action, "
        f"{sum(1 for f in findings if f.severity == 'watch')} worth watching, "
        f"{len(good)} going well.",
        "",
        "This is the rule-based summary. Add an `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` "
        "to `.env` for the coaching layer, which weighs these findings against each "
        "other and turns them into a plan.",
    ]
    if good:
        lines += ["", "**Going well:** " + "; ".join(f.title for f in good[:3]) + "."]

    return CoachReport(
        headline=headline,
        assessment="\n".join(lines),
        priorities=priorities,
        quick_wins=[f.recommendation for f in ranked[:3] if f.recommendation],
        ignore_for_now=[],
        four_week_plan=[],
        data_gaps=["Connect an API key to get a prioritised, explained plan."],
        source="rules",
    )


def severity_label(severity: str) -> str:
    return SEVERITY_LABEL.get(severity, severity.title())
