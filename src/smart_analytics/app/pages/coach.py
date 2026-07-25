"""AI Coach: Claude interprets the computed briefing and answers follow-ups."""

from __future__ import annotations

import json

import streamlit as st

from ...ai import CoachUnavailable, ask, coach_report, fallback_report
from ...config import settings
from .. import components as ui
from ..state import current_report, data_version, palette

REPORT_KEY = "coach_report"
REPORT_VERSION_KEY = "coach_report_version"
CHAT_KEY = "coach_chat"


def render() -> None:
    report = current_report()
    colors = palette()

    st.title("AI coach")

    if not report.has_data:
        ui.no_data_notice("activities")
        return

    briefing = report.briefing()

    st.markdown(
        f'<div style="color:{colors.ink_secondary};font-size:0.94rem;line-height:1.55;">'
        f"Claude reads the <strong>computed</strong> metrics — not raw activity files — and "
        f"interprets them: what matters most, why, and what to change. Every number it "
        f"quotes comes from the analytics engine, so the arithmetic stays reproducible."
        f"</div>",
        unsafe_allow_html=True,
    )

    if not settings.has_anthropic_key:
        st.info(
            "No `ANTHROPIC_API_KEY` found in `.env`. Showing the rule-based summary instead — "
            "all metrics and findings are still computed, you just don't get the narrative "
            "layer that weighs them against each other."
        )

    controls = st.columns([1.2, 1, 2])
    with controls[0]:
        generate = st.button("Generate assessment", type="primary", use_container_width=True)
    with controls[1]:
        if st.button("Clear", use_container_width=True):
            st.session_state.pop(REPORT_KEY, None)
            st.session_state.pop(CHAT_KEY, None)
            st.rerun()
    with controls[2]:
        if settings.has_anthropic_key:
            st.caption(f"Model: `{settings.model}`")

    stale = st.session_state.get(REPORT_VERSION_KEY) != data_version()
    if REPORT_KEY in st.session_state and stale:
        st.warning("Your data changed since this assessment was generated — regenerate it.")

    if generate:
        with st.spinner("Claude is reviewing your training…"):
            if settings.has_anthropic_key:
                try:
                    result = coach_report(briefing)
                except CoachUnavailable as exc:
                    result = fallback_report(report.findings)
                    st.warning(str(exc))
            else:
                result = fallback_report(report.findings)
        st.session_state[REPORT_KEY] = result
        st.session_state[REPORT_VERSION_KEY] = data_version()
        st.session_state.pop(CHAT_KEY, None)

    result = st.session_state.get(REPORT_KEY)
    if result is None:
        st.divider()
        _briefing_preview(briefing, colors)
        return

    if result.error:
        st.error(f"The coaching request failed: {result.error}")
        st.caption("The analytics on the other pages are unaffected.")
        return

    st.divider()
    ui.hero(result.headline, "Assessment", colors)
    st.markdown(result.assessment)

    if result.priorities:
        st.divider()
        ui.section("Priorities", "Ranked by expected impact.", colors)
        for index, priority in enumerate(result.priorities, start=1):
            with st.container(border=True):
                st.markdown(
                    f'<div style="font-size:0.78rem;font-weight:600;letter-spacing:0.04em;'
                    f'text-transform:uppercase;color:{colors.ink_muted};">'
                    f'{index} · {priority.get("area", "").title()} · '
                    f'{priority.get("timeframe", "")}</div>'
                    f'<div style="font-weight:650;font-size:1.05rem;color:{colors.ink};'
                    f'margin:2px 0 6px;">{priority.get("title", "")}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="color:{colors.ink_secondary};font-size:0.92rem;">'
                    f'<strong>Evidence:</strong> {priority.get("evidence", "")}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="color:{colors.ink_secondary};font-size:0.92rem;'
                    f'margin-top:4px;">{priority.get("why_it_matters", "")}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="margin-top:8px;padding-left:10px;'
                    f'border-left:2px solid {colors.series(0)};color:{colors.ink};'
                    f'font-size:0.94rem;"><strong>Action:</strong> '
                    f'{priority.get("action", "")}</div>',
                    unsafe_allow_html=True,
                )

    columns = st.columns(2)
    if result.quick_wins:
        with columns[0]:
            ui.section("Quick wins", None, colors)
            for item in result.quick_wins:
                st.markdown(f"- {item}")
    if result.ignore_for_now:
        with columns[1]:
            ui.section("Not worth acting on yet", None, colors)
            for item in result.ignore_for_now:
                st.markdown(f"- {item}")

    if result.four_week_plan:
        st.divider()
        ui.section("Next four weeks", None, colors)
        for week in result.four_week_plan:
            with st.expander(f"Week {week.get('week', '?')} — {week.get('focus', '')}",
                             expanded=week.get("week") == 1):
                for change in week.get("key_changes", []):
                    st.markdown(f"- {change}")

    if result.data_gaps:
        with st.expander("What would sharpen this analysis"):
            for gap in result.data_gaps:
                st.markdown(f"- {gap}")

    if result.usage:
        st.caption(
            f"Model {result.model} · {result.usage.get('input_tokens') or 0} input tokens "
            f"({result.usage.get('cache_read_input_tokens') or 0} from cache), "
            f"{result.usage.get('output_tokens') or 0} output tokens."
        )

    if settings.has_anthropic_key:
        st.divider()
        _chat(briefing, colors)

    st.divider()
    _briefing_preview(briefing, colors)


def _chat(briefing: dict, colors) -> None:
    ui.section("Ask a follow-up", "Answers are grounded in the same briefing.", colors)
    history = st.session_state.setdefault(CHAT_KEY, [])

    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input("e.g. why are my hamstrings behind, and what would fix it fastest?")
    if not question:
        return

    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(ask(question, briefing, history[:-1]))
        except Exception as exc:
            answer = f"That request failed: `{type(exc).__name__}: {exc}`"
            st.error(answer)
    history.append({"role": "assistant", "content": answer})


def _briefing_preview(briefing: dict, colors) -> None:
    with st.expander("Inspect the exact briefing sent to Claude"):
        st.caption(
            "This is the whole input. If a number isn't in here, the model has no way to "
            "know it — which is the point: no raw files, no room to invent statistics."
        )
        st.code(json.dumps(briefing, indent=2, sort_keys=True, default=str), language="json")
