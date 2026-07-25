"""Overview: the state of training in one screen."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...viz import activity_mix, fitness_fatigue
from .. import components as ui
from ..state import current_report, palette


def render() -> None:
    report = current_report()
    colors = palette()

    st.title("Training overview")

    if not report.has_data:
        ui.no_data_notice("activities")
        return

    top = report.top_findings(1)
    headline = top[0].title if top else "Nothing needs attention"
    window = _window_text(report)
    ui.hero(headline, "Biggest thing right now", colors, sub=window)

    ui.stat_tiles(_tiles(report), palette=colors)

    _next_session_callout(report, colors)
    _progress_callout(report, colors)

    st.divider()
    ui.section("What the data says", "Ranked by severity, most urgent first.", colors)
    ui.findings_list(report.top_findings(7), colors)

    with st.expander(f"All {len(report.findings)} findings"):
        ui.findings_list(report.findings_for(), colors)

    st.divider()
    left, right = st.columns(2)
    with left:
        ui.section("Where the time goes", None, colors)
        ui.chart(activity_mix(report.activity_mix, colors), colors,
                 table=report.activity_mix, key="overview_mix")
    with right:
        ui.section("Fitness and fatigue", None, colors)
        table = None
        if not report.load_series.empty:
            table = (report.load_series.tail(21)[["date", "load", "ctl", "atl", "tsb", "acwr"]]
                     .round(2).iloc[::-1])
        ui.chart(fitness_fatigue(report.load_series, colors), colors, table=table,
                 key="overview_ff")


def _window_text(report) -> str:
    first, last = report.meta.get("first_date"), report.meta.get("last_date")
    count = report.meta.get("activity_count") or 0
    if not first or not last:
        return f"{count} activities"
    return f"{count} activities · {first} to {last}"


def _tiles(report) -> list[dict]:
    tiles: list[dict] = []

    if not report.weekly_runs.empty:
        recent = report.weekly_runs.tail(4)["distance_km"].mean()
        prior = report.weekly_runs.tail(8).head(4)["distance_km"].mean()
        delta = None
        if pd.notna(prior) and prior > 0:
            delta = f"{(recent - prior) / prior * 100:+.0f}% vs prior 4 wk"
        tiles.append({"label": "Running volume", "value": f"{recent:.0f} km/wk",
                      "delta": delta,
                      "help": "Mean weekly distance over the last four weeks."})

    if not report.lagging.empty:
        weekly_sets = report.lagging["weekly_sets"].sum()
        behind = int((report.lagging["attention_score"] >= 60).sum())
        tiles.append({
            "label": "Strength volume", "value": f"{weekly_sets:.0f} sets/wk",
            "note": f"{behind} muscle{'s' if behind != 1 else ''} falling behind",
            "help": "Total effective sets per week across all muscles.",
        })

    if not report.load_series.empty:
        latest = report.load_series.dropna(subset=["acwr"]).tail(1)
        if not latest.empty:
            row = latest.iloc[0]
            tiles.append({"label": "Load balance", "value": f"{row['acwr']:.2f}",
                          "note": str(row["acwr_status"]),
                          "help": "Acute:chronic workload ratio. 0.8–1.3 is productive."})
        last = report.load_series.tail(1).iloc[0]
        if pd.notna(last["ctl"]):
            tiles.append({"label": "Fitness", "value": f"{last['ctl']:.0f}",
                          "note": f"form {last['tsb']:+.0f}" if pd.notna(last["tsb"]) else None,
                          "help": "42-day exponentially weighted training load."})

    if not report.athlete_metrics.empty:
        latest = report.athlete_metrics.tail(1).iloc[0]
        if pd.notna(latest.get("training_status")):
            note = (f"readiness {latest['readiness_score']:.0f}/100"
                    if pd.notna(latest.get("readiness_score")) else "from Garmin")
            tiles.append({"label": "Garmin status", "value": str(latest["training_status"]),
                          "note": note,
                          "help": "Garmin's own training-status verdict from the watch."})

    efficiency = report.run_trends.get("aerobic_efficiency", {})
    if efficiency.get("reliable"):
        tiles.append({
            "label": "Aerobic efficiency", "value": f"{efficiency['pct_per_month']:+.1f}%/mo",
            "note": f"over {efficiency['span_days']} days",
            "help": "Trend in metres covered per heartbeat — higher is better.",
        })

    return tiles[:5]


def _next_session_callout(report, colors) -> None:
    """The recommendation, surfaced here because it's the reason to open the app."""
    rec = report.recommendation
    if rec is None:
        return
    icons = {"rest": "🛌", "easy_run": "🏃", "long_run": "🏔", "quality_run": "⚡",
             "strength": "🏋", "mobility": "🧘"}
    with st.container(border=True):
        left, right = st.columns([4, 1])
        with left:
            st.markdown(
                f'<div style="font-size:0.75rem;font-weight:600;letter-spacing:0.05em;'
                f'text-transform:uppercase;color:{colors.ink_muted};">Suggested next session'
                f'</div>'
                f'<div style="font-size:1.15rem;font-weight:650;color:{colors.ink};'
                f'margin:2px 0 4px;">{icons.get(rec.kind, "•")} {rec.title}</div>'
                f'<div style="color:{colors.ink_secondary};font-size:0.9rem;">'
                f'{rec.reasons[0] if rec.reasons else ""}</div>',
                unsafe_allow_html=True)
        with right:
            plan_page = (st.session_state.get("nav_pages") or {}).get("plan")
            if plan_page is not None:
                st.page_link(plan_page, label="Open plan",
                             icon=":material/arrow_forward:")


def _progress_callout(report, colors) -> None:
    deltas = report.progress_deltas
    if not deltas or not deltas.get("metrics"):
        return
    improved = [m for m in deltas["metrics"] if m["direction"] == "improved"]
    worsened = [m for m in deltas["metrics"] if m["direction"] == "worsened"]
    catching_up = [m for m in deltas["muscles"] if m["score_change"] <= -8]

    parts = []
    if improved:
        parts.append(f"**{len(improved)} metric{'s' if len(improved) != 1 else ''} improved**")
    if worsened:
        parts.append(f"{len(worsened)} worsened")
    if catching_up:
        parts.append(f"{len(catching_up)} muscle{'s' if len(catching_up) != 1 else ''} "
                     f"catching up")
    if not parts:
        return
    st.caption(f"Since {deltas['from']} ({deltas['days_between']} days ago): "
               + ", ".join(parts) + ". See **Progress** for the detail.")
