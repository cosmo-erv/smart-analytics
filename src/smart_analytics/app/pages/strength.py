"""Strength: which muscles are covered, which are falling behind, and why."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...analytics.strength import exercise_sessions
from ...viz import (
    attention_scores,
    exercise_progression,
    muscle_volume_vs_target,
    muscle_week_heatmap,
)
from .. import components as ui
from ..state import current_report, palette


def render() -> None:
    report = current_report()
    colors = palette()

    st.title("Strength")

    if not report.has_strength:
        if report.has_data:
            st.warning(
                "No per-set strength data found. The muscle model needs Garmin's exercise "
                "detail, which comes from strength workouts recorded on a watch with "
                "exercise tracking on. Manually logged sessions carry no set data."
            )
            if not report.unmapped.empty:
                st.caption("Unmapped exercises found in the cache:")
                st.dataframe(report.unmapped, use_container_width=True, hide_index=True)
        else:
            ui.no_data_notice("strength activities")
        return

    lagging = report.lagging
    worst = lagging.iloc[0]
    ui.hero(
        str(worst["muscle_label"]),
        "Most neglected muscle",
        colors,
        sub=f"{worst['reasons']} — attention score {worst['attention_score']:.0f}/100",
    )

    ui.stat_tiles(_tiles(report), palette=colors)

    st.divider()
    ui.section(
        "Volume per muscle",
        f"Effective sets per week over the last {report.meta['lookback_days']} days, against "
        f"your {report.settings.weekly_sets_min}–{report.settings.weekly_sets_max} set target "
        f"band. Secondary involvement counts fractionally: a row gives the upper back a full "
        f"set and the biceps half of one.",
        colors,
    )
    volume_table = lagging[["muscle_label", "region", "weekly_sets", "pct_of_target",
                            "pct_per_month", "days_since", "verdict"]].rename(columns={
        "muscle_label": "Muscle", "region": "Region", "weekly_sets": "Sets/week",
        "pct_of_target": "% of target", "pct_per_month": "Strength %/month",
        "days_since": "Days since trained", "verdict": "Verdict"})
    ui.chart(muscle_volume_vs_target(lagging, colors, report.settings.weekly_sets_min,
                                     report.settings.weekly_sets_max),
             colors, table=volume_table, key="strength_volume")

    st.divider()
    ui.section(
        "Which muscles are falling behind",
        "Four independent signals combined: volume against target, estimated-1RM trend, "
        "days since the muscle was last loaded, and how it compares with its structural "
        "counterpart. Hover a bar for the reasons that fired.",
        colors,
    )
    score_table = lagging[["muscle_label", "attention_score", "volume_component",
                           "trend_component", "recency_component", "balance_component",
                           "reasons"]].rename(columns={
        "muscle_label": "Muscle", "attention_score": "Score",
        "volume_component": "Volume", "trend_component": "Trend",
        "recency_component": "Recency", "balance_component": "Balance",
        "reasons": "Reasons"})
    ui.chart(attention_scores(lagging, colors), colors, table=score_table,
             table_label="View scores and component breakdown", key="strength_scores")

    st.divider()
    left, right = st.columns([1.1, 1])
    with left:
        ui.section("Balance ratios", "Antagonist and structural pairs.", colors)
        if report.balance.empty:
            st.info("Not enough variety in the logged exercises to compute ratios.")
        else:
            display = report.balance[["pair", "ratio", "low", "high", "status",
                                      "numerator_sets", "denominator_sets"]].rename(columns={
                "pair": "Pair", "ratio": "Ratio", "low": "Healthy min", "high": "Healthy max",
                "status": "Status", "numerator_sets": "First side sets/wk",
                "denominator_sets": "Second side sets/wk"})
            st.dataframe(display, use_container_width=True, hide_index=True)
    with right:
        ui.section("Movement patterns", "Whole patterns missing is the loudest signal.", colors)
        patterns = report.patterns.copy()
        patterns["pattern"] = patterns["pattern"].str.replace("_", " ").str.title()
        st.dataframe(patterns.rename(columns={
            "pattern": "Pattern", "weekly_sets": "Sets/week", "sessions": "Sessions"}),
            use_container_width=True, hide_index=True)

    st.divider()
    ui.section("Strength progression", "Estimated 1RM (Epley) from the best set each session.",
               colors)
    sessions = exercise_sessions(report.expanded)
    options = (report.progress.sort_values("sessions", ascending=False)["exercise"].tolist()
               if not report.progress.empty else sorted(sessions["exercise"].unique()))
    default = options[:3]
    chosen = st.multiselect("Exercises", options, default=default, max_selections=6,
                            help="Up to six; each line is labelled directly on the chart.")
    ui.chart(exercise_progression(sessions, chosen, colors), colors,
             table=_progress_table(report.progress), table_label="View all exercise trends",
             key="strength_progress")

    st.divider()
    ui.section("Findings", None, colors)
    ui.findings_list(report.findings_for("strength"), colors)

    if not report.unmapped.empty:
        with st.expander(f"{len(report.unmapped)} unmapped exercises "
                         f"({int(report.unmapped['sets'].sum())} sets excluded)"):
            st.caption(
                "These exercises aren't in the muscle map, so their volume is excluded. "
                "Add them to `NAME_PROFILES` in `src/smart_analytics/domain/exercises.py` "
                "to include them."
            )
            st.dataframe(report.unmapped, use_container_width=True, hide_index=True)


def _tiles(report) -> list[dict]:
    lagging = report.lagging
    behind = int((lagging["attention_score"] >= 60).sum())
    watch = int((lagging["attention_score"].between(40, 60)).sum())
    covered = int((lagging["attention_score"] < 22).sum())

    weeks = max(report.meta["lookback_days"] / 7.0, 1)
    sessions = int(report.expanded["activity_id"].nunique())
    window_sessions = report.expanded[
        report.expanded["local_date"] >= report.expanded["local_date"].max()
        - pd.Timedelta(days=report.meta["lookback_days"])]["activity_id"].nunique()

    tiles = [
        {"label": "Falling behind", "value": str(behind),
         "note": f"{watch} more to watch",
         "help": "Muscles scoring 60+ on the attention model."},
        {"label": "Well covered", "value": str(covered),
         "note": f"of {len(lagging)} muscles"},
        {"label": "Sessions", "value": f"{window_sessions / weeks:.1f}/wk",
         "note": f"{sessions} logged in total"},
    ]

    reliable = report.progress[report.progress["reliable"]] if not report.progress.empty \
        else report.progress
    if not reliable.empty:
        best = reliable.sort_values("pct_per_month", ascending=False).iloc[0]
        # Value carries the number; the exercise name goes in the note, which wraps
        # instead of truncating the way a long metric value does.
        tiles.append({"label": "Fastest gaining lift",
                      "value": f"{best['pct_per_month']:+.1f}%/mo",
                      "note": str(best["exercise"]),
                      "help": "Estimated 1RM trend over the progression window."})
        stalled = int(reliable["status"].isin(["stalled", "regressing"]).sum())
        tiles.append({"label": "Stalled lifts", "value": str(stalled),
                      "note": f"of {len(reliable)} tracked"})
    return tiles


def _progress_table(progress: pd.DataFrame) -> pd.DataFrame:
    if progress is None or progress.empty:
        return pd.DataFrame()
    table = progress[["exercise", "sessions", "current_e1rm", "best_e1rm", "kg_per_month",
                      "pct_per_month", "days_since", "status"]].copy()
    return table.rename(columns={
        "exercise": "Exercise", "sessions": "Sessions", "current_e1rm": "Current e1RM (kg)",
        "best_e1rm": "Best e1RM (kg)", "kg_per_month": "kg/month",
        "pct_per_month": "%/month", "days_since": "Days since", "status": "Status"})
