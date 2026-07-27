"""Progress: whether the things flagged before are actually getting better."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...analytics.snapshots import (
    TRACKED_METRICS,
    format_unit,
    metric_history,
    muscle_score_history,
)
from ...domain.muscles import label as muscle_label
from ...viz import muscle_score_history_chart, snapshot_metric_chart
from .. import components as ui
from ..state import current_report, palette


def render() -> None:
    report = current_report()
    colors = palette()

    st.title("Progress")

    history = report.snapshot_history
    if len(history) < 2:
        st.info(
            f"Progress tracking needs at least two snapshots and you have {len(history)}. "
            f"A snapshot is stored automatically each time you sync, so this page fills in "
            f"as you use the app — come back after a week or two of syncs."
        )
        st.caption(
            "Snapshots record the headline numbers only (muscle attention scores, weekly "
            "volume, efficiency trend, load balance), so the diagnostics themselves can be "
            "trended rather than just the raw activities."
        )
        return

    deltas = report.progress_deltas
    weeks = st.select_slider("Compare against", options=[2, 4, 6, 8, 12, 16],
                             value=deltas.get("weeks_back", 6) if deltas else 6,
                             format_func=lambda w: f"{w} weeks ago")
    if weeks != (deltas.get("weeks_back") if deltas else None):
        from ...analytics.snapshots import compare
        deltas = compare(history, weeks_back=weeks)

    if not deltas:
        st.warning(
            f"No snapshot close enough to {weeks} weeks ago to compare against. "
            f"You have {len(history)} snapshots spanning "
            f"{history[0]['taken_on']} → {history[-1]['taken_on']}."
        )
        return

    ui.hero(f"{deltas['days_between']} days",
            f"Comparing {deltas['from']} → {deltas['to']}", colors,
            sub=f"{len(history)} snapshots stored.")

    _metric_deltas(deltas, colors)

    st.divider()
    ui.section("Are the flagged muscles catching up?",
               "Attention score over time. Falling is good — it means volume, strength "
               "trend, recency or balance improved.", colors)

    scores = muscle_score_history(history)
    if scores.empty:
        st.info("No muscle scores in the stored snapshots yet.")
    else:
        current_worst = (report.lagging.head(4)["muscle"].tolist()
                         if not report.lagging.empty else [])
        options = sorted(scores["muscle"].unique(), key=lambda m: muscle_label(m))
        chosen = st.multiselect(
            "Muscles", options, default=current_worst or options[:3], max_selections=6,
            format_func=muscle_label)
        table = _muscle_delta_table(deltas)
        ui.chart(muscle_score_history_chart(scores, chosen,
                                            {m: muscle_label(m) for m in options}, colors),
                 colors, table=table, table_label="View all muscle changes",
                 key="progress_muscles")

    st.divider()
    ui.section("Tracked metrics over time", None, colors)
    available = [(key, label, unit) for key, label, _better, unit, _fmt in TRACKED_METRICS
                 if len(metric_history(history, key)) >= 2]
    if not available:
        st.info("Not enough repeated metrics across snapshots yet.")
        return

    columns = st.columns(2)
    for index, (key, label, unit) in enumerate(available):
        with columns[index % 2]:
            ui.chart(snapshot_metric_chart(metric_history(history, key), label, unit, colors),
                     colors, key=f"progress_metric_{key}")

    st.divider()
    ui.section("Progress findings", None, colors)
    ui.findings_list(report.findings_for("progress"), colors,
                     empty_message="No significant movement either way in this window.")


def _metric_deltas(deltas: dict, colors) -> None:
    if not deltas.get("metrics"):
        st.caption("No metrics appear in both snapshots yet.")
        return

    improved = [m for m in deltas["metrics"] if m["direction"] == "improved"]
    worsened = [m for m in deltas["metrics"] if m["direction"] == "worsened"]

    ui.stat_tiles([
        {"label": "Improved", "value": str(len(improved)),
         "note": ", ".join(m["label"] for m in improved[:2]) or "—"},
        {"label": "Worsened", "value": str(len(worsened)),
         "note": ", ".join(m["label"] for m in worsened[:2]) or "—"},
        {"label": "Muscles catching up",
         "value": str(len([m for m in deltas["muscles"] if m["score_change"] <= -8]))},
        {"label": "Muscles slipping",
         "value": str(len([m for m in deltas["muscles"] if m["score_change"] >= 8]))},
    ], palette=colors)

    frame = pd.DataFrame([{
        "Metric": m["label"],
        "Then": format_unit(m["then_label"], m["unit"]),
        "Now": format_unit(m["now_label"], m["unit"]),
        "Change": m["change_label"],
        "Direction": m["direction"],
    } for m in deltas["metrics"]])
    st.dataframe(frame, use_container_width=True, hide_index=True)


def _muscle_delta_table(deltas: dict) -> pd.DataFrame:
    if not deltas.get("muscles"):
        return pd.DataFrame()
    return pd.DataFrame([{
        "Muscle": muscle_label(m["muscle"]),
        "Score then": round(m["then_score"], 0),
        "Score now": round(m["now_score"], 0),
        "Change": round(m["score_change"], 0),
        "Sets/wk then": (None if pd.isna(m["then_sets"]) else round(m["then_sets"], 1)),
        "Sets/wk now": (None if pd.isna(m["now_sets"]) else round(m["now_sets"], 1)),
    } for m in deltas["muscles"]])
