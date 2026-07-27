"""Running: performance over time, and where the improvements are."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...analytics.running import format_duration, format_pace
from ...viz import (
    decoupling_chart,
    efficiency_trend,
    intensity_split,
    pace_by_distance,
    weekly_running_volume,
    zone_distribution_chart,
)
from .. import components as ui
from ..state import current_report, palette


def render() -> None:
    report = current_report()
    colors = palette()

    st.title("Running")

    if not report.has_running:
        ui.no_data_notice("runs") if not report.has_data else st.warning(
            "No running activities in the cache yet.")
        return

    efficiency = report.run_trends.get("aerobic_efficiency", {})
    if efficiency.get("reliable"):
        pct = efficiency["pct_per_month"]
        verdict = "improving" if pct >= 1 else ("declining" if pct <= -1 else "plateaued")
        ui.hero(f"{pct:+.1f}%/month", f"Aerobic efficiency — {verdict}", colors,
                sub=(f"Metres per heartbeat across {efficiency['n']} runs over "
                     f"{efficiency['span_days']} days. This is the progress signal that "
                     f"isn't confounded by how hard you ran."))
    else:
        ui.hero(f"{len(report.runs)}", "Runs in the cache", colors,
                sub="Not enough history yet for a reliable efficiency trend.")

    ui.stat_tiles(_tiles(report), palette=colors)

    st.divider()
    ui.section(
        "Is fitness actually improving?",
        "Pace alone is confounded — a faster run at a higher heart rate isn't necessarily "
        "fitness. Metres per heartbeat normalises pace by cardiac cost, so it only rises "
        "when fitness genuinely does.",
        colors,
    )
    ui.chart(efficiency_trend(report.runs, colors), colors,
             table=_runs_table(report.runs), table_label="View recent runs",
             key="running_efficiency")

    st.divider()
    left, right = st.columns([1.4, 1])
    with left:
        ui.section("Weekly volume", None, colors)
        weekly = report.weekly_runs.copy()
        if not weekly.empty:
            weekly_display = weekly.tail(20).iloc[::-1].copy()
            weekly_display["avg_pace"] = weekly_display["avg_pace_s_per_km"].map(format_pace)
            weekly_display = weekly_display[["week", "distance_km", "runs", "avg_pace",
                                             "longest_km", "m_per_beat"]].round(2)
            weekly_display = weekly_display.rename(columns={
                "week": "Week", "distance_km": "km", "runs": "Runs", "avg_pace": "Avg pace",
                "longest_km": "Longest km", "m_per_beat": "m/beat"})
        else:
            weekly_display = pd.DataFrame()
        ui.chart(weekly_running_volume(weekly, colors), colors, table=weekly_display,
                 key="running_volume")
    with right:
        ui.section("Consistency", None, colors)
        stability = report.consistency
        if stability:
            st.dataframe(pd.DataFrame([
                {"Metric": "Mean weekly distance", "Value": f"{stability['mean_weekly_km']:.1f} km"},
                {"Metric": "Runs per week", "Value": f"{stability['runs_per_week']:.1f}"},
                {"Metric": "Weeks with no runs",
                 "Value": f"{stability['zero_weeks']} of {stability['weeks']}"},
                {"Metric": "Week-to-week variation", "Value": f"{stability['cv_pct']:.0f}%"},
                {"Metric": "Long run share of volume",
                 "Value": f"{stability['longest_run_share_pct']:.0f}%"},
            ]), use_container_width=True, hide_index=True)
        else:
            st.info("Not enough weeks of running to assess consistency.")

    st.divider()
    ui.section(
        "Intensity distribution",
        "Most self-coached runners spend too long at moderate intensity — hard enough to "
        "accumulate fatigue, not hard enough to drive adaptation. Roughly 75–80% of time "
        "should be genuinely easy.",
        colors,
    )
    ui.chart(intensity_split(report.intensity, colors), colors, key="running_intensity")
    if report.intensity:
        source = ("heart-rate zone time from the device" if report.intensity["source"] == "hr_zones"
                  else "per-run intensity labels (no zone data available)")
        st.caption(f"Based on {report.intensity['total_hours']:.0f} hours of running, using {source}.")

    st.divider()
    _zones(report, colors)

    st.divider()
    _splits(report, colors)

    st.divider()
    ui.section("Personal bests and race predictions", None, colors)
    tab_best, tab_predict = st.tabs(["Personal bests", "Predictions"])
    with tab_best:
        if report.bests.empty:
            st.info("No efforts long enough to rank into distance buckets yet.")
        else:
            ui.chart(pace_by_distance(report.bests, colors), colors,
                     table=_bests_table(report.bests), key="running_bests")
            st.caption(
                "Efforts inside each bucket are compared on Riegel-equivalent time for the "
                "exact distance, so a 10.8 km run doesn't beat a 10.0 km one by being longer."
            )
    with tab_predict:
        if report.predictions.empty:
            st.info("Need a recent effort of 5 km or more to predict from.")
        else:
            predictions = report.predictions.copy()
            predictions["Predicted time"] = predictions["predicted_time_s"].map(format_duration)
            predictions["Pace"] = predictions["predicted_pace_s_per_km"].map(format_pace)
            source = predictions.iloc[0]
            st.dataframe(
                predictions[["bucket", "Predicted time", "Pace"]].rename(
                    columns={"bucket": "Distance"}),
                use_container_width=True, hide_index=True)
            st.caption(
                f"Riegel projection from your strongest recent effort "
                f"({source['source_distance_km']:.1f} km on "
                f"{pd.Timestamp(source['source_date']).strftime('%d %b %Y')}). Long-distance "
                f"predictions assume endurance matches the reference effort — treat the "
                f"marathon figure as optimistic unless you've trained the volume."
            )

    st.divider()
    ui.section("Findings", None, colors)
    ui.findings_list(report.findings_for("running"), colors)

    with st.expander("Trend detail"):
        st.dataframe(_trend_table(report.run_trends), use_container_width=True, hide_index=True)


def _tiles(report) -> list[dict]:
    runs, weekly = report.runs, report.weekly_runs
    tiles: list[dict] = []

    if not weekly.empty:
        recent = weekly.tail(4)["distance_km"].mean()
        prior = weekly.tail(8).head(4)["distance_km"].mean()
        delta = (f"{(recent - prior) / prior * 100:+.0f}% vs prior 4 wk"
                 if pd.notna(prior) and prior > 0 else None)
        tiles.append({"label": "Volume", "value": f"{recent:.0f} km/wk", "delta": delta})

    easy = runs[runs["intensity"] == "easy"]
    if not easy.empty:
        pace = easy.tail(10)["pace_s_per_km"].mean()
        tiles.append({"label": "Easy pace", "value": format_pace(pace),
                      "note": f"last {min(len(easy), 10)} easy runs",
                      "help": "Mean pace of recent easy-intensity runs."})

    if runs["avg_cadence"].notna().any():
        cadence = runs.tail(20)["avg_cadence"].mean()
        tiles.append({"label": "Cadence", "value": f"{cadence:.0f} spm",
                      "note": "target 170–180"})

    if runs["vo2max"].notna().any():
        vo2 = runs["vo2max"].dropna()
        trend = report.run_trends.get("vo2max", {})
        tiles.append({"label": "VO2max (Garmin)", "value": f"{vo2.iloc[-1]:.0f}",
                      "delta": (f"{trend['per_month']:+.2f}/month"
                                if trend.get("reliable") else None),
                      "help": "Garmin's own estimate — a coarse corroborating signal."})

    if report.intensity:
        tiles.append({"label": "Easy time share", "value": f"{report.intensity['easy_pct']:.0f}%",
                      "note": f"{report.intensity['moderate_pct']:.0f}% moderate",
                      "help": "Target roughly 75–80% easy."})
    return tiles[:5]


def _runs_table(runs: pd.DataFrame) -> pd.DataFrame:
    table = runs.tail(30).iloc[::-1][["local_date", "name", "distance_km", "pace_label",
                                     "avg_hr", "m_per_beat", "avg_cadence", "intensity"]].copy()
    table["local_date"] = pd.to_datetime(table["local_date"]).dt.strftime("%d %b %Y")
    return table.round(2).rename(columns={
        "local_date": "Date", "name": "Run", "distance_km": "km", "pace_label": "Pace",
        "avg_hr": "Avg HR", "m_per_beat": "m/beat", "avg_cadence": "Cadence",
        "intensity": "Intensity"})


def _bests_table(bests: pd.DataFrame) -> pd.DataFrame:
    table = bests.copy()
    table["Best"] = table["best_time_s"].map(format_duration)
    table["Best pace"] = table["best_pace_s_per_km"].map(format_pace)
    table["Recent best"] = table["recent_best_time_s"].map(format_duration)
    table["Set"] = pd.to_datetime(table["best_date"]).dt.strftime("%d %b %Y")
    return table[["bucket", "Best", "Best pace", "Set", "Recent best", "pct_off_best",
                  "attempts"]].rename(columns={
        "bucket": "Distance", "pct_off_best": "% off best", "attempts": "Attempts"})


def _trend_table(trends: dict) -> pd.DataFrame:
    labels = {
        "aerobic_efficiency": "Aerobic efficiency (m/beat)",
        "easy_pace": "Easy-run pace (s/km)",
        "cadence": "Cadence (spm)",
        "weekly_distance": "Weekly distance (km)",
        "vo2max": "VO2max estimate",
    }
    rows = []
    for key, label in labels.items():
        trend = trends.get(key)
        if not trend:
            continue
        direction = "—"
        if trend.get("reliable") and trend.get("pct_per_month") is not None:
            improving = (trend["pct_per_month"] > 0) == trend["higher_is_better"]
            direction = "improving" if trend["pct_per_month"] and improving else (
                "flat" if abs(trend["pct_per_month"]) < 0.5 else "worsening")
        rows.append({
            "Metric": label,
            "Change per month": ("—" if not trend.get("reliable")
                                 else f"{trend['per_month']:+.2f} {trend['unit']}"),
            "% per month": ("—" if not trend.get("reliable")
                            else f"{trend['pct_per_month']:+.1f}%"),
            "Direction": direction,
            "Fit quality (r²)": "—" if not trend.get("reliable") else f"{trend['r_squared']:.2f}",
            "Data points": trend.get("n", 0),
        })
    return pd.DataFrame(rows)


def _zones(report, colors) -> None:
    model = report.zone_model
    ui.section(
        "Your training paces",
        "Built from Garmin's own lactate-threshold estimate rather than a guessed "
        "max heart rate — threshold is the most reproducible anchor there is.",
        colors,
    )

    if model is None or not model.has_pace_zones:
        st.info(
            "Garmin hasn't produced a lactate-threshold estimate for this account yet, so "
            "personal pace zones aren't available. It usually appears after a few hard runs "
            "with heart rate recorded. Everything else on this page works without it."
        )
        return

    tiles = [
        {"label": "Threshold pace", "value": format_pace(model.lt_pace_s),
         "note": "roughly one-hour race effort"},
        {"label": "Easy range", "value": model.get("easy").compact_label,
         "note": "the bulk of weekly volume"},
        {"label": "Tempo range", "value": model.get("threshold").compact_label,
         "note": "comfortably hard"},
    ]
    if model.lt_hr:
        tiles.append({"label": "Threshold heart rate", "value": f"{model.lt_hr:.0f} bpm"})
    ui.stat_tiles(tiles, palette=colors)

    discipline = report.easy_discipline
    if discipline:
        if discipline["too_fast_pct"] >= 40:
            st.error(
                f"**{discipline['too_fast_pct']:.0f}% of your easy and moderate runs are too "
                f"fast.** They average {discipline['mean_pace_label']} against an easy range of "
                f"{discipline['easy_range_label']} — about "
                f"{abs(discipline['seconds_too_fast']):.0f} s/km too quick. This is the "
                f"mechanism behind the grey-zone problem: slowing these down is the single "
                f"biggest change available to you."
            )
        else:
            st.success(
                f"Easy and moderate runs average {discipline['mean_pace_label']}, inside your "
                f"{discipline['easy_range_label']} easy range."
            )

    columns = st.columns([1.25, 1])
    with columns[0]:
        st.dataframe(model.summary_table(), use_container_width=True, hide_index=True)
    with columns[1]:
        ui.chart(zone_distribution_chart(report.zone_distribution, colors), colors,
                 table=report.zone_distribution, key="running_zone_dist")

    if model.has_hr_zones:
        with st.expander("Heart-rate zones from your device"):
            hr = model.hr_zones.copy()
            hr = hr.rename(columns={"zone": "Zone", "floor_bpm": "From (bpm)",
                                    "ceiling_bpm": "To (bpm)", "purpose": "What it's for",
                                    "pct_of_threshold": "% of threshold HR"})
            st.dataframe(hr.drop(columns=["sport"], errors="ignore"),
                         use_container_width=True, hide_index=True)


def _splits(report, colors) -> None:
    ui.section(
        "Split-level analysis",
        "What averages hide. Aerobic decoupling asks whether heart rate drifts upward at "
        "constant pace through a long run — a run averaging 150 bpm might hold 145 "
        "throughout or climb from 138 to 162, and only the second means the distance is "
        "past your current endurance.",
        colors,
    )

    if report.splits.empty:
        st.info(
            "No per-lap data synced yet. Enable **Fetch per-lap splits** in Sync & Settings — "
            "it costs one request per run and unlocks decoupling and interval analysis."
        )
        return

    tiles = []
    if not report.decoupling.empty:
        recent = float(report.decoupling.tail(6)["decoupling_pct"].mean())
        verdict = ("comfortable" if recent <= 5 else
                   "moderate drift" if recent <= 10 else "beyond endurance")
        tiles.append({"label": "Decoupling (recent long runs)", "value": f"{recent:.1f}%",
                      "note": f"{verdict} · under 5% is good"})
    if report.decoupling_trend:
        trend = report.decoupling_trend
        tiles.append({"label": "Durability trend", "value": f"{trend['per_month']:+.2f} pp/mo",
                      "note": f"across {trend['n']} long runs"})
    if not report.intervals.empty:
        fade = float(report.intervals.tail(5)["fade_pct"].mean())
        tiles.append({"label": "Interval fade", "value": f"{fade:+.1f}%",
                      "note": "first rep to last"})
    if report.negative_splits:
        neg = report.negative_splits
        tiles.append({"label": "Negative splits", "value": f"{neg['negative_pct']:.0f}%",
                      "note": f"of {neg['runs']} runs"})
    if tiles:
        ui.stat_tiles(tiles, palette=colors)

    decoupling_table = None
    if not report.decoupling.empty:
        decoupling_table = report.decoupling.tail(25).iloc[::-1][
            ["local_date", "distance_km", "laps", "decoupling_pct", "hr_drift_bpm", "verdict"]
        ].copy()
        decoupling_table["local_date"] = pd.to_datetime(
            decoupling_table["local_date"]).dt.strftime("%d %b %Y")
        decoupling_table = decoupling_table.rename(columns={
            "local_date": "Date", "distance_km": "km", "laps": "Laps",
            "decoupling_pct": "Decoupling %", "hr_drift_bpm": "HR drift (bpm)",
            "verdict": "Verdict"})
    ui.chart(decoupling_chart(report.decoupling, colors), colors, table=decoupling_table,
             table_label="View long runs analysed", key="running_decoupling")

    if not report.intervals.empty:
        st.markdown("**Interval sessions**")
        intervals = report.intervals.tail(12).iloc[::-1].copy()
        intervals["Date"] = pd.to_datetime(intervals["local_date"]).dt.strftime("%d %b %Y")
        intervals["Mean rep pace"] = intervals["mean_rep_pace_s"].map(format_pace)
        intervals["First → last"] = (intervals["first_rep_pace_s"].map(format_pace) + " → "
                                     + intervals["last_rep_pace_s"].map(format_pace))
        st.dataframe(
            intervals[["Date", "reps", "rep_distance_m", "Mean rep pace", "First → last",
                       "fade_pct", "pace_cv_pct", "verdict"]].rename(columns={
                "reps": "Reps", "rep_distance_m": "Rep distance (m)", "fade_pct": "Fade %",
                "pace_cv_pct": "Pace variation %", "verdict": "Verdict"}),
            use_container_width=True, hide_index=True)
        st.caption(
            "Fade is how much slower the final rep is than the first. Under 1.5% is well "
            "paced; above 4% means the early reps were faster than the session's real target."
        )
