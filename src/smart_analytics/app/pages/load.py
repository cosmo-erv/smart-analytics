"""Load & recovery: every activity type on one scale, plus wellness context."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...analytics.running import format_pace
from ...viz import (
    acwr_chart,
    activity_mix,
    discipline_split_chart,
    fitness_fatigue,
    readiness_chart,
    recovery_trend,
)
from .. import components as ui
from ..state import current_report, palette


def render() -> None:
    report = current_report()
    colors = palette()

    st.title("Load & recovery")

    if not report.has_data:
        ui.no_data_notice("activities")
        return

    latest = (report.load_series.dropna(subset=["acwr"]).tail(1)
              if not report.load_series.empty else pd.DataFrame())
    if not latest.empty:
        row = latest.iloc[0]
        ui.hero(f"{row['acwr']:.2f}", f"Load balance — {row['acwr_status']}", colors,
                sub=(f"7-day load {row['acute_7d']:.0f} against a 28-day weekly average of "
                     f"{row['chronic_weekly']:.0f}. Productive sits between 0.8 and 1.3."))
    else:
        ui.hero("—", "Load balance", colors,
                sub="Needs 28 consecutive days of activity history.")

    ui.stat_tiles(_tiles(report), palette=colors)

    estimated = (report.daily_load["estimated_share"].mean()
                 if not report.daily_load.empty else 0)
    if estimated > 0.2:
        st.caption(
            f"{estimated * 100:.0f}% of the load in this window is estimated from heart rate "
            f"(Banister TRIMP) rather than reported by the device — common for strength "
            f"sessions and older watches. Inferred max HR "
            f"{report.meta.get('max_hr', 0):.0f}, resting {report.meta.get('resting_hr', 0):.0f}."
        )

    st.divider()
    _garmin_view(report, colors)

    st.divider()
    _hybrid(report, colors)

    st.divider()
    ui.section(
        "Acute versus chronic load",
        "The 7-day load compared with the 28-day weekly average. Ramping faster than your "
        "body has adapted to is the classic precursor to overuse injury; sitting below 0.8 "
        "for long means you're detraining.",
        colors,
    )
    load_table = None
    if not report.load_series.empty:
        load_table = (report.load_series.dropna(subset=["acwr"]).tail(30).iloc[::-1]
                      [["date", "load", "acute_7d", "chronic_weekly", "acwr", "acwr_status",
                        "monotony"]].round(2))
        load_table["date"] = pd.to_datetime(load_table["date"]).dt.strftime("%d %b %Y")
        load_table = load_table.rename(columns={
            "date": "Date", "load": "Daily load", "acute_7d": "7-day",
            "chronic_weekly": "28-day weekly", "acwr": "ACWR", "acwr_status": "Status",
            "monotony": "Monotony"})
    ui.chart(acwr_chart(report.load_series, colors), colors, table=load_table, key="load_acwr")

    st.divider()
    left, right = st.columns(2)
    with left:
        ui.section("Fitness and fatigue", "Exponentially weighted over 42 and 7 days.", colors)
        ui.chart(fitness_fatigue(report.load_series, colors), colors, key="load_ff")
    with right:
        ui.section("Training mix", f"Last {report.meta['lookback_days']} days.", colors)
        ui.chart(activity_mix(report.activity_mix, colors), colors,
                 table=report.activity_mix, key="load_mix")

    st.divider()
    ui.section(
        "Recovery markers",
        "Sustained movement against baseline matters more than any single day.",
        colors,
    )
    if report.daily_metrics.empty:
        st.info(
            "No wellness data synced. Enable wellness days in **Sync & Settings** to pull "
            "resting heart rate, HRV, sleep and body battery."
        )
    else:
        if report.recovery:
            st.dataframe(_recovery_table(report.recovery), use_container_width=True,
                         hide_index=True)
        col_a, col_b = st.columns(2)
        with col_a:
            ui.chart(recovery_trend(report.daily_metrics, colors, "resting_hr",
                                    "Resting heart rate (bpm)"), colors, key="rec_rhr")
            ui.chart(recovery_trend(report.daily_metrics, colors, "sleep_hours",
                                    "Sleep (hours)"), colors, key="rec_sleep")
        with col_b:
            ui.chart(recovery_trend(report.daily_metrics, colors, "hrv_ms",
                                    "HRV (ms)"), colors, key="rec_hrv")
            ui.chart(recovery_trend(report.daily_metrics, colors, "body_battery_high",
                                   "Body battery peak"), colors, key="rec_bb")

    st.divider()
    ui.section("Findings", None, colors)
    ui.findings_list(report.findings_for("load", "recovery"), colors)


def _tiles(report) -> list[dict]:
    tiles: list[dict] = []
    if not report.load_series.empty:
        last = report.load_series.tail(1).iloc[0]
        if pd.notna(last["ctl"]):
            tiles.append({"label": "Fitness (42-day)", "value": f"{last['ctl']:.0f}"})
        if pd.notna(last["atl"]):
            tiles.append({"label": "Fatigue (7-day)", "value": f"{last['atl']:.0f}"})
        if pd.notna(last["tsb"]):
            state = ("fresh" if last["tsb"] > 5 else
                     "loaded" if last["tsb"] > -12 else "deeply fatigued")
            tiles.append({"label": "Form balance", "value": f"{last['tsb']:+.0f}",
                          "note": state})
        if pd.notna(last["monotony"]):
            tiles.append({"label": "Monotony", "value": f"{last['monotony']:.1f}",
                          "note": "keep under 2.0",
                          "help": "Weekly mean load over its standard deviation."})

    if report.recovery.get("resting_hr_recent") is not None:
        delta = report.recovery.get("resting_hr_delta")
        tiles.append({
            "label": "Resting HR", "value": f"{report.recovery['resting_hr_recent']:.0f} bpm",
            "delta": f"{delta:+.0f} vs baseline" if delta is not None else None,
            "delta_color": "inverse",
        })
    return tiles[:5]


def _recovery_table(recovery: dict) -> pd.DataFrame:
    labels = [
        ("resting_hr", "Resting heart rate", "bpm", True),
        ("hrv", "HRV", "ms", False),
        ("sleep_hours", "Sleep", "hours", False),
        ("sleep_score", "Sleep score", "", False),
        ("body_battery", "Body battery peak", "", False),
    ]
    rows = []
    for key, label, unit, lower_is_better in labels:
        recent = recovery.get(f"{key}_recent")
        if recent is None:
            continue
        baseline = recovery.get(f"{key}_baseline")
        delta = recovery.get(f"{key}_delta")
        direction = "—"
        if delta is not None:
            if abs(delta) < 0.5:
                direction = "steady"
            else:
                improving = (delta < 0) if lower_is_better else (delta > 0)
                direction = "improving" if improving else "worsening"
        rows.append({
            "Marker": label,
            "Last 28 days": f"{recent:.1f} {unit}".strip(),
            "Baseline": f"{baseline:.1f} {unit}".strip() if baseline is not None else "—",
            "Change": f"{delta:+.1f}" if delta is not None else "—",
            "Direction": direction,
        })
    return pd.DataFrame(rows)


def _garmin_view(report, colors) -> None:
    ui.section(
        "Garmin's own assessment",
        "Straight from the watch's longitudinal model, not recomputed here. Where this "
        "and the local analysis disagree, both numbers are shown — they use different "
        "windows and weightings, and the disagreement is itself informative.",
        colors,
    )

    metrics = report.athlete_metrics
    if metrics is None or metrics.empty:
        st.info(
            "No Garmin training metrics synced. Set **Days of Garmin training metrics** above "
            "zero in Sync & Settings to pull training status, readiness, lactate threshold, "
            "VO2max, FTP and race predictions."
        )
        return

    latest = metrics.tail(1).iloc[0]
    tiles = []
    if pd.notna(latest.get("training_status")):
        tiles.append({"label": "Training status", "value": str(latest["training_status"]),
                      "note": "Garmin's verdict"})
    if pd.notna(latest.get("readiness_score")):
        note = str(latest["readiness_level"]) if pd.notna(latest.get("readiness_level")) else None
        tiles.append({"label": "Readiness today", "value": f"{latest['readiness_score']:.0f}/100",
                      "note": note})
    if pd.notna(latest.get("recovery_time_h")):
        tiles.append({"label": "Recovery remaining", "value": f"{latest['recovery_time_h']:.0f} h",
                      "note": "before you're fully recovered"})
    if pd.notna(latest.get("load_ratio")):
        tiles.append({"label": "Garmin load ratio", "value": f"{latest['load_ratio']:.2f}",
                      "note": "its own acute:chronic"})
    if pd.notna(latest.get("vo2max_running")):
        tiles.append({"label": "VO2max", "value": f"{latest['vo2max_running']:.1f}",
                      "note": "running estimate"})
    if tiles:
        ui.stat_tiles(tiles[:5], palette=colors)

    columns = st.columns([1.3, 1])
    with columns[0]:
        ui.chart(readiness_chart(metrics, colors), colors, key="load_readiness")
    with columns[1]:
        st.markdown("**Physiological markers**")
        rows = []
        for column, label, spec in [
            ("lt_hr", "Lactate threshold HR", "{:.0f} bpm"),
            ("lt_speed_mps", "Lactate threshold pace", None),
            ("ftp_watts", "Cycling FTP", "{:.0f} W"),
            ("endurance_score", "Endurance score", "{:.0f}"),
            ("hill_score", "Hill score", "{:.0f}"),
            ("fitness_age", "Fitness age", "{:.0f}"),
            ("running_tolerance_km", "Running tolerance", "{:.0f} km/week"),
            ("acute_load", "Garmin 7-day load", "{:.0f}"),
            ("chronic_load", "Garmin chronic load", "{:.0f}"),
        ]:
            value = latest.get(column)
            if value is None or pd.isna(value):
                continue
            if column == "lt_speed_mps":
                rows.append({"Marker": label, "Value": format_pace(1000.0 / float(value))})
            else:
                rows.append({"Marker": label, "Value": spec.format(float(value))})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No threshold metrics available for this account yet.")

    if pd.notna(latest.get("load_target_low")) and pd.notna(latest.get("load_target_high")):
        st.caption(
            f"Garmin's optimal weekly load range for you is "
            f"{latest['load_target_low']:.0f}–{latest['load_target_high']:.0f}, and it currently "
            f"reads your 7-day load as {latest.get('acute_load', float('nan')):.0f}."
        )


def _hybrid(report, colors) -> None:
    ui.section(
        "Strength and running together",
        "Concurrent training competes: heavy legs degrade running quality for 24–48 hours, "
        "and running before lifting costs more strength adaptation than the reverse. These "
        "are scheduling problems, not volume problems.",
        colors,
    )

    if report.discipline_split.empty:
        st.info("Needs both strength and running data in the same period.")
        return

    recent = report.discipline_split.tail(4)
    ui.stat_tiles([
        {"label": "Load split", "value": f"{recent['strength_pct'].mean():.0f}% strength",
         "note": f"{recent['running_pct'].mean():.0f}% running, last 4 weeks"},
        {"label": "Strength time", "value": f"{recent['strength_hours'].mean():.1f} h/wk"},
        {"label": "Running time", "value": f"{recent['running_hours'].mean():.1f} h/wk"},
        {"label": "Session collisions", "value": str(len(report.hybrid_events)),
         "note": "quality run within 30 h of heavy legs"},
    ], palette=colors)

    ui.chart(discipline_split_chart(report.discipline_split, colors), colors,
             table=report.discipline_split.round(1), key="load_discipline")

    columns = st.columns([1.2, 1])
    with columns[0]:
        st.markdown("**Recent collisions**")
        if report.hybrid_events.empty:
            st.success(
                "No leg session landed inside the interference window before a quality run — "
                "the two disciplines are getting fresh legs each.")
        else:
            st.dataframe(
                report.hybrid_events[["date", "kind", "gap_hours", "first", "second"]].rename(
                    columns={"date": "Date", "kind": "Type", "gap_hours": "Gap (h)",
                             "first": "First session", "second": "Second session"}),
                use_container_width=True, hide_index=True)
    with columns[1]:
        st.markdown("**Week structure**")
        structure = report.week_structure
        if structure:
            st.dataframe(pd.DataFrame([
                {"Metric": "Hard sessions per week",
                 "Value": f"{structure['hard_sessions_per_week']:.1f}"},
                {"Metric": "Separate hard days", "Value": str(structure["hard_days"])},
                {"Metric": "Days with two hard sessions",
                 "Value": str(structure["shared_hard_days"])},
                {"Metric": "Back-to-back hard days",
                 "Value": str(structure["back_to_back_hard_days"])},
                {"Metric": "Mean gap between hard days",
                 "Value": f"{structure['mean_gap_days']:.1f} days"},
                {"Metric": "Full rest days per week",
                 "Value": f"{structure['rest_days_per_week']:.1f}"},
            ]), use_container_width=True, hide_index=True)
        else:
            st.caption("Not enough sessions to assess week structure.")

    ui.findings_list(report.findings_for("hybrid"), colors,
                     empty_message="No concurrent-training issues found.")
