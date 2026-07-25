"""Snapshots of the computed report, so the diagnostics themselves can be trended.

A dashboard that only shows the present tells you rear delts are behind, and says
the same thing next month whether or not you acted. What makes the tool worth
opening again is the *delta*: "rear delts 72 → 38 over six weeks, because volume
went from 2.4 to 9.1 sets — that worked, keep going."

So each sync stores a compact snapshot of the headline numbers (one row per day at
most), and this module diffs the current state against a chosen number of weeks
back. Nothing here recomputes anything — it compares what was already computed.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from .. import db
from .findings import Finding

# Units that attach directly to the number; everything else takes a space.
TIGHT_UNITS = {"%", "%/mo", ""}


def format_unit(value_label: str, unit: str) -> str:
    """Join a value and its unit with the spacing that unit conventionally takes."""
    return f"{value_label}{unit}" if unit in TIGHT_UNITS else f"{value_label} {unit}"


# Metrics tracked across snapshots: (key, label, higher_is_better, unit, format)
TRACKED_METRICS: list[tuple[str, str, bool, str, str]] = [
    ("weekly_km", "Weekly running volume", True, "km", "{:.1f}"),
    ("efficiency_pct_per_month", "Aerobic efficiency trend", True, "%/mo", "{:+.2f}"),
    ("easy_pct", "Easy-running share", True, "%", "{:.0f}"),
    ("decoupling_pct", "Aerobic decoupling", False, "%", "{:.1f}"),
    ("acwr", "Load balance (ACWR)", None, "", "{:.2f}"),
    ("ctl", "Fitness", True, "", "{:.0f}"),
    ("weekly_sets_total", "Weekly effective sets", True, "sets", "{:.0f}"),
    ("muscles_behind", "Muscles falling behind", False, "", "{:.0f}"),
    ("mean_attention", "Mean attention score", False, "", "{:.1f}"),
    ("readiness", "Garmin readiness", True, "", "{:.0f}"),
]


def snapshot_payload(report) -> dict[str, Any]:
    """The compact set of numbers worth keeping per day."""
    payload: dict[str, Any] = {"generated_at": datetime.now().isoformat(timespec="seconds")}

    if not report.lagging.empty:
        payload["muscle_scores"] = {
            row.muscle: round(float(row.attention_score), 1)
            for row in report.lagging.itertuples()
        }
        payload["muscle_weekly_sets"] = {
            row.muscle: round(float(row.weekly_sets), 2)
            for row in report.lagging.itertuples()
        }
        payload["weekly_sets_total"] = round(float(report.lagging["weekly_sets"].sum()), 1)
        payload["muscles_behind"] = int((report.lagging["attention_score"] >= 60).sum())
        payload["mean_attention"] = round(float(report.lagging["attention_score"].mean()), 1)

    if not report.progress.empty:
        payload["e1rm"] = {
            str(row.exercise): round(float(row.current_e1rm), 1)
            for row in report.progress.itertuples()
            if pd.notna(row.current_e1rm)
        }

    if not report.weekly_runs.empty:
        payload["weekly_km"] = round(float(report.weekly_runs.tail(4)["distance_km"].mean()), 1)

    efficiency = report.run_trends.get("aerobic_efficiency", {})
    if efficiency.get("reliable"):
        payload["efficiency_pct_per_month"] = efficiency["pct_per_month"]

    if report.intensity:
        payload["easy_pct"] = report.intensity["easy_pct"]

    if getattr(report, "decoupling", None) is not None and not report.decoupling.empty:
        payload["decoupling_pct"] = round(
            float(report.decoupling.tail(6)["decoupling_pct"].mean()), 1)

    if not report.load_series.empty:
        latest = report.load_series.dropna(subset=["acwr"]).tail(1)
        if not latest.empty:
            payload["acwr"] = round(float(latest.iloc[0]["acwr"]), 2)
        last = report.load_series.tail(1).iloc[0]
        if pd.notna(last["ctl"]):
            payload["ctl"] = round(float(last["ctl"]), 1)

    if getattr(report, "athlete_metrics", None) is not None \
            and not report.athlete_metrics.empty:
        readiness = report.athlete_metrics["readiness_score"].dropna()
        if not readiness.empty:
            payload["readiness"] = round(float(readiness.iloc[-1]), 0)

    return payload


def save_snapshot(conn: sqlite3.Connection, report, taken_on: date | None = None) -> None:
    """Store today's snapshot, overwriting any earlier one from the same day."""
    taken_on = taken_on or date.today()
    db.save_snapshot(conn, taken_on.isoformat(), snapshot_payload(report))


def load_history(conn: sqlite3.Connection, limit: int = 120) -> list[dict[str, Any]]:
    """Snapshots, oldest first."""
    return sorted(db.load_snapshots(conn, limit), key=lambda s: s["taken_on"])


def _nearest(history: list[dict[str, Any]], target: date) -> dict[str, Any] | None:
    """The snapshot closest to ``target`` — snapshots only exist on sync days."""
    if not history:
        return None
    best, best_gap = None, None
    for entry in history:
        try:
            taken = date.fromisoformat(entry["taken_on"])
        except (ValueError, KeyError):
            continue
        gap = abs((taken - target).days)
        if best_gap is None or gap < best_gap:
            best, best_gap = entry, gap
    # Beyond a fortnight either side it isn't the comparison the user asked for.
    return best if best_gap is not None and best_gap <= 14 else None


def compare(history: list[dict[str, Any]], weeks_back: int = 6) -> dict[str, Any]:
    """Diff the newest snapshot against one from roughly ``weeks_back`` ago."""
    if len(history) < 2:
        return {}
    current = history[-1]
    try:
        current_date = date.fromisoformat(current["taken_on"])
    except (ValueError, KeyError):
        return {}
    previous = _nearest(history[:-1], current_date - pd.Timedelta(weeks=weeks_back).to_pytimedelta())
    if not previous or previous["taken_on"] == current["taken_on"]:
        return {}

    metrics = []
    for key, label, higher_is_better, unit, fmt in TRACKED_METRICS:
        now, then = current.get(key), previous.get(key)
        if now is None or then is None:
            continue
        change = float(now) - float(then)
        if higher_is_better is None:
            direction = "changed"
        elif abs(change) < 1e-9:
            direction = "flat"
        else:
            improving = (change > 0) if higher_is_better else (change < 0)
            direction = "improved" if improving else "worsened"
        metrics.append({
            "key": key, "label": label, "unit": unit,
            "then": float(then), "now": float(now), "change": round(change, 2),
            "then_label": fmt.format(float(then)), "now_label": fmt.format(float(now)),
            "change_label": ("—" if higher_is_better is None
                             else f"{change:+.2f}".rstrip("0").rstrip(".")),
            "direction": direction,
        })

    muscles = []
    now_scores = current.get("muscle_scores") or {}
    then_scores = previous.get("muscle_scores") or {}
    now_sets = current.get("muscle_weekly_sets") or {}
    then_sets = previous.get("muscle_weekly_sets") or {}
    for muscle, now_score in now_scores.items():
        if muscle not in then_scores:
            continue
        muscles.append({
            "muscle": muscle,
            "then_score": float(then_scores[muscle]),
            "now_score": float(now_score),
            "score_change": round(float(now_score) - float(then_scores[muscle]), 1),
            "then_sets": float(then_sets.get(muscle, np.nan)),
            "now_sets": float(now_sets.get(muscle, np.nan)),
        })

    return {
        "from": previous["taken_on"], "to": current["taken_on"],
        "weeks_back": weeks_back,
        "days_between": (current_date - date.fromisoformat(previous["taken_on"])).days,
        "metrics": metrics,
        "muscles": sorted(muscles, key=lambda m: m["score_change"]),
        "snapshots": len(history),
    }


def metric_history(history: list[dict[str, Any]], key: str) -> pd.DataFrame:
    """One tracked metric as a time series, for charting."""
    rows = [{"taken_on": entry["taken_on"], "value": entry[key]}
            for entry in history if entry.get(key) is not None]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["taken_on"] = pd.to_datetime(frame["taken_on"])
    return frame


def muscle_score_history(history: list[dict[str, Any]]) -> pd.DataFrame:
    """Attention score per muscle over time."""
    rows = []
    for entry in history:
        for muscle, score in (entry.get("muscle_scores") or {}).items():
            rows.append({"taken_on": entry["taken_on"], "muscle": muscle, "score": score})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["taken_on"] = pd.to_datetime(frame["taken_on"])
    return frame


# --- findings ---------------------------------------------------------------

def progress_findings(deltas: dict[str, Any], muscle_label) -> list[Finding]:
    """Turn the comparison into "this worked" / "this didn't" statements."""
    if not deltas:
        return []
    findings: list[Finding] = []
    span = deltas["days_between"]

    improved = [m for m in deltas["muscles"] if m["score_change"] <= -8]
    worsened = [m for m in deltas["muscles"] if m["score_change"] >= 8]

    for entry in improved[:3]:
        sets_text = ""
        if not np.isnan(entry["then_sets"]) and not np.isnan(entry["now_sets"]):
            sets_text = (f" — weekly volume went from {entry['then_sets']:.1f} to "
                         f"{entry['now_sets']:.1f} effective sets")
        findings.append(Finding(
            area="progress", subject=entry["muscle"],
            title=f"{muscle_label(entry['muscle'])} is catching up",
            detail=(f"Attention score {entry['then_score']:.0f} → {entry['now_score']:.0f} over "
                    f"{span} days{sets_text}. Whatever you changed here is working."),
            severity="good", metric=f"{entry['score_change']:+.0f} points",
            evidence=entry))

    for entry in worsened[:3]:
        findings.append(Finding(
            area="progress", subject=entry["muscle"],
            title=f"{muscle_label(entry['muscle'])} has slipped further behind",
            detail=(f"Attention score {entry['then_score']:.0f} → {entry['now_score']:.0f} over "
                    f"{span} days. It was already flagged and has got worse, not better."),
            severity="act", metric=f"{entry['score_change']:+.0f} points",
            recommendation="This one needs a scheduled slot, not an opportunistic one.",
            evidence=entry))

    for metric in deltas["metrics"]:
        if metric["direction"] not in {"improved", "worsened"}:
            continue
        # Only report movement large enough to be a signal rather than noise.
        relative = abs(metric["change"]) / max(abs(metric["then"]), 1e-6)
        if relative < 0.12:
            continue
        findings.append(Finding(
            area="progress", subject=metric["key"],
            title=f"{metric['label']} has {metric['direction']}",
            detail=(f"{format_unit(metric['then_label'], metric['unit'])} → "
                    f"{format_unit(metric['now_label'], metric['unit'])} over "
                    f"{span} days."),
            severity="good" if metric["direction"] == "improved" else "watch",
            metric=format_unit(metric["now_label"], metric["unit"]), evidence=metric))

    return findings
