"""Builds the single report object the GUI renders and the AI layer reasons over.

One entry point — :func:`build_report` — so the GUI, the CLI and the coaching
prompt all see exactly the same numbers. :meth:`TrainingReport.briefing` is the
compact, token-bounded JSON view handed to Claude; if a number isn't in the
briefing, the model has no business mentioning it.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .. import db
from ..config import Settings, settings as default_settings
from . import hybrid as hybrid_mod
from . import load as load_mod
from . import niggles as niggles_mod
from . import prescription as prescription_mod
from . import running as run_mod
from . import snapshots as snapshots_mod
from . import splits as splits_mod
from . import strength as strength_mod
from . import zones as zones_mod
from ..domain import exercises
from ..domain.muscles import label as muscle_label
from .findings import Finding, sort_findings, to_records

log = logging.getLogger(__name__)


@dataclass
class TrainingReport:
    generated_at: datetime
    settings: Settings

    activities: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Running
    runs: pd.DataFrame = field(default_factory=pd.DataFrame)
    weekly_runs: pd.DataFrame = field(default_factory=pd.DataFrame)
    run_trends: dict[str, Any] = field(default_factory=dict)
    intensity: dict[str, Any] = field(default_factory=dict)
    bests: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    consistency: dict[str, Any] = field(default_factory=dict)

    # Strength
    sets: pd.DataFrame = field(default_factory=pd.DataFrame)
    expanded: pd.DataFrame = field(default_factory=pd.DataFrame)
    muscle_volume: pd.DataFrame = field(default_factory=pd.DataFrame)
    weekly_muscle: pd.DataFrame = field(default_factory=pd.DataFrame)
    muscle_trends: pd.DataFrame = field(default_factory=pd.DataFrame)
    lagging: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance: pd.DataFrame = field(default_factory=pd.DataFrame)
    progress: pd.DataFrame = field(default_factory=pd.DataFrame)
    patterns: pd.DataFrame = field(default_factory=pd.DataFrame)
    unmapped: pd.DataFrame = field(default_factory=pd.DataFrame)
    muscle_sources: dict[str, Any] = field(default_factory=dict)

    # Cross-discipline
    daily_load: pd.DataFrame = field(default_factory=pd.DataFrame)
    load_series: pd.DataFrame = field(default_factory=pd.DataFrame)
    activity_mix: pd.DataFrame = field(default_factory=pd.DataFrame)
    recovery: dict[str, Any] = field(default_factory=dict)

    # Garmin's own physiological model
    athlete_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    garmin_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    personal_records: pd.DataFrame = field(default_factory=pd.DataFrame)
    zone_model: Any = None
    zone_distribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    easy_discipline: dict[str, Any] = field(default_factory=dict)

    # Split-level running detail
    splits: pd.DataFrame = field(default_factory=pd.DataFrame)
    decoupling: pd.DataFrame = field(default_factory=pd.DataFrame)
    intervals: pd.DataFrame = field(default_factory=pd.DataFrame)
    negative_splits: dict[str, Any] = field(default_factory=dict)
    decoupling_trend: dict[str, Any] = field(default_factory=dict)

    # Concurrent-training (hybrid) analysis
    hybrid_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    discipline_split: pd.DataFrame = field(default_factory=pd.DataFrame)
    week_structure: dict[str, Any] = field(default_factory=dict)

    # Niggles and progress over time
    niggle_context: pd.DataFrame = field(default_factory=pd.DataFrame)
    niggle_recurrence: pd.DataFrame = field(default_factory=pd.DataFrame)
    snapshot_history: list[dict[str, Any]] = field(default_factory=list)
    progress_deltas: dict[str, Any] = field(default_factory=dict)

    # Prescription
    recommendation: Any = None
    weekly_targets: dict[str, Any] = field(default_factory=dict)

    findings: list[Finding] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # --- convenience -------------------------------------------------------

    @property
    def has_data(self) -> bool:
        return not self.activities.empty

    @property
    def has_strength(self) -> bool:
        return not self.expanded.empty

    @property
    def has_running(self) -> bool:
        return not self.runs.empty

    def findings_for(self, *areas: str) -> list[Finding]:
        if not areas:
            return sort_findings(self.findings)
        return sort_findings([f for f in self.findings if f.area in areas])

    def top_findings(self, limit: int = 8) -> list[Finding]:
        return sort_findings(self.findings)[:limit]

    # --- the LLM's view ----------------------------------------------------

    def briefing(self, max_findings: int = 26) -> dict[str, Any]:
        """Compact, self-describing JSON for the coaching prompt."""
        window = self.meta.get("lookback_days")
        brief: dict[str, Any] = {
            "generated_at": self.generated_at.isoformat(timespec="seconds"),
            "data_window": {
                "first_activity": self.meta.get("first_date"),
                "last_activity": self.meta.get("last_date"),
                "total_activities": int(len(self.activities)),
                "analysis_lookback_days": window,
            },
            "units": {
                "effective_sets": "sets/week, fractional credit for secondary muscles",
                "pace": "seconds per km",
                "m_per_beat": "metres covered per heartbeat (higher is better)",
                "e1rm": "estimated 1-rep max in kg (Epley)",
                "acwr": "7-day load / 28-day weekly average load",
            },
            "targets": {
                "weekly_sets_per_muscle": [self.settings.weekly_sets_min,
                                           self.settings.weekly_sets_max],
            },
            "findings": to_records(sort_findings(self.findings)[:max_findings]),
        }

        if not self.activity_mix.empty:
            brief["activity_mix"] = self.activity_mix.to_dict("records")

        if self.has_strength:
            brief["strength"] = {
                "muscles": _records(
                    self.lagging[["muscle_label", "weekly_sets", "pct_of_target",
                                  "pct_per_month", "days_since", "attention_score", "verdict"]],
                    limit=20),
                "balance_ratios": _records(
                    self.balance[["pair", "ratio", "low", "high", "status"]], limit=8),
                "exercise_progress": _records(
                    self.progress[self.progress["reliable"]][
                        ["exercise", "sessions", "current_e1rm", "best_e1rm",
                         "kg_per_month", "pct_per_month", "status"]], limit=14),
                "movement_patterns": _records(self.patterns, limit=10),
                "muscle_attribution": (
                    "Garmin's own muscle assignments from structured workouts, "
                    "falling back to a curated table where Garmin has none"
                    if self.muscle_sources.get("garmin_entries")
                    else "curated exercise → muscle table (no Garmin workout data synced)"),
            }

        if self.has_running:
            brief["running"] = {
                "trends": self.run_trends,
                "intensity_distribution": self.intensity,
                "consistency": self.consistency,
                "personal_bests": _records(
                    self.bests[["bucket", "best_time_s", "best_date", "recent_best_time_s",
                                "pct_off_best", "days_since_best", "attempts"]], limit=6),
                "race_predictions": _records(
                    self.predictions[["bucket", "predicted_time_s",
                                      "predicted_pace_s_per_km"]], limit=6),
                "recent_weeks": _records(
                    self.weekly_runs.tail(8)[["week", "distance_km", "runs",
                                              "avg_pace_s_per_km", "m_per_beat"]], limit=8),
            }

        if not self.load_series.empty:
            latest = self.load_series.dropna(subset=["acwr"]).tail(1)
            if not latest.empty:
                row = latest.iloc[0]
                brief["load"] = {
                    "date": str(pd.to_datetime(row["date"]).date()),
                    "acwr": _num(row["acwr"]),
                    "acwr_status": row["acwr_status"],
                    "acute_7d": _num(row["acute_7d"]),
                    "chronic_weekly": _num(row["chronic_weekly"]),
                    "ctl_fitness": _num(row["ctl"]),
                    "atl_fatigue": _num(row["atl"]),
                    "tsb_form": _num(row["tsb"]),
                    "monotony": _num(row["monotony"]),
                }

        if self.recovery:
            brief["recovery"] = self.recovery

        # --- Garmin's own model, which outranks anything we could infer ------
        garmin: dict[str, Any] = {}
        if not self.athlete_metrics.empty:
            latest = self.athlete_metrics.tail(1).iloc[0]
            for key, column in [
                ("vo2max_running", "vo2max_running"), ("lt_hr", "lt_hr"),
                ("ftp_watts", "ftp_watts"), ("endurance_score", "endurance_score"),
                ("hill_score", "hill_score"), ("training_status", "training_status"),
                ("readiness_score", "readiness_score"), ("readiness_level", "readiness_level"),
                ("recovery_time_h", "recovery_time_h"), ("load_ratio", "load_ratio"),
                ("acute_load", "acute_load"), ("chronic_load", "chronic_load"),
            ]:
                value = latest.get(column)
                if value is not None and not (isinstance(value, float) and pd.isna(value)):
                    garmin[key] = value if isinstance(value, str) else _num(value)
        if self.zone_model is not None and self.zone_model.has_pace_zones:
            garmin["threshold_pace_s_per_km"] = round(self.zone_model.lt_pace_s, 1)
            garmin["pace_zones"] = {
                zone.key: {"fast_s_per_km": round(zone.fast_pace_s, 0),
                           "slow_s_per_km": round(zone.slow_pace_s, 0)}
                for zone in self.zone_model.pace_zones
            }
        if not self.garmin_predictions.empty:
            latest_day = self.garmin_predictions["local_date"].max()
            current = self.garmin_predictions[
                self.garmin_predictions["local_date"] == latest_day]
            garmin["race_predictions"] = {
                f"{int(row.distance_m / 1000)}km": round(float(row.predicted_time_s), 0)
                for row in current.itertuples()
            }
        if not self.personal_records.empty:
            garmin["personal_records"] = _records(
                self.personal_records[["label", "value", "unit", "achieved_on"]], limit=8)
        if garmin:
            brief["garmin_metrics"] = garmin

        if self.easy_discipline:
            brief["easy_pace_discipline"] = self.easy_discipline

        # --- split-level detail ---------------------------------------------
        splits_brief: dict[str, Any] = {}
        if not self.decoupling.empty:
            recent = self.decoupling.tail(6)
            splits_brief["recent_decoupling_pct"] = round(
                float(recent["decoupling_pct"].mean()), 1)
            splits_brief["decoupling_note"] = ("percent efficiency loss first half to second "
                                               "half of long runs; under 5 is good")
            splits_brief["runs_analysed"] = int(len(self.decoupling))
        if self.decoupling_trend:
            splits_brief["decoupling_trend"] = self.decoupling_trend
        if not self.intervals.empty:
            recent = self.intervals.tail(5)
            splits_brief["interval_sessions"] = _records(
                recent[["local_date", "reps", "mean_rep_pace_s", "fade_pct", "verdict"]], limit=5)
        if self.negative_splits:
            splits_brief["negative_split_rate"] = self.negative_splits
        if splits_brief:
            brief["split_analysis"] = splits_brief

        # --- concurrent training --------------------------------------------
        hybrid_brief: dict[str, Any] = {}
        if not self.hybrid_events.empty:
            hybrid_brief["collisions"] = _records(
                self.hybrid_events[["date", "kind", "gap_hours"]], limit=8)
            hybrid_brief["collision_count"] = int(len(self.hybrid_events))
        if not self.discipline_split.empty:
            recent = self.discipline_split.tail(4)
            hybrid_brief["load_split_recent_4wk"] = {
                "strength_pct": round(float(recent["strength_pct"].mean()), 0),
                "running_pct": round(float(recent["running_pct"].mean()), 0),
                "strength_hours_per_week": round(float(recent["strength_hours"].mean()), 1),
                "running_hours_per_week": round(float(recent["running_hours"].mean()), 1),
            }
        if self.week_structure:
            hybrid_brief["week_structure"] = self.week_structure
        if hybrid_brief:
            hybrid_brief["athlete_goal"] = "concurrent strength and running, roughly equal"
            brief["concurrent_training"] = hybrid_brief

        # --- niggles: context Garmin cannot supply --------------------------
        if not self.niggle_context.empty:
            brief["niggles"] = {
                "open": _records(
                    self.niggle_context[self.niggle_context["status"] == "open"][
                        ["noted_on", "area", "severity", "severity_label", "note", "days_open",
                         "acwr_at_onset", "load_vs_chronic_pct"]], limit=6),
                "recurring": _records(
                    self.niggle_recurrence[["area", "entries", "worst_severity", "last_noted"]],
                    limit=5),
                "note": ("self-reported; severity 1 = aware of it, 5 = stopping training. "
                         "Treat as the athlete's ground truth about how the body feels."),
            }

        # --- movement since last time ---------------------------------------
        if self.progress_deltas:
            brief["progress_since_last_check"] = {
                "from": self.progress_deltas["from"],
                "to": self.progress_deltas["to"],
                "days_between": self.progress_deltas["days_between"],
                "metrics": self.progress_deltas["metrics"],
                "muscles_most_improved": self.progress_deltas["muscles"][:4],
                "muscles_most_worsened": self.progress_deltas["muscles"][-4:],
            }

        if self.recommendation is not None:
            brief["local_recommendation"] = self.recommendation.to_dict()
        if self.weekly_targets:
            brief["weekly_targets"] = self.weekly_targets

        # One sweep at the end, so every section is JSON-valid by construction.
        return json_safe(brief)


def _num(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 2)


def json_safe(value: Any) -> Any:
    """Recursively replace values that ``json.dumps`` would render invalidly.

    ``json.dumps`` emits bare ``NaN``/``Infinity`` for float sentinels, which is
    not valid JSON — the API rejects it. Pandas also can't hold ``None`` in a
    float column, so NaN survives ``DataFrame.where``; it has to be stripped from
    the assembled structure instead.
    """
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return None if (value != value or value in (float("inf"), float("-inf"))) else value
    if isinstance(value, (np.integer, np.floating)):
        return json_safe(float(value))
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat(sep=" ")[:19]
    try:
        if value is not None and not isinstance(value, (str, bool, int)) and pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _records(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    """DataFrame → JSON-safe records, rounded and truncated for the prompt."""
    if frame is None or frame.empty:
        return []
    subset = frame.head(limit).copy()
    for column in subset.columns:
        if pd.api.types.is_datetime64_any_dtype(subset[column]):
            subset[column] = subset[column].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(subset[column]):
            subset[column] = subset[column].round(2)
    return subset.where(pd.notna(subset), None).to_dict("records")


def _install_garmin_muscles(conn: sqlite3.Connection) -> dict[str, Any]:
    """Load synced Garmin muscle assignments and make them the preferred source.

    Returns a diagnostics dict for the data page: how many entries Garmin
    supplied, and any anatomical names it used that our taxonomy can't express.
    """
    try:
        records = db.load_exercise_muscles(conn)
    except sqlite3.Error as exc:  # pre-migration database
        log.debug("no Garmin muscle table yet: %s", exc)
        records = []

    muscle_map = exercises.GarminMuscleMap(records) if records else None
    exercises.set_garmin_muscle_map(muscle_map)

    diagnostics: dict[str, Any] = {
        "garmin_records": len(records),
        "unmatched_names": {},
        **exercises.coverage_report(muscle_map),
    }
    if muscle_map is not None:
        diagnostics["unmatched_names"] = dict(
            sorted(muscle_map.unmatched_names.items(), key=lambda kv: -kv[1])[:20])
    return diagnostics


def build_report(conn: sqlite3.Connection, config: Settings | None = None,
                 lookback_days: int = strength_mod.DEFAULT_LOOKBACK_DAYS,
                 progress_lookback_days: int = 180) -> TrainingReport:
    """Run every analytics engine over the cached data and collect the results."""
    config = config or default_settings
    activities = db.load_activities(conn)
    sets = db.load_strength_sets(conn)
    daily_metrics = db.load_daily_metrics(conn)

    report = TrainingReport(generated_at=datetime.now(), settings=config,
                            activities=activities, daily_metrics=daily_metrics, sets=sets)

    counts = db.counts(conn)
    report.meta = {
        "lookback_days": lookback_days,
        "progress_lookback_days": progress_lookback_days,
        "first_date": counts.get("first_date"),
        "last_date": counts.get("last_date"),
        "activity_count": counts.get("activities"),
        "strength_activity_count": counts.get("strength_activities"),
        "set_count": counts.get("sets"),
    }

    findings: list[Finding] = []

    # --- strength ----------------------------------------------------------
    # Garmin's own muscle assignments, synced from structured workouts, are
    # installed before any set is expanded so they take precedence over the
    # curated fallback table for every exercise Garmin has an opinion about.
    report.muscle_sources = _install_garmin_muscles(conn)
    report.expanded = strength_mod.expand_sets(sets)
    report.unmapped = strength_mod.unmapped_exercises(sets)
    if not report.expanded.empty:
        report.weekly_muscle = strength_mod.weekly_muscle_volume(report.expanded)
        report.muscle_volume = strength_mod.muscle_volume_summary(report.expanded, lookback_days)
        report.progress = strength_mod.exercise_progress(report.expanded, progress_lookback_days)
        report.muscle_trends = strength_mod.muscle_strength_trend(
            report.expanded, report.progress, progress_lookback_days)
        report.balance = strength_mod.balance_ratios(report.muscle_volume)
        report.patterns = strength_mod.pattern_coverage(report.expanded, lookback_days)
        report.lagging = strength_mod.lagging_muscles(
            report.muscle_volume, report.muscle_trends,
            config.weekly_sets_min, config.weekly_sets_max)
        findings += strength_mod.strength_findings(
            report.lagging, report.balance, report.progress, report.patterns,
            report.unmapped, config.weekly_sets_max)

    # --- running -----------------------------------------------------------
    report.runs = run_mod.prepare_runs(activities)
    if not report.runs.empty:
        report.weekly_runs = run_mod.weekly_running(report.runs)
        report.run_trends = run_mod.performance_trends(report.runs, progress_lookback_days)
        report.intensity = run_mod.intensity_distribution(report.runs, lookback_days)
        report.bests = run_mod.personal_bests(report.runs)
        report.predictions = run_mod.race_predictions(report.runs)
        report.consistency = run_mod.consistency(report.runs, lookback_days)
        findings += run_mod.running_findings(
            report.runs, report.run_trends, report.intensity, report.bests, report.consistency)

    # --- load & recovery ---------------------------------------------------
    if not activities.empty:
        max_hr, resting_hr = load_mod.infer_hr_bounds(
            activities, daily_metrics, config.max_hr, config.resting_hr)
        report.meta["max_hr"] = max_hr
        report.meta["resting_hr"] = resting_hr
        report.daily_load = load_mod.daily_load(activities, max_hr, resting_hr)
        report.load_series = load_mod.load_series(report.daily_load)
        report.activity_mix = load_mod.activity_mix(activities, lookback_days)
        report.recovery = load_mod.recovery_trends(daily_metrics)
        findings += load_mod.load_findings(report.load_series, report.activity_mix, report.recovery)

    # --- Garmin's own physiology, and the zones built from it ---------------
    report.athlete_metrics = db.load_athlete_metrics(conn)
    report.garmin_predictions = db.load_race_predictions(conn)
    report.personal_records = db.load_personal_records(conn)
    report.zone_model = zones_mod.build_zone_model(
        report.athlete_metrics, db.load_hr_zones(conn, "running"))
    if report.has_running:
        report.zone_distribution = zones_mod.zone_distribution(
            report.runs, report.zone_model, lookback_days)
        report.easy_discipline = zones_mod.easy_run_discipline(
            report.runs, report.zone_model, lookback_days)
    findings += zones_mod.zone_findings(report.zone_model, report.easy_discipline)

    # --- split-level running detail ----------------------------------------
    report.splits = db.load_splits(conn, "running")
    if not report.splits.empty:
        report.decoupling = splits_mod.decoupling(report.splits)
        report.intervals = splits_mod.interval_sessions(report.splits)
        report.negative_splits = splits_mod.negative_split_rate(report.splits)
        report.decoupling_trend = splits_mod.decoupling_trend(report.decoupling)
    findings += splits_mod.split_findings(
        report.decoupling, report.intervals, report.negative_splits, report.decoupling_trend)

    # --- concurrent training -----------------------------------------------
    if report.has_strength and report.has_running:
        report.hybrid_events = hybrid_mod.interference_events(
            activities, report.expanded, report.runs, lookback_days)
        report.discipline_split = hybrid_mod.discipline_split(activities, lookback_days)
        report.week_structure = hybrid_mod.hard_day_structure(
            activities, report.expanded, report.runs, lookback_days)
        findings += hybrid_mod.hybrid_findings(
            report.hybrid_events, report.discipline_split, report.week_structure, lookback_days)

    # --- niggles -----------------------------------------------------------
    niggle_log = db.load_niggles(conn)
    if not niggle_log.empty:
        report.niggle_context = niggles_mod.context_for_niggles(
            niggle_log, report.load_series, report.runs)
        report.niggle_recurrence = niggles_mod.recurrence(report.niggle_context)
        findings += niggles_mod.niggle_findings(report.niggle_context, report.niggle_recurrence)

    # --- progress since last time ------------------------------------------
    report.snapshot_history = snapshots_mod.load_history(conn)
    report.progress_deltas = snapshots_mod.compare(report.snapshot_history, weeks_back=6)
    findings += snapshots_mod.progress_findings(report.progress_deltas, muscle_label)

    # --- what to do next ---------------------------------------------------
    if report.has_data:
        report.recommendation = prescription_mod.next_session(report)
        report.weekly_targets = prescription_mod.weekly_targets(report)
        findings += prescription_mod.prescription_findings(
            report.recommendation, report.weekly_targets)

    report.findings = sort_findings(findings)
    return report
