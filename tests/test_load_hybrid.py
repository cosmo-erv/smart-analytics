"""Training load, concurrent-training interference, niggles and prescription."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smart_analytics.analytics import hybrid, load, niggles, prescription
from smart_analytics.analytics.zones import build_zone_model


def make_activities(rows):
    """Rows are (date, type, duration_s, avg_hr, training_load)."""
    return pd.DataFrame([{
        "activity_id": f"x{index}",
        "local_date": pd.Timestamp(date),
        "start_time": pd.Timestamp(date),
        "activity_type": activity_type,
        "duration_s": duration,
        "distance_m": 10_000.0 if activity_type == "running" else 0.0,
        "avg_hr": avg_hr,
        "max_hr": (avg_hr + 20) if avg_hr else None,
        "training_load": training_load,
    } for index, (date, activity_type, duration, avg_hr, training_load) in enumerate(rows)])


# --- load -------------------------------------------------------------------

def test_trimp_rises_with_intensity_and_duration():
    easy = load.trimp(3600, 130, 190, 50)
    hard = load.trimp(3600, 175, 190, 50)
    longer = load.trimp(7200, 130, 190, 50)
    assert hard > easy
    assert longer == pytest.approx(easy * 2)
    assert load.trimp(0, 150, 190, 50) == 0
    assert load.trimp(3600, 0, 190, 50) == 0


def test_daily_load_prefers_reported_load_and_fills_rest_days():
    activities = make_activities([
        ("2026-06-01", "running", 3600, 150, 120.0),
        ("2026-06-04", "strength_training", 3600, 110, None),  # estimated
    ])
    daily = load.daily_load(activities, max_hr=190, resting_hr=50)

    # Gap-free index: 1st to 4th June is four rows, including two zero-load days.
    assert len(daily) == 4
    assert daily.iloc[0]["load"] == 120.0
    assert daily.iloc[0]["estimated_share"] == 0.0
    assert (daily["load"] == 0).sum() == 2
    # The strength session had no reported load, so it was estimated.
    assert daily.iloc[-1]["estimated_share"] == 1.0
    assert daily.iloc[-1]["load"] > 0


def test_acwr_and_status_bands():
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    steady = pd.DataFrame({"date": dates, "load": [100.0] * 40,
                           "estimated_share": 0.0, "activities": 1, "duration_min": 60.0})
    series = load.load_series(steady)
    latest = series.dropna(subset=["acwr"]).iloc[-1]
    # Constant load means acute equals chronic.
    assert latest["acwr"] == pytest.approx(1.0, abs=0.01)
    assert latest["acwr_status"] == "productive"

    assert load.acwr_status(0.5) == "detraining"
    assert load.acwr_status(1.4) == "caution"
    assert load.acwr_status(1.8) == "high risk"
    assert load.acwr_status(None) == "insufficient history"


def test_acwr_requires_full_windows_before_reporting():
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    short = pd.DataFrame({"date": dates, "load": [100.0] * 20,
                          "estimated_share": 0.0, "activities": 1, "duration_min": 60.0})
    series = load.load_series(short)
    # 28-day chronic window can't be filled in 20 days, so ACWR stays null.
    assert series["acwr"].isna().all()


def test_infer_hr_bounds_prefers_configured_values():
    activities = make_activities([("2026-06-01", "running", 3600, 150, 100.0)])
    activities.loc[0, "max_hr"] = 185
    max_hr, resting = load.infer_hr_bounds(activities, None, max_hr=200, resting_hr=45)
    assert (max_hr, resting) == (200.0, 45.0)
    # Implausible spread falls back to defaults rather than producing nonsense load.
    max_hr, resting = load.infer_hr_bounds(activities, None, max_hr=100, resting_hr=95)
    assert max_hr == load.DEFAULT_MAX_HR


# --- hybrid -----------------------------------------------------------------

def make_expanded(rows):
    """Rows are (activity_id, date, muscle, effective_sets)."""
    return pd.DataFrame([{
        "activity_id": activity_id,
        "local_date": pd.Timestamp(date),
        "muscle": muscle,
        "effective_sets": sets,
        "exercise": "Test",
        "reps": 8.0,
        "weight_kg": 100.0,
        "volume_kg": 800.0,
        "e1rm_kg": 120.0,
        "share": 1.0,
        "pattern": "squat",
        "category": "SQUAT",
        "equipment": None,
    } for activity_id, date, muscle, sets in rows])


def test_leg_sessions_identified_by_lower_body_share():
    expanded = make_expanded([
        ("leg1", "2026-06-01", "quads", 6.0),
        ("leg1", "2026-06-01", "chest", 1.0),
        ("push1", "2026-06-02", "chest", 8.0),
        ("push1", "2026-06-02", "quads", 1.0),
    ])
    legs = hybrid.leg_sessions(expanded)
    assert list(legs["activity_id"]) == ["leg1"]
    assert legs.iloc[0]["leg_share"] > hybrid.LEG_SHARE_THRESHOLD


def test_interference_detects_quality_run_soon_after_legs():
    activities = make_activities([
        ("2026-06-01 18:00", "strength_training", 3600, 120, 80.0),
        ("2026-06-02 07:00", "running", 2700, 175, 130.0),
    ])
    activities.loc[1, "activity_id"] = "run1"
    activities.loc[0, "activity_id"] = "leg1"
    expanded = make_expanded([("leg1", "2026-06-01", "quads", 8.0)])
    runs = pd.DataFrame([{
        "activity_id": "run1", "local_date": pd.Timestamp("2026-06-02"),
        "start_time": pd.Timestamp("2026-06-02 07:00"), "name": "Intervals",
        "distance_km": 9.0, "intensity": "hard",
    }])

    events = hybrid.interference_events(activities, expanded, runs, lookback_days=365)
    assert len(events) == 1
    assert events.iloc[0]["gap_hours"] == pytest.approx(13.0, abs=0.5)


def test_no_interference_when_sessions_are_well_separated():
    activities = make_activities([
        ("2026-06-01 18:00", "strength_training", 3600, 120, 80.0),
        ("2026-06-04 07:00", "running", 2700, 175, 130.0),
    ])
    activities.loc[0, "activity_id"] = "leg1"
    activities.loc[1, "activity_id"] = "run1"
    expanded = make_expanded([("leg1", "2026-06-01", "quads", 8.0)])
    runs = pd.DataFrame([{
        "activity_id": "run1", "local_date": pd.Timestamp("2026-06-04"),
        "start_time": pd.Timestamp("2026-06-04 07:00"), "name": "Intervals",
        "distance_km": 9.0, "intensity": "hard",
    }])
    assert hybrid.interference_events(activities, expanded, runs).empty


def test_discipline_split_percentages_sum_to_100():
    activities = make_activities([
        ("2026-06-01", "running", 3600, 150, 300.0),
        ("2026-06-02", "strength_training", 3600, 110, 100.0),
        ("2026-06-03", "cycling", 3600, 130, 100.0),
    ])
    split = hybrid.discipline_split(activities, lookback_days=365)
    row = split.iloc[0]
    assert row["strength_pct"] == 20
    assert row["running_pct"] == 60


# --- niggles ----------------------------------------------------------------

def test_niggle_context_attaches_load_at_onset():
    dates = pd.date_range("2026-05-01", periods=40, freq="D")
    daily = pd.DataFrame({"date": dates, "load": [100.0] * 40,
                          "estimated_share": 0.0, "activities": 1, "duration_min": 60.0})
    series = load.load_series(daily)
    log = pd.DataFrame([{
        "id": 1, "noted_on": pd.Timestamp("2026-06-05"), "area": "Achilles",
        "severity": 3, "note": "tight", "resolved_on": pd.NaT,
    }])
    context = niggles.context_for_niggles(log, series)
    row = context.iloc[0]
    assert row["status"] == "open"
    assert row["acwr_at_onset"] == pytest.approx(1.0, abs=0.05)
    assert row["severity_label"] == "Changes how I train"


def test_recurring_niggle_is_flagged():
    log = pd.DataFrame([
        {"id": 1, "noted_on": pd.Timestamp.today() - pd.Timedelta(days=10),
         "area": "Achilles", "severity": 3, "note": None, "resolved_on": pd.NaT},
        {"id": 2, "noted_on": pd.Timestamp.today() - pd.Timedelta(days=60),
         "area": "Achilles", "severity": 2, "note": None,
         "resolved_on": pd.Timestamp.today() - pd.Timedelta(days=50)},
    ])
    context = niggles.context_for_niggles(log, pd.DataFrame())
    repeats = niggles.recurrence(context)
    assert repeats.iloc[0]["entries"] == 2

    findings = niggles.niggle_findings(context, repeats)
    assert any("keeps coming back" in f.title for f in findings)
    # Severity 3 for over a fortnight must point at a professional, not self-manage.
    assert any("physio" in (f.recommendation or "") for f in findings)


# --- prescription -----------------------------------------------------------

class FakeReport:
    """Minimal stand-in exercising the recommendation gates in isolation."""

    def __init__(self, **overrides):
        self.athlete_metrics = pd.DataFrame()
        self.load_series = pd.DataFrame()
        self.lagging = pd.DataFrame()
        self.expanded = pd.DataFrame()
        self.runs = pd.DataFrame()
        self.activities = pd.DataFrame()
        self.intensity = {}
        self.niggle_context = pd.DataFrame()
        self.weekly_runs = pd.DataFrame()
        self.hybrid_events = pd.DataFrame()
        self.zone_model = build_zone_model(pd.DataFrame())
        self.settings = type("S", (), {"weekly_sets_min": 10, "weekly_sets_max": 20})()
        self.has_data = True
        for key, value in overrides.items():
            setattr(self, key, value)


def test_severe_niggle_overrides_everything():
    report = FakeReport(niggle_context=pd.DataFrame([{
        "status": "open", "severity": 4, "area": "Knee", "days_open": 3,
    }]))
    recommendation = prescription.next_session(report)
    assert recommendation.kind == "rest"
    assert recommendation.confidence == "high"
    assert "Knee" in recommendation.reasons[0]


def test_very_low_readiness_forces_rest():
    report = FakeReport(athlete_metrics=pd.DataFrame([{
        "local_date": pd.Timestamp.today(), "readiness_score": 20.0,
        "recovery_time_h": 30.0,
    }]))
    recommendation = prescription.next_session(report)
    assert recommendation.kind == "rest"


def test_lagging_muscle_earns_a_strength_slot():
    report = FakeReport(
        lagging=pd.DataFrame([{
            "muscle": "hamstrings", "muscle_label": "Hamstrings", "weekly_sets": 1.0,
            "attention_score": 70.0, "days_since": 12, "deficit": 9.0,
        }]),
        activities=make_activities([("2026-06-01", "strength_training", 3600, 110, 80.0)]),
    )
    recommendation = prescription.next_session(report, as_of=pd.Timestamp("2026-06-05").date())
    assert recommendation.kind == "strength"
    assert "Hamstrings" in recommendation.targets[0]


def test_quality_run_recommended_when_recovered_and_stale():
    report = FakeReport(
        athlete_metrics=pd.DataFrame([{
            "local_date": pd.Timestamp.today(), "readiness_score": 85.0,
            "recovery_time_h": 2.0,
        }]),
        runs=pd.DataFrame([{
            "activity_id": "r1", "local_date": pd.Timestamp("2026-06-01"),
            "intensity": "hard", "distance_km": 10.0, "pace_s_per_km": 300.0,
        }]),
    )
    recommendation = prescription.next_session(report, as_of=pd.Timestamp("2026-06-10").date())
    assert recommendation.kind == "quality_run"


def test_weekly_targets_quantify_the_deficit(report):
    """Against the demo athlete: targets must be countable, not advisory."""
    targets = prescription.weekly_targets(report)
    assert targets["strength"]
    first = targets["strength"][0]
    assert first["add_sets"] > 0
    assert first["target"] == report.settings.weekly_sets_min
    assert any("km" in str(item.get("unit", "")) for item in targets["running"])
