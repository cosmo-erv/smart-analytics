"""Strength analytics: fractional credit, estimated 1RM, trends, lag detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smart_analytics.analytics import strength


def make_sets(rows):
    """Build a strength_sets-shaped frame from (date, category, name, reps, kg) tuples."""
    return pd.DataFrame([{
        "activity_id": f"a{index // 4}",
        "set_index": index,
        "local_date": pd.Timestamp(date),
        "start_time": pd.Timestamp(date),
        "set_type": "ACTIVE",
        "category": category,
        "exercise_name": name,
        "reps": reps,
        "weight_kg": weight,
        "duration_s": 30.0,
    } for index, (date, category, name, reps, weight) in enumerate(rows)])


def test_expand_sets_applies_fractional_credit():
    sets = make_sets([("2026-01-05", "BENCH_PRESS", "BARBELL_BENCH_PRESS", 5, 100.0)])
    expanded = strength.expand_sets(sets)
    by_muscle = expanded.set_index("muscle")

    assert by_muscle.loc["chest", "effective_sets"] == 1.0
    assert by_muscle.loc["triceps", "effective_sets"] == 0.5
    # Tonnage is scaled by the same share, so secondary work isn't double counted.
    assert by_muscle.loc["chest", "volume_kg"] == pytest.approx(500.0)
    assert by_muscle.loc["triceps", "volume_kg"] == pytest.approx(250.0)


def test_rest_sets_and_zero_rep_sets_are_excluded():
    sets = make_sets([("2026-01-05", "SQUAT", "BARBELL_BACK_SQUAT", 5, 100.0)])
    sets.loc[0, "set_type"] = "REST"
    assert strength.expand_sets(sets).empty

    zero = make_sets([("2026-01-05", "SQUAT", "BARBELL_BACK_SQUAT", 0, 100.0)])
    assert strength.expand_sets(zero).empty


def test_bodyweight_sets_count_as_volume_but_not_tonnage():
    """A pull-up must still earn its effective set even with no weight recorded."""
    sets = make_sets([("2026-01-05", "PULL_UP", "PULL_UP", 8, None)])
    expanded = strength.expand_sets(sets)
    lats = expanded[expanded["muscle"] == "lats"].iloc[0]
    assert lats["effective_sets"] == 1.0
    assert lats["volume_kg"] == 0.0
    assert np.isnan(lats["e1rm_kg"])


def test_estimate_1rm_epley_and_guards():
    assert strength.estimate_1rm(100.0, 1) == pytest.approx(100 * (1 + 1 / 30))
    assert strength.estimate_1rm(100.0, 10) == pytest.approx(100 * (1 + 10 / 30))
    # Reps are capped so a 30-rep set can't imply a 2x 1RM.
    assert strength.estimate_1rm(50.0, 40) == strength.estimate_1rm(
        50.0, strength.E1RM_REP_CAP)
    assert np.isnan(strength.estimate_1rm(float("nan"), 5))
    assert np.isnan(strength.estimate_1rm(0.0, 5))


def test_unmapped_exercises_are_reported_not_silently_dropped():
    sets = make_sets([
        ("2026-01-05", "BENCH_PRESS", "BARBELL_BENCH_PRESS", 5, 100.0),
        ("2026-01-05", "MYSTERY", "SOME_MACHINE", 10, 40.0),
    ])
    expanded = strength.expand_sets(sets)
    unmapped = strength.unmapped_exercises(sets)

    assert set(expanded["exercise"]) == {"Barbell Bench Press"}
    assert unmapped.iloc[0]["exercise_name"] == "SOME_MACHINE"
    assert int(unmapped.iloc[0]["sets"]) == 1


def test_linear_trend_detects_progression_and_needs_enough_span():
    dates = pd.date_range("2026-01-01", periods=10, freq="7D")
    rising = pd.Series(np.linspace(100, 118, 10))
    trend = strength.linear_trend(dates, rising)
    assert trend.reliable
    assert trend.slope_per_month > 0
    assert trend.r_squared > 0.99

    # Too few points, and too short a span, are both rejected rather than fitted.
    assert not strength.linear_trend(dates[:3], rising[:3]).reliable
    assert not strength.linear_trend(
        pd.date_range("2026-01-01", periods=6, freq="D"), rising[:6]).reliable


def test_lagging_muscles_flags_zero_volume_muscle_highest():
    volume = strength.muscle_volume_summary(strength.expand_sets(make_sets([
        ("2026-06-01", "BENCH_PRESS", "BARBELL_BENCH_PRESS", 5, 100.0),
        ("2026-06-01", "BENCH_PRESS", "BARBELL_BENCH_PRESS", 5, 100.0),
    ] * 6)))
    lagging = strength.lagging_muscles(volume, pd.DataFrame(), weekly_sets_min=10)

    # Every muscle appears, including ones with no volume at all.
    assert len(lagging) == len(volume)
    worst = lagging.iloc[0]
    assert worst["weekly_sets"] == 0
    assert worst["verdict"] == "falling behind"
    # Chest got trained, so it must score better than an untouched muscle.
    chest = lagging[lagging["muscle"] == "chest"].iloc[0]
    assert chest["attention_score"] < worst["attention_score"]


def test_lagging_score_components_are_bounded():
    volume = strength.muscle_volume_summary(strength.expand_sets(make_sets([
        ("2026-06-01", "SQUAT", "BARBELL_BACK_SQUAT", 5, 120.0),
    ])))
    lagging = strength.lagging_muscles(volume, pd.DataFrame(), weekly_sets_min=10)
    for column in ("volume_component", "trend_component", "recency_component",
                   "balance_component"):
        assert lagging[column].between(0, 1).all()
    assert lagging["attention_score"].between(0, 100).all()


def test_balance_ratios_identify_the_side_that_is_behind():
    """Bench only: chest/triceps get volume, back gets none — push must read high."""
    volume = strength.muscle_volume_summary(strength.expand_sets(make_sets([
        ("2026-06-01", "BENCH_PRESS", "BARBELL_BENCH_PRESS", 5, 100.0),
    ] * 12)))
    balance = strength.balance_ratios(volume)
    push_pull = balance[balance["pair"] == "Push : Pull"].iloc[0]
    assert push_pull["status"] == "no antagonist work"


def test_empty_input_produces_empty_frames_not_errors():
    empty = pd.DataFrame()
    assert strength.expand_sets(empty).empty
    assert strength.exercise_progress(strength.expand_sets(empty)).empty
    assert strength.balance_ratios(pd.DataFrame()).empty
    # The volume summary still lists every muscle, so "nothing trained" is visible.
    assert len(strength.muscle_volume_summary(strength.expand_sets(empty))) > 0


def test_report_finds_the_planted_weaknesses(report):
    """The demo athlete neglects hamstrings and rear delts; the model must see it."""
    lagging = report.lagging.set_index("muscle")
    assert lagging.loc["hamstrings", "weekly_sets"] < lagging.loc["quads", "weekly_sets"]
    assert lagging.loc["rear_delts", "weekly_sets"] < lagging.loc["front_delts", "weekly_sets"]

    # Rowing was planted as stalled while squat progresses.
    progress = report.progress.set_index("exercise")
    if "Barbell Row" in progress.index and "Barbell Back Squat" in progress.index:
        assert (progress.loc["Barbell Row", "pct_per_month"]
                < progress.loc["Barbell Back Squat", "pct_per_month"])
