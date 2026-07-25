"""Zones built from Garmin's threshold, and the split-level analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smart_analytics.analytics import splits as splits_mod
from smart_analytics.analytics import zones


def metrics_with(lt_speed=3.33, lt_hr=170.0):
    return pd.DataFrame([{
        "local_date": pd.Timestamp("2026-06-01"),
        "lt_speed_mps": lt_speed,
        "lt_hr": lt_hr,
    }])


# --- zones ------------------------------------------------------------------

def test_zone_model_built_from_threshold_speed():
    model = zones.build_zone_model(metrics_with(lt_speed=3.33))
    assert model.source == "garmin"
    # Threshold speed 3.33 m/s = 5:00/km.
    assert model.lt_pace_s == pytest.approx(300.0, abs=1)

    easy = model.get("easy")
    threshold = model.get("threshold")
    # Easy must be slower (higher s/km) than threshold, and ranges must be ordered.
    assert easy.fast_pace_s > threshold.slow_pace_s
    assert easy.fast_pace_s < easy.slow_pace_s
    for zone in model.pace_zones:
        assert zone.fast_pace_s < zone.slow_pace_s


def test_zone_model_unavailable_without_threshold():
    model = zones.build_zone_model(pd.DataFrame())
    assert model.source == "unavailable"
    assert not model.has_pace_zones
    assert model.summary_table().empty
    # And the finding says so rather than inventing zones.
    findings = zones.zone_findings(model, {})
    assert findings[0].severity == "info"
    assert "threshold" in findings[0].title.lower()


def test_zone_for_pace_assigns_the_right_band():
    model = zones.build_zone_model(metrics_with(lt_speed=3.33))
    assert model.zone_for_pace(300).key == "threshold"     # at threshold
    assert model.zone_for_pace(370).key == "easy"          # 6:10/km
    assert model.zone_for_pace(500).key == "recovery"      # very slow
    assert model.zone_for_pace(None) is None


def test_easy_run_discipline_quantifies_running_easy_runs_too_fast():
    """The point of the zone model: turn "too fast" into seconds per km."""
    model = zones.build_zone_model(metrics_with(lt_speed=3.33))
    easy = model.get("easy")
    too_fast_pace = easy.fast_pace_s - 25  # 25 s/km quicker than the easy ceiling

    runs = pd.DataFrame([{
        "activity_id": f"r{i}",
        "local_date": pd.Timestamp("2026-06-01") + pd.Timedelta(days=i),
        "distance_km": 10.0,
        "pace_s_per_km": too_fast_pace,
        "intensity": "easy",
    } for i in range(6)])

    discipline = zones.easy_run_discipline(runs, model, lookback_days=90)
    assert discipline["too_fast_pct"] == 100
    assert discipline["seconds_too_fast"] == pytest.approx(25, abs=1)

    findings = zones.zone_findings(model, discipline)
    assert any(f.severity == "act" and "too fast" in f.title for f in findings)


def test_easy_discipline_praises_correct_pacing():
    model = zones.build_zone_model(metrics_with(lt_speed=3.33))
    easy = model.get("easy")
    runs = pd.DataFrame([{
        "activity_id": f"r{i}",
        "local_date": pd.Timestamp("2026-06-01") + pd.Timedelta(days=i),
        "distance_km": 10.0,
        "pace_s_per_km": (easy.fast_pace_s + easy.slow_pace_s) / 2,
        "intensity": "easy",
    } for i in range(6)])
    discipline = zones.easy_run_discipline(runs, model, lookback_days=90)
    assert discipline["too_fast_pct"] == 0
    assert any(f.severity == "good" for f in zones.zone_findings(model, discipline))


# --- splits -----------------------------------------------------------------

def make_splits(activity_id, laps, date="2026-06-01", split_type="ACTIVE"):
    """Laps are (distance_m, duration_s, avg_hr) tuples."""
    return pd.DataFrame([{
        "activity_id": activity_id,
        "split_index": index + 1,
        "split_type": split_type,
        "local_date": pd.Timestamp(date),
        "distance_m": distance,
        "duration_s": duration,
        "moving_s": duration,
        "avg_hr": avg_hr,
        "max_hr": avg_hr + 5,
        "avg_speed_mps": distance / duration,
        "avg_cadence": 170.0,
        "avg_power": None,
        "elevation_gain_m": 2.0,
        "elevation_loss_m": 2.0,
        "activity_type": "running",
        "name": "Long run",
    } for index, (distance, duration, avg_hr) in enumerate(laps)])


def test_decoupling_is_positive_when_heart_rate_drifts_up_at_constant_pace():
    # Same pace throughout, heart rate climbing 140 → 160.
    laps = [(1000.0, 300.0, hr) for hr in np.linspace(140, 160, 12)]
    table = splits_mod.decoupling(make_splits("a1", laps))
    assert len(table) == 1
    row = table.iloc[0]
    assert row["decoupling_pct"] > 5
    assert row["hr_drift_bpm"] > 0
    assert row["verdict"] in {"moderate drift", "beyond current endurance"}


def test_decoupling_near_zero_for_a_steady_run():
    laps = [(1000.0, 300.0, 145.0) for _ in range(12)]
    row = splits_mod.decoupling(make_splits("a2", laps)).iloc[0]
    assert abs(row["decoupling_pct"]) < 1
    assert row["verdict"] == "aerobically comfortable"


def test_decoupling_skips_short_runs_and_interval_sessions():
    short = make_splits("a3", [(1000.0, 300.0, 150.0)] * 4)
    assert splits_mod.decoupling(short).empty

    intervals = make_splits("a4", [(1000.0, 300.0, 150.0)] * 12, split_type="INTERVAL")
    assert splits_mod.decoupling(intervals).empty


def test_interval_session_detects_fade_from_labelled_laps():
    """Reps slowing from 200s to 212s must read as "started too hard"."""
    laps = [(800.0, duration, 175.0) for duration in (200, 203, 206, 209, 212)]
    frame = make_splits("a5", laps, split_type="INTERVAL")
    table = splits_mod.interval_sessions(frame)
    row = table.iloc[0]
    assert row["reps"] == 5
    assert row["fade_pct"] == pytest.approx(6.0, abs=0.5)
    assert row["verdict"] == "started too hard"


def test_interval_session_inferred_from_unlabelled_bimodal_pacing():
    """Devices that don't label lap intensity still expose the fast/slow pattern."""
    laps = []
    for _ in range(5):
        laps.append((800.0, 200.0, 175.0))   # work
        laps.append((400.0, 200.0, 140.0))   # recovery, half the speed
    table = splits_mod.interval_sessions(make_splits("a6", laps))
    assert not table.empty
    assert table.iloc[0]["reps"] == 5


def test_steady_run_is_not_mistaken_for_an_interval_session():
    laps = [(1000.0, 300.0 + jitter, 150.0) for jitter in (0, 2, -2, 1, -1, 3, -3, 0)]
    assert splits_mod.interval_sessions(make_splits("a7", laps)).empty


def test_negative_split_rate_counts_faster_second_halves():
    fading = make_splits("a8", [(1000.0, d, 150.0) for d in (295, 297, 302, 306)])
    negative = make_splits("a9", [(1000.0, d, 150.0) for d in (306, 302, 297, 295)])
    stats = splits_mod.negative_split_rate(pd.concat([fading, negative]))
    assert stats["runs"] == 2
    assert stats["negative_splits"] == 1
    assert stats["negative_pct"] == 50


def test_split_findings_report_absence_of_data_honestly():
    findings = splits_mod.split_findings(pd.DataFrame(), pd.DataFrame(), {}, {})
    assert len(findings) == 1
    assert findings[0].severity == "info"


def test_demo_report_finds_planted_long_run_drift_and_interval_fade(report):
    """The demo generator plants 7% drift on long runs and a rep fade."""
    assert not report.decoupling.empty
    assert report.decoupling["decoupling_pct"].max() > 3
    assert not report.intervals.empty
    assert report.intervals["fade_pct"].mean() > 2
