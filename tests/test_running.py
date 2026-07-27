"""Running analytics: pace derivation, efficiency, intensity, bests and predictions."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from smart_analytics.analytics import running


def make_runs(rows):
    """Rows are (date, distance_m, duration_s, avg_hr, zone seconds tuple or None)."""
    records = []
    for index, (date, distance, duration, avg_hr, zones) in enumerate(rows):
        zone_json = None
        if zones:
            zone_json = json.dumps({f"z{i + 1}": value for i, value in enumerate(zones)})
        records.append({
            "activity_id": f"r{index}",
            "local_date": pd.Timestamp(date),
            "start_time": pd.Timestamp(date),
            "activity_type": "running",
            "name": "Run",
            "distance_m": distance,
            "duration_s": duration,
            "moving_s": duration,
            "avg_hr": avg_hr,
            "max_hr": (avg_hr + 15) if avg_hr else None,
            "avg_cadence": 170.0,
            "avg_stride_m": 1.1,
            "elevation_gain_m": 20.0,
            "training_load": 90.0,
            "vo2max": 50.0,
            "avg_gct_ms": 250.0,
            "avg_vert_osc_cm": 9.0,
            "hr_zone_json": zone_json,
        })
    return pd.DataFrame(records)


def test_format_pace_and_duration():
    assert running.format_pace(300) == "5:00/km"
    assert running.format_pace(305.4) == "5:05/km"
    assert running.format_pace(None) == "—"
    assert running.format_pace(0) == "—"
    assert running.format_duration(3725) == "1:02:05"
    assert running.format_duration(125) == "2:05"
    assert running.format_duration(np.nan) == "—"


def test_prepare_runs_derives_pace_and_efficiency():
    runs = running.prepare_runs(make_runs([("2026-01-01", 10_000, 3000, 150, None)]))
    row = runs.iloc[0]
    assert row["distance_km"] == pytest.approx(10.0)
    assert row["pace_s_per_km"] == pytest.approx(300.0)
    assert row["pace_label"] == "5:00/km"
    # 10,000 m over 50 min at 150 bpm = 7500 beats.
    assert row["m_per_beat"] == pytest.approx(10_000 / (50 * 150))


def test_prepare_runs_filters_walk_breaks_and_non_runs():
    """Sub-400 m or sub-2-minute entries are GPS artefacts, not runs."""
    runs = running.prepare_runs(make_runs([
        ("2026-01-01", 300, 200, 120, None),      # too short
        ("2026-01-02", 5_000, 60, 150, None),     # implausibly brief
        ("2026-01-03", 5_000, 1500, 150, None),   # keep
    ]))
    assert len(runs) == 1


def test_intensity_uses_zone_data_when_present():
    # Mostly zone 1-2 → easy; mostly zone 4-5 → hard.
    runs = running.prepare_runs(make_runs([
        ("2026-01-01", 10_000, 3000, 140, (600, 1800, 600, 0, 0)),
        ("2026-01-02", 8_000, 2400, 175, (100, 200, 300, 1200, 600)),
    ]))
    assert runs.iloc[0]["intensity"] == "easy"
    assert runs.iloc[1]["intensity"] == "hard"


def test_intensity_falls_back_to_hr_fraction_without_zones():
    runs = running.prepare_runs(make_runs([("2026-01-01", 10_000, 3000, 140, None)]))
    assert runs.iloc[0]["intensity"] in {"easy", "moderate", "hard"}


def test_intensity_distribution_reports_source_and_sums_to_100():
    runs = running.prepare_runs(make_runs([
        ("2026-06-01", 10_000, 3000, 140, (600, 1800, 600, 0, 0)),
        ("2026-06-03", 8_000, 2400, 175, (100, 200, 300, 1200, 600)),
    ]))
    distribution = running.intensity_distribution(runs, lookback_days=365)
    assert distribution["source"] == "hr_zones"
    total = (distribution["easy_pct"] + distribution["moderate_pct"]
             + distribution["hard_pct"])
    assert total == pytest.approx(100.0, abs=0.2)


def test_weekly_running_uses_distance_weighted_pace():
    """A short fast run must not drag the weekly average as much as a long slow one."""
    runs = running.prepare_runs(make_runs([
        ("2026-06-01", 2_000, 480, 170, None),    # 4:00/km
        ("2026-06-02", 18_000, 6_300, 145, None),  # 5:50/km
    ]))
    weekly = running.weekly_running(runs)
    expected = (480 + 6300) / 20.0
    assert weekly.iloc[0]["avg_pace_s_per_km"] == pytest.approx(expected)


def test_personal_bests_normalise_with_riegel_so_longer_is_not_better():
    """A 10.8 km run at the same pace must not beat a clean 10.0 km effort."""
    runs = running.prepare_runs(make_runs([
        ("2026-03-01", 10_000, 2_400, 165, None),   # 4:00/km
        ("2026-04-01", 10_800, 2_620, 165, None),   # ~4:03/km, longer
    ]))
    bests = running.personal_bests(runs, recent_days=3650)
    ten_k = bests[bests["bucket"] == "10K"].iloc[0]
    assert ten_k["attempts"] == 2
    assert ten_k["best_time_s"] == pytest.approx(2400, abs=5)


def test_race_predictions_scale_with_riegel_exponent():
    runs = running.prepare_runs(make_runs([("2026-06-01", 10_000, 2_400, 165, None)]))
    predictions = running.race_predictions(runs).set_index("bucket")
    half = predictions.loc["Half marathon", "predicted_time_s"]
    expected = 2400 * (21_097 / 10_000) ** running.RIEGEL_EXPONENT
    assert half == pytest.approx(expected, rel=0.01)
    # Longer distances must always predict slower paces.
    assert (predictions.loc["Marathon", "predicted_pace_s_per_km"]
            > predictions.loc["5K", "predicted_pace_s_per_km"])


def test_consistency_counts_weeks_with_no_running():
    runs = running.prepare_runs(make_runs([
        ("2026-06-01", 10_000, 3_000, 150, None),
        ("2026-06-29", 10_000, 3_000, 150, None),
    ]))
    stability = running.consistency(runs, lookback_days=28)
    assert stability["zero_weeks"] >= 1
    assert stability["weeks"] == 4


def test_empty_running_input_is_safe():
    assert running.prepare_runs(pd.DataFrame()).empty
    assert running.intensity_distribution(pd.DataFrame()) == {}
    assert running.personal_bests(pd.DataFrame()).empty
    assert running.race_predictions(pd.DataFrame()).empty
    assert running.consistency(pd.DataFrame()) == {}


def test_demo_report_detects_the_planted_grey_zone(report):
    """Demo runs are deliberately moderate-heavy; the finding must fire."""
    assert report.intensity["moderate_pct"] > report.intensity["easy_pct"]
    titles = [f.title for f in report.findings_for("running")]
    assert any("grey zone" in title for title in titles)
