"""End-to-end: ingest normalisation, storage, snapshots, briefing and digest."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import pytest

from smart_analytics import db
from smart_analytics.analytics import build_report, snapshots
from smart_analytics.garmin import SampleGarminClient, normalise_activity, normalise_sets, sync
from smart_analytics.garmin import physiology
from smart_analytics.reporting import weekly_digest_html, weekly_digest_markdown


# --- normalisation ----------------------------------------------------------

def test_normalise_activity_converts_garmin_units():
    raw = {
        "activityId": 123,
        "activityName": "Morning Run",
        "activityType": {"typeKey": "trail_running"},
        "startTimeLocal": "2026-06-01 07:30:00",
        "distance": 10_000.0,
        "duration": 3000.0,
        "averageSpeed": 3.33,
        "averageHR": 150,
        "avgStrideLength": 110.0,          # centimetres
        "hrTimeInZone_1": 300.0,
        "hrTimeInZone_2": 1200.0,
    }
    row = normalise_activity(raw)
    assert row["activity_id"] == "123"
    assert row["activity_type"] == "running"      # trail_running folded to family
    assert row["activity_subtype"] == "trail_running"
    assert row["avg_stride_m"] == pytest.approx(1.1)   # cm → m
    assert json.loads(row["hr_zone_json"]) == {"z1": 300.0, "z2": 1200.0}


def test_normalise_sets_converts_grams_and_keeps_rest_sets():
    raw = [
        {"setType": "ACTIVE", "repetitionCount": 8, "weight": 82_500.0, "duration": 30.0,
         "startTime": "2026-06-01 18:00:00",
         "exercises": [{"category": "SQUAT", "name": "BARBELL_BACK_SQUAT",
                        "probability": 100}]},
        {"setType": "REST", "repetitionCount": None, "weight": None, "duration": 120.0,
         "startTime": "2026-06-01 18:00:30", "exercises": []},
    ]
    rows = normalise_sets("123", "2026-06-01", raw)
    assert rows[0]["weight_kg"] == pytest.approx(82.5)   # grams → kg
    assert rows[0]["category"] == "SQUAT"
    assert rows[1]["set_type"] == "REST"                 # rest structure preserved
    assert rows[1]["weight_kg"] is None


def test_normalise_activity_rejects_unusable_payloads():
    assert normalise_activity({}) is None
    assert normalise_activity({"activityId": 1}) is None          # no start time
    assert normalise_activity({"startTimeLocal": "2026-06-01 07:00:00"}) is None


def test_physiology_normalisers_survive_nesting_and_missing_fields():
    day = date(2026, 6, 1)
    nested = {"mostRecentTrainingStatus": {"latestTrainingStatusData": {"3419": {
        "trainingStatus": 4, "acuteTrainingLoadDTO": {
            "dailyTrainingLoadAcute": 500, "dailyAcuteChronicWorkloadRatio": 1.1}}}}}
    row = physiology.normalise_training_status(nested, day)
    assert row["training_status"] == "Productive"
    assert row["acute_load"] == 500
    assert row["load_ratio"] == 1.1

    # Absent data yields Nones, never fabricated numbers or exceptions.
    for normaliser in (physiology.normalise_training_status,
                       physiology.normalise_max_metrics,
                       physiology.normalise_lactate_threshold,
                       physiology.normalise_training_readiness):
        result = normaliser(None, day)
        assert result["local_date"] == "2026-06-01"
        assert all(value is None for key, value in result.items() if key != "local_date")


def test_lactate_threshold_accepts_pace_or_speed():
    day = date(2026, 6, 1)
    as_speed = physiology.normalise_lactate_threshold({"lactateThresholdSpeed": 3.4}, day)
    as_pace = physiology.normalise_lactate_threshold({"lactateThresholdSpeed": 294.0}, day)
    assert as_speed["lt_speed_mps"] == pytest.approx(3.4)
    # 294 s/km is a pace, not a speed — it must be converted, not stored as m/s.
    assert as_pace["lt_speed_mps"] == pytest.approx(1000 / 294, rel=0.01)


def test_hr_zones_from_activity_zone_report():
    payload = [{"zoneNumber": 1, "zoneLowBoundary": 100},
               {"zoneNumber": 2, "zoneLowBoundary": 125},
               {"zoneNumber": 3, "zoneLowBoundary": 145}]
    zones = physiology.normalise_hr_zones(payload)
    assert [z["zone"] for z in zones] == [1, 2, 3]
    assert zones[0]["ceiling_bpm"] == 125       # ceiling inferred from the next floor
    assert zones[-1]["ceiling_bpm"] is None


# --- storage ----------------------------------------------------------------

def test_sync_is_idempotent(tmp_path):
    conn = db.connect(tmp_path / "idem.db")
    client = SampleGarminClient(days=60, seed=3)
    first = sync(conn, client, history_days=60, wellness_days=5, physiology_days=2,
                 throttle_s=0.0)
    before = db.counts(conn)
    second = sync(conn, client, history_days=60, wellness_days=5, physiology_days=2,
                  throttle_s=0.0)
    after = db.counts(conn)

    assert before["activities"] == after["activities"]
    assert before["sets"] == after["sets"]
    assert before["splits"] == after["splits"]
    # Second pass re-reads summaries but skips detail it already has.
    assert second.strength_details == 0
    assert second.splits_activities == 0
    conn.close()


def test_migration_adds_new_columns_to_an_existing_database(tmp_path):
    """A database created before splits existed must upgrade, not break."""

    import sqlite3

    path = tmp_path / "old.db"
    legacy = sqlite3.connect(path)
    # The v1 activities table: everything except the columns added later.
    v1_columns = [c for c in db.ACTIVITY_COLUMNS if c != "splits_fetched"]
    legacy.execute(
        "CREATE TABLE activities (activity_id TEXT PRIMARY KEY, "
        + ", ".join(f"{c} TEXT" for c in v1_columns if c != "activity_id") + ")")
    legacy.execute(
        "INSERT INTO activities (activity_id, start_time, local_date, activity_type) "
        "VALUES ('1', '2026-01-01 07:00:00', '2026-01-01', 'running')")
    legacy.commit()
    legacy.close()

    conn = db.connect(path)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(activities)")}
    assert "splits_fetched" in columns
    # The existing row survives the upgrade.
    assert conn.execute("SELECT COUNT(*) FROM activities").fetchone()[0] == 1
    conn.close()


def test_niggle_lifecycle(empty_db):
    niggle_id = db.add_niggle(empty_db, "2026-06-01", "Calf", 2, "mild")
    log = db.load_niggles(empty_db)
    assert len(log) == 1
    assert pd.isna(log.iloc[0]["resolved_on"])

    db.resolve_niggle(empty_db, niggle_id, "2026-06-10")
    assert pd.notna(db.load_niggles(empty_db).iloc[0]["resolved_on"])

    db.delete_niggle(empty_db, niggle_id)
    assert db.load_niggles(empty_db).empty


def test_clear_all_empties_every_table(synced_db, tmp_path):
    conn = db.connect(tmp_path / "clear.db")
    sync(conn, SampleGarminClient(days=40, seed=5), history_days=40, wellness_days=3,
         physiology_days=1, throttle_s=0.0)
    assert db.counts(conn)["activities"] > 0
    db.clear_all(conn)
    counts = db.counts(conn)
    for key in ("activities", "sets", "splits", "athlete_metrics", "daily_metrics"):
        assert counts[key] == 0
    conn.close()


# --- report and outputs -----------------------------------------------------

def test_report_builds_all_sections(report):
    assert report.has_data and report.has_running and report.has_strength
    assert not report.lagging.empty
    assert not report.load_series.empty
    assert not report.splits.empty
    assert report.zone_model.has_pace_zones
    assert report.recommendation is not None
    assert len(report.findings) > 10


def test_report_on_an_empty_database_is_safe(empty_db):
    empty = build_report(empty_db)
    assert not empty.has_data
    assert empty.findings == [] or all(f.severity == "info" for f in empty.findings)
    # The briefing must still be serialisable so the coach page can render.
    assert json.dumps(empty.briefing(), default=str)


def test_briefing_is_json_safe_and_bounded(report):
    brief = report.briefing()
    encoded = json.dumps(brief, default=str)
    assert "NaN" not in encoded, "NaN is not valid JSON and breaks the API call"
    # Bounded so the prompt can't grow without limit as history accumulates.
    assert len(encoded) < 120_000
    for key in ("findings", "units", "data_window", "garmin_metrics", "split_analysis"):
        assert key in brief


def test_briefing_contains_no_raw_activity_rows(report):
    """The coach sees computed metrics only — never a dump of activities."""
    brief = report.briefing()
    assert "activities" not in brief
    assert "raw_json" not in json.dumps(brief, default=str)


def test_snapshot_round_trip_and_comparison(empty_db, report):
    payload = snapshots.snapshot_payload(report)
    assert payload["muscle_scores"]

    today = date.today()
    db.save_snapshot(empty_db, (today - timedelta(weeks=6)).isoformat(),
                     {**payload,
                      "muscle_scores": {k: min(v + 20, 100)
                                        for k, v in payload["muscle_scores"].items()},
                      "weekly_km": payload.get("weekly_km", 30) - 10})
    db.save_snapshot(empty_db, today.isoformat(), payload)

    history = snapshots.load_history(empty_db)
    assert len(history) == 2
    deltas = snapshots.compare(history, weeks_back=6)
    # Scores fell, so every muscle should read as improved.
    assert all(m["score_change"] <= 0 for m in deltas["muscles"])
    assert any(m["label"] == "Weekly running volume" and m["direction"] == "improved"
               for m in deltas["metrics"])

    findings = snapshots.progress_findings(deltas, lambda m: m.title())
    assert any("catching up" in f.title for f in findings)


def test_snapshot_comparison_needs_a_nearby_snapshot(empty_db, report):
    db.save_snapshot(empty_db, date.today().isoformat(), snapshots.snapshot_payload(report))
    # A single snapshot can't be compared against anything.
    assert snapshots.compare(snapshots.load_history(empty_db)) == {}


def test_digest_markdown_and_html(report):
    markdown = weekly_digest_markdown(report)
    for heading in ("# Training digest", "## Next session", "## Where things stand"):
        assert heading in markdown

    html = weekly_digest_html(report)
    assert html.startswith("<!doctype html>")
    assert "<table>" in html and "</html>" in html
    # Markdown tables must become real tables, not literal pipes.
    assert "| ---" not in html


def test_digest_escapes_user_supplied_text(empty_db, report):
    """A niggle note is user input and must not be able to inject markup."""
    db.add_niggle(empty_db, date.today().isoformat(), "Knee", 3,
                  "<script>alert('x')</script>")
    injected = build_report(empty_db)
    html = weekly_digest_html(injected)
    assert "<script>" not in html
