"""Structured workouts and Garmin's own exercise → muscle assignments.

The user always trains from a structured workout, so Garmin's own muscle
assignment is available for nearly every set — and it should beat the curated
fallback table. These tests cover the whole path: the shape of Garmin's workout
payload, the translation of anatomical names into the app's taxonomy, the
resolver's precedence rules, and the sync/report wiring that installs it.
"""

from __future__ import annotations

import json

import pytest

from smart_analytics import db
from smart_analytics.analytics import build_report
from smart_analytics.domain import exercises, garmin_muscles
from smart_analytics.garmin import SampleGarminClient, sync
from smart_analytics.garmin.sample import LIFT_PLANS
from smart_analytics.garmin.workouts import (
    muscle_records_from_steps,
    normalise_exercise_library,
    normalise_workout,
)


# --- Garmin's payload shape --------------------------------------------------

def test_repeat_groups_are_flattened_and_executable_steps_kept():
    """Each exercise sits inside a repeat group; only the inner step is a set."""
    payload = SampleGarminClient(days=10).workout("880003")
    summary, steps = normalise_workout(payload)

    assert summary["workout_id"] == "880003"
    assert summary["sport"] == "strength_training"
    # Four exercises in the legs plan, each wrapped in its own repeat group.
    assert len(steps) == 4
    assert summary["step_count"] == 4
    assert [s["exercise_name"] for s in steps] == [
        "BARBELL_BACK_SQUAT", "LEG_PRESS", "STANDING_CALF_RAISE", "SEATED_LEG_CURL"]
    # The repeat wrappers themselves name no exercise and must not become steps.
    assert all(s["category"] for s in steps)


def test_step_indices_are_unique_so_no_step_is_lost():
    _, steps = normalise_workout(SampleGarminClient(days=10).workout("880001"))
    indices = [s["step_index"] for s in steps]
    assert len(set(indices)) == len(indices)


def test_reps_come_from_the_end_condition_not_the_repeat_count():
    """A strength step states reps as ``endCondition: reps``; iterations are sets."""
    _, steps = normalise_workout(SampleGarminClient(days=10).workout("880003"))
    squat = next(s for s in steps if s["exercise_name"] == "BARBELL_BACK_SQUAT")
    assert squat["target_reps"] == 5      # 5 reps, not the 5 sets


@pytest.mark.parametrize("name, expected_kg", [
    ("BARBELL_BACK_SQUAT", 95.0),                  # grams in the payload
    ("INCLINE_DUMBBELL_BENCH_PRESS", 24.0),        # pounds in the payload
])
def test_planned_weights_are_normalised_to_kilograms(name, expected_kg):
    client = SampleGarminClient(days=10)
    steps = [step for workout_id in ("880001", "880003")
             for step in normalise_workout(client.workout(workout_id))[1]]
    step = next(s for s in steps if s["exercise_name"] == name)
    assert step["target_weight_kg"] == pytest.approx(expected_kg, abs=0.1)


def test_muscle_fields_survive_the_round_trip_to_records():
    _, steps = normalise_workout(SampleGarminClient(days=10).workout("880002"))
    records = muscle_records_from_steps(steps)
    row = next(r for r in records if r["exercise_name"] == "CABLE_FACE_PULL")
    assert row["primary_muscles"] == ["DELTOID_POSTERIOR"]
    assert row["source"] == "workout"


def test_workout_without_an_id_is_rejected_rather_than_stored():
    assert normalise_workout({"workoutName": "no id"}) == (None, [])
    assert normalise_workout("not a workout at all") == (None, [])


def test_exercise_library_walks_an_unknown_shape_and_dedupes():
    payload = {"categories": [
        {"categoryKey": "BENCH_PRESS", "primaryMuscles": ["PECTORALIS_MAJOR"],
         "exercises": [
             {"name": "BARBELL_BENCH_PRESS", "primaryMuscles": [{"name": "PECTORALIS_MAJOR"}],
              "secondaryMuscles": "TRICEPS_BRACHII, DELTOID_ANTERIOR"},
             # Same key twice — the later entry wins rather than duplicating.
             {"name": "BARBELL_BENCH_PRESS", "primaryMuscles": ["PECTORALIS_MAJOR"]},
         ]},
    ]}
    records = normalise_exercise_library(payload)
    keys = [(r["category"], r["exercise_name"]) for r in records]
    assert len(keys) == len(set(keys))
    assert ("BENCH_PRESS", "") in keys
    named = next(r for r in records if r["exercise_name"] == "BARBELL_BENCH_PRESS")
    assert named["category"] == "BENCH_PRESS"


# --- translating Garmin's anatomy into the 18-muscle model -------------------

def test_specific_muscle_names_win_over_the_general_ones_they_contain():
    """``BICEPS_FEMORIS`` is a hamstring, and must not match the "biceps" rule."""
    assert garmin_muscles.translate_one("BICEPS_FEMORIS") == "hamstrings"
    assert garmin_muscles.translate_one("BICEPS_BRACHII") == "biceps"
    assert garmin_muscles.translate_one("DELTOID_ANTERIOR") == "front_delts"
    assert garmin_muscles.translate_one("DELTOID_POSTERIOR") == "rear_delts"


def test_unqualified_deltoid_is_reported_rather_than_guessed():
    """Front, side and rear delts are separate balance targets — no guessing."""
    matched, unmatched = garmin_muscles.translate(
        ["PECTORALIS_MAJOR", "DELTOID", "TRICEPS_BRACHII"])
    assert matched == {"chest", "triceps"}
    assert unmatched == ["DELTOID"]
    assert "can't tell front from side from rear" in garmin_muscles.unmappable_reason("DELTOID")


def test_secondary_muscles_get_half_credit_and_primary_wins_a_tie():
    profile, unmatched = garmin_muscles.build_profile(
        ["PECTORALIS_MAJOR"], ["PECTORALIS_MAJOR", "TRICEPS_BRACHII"])
    assert unmatched == []
    assert profile == {"chest": 1.0, "triceps": 0.5}


# --- resolver precedence ----------------------------------------------------

def test_garmin_assignment_beats_the_curated_table():
    curated = exercises.resolve("SHOULDER_STABILITY", "CABLE_FACE_PULL")
    garmin_map = exercises.GarminMuscleMap([{
        "category": "SHOULDER_STABILITY", "exercise_name": "CABLE_FACE_PULL",
        "primary_muscles": ["DELTOID_POSTERIOR"], "secondary_muscles": ["TRAPEZIUS"],
    }])
    from_garmin = exercises.resolve("SHOULDER_STABILITY", "CABLE_FACE_PULL",
                                    garmin_map=garmin_map)

    assert from_garmin.source == "garmin_name"
    assert from_garmin.from_garmin
    assert from_garmin.muscles == {"rear_delts": 1.0, "upper_back": 0.5}
    assert not curated.from_garmin
    # The movement pattern still comes from our tables — Garmin doesn't classify
    # movements, only muscles.
    assert from_garmin.pattern == curated.pattern


def test_a_named_exercise_is_preferred_over_its_category():
    garmin_map = exercises.GarminMuscleMap([
        {"category": "SQUAT", "exercise_name": "",
         "primary_muscles": ["QUADRICEPS"], "secondary_muscles": []},
        {"category": "SQUAT", "exercise_name": "LEG_PRESS",
         "primary_muscles": ["QUADRICEPS"], "secondary_muscles": ["GLUTEUS_MAXIMUS"]},
    ])
    named = exercises.resolve("SQUAT", "LEG_PRESS", garmin_map=garmin_map)
    category = exercises.resolve("SQUAT", None, garmin_map=garmin_map)

    assert named.source == "garmin_name"
    assert named.muscles == {"quads": 1.0, "glutes": 0.5}
    assert category.source == "garmin_category"
    assert category.muscles == {"quads": 1.0}


def test_a_name_match_survives_a_category_mismatch():
    """Garmin logs an exercise under a different category than the workout used."""
    garmin_map = exercises.GarminMuscleMap([{
        "category": "ROW", "exercise_name": "BARBELL_ROW",
        "primary_muscles": ["LATISSIMUS_DORSI"], "secondary_muscles": [],
    }])
    resolved = exercises.resolve("UNKNOWN_CATEGORY", "BARBELL_ROW", garmin_map=garmin_map)
    assert resolved.source == "garmin_name"
    assert resolved.muscles == {"lats": 1.0}


def test_an_entry_garmin_gave_no_placeable_muscles_for_falls_back():
    """All-unmatched assignments must not shadow the curated table with nothing."""
    garmin_map = exercises.GarminMuscleMap([{
        "category": "BENCH_PRESS", "exercise_name": "BARBELL_BENCH_PRESS",
        "primary_muscles": ["TIBIALIS_ANTERIOR"], "secondary_muscles": [],
    }])
    resolved = exercises.resolve("BENCH_PRESS", "BARBELL_BENCH_PRESS", garmin_map=garmin_map)
    assert not resolved.from_garmin
    assert resolved.muscles["chest"] == pytest.approx(1.0)
    assert garmin_map.unmatched_names == {"TIBIALIS_ANTERIOR": 1}


# --- sync and report wiring -------------------------------------------------

@pytest.fixture(scope="module")
def workout_db(tmp_path_factory):
    conn = db.connect(tmp_path_factory.mktemp("workouts") / "w.db")
    sync(conn, SampleGarminClient(days=60, seed=4), history_days=60,
         detail_batch=100, split_batch=0, wellness_days=0, physiology_days=0,
         throttle_s=0.0)
    yield conn
    conn.close()


def test_sync_stores_workouts_steps_and_muscle_entries(workout_db):
    counts = db.counts(workout_db)
    assert counts["workouts"] == 3
    assert counts["workout_steps"] == sum(len(plan) for plan in LIFT_PLANS.values())
    records = db.load_exercise_muscles(workout_db)
    assert records
    # Named exercises from the workouts, plus category-level library entries.
    assert any(r["exercise_name"] == "BARBELL_BACK_SQUAT" for r in records)
    assert any(not r["exercise_name"] for r in records)


def test_muscle_lists_are_stored_as_decodable_json(workout_db):
    row = workout_db.execute(
        "SELECT primary_muscles FROM workout_steps WHERE exercise_name = ?",
        ("SEATED_LEG_CURL",)).fetchone()
    assert json.loads(row["primary_muscles"]) == ["BICEPS_FEMORIS", "SEMITENDINOSUS"]


def test_resyncing_does_not_refetch_workout_detail(workout_db):
    """Detail costs one request per workout, so it must only be fetched once."""
    again = sync(workout_db, SampleGarminClient(days=60, seed=4), history_days=60,
                 fetch_details=False, split_batch=0, wellness_days=0,
                 physiology_days=0, throttle_s=0.0)
    assert again.workouts_written == 0
    assert again.errors == []


def test_strength_activities_link_to_the_workout_they_were_run_against(workout_db):
    linked = workout_db.execute(
        "SELECT COUNT(*) FROM activities WHERE activity_type = 'strength_training' "
        "AND workout_id IS NOT NULL").fetchone()[0]
    total = workout_db.execute(
        "SELECT COUNT(*) FROM activities WHERE activity_type = 'strength_training'"
    ).fetchone()[0]
    assert linked == total > 0
    assert db.workout_ids_needing_fetch(workout_db) == []


def test_report_installs_the_garmin_map_and_reports_its_provenance(workout_db):
    report = build_report(workout_db)

    assert report.muscle_sources["garmin_records"] > 0
    assert report.muscle_sources["garmin_named"] > 0
    # The unqualified DELTOID that Garmin attaches to the row is surfaced, not
    # silently dropped.
    assert report.muscle_sources["unmatched_names"] == {"DELTOID": 2}
    assert exercises.active_garmin_muscle_map() is not None
    assert "Garmin's own muscle assignments" in \
        report.briefing()["strength"]["muscle_attribution"]


def test_attribution_uses_garmin_for_exercises_it_labelled(workout_db):
    build_report(workout_db)          # installs the map as the resolver default
    resolved = exercises.resolve("LEG_CURL", "SEATED_LEG_CURL")
    assert resolved.from_garmin
    # Garmin lists both hamstring heads plus calves as secondary.
    assert resolved.muscles["hamstrings"] == pytest.approx(1.0)
    assert resolved.muscles["calves"] == pytest.approx(0.5)


def test_report_falls_back_cleanly_when_no_workouts_were_synced(empty_db):
    report = build_report(empty_db)
    assert report.muscle_sources["garmin_records"] == 0
    assert exercises.active_garmin_muscle_map() is None
