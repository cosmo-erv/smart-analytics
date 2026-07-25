"""The exercise → muscle mapping, which everything in the strength analysis rests on."""

from __future__ import annotations

from smart_analytics.domain import exercises as ex
from smart_analytics.domain.muscles import MUSCLE_IDS, label, region_of


def test_category_default_maps_to_prime_movers():
    resolved = ex.resolve("BENCH_PRESS", "BARBELL_BENCH_PRESS")
    assert resolved.source == "category"
    assert resolved.muscles["chest"] == 1.0
    assert resolved.muscles["triceps"] == 0.5
    assert resolved.pattern == ex.H_PUSH


def test_name_override_beats_category():
    """A Romanian deadlift must not be credited to quads like a conventional pull."""
    conventional = ex.resolve("DEADLIFT", "BARBELL_DEADLIFT")
    romanian = ex.resolve("DEADLIFT", "BARBELL_ROMANIAN_DEADLIFT")
    assert conventional.source == "category"
    assert romanian.source == "name"
    assert "quads" in conventional.muscles
    assert "quads" not in romanian.muscles
    assert romanian.muscles["hamstrings"] == 1.0


def test_most_specific_name_key_wins():
    """CLOSE_GRIP_BENCH_PRESS (3 tokens) must beat DIP-style shorter keys."""
    resolved = ex.resolve("BENCH_PRESS", "CLOSE_GRIP_BARBELL_BENCH_PRESS")
    assert resolved.muscles["triceps"] == 1.0
    assert resolved.muscles["chest"] < 1.0


def test_unmapped_and_non_loading_are_distinguished():
    assert ex.resolve("WARM_UP", None).source == "non_loading"
    unknown = ex.resolve("SOME_NEW_CATEGORY", "MYSTERY_MOVE")
    assert unknown.source == "unmapped"
    assert not unknown.is_mapped


def test_equipment_detection():
    assert ex.resolve("CURL", "DUMBBELL_CURL").equipment == "Dumbbell"
    assert ex.resolve("ROW", "CABLE_SEATED_ROW").equipment == "Cable"
    assert ex.resolve("PULL_UP", "PULL_UP").equipment is None


def test_every_mapped_muscle_exists_in_the_taxonomy():
    """A typo in a profile would silently drop volume, so assert the ids are real."""
    referenced = set()
    for profile in ex.CATEGORY_PROFILES.values():
        referenced |= set(profile.muscles)
    for profile in ex.NAME_PROFILES.values():
        referenced |= set(profile.muscles)
    assert referenced <= set(MUSCLE_IDS), referenced - set(MUSCLE_IDS)


def test_all_weights_are_in_the_documented_range():
    for name, profile in {**ex.CATEGORY_PROFILES, **ex.NAME_PROFILES}.items():
        for muscle, weight in profile.muscles.items():
            assert 0 < weight <= 1.0, f"{name}/{muscle} = {weight}"


# TOTAL_BODY is a catch-all for movements Garmin couldn't classify (thrusters,
# burpees, complexes). It intentionally spreads partial credit rather than naming
# a prime mover it can't know, so it's exempt from the rule below.
DIFFUSE_CATEGORIES = {"TOTAL_BODY"}


def test_every_profile_has_a_prime_mover():
    """A profile of only fractional credits would understate every exercise."""
    profiles = {**ex.CATEGORY_PROFILES, **ex.NAME_PROFILES}
    for name, profile in profiles.items():
        if name in DIFFUSE_CATEGORIES:
            continue
        assert max(profile.muscles.values()) == 1.0, name


def test_labels_and_regions_resolve():
    assert label("rear_delts") == "Rear delts"
    assert region_of("quads") == "legs"
    assert label("not_a_muscle") == "Not A Muscle"  # falls back, doesn't raise
