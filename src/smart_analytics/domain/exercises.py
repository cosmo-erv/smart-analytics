"""Garmin exercise → muscle mapping.

Garmin's strength activities expose each set as an *exercise category* (the FIT
`exercise_category` enum, e.g. ``BENCH_PRESS``) plus an optional, more specific
*exercise name* (e.g. ``INCLINE_DUMBBELL_BENCH_PRESS``). Neither says anything
about muscles, so this module supplies that layer.

Resolution order for a set:

0. **Garmin's own muscle data**, when available. Structured workouts and Garmin's
   exercise library state which muscles each exercise works; that beats anything
   inferred here, so a :class:`GarminMuscleMap` built from synced data is checked
   first. An exact exercise-name match wins over a category-level one.
1. **Name profile** — the most specific matching entry in :data:`NAME_PROFILES`
   (matched on token subset, so ``ROMANIAN_DEADLIFT`` catches
   ``BARBELL_ROMANIAN_DEADLIFT``). More specific keys win.
2. **Category profile** — :data:`CATEGORY_PROFILES`, the FIT category default.
3. **Unmapped** — recorded as such so the UI can tell you what it ignored,
   rather than silently dropping volume.

Weights are *fractional credit*, not percentages of force: 1.0 = the muscle is
a prime mover, 0.5 = meaningful secondary involvement, 0.25 = minor/stabiliser.
Summing weight × sets gives "effective sets", the unit the volume model uses,
which is how training literature usually counts indirect work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .muscles import (
    ABS,
    ADDUCTORS,
    BICEPS,
    CALVES,
    CHEST,
    FOREARMS,
    FRONT_DELTS,
    GLUTES,
    HAMSTRINGS,
    HIP_FLEXORS,
    LATS,
    LOWER_BACK,
    OBLIQUES,
    QUADS,
    REAR_DELTS,
    SIDE_DELTS,
    TRICEPS,
    UPPER_BACK,
)

# Movement patterns, used for the pattern-coverage view.
H_PUSH = "horizontal_push"
V_PUSH = "vertical_push"
H_PULL = "horizontal_pull"
V_PULL = "vertical_pull"
SQUAT_P = "squat"
HINGE = "hinge"
LUNGE_P = "lunge"
CARRY_P = "carry"
CORE_P = "core"
ISOLATION = "isolation"
CONDITIONING = "conditioning"

PATTERNS = [H_PUSH, V_PUSH, H_PULL, V_PULL, SQUAT_P, HINGE, LUNGE_P, CARRY_P, CORE_P, ISOLATION]

# Categories that carry no meaningful resistance volume.
NON_LOADING_CATEGORIES = {"WARM_UP", "CARDIO", "RUN", "UNKNOWN", "REST", "BREATHING", "STRETCH"}


@dataclass(frozen=True)
class Profile:
    """A muscle-activation profile for an exercise or category."""

    muscles: dict[str, float]
    pattern: str = ISOLATION


def _p(pattern: str, **muscles: float) -> Profile:
    return Profile(muscles=dict(muscles), pattern=pattern)


# --- Category-level defaults (FIT exercise_category) -------------------------

CATEGORY_PROFILES: dict[str, Profile] = {
    "BENCH_PRESS": _p(H_PUSH, chest=1.0, front_delts=0.5, triceps=0.5),
    "PUSH_UP": _p(H_PUSH, chest=1.0, triceps=0.5, front_delts=0.5, abs=0.25),
    "FLYE": _p(ISOLATION, chest=1.0, front_delts=0.25),
    "SHOULDER_PRESS": _p(V_PUSH, front_delts=1.0, side_delts=0.5, triceps=0.5, upper_back=0.25),
    "TRICEPS_EXTENSION": _p(ISOLATION, triceps=1.0),
    "LATERAL_RAISE": _p(ISOLATION, side_delts=1.0, upper_back=0.25),
    "SHOULDER_STABILITY": _p(ISOLATION, rear_delts=1.0, upper_back=0.5, side_delts=0.25),
    "ROW": _p(H_PULL, upper_back=1.0, lats=0.5, biceps=0.5, rear_delts=0.5, lower_back=0.25),
    "PULL_UP": _p(V_PULL, lats=1.0, upper_back=0.5, biceps=0.5, forearms=0.25),
    "CURL": _p(ISOLATION, biceps=1.0, forearms=0.5),
    "SHRUG": _p(ISOLATION, upper_back=1.0, forearms=0.25),
    "DEADLIFT": _p(HINGE, hamstrings=1.0, glutes=1.0, lower_back=1.0, upper_back=0.5,
                   forearms=0.5, quads=0.25),
    "HYPEREXTENSION": _p(HINGE, lower_back=1.0, glutes=0.5, hamstrings=0.5),
    "HIP_SWING": _p(HINGE, glutes=1.0, hamstrings=0.5, lower_back=0.5, upper_back=0.25),
    "HIP_RAISE": _p(HINGE, glutes=1.0, hamstrings=0.5, lower_back=0.25),
    "HIP_STABILITY": _p(ISOLATION, glutes=1.0, adductors=0.5, obliques=0.25),
    "SQUAT": _p(SQUAT_P, quads=1.0, glutes=1.0, hamstrings=0.5, lower_back=0.25, adductors=0.25),
    "LUNGE": _p(LUNGE_P, quads=1.0, glutes=1.0, hamstrings=0.5, adductors=0.25),
    "LEG_CURL": _p(ISOLATION, hamstrings=1.0, calves=0.25),
    "CALF_RAISE": _p(ISOLATION, calves=1.0),
    "OLYMPIC_LIFT": _p(HINGE, quads=1.0, glutes=1.0, upper_back=0.5, hamstrings=0.5,
                       lower_back=0.5, side_delts=0.25, calves=0.25, forearms=0.25),
    "PLYO": _p(SQUAT_P, quads=1.0, glutes=0.5, calves=0.5),
    "CARRY": _p(CARRY_P, forearms=1.0, upper_back=0.5, obliques=0.5, abs=0.25),
    "CORE": _p(CORE_P, abs=1.0, obliques=0.5),
    "CRUNCH": _p(CORE_P, abs=1.0, hip_flexors=0.25),
    "SIT_UP": _p(CORE_P, abs=1.0, hip_flexors=0.5),
    "PLANK": _p(CORE_P, abs=1.0, obliques=0.5),
    "LEG_RAISE": _p(CORE_P, abs=1.0, hip_flexors=0.5),
    "CHOP": _p(CORE_P, obliques=1.0, abs=0.5),
    "TOTAL_BODY": _p(SQUAT_P, quads=0.5, glutes=0.5, chest=0.5, upper_back=0.5, abs=0.5,
                     front_delts=0.25, triceps=0.25, hamstrings=0.25),
}

# --- Name-level overrides ---------------------------------------------------
# Keys are token sets; a key matches when every token appears in the exercise
# name. The key with the most matched tokens wins, so add specific variants
# freely — they take precedence over shorter ones automatically.

NAME_PROFILES: dict[str, Profile] = {
    # Horizontal push
    "INCLINE_BENCH_PRESS": _p(H_PUSH, chest=1.0, front_delts=0.75, triceps=0.5),
    "DECLINE_BENCH_PRESS": _p(H_PUSH, chest=1.0, triceps=0.5, front_delts=0.25),
    "CLOSE_GRIP_BENCH_PRESS": _p(H_PUSH, triceps=1.0, chest=0.75, front_delts=0.5),
    "DIP": _p(H_PUSH, triceps=1.0, chest=0.75, front_delts=0.5),
    "BENCH_DIP": _p(ISOLATION, triceps=1.0, chest=0.25, front_delts=0.25),
    "PULLOVER": _p(ISOLATION, lats=1.0, chest=0.5, triceps=0.25),
    # Vertical push
    "ARNOLD_PRESS": _p(V_PUSH, front_delts=1.0, side_delts=0.75, triceps=0.5),
    "UPRIGHT_ROW": _p(ISOLATION, side_delts=1.0, upper_back=0.75, biceps=0.25),
    # Pull
    "REAR_DELT": _p(ISOLATION, rear_delts=1.0, upper_back=0.5),
    "REVERSE_FLYE": _p(ISOLATION, rear_delts=1.0, upper_back=0.5),
    "FACE_PULL": _p(ISOLATION, rear_delts=1.0, upper_back=0.75),
    "CHIN_UP": _p(V_PULL, lats=1.0, biceps=0.75, upper_back=0.5, forearms=0.25),
    "LAT_PULLDOWN": _p(V_PULL, lats=1.0, upper_back=0.5, biceps=0.5, forearms=0.25),
    "STRAIGHT_ARM_PULLDOWN": _p(ISOLATION, lats=1.0, triceps=0.25),
    "HAMMER_CURL": _p(ISOLATION, biceps=1.0, forearms=0.75),
    "PREACHER_CURL": _p(ISOLATION, biceps=1.0),
    "WRIST_CURL": _p(ISOLATION, forearms=1.0),
    "REVERSE_CURL": _p(ISOLATION, forearms=1.0, biceps=0.75),
    # Hinge
    "ROMANIAN_DEADLIFT": _p(HINGE, hamstrings=1.0, glutes=1.0, lower_back=0.75, forearms=0.5),
    "STIFF_LEG_DEADLIFT": _p(HINGE, hamstrings=1.0, glutes=0.75, lower_back=0.75, forearms=0.5),
    "SUMO_DEADLIFT": _p(HINGE, glutes=1.0, quads=0.75, adductors=0.75, hamstrings=0.75,
                        lower_back=0.75, upper_back=0.5, forearms=0.5),
    "GOOD_MORNING": _p(HINGE, hamstrings=1.0, lower_back=1.0, glutes=0.5),
    "HIP_THRUST": _p(HINGE, glutes=1.0, hamstrings=0.5),
    "NORDIC": _p(ISOLATION, hamstrings=1.0, calves=0.25),
    "KETTLEBELL_SWING": _p(HINGE, glutes=1.0, hamstrings=0.75, lower_back=0.5, upper_back=0.25),
    # Squat / legs
    "FRONT_SQUAT": _p(SQUAT_P, quads=1.0, glutes=0.75, abs=0.5, upper_back=0.5, lower_back=0.25),
    "HACK_SQUAT": _p(SQUAT_P, quads=1.0, glutes=0.5),
    "GOBLET_SQUAT": _p(SQUAT_P, quads=1.0, glutes=0.75, abs=0.5),
    "BULGARIAN_SPLIT_SQUAT": _p(LUNGE_P, quads=1.0, glutes=1.0, adductors=0.25, hamstrings=0.25),
    "SPLIT_SQUAT": _p(LUNGE_P, quads=1.0, glutes=0.75, hamstrings=0.25),
    "LEG_PRESS": _p(SQUAT_P, quads=1.0, glutes=0.5, hamstrings=0.25),
    "LEG_EXTENSION": _p(ISOLATION, quads=1.0),
    "STEP_UP": _p(LUNGE_P, quads=1.0, glutes=1.0, calves=0.25),
    "ADDUCTOR": _p(ISOLATION, adductors=1.0),
    "ABDUCTOR": _p(ISOLATION, glutes=1.0, adductors=0.25),
    "GLUTE_BRIDGE": _p(HINGE, glutes=1.0, hamstrings=0.5),
    # Core
    "SIDE_PLANK": _p(CORE_P, obliques=1.0, abs=0.5),
    "RUSSIAN_TWIST": _p(CORE_P, obliques=1.0, abs=0.5),
    "WOOD_CHOP": _p(CORE_P, obliques=1.0, abs=0.5),
    "PALLOF": _p(CORE_P, obliques=1.0, abs=0.5),
    "HANGING_LEG_RAISE": _p(CORE_P, abs=1.0, hip_flexors=0.5, forearms=0.25),
    "AB_WHEEL": _p(CORE_P, abs=1.0, lats=0.25),
    # Carry
    "FARMERS_CARRY": _p(CARRY_P, forearms=1.0, upper_back=0.5, obliques=0.5),
    "SUITCASE_CARRY": _p(CARRY_P, forearms=1.0, obliques=1.0, upper_back=0.5),
}

EQUIPMENT_TOKENS = [
    "BARBELL", "DUMBBELL", "KETTLEBELL", "CABLE", "MACHINE", "SMITH", "BAND",
    "BODYWEIGHT", "WEIGHTED", "ASSISTED", "TRAP_BAR", "EZ_BAR", "LANDMINE",
]

_TOKEN_SPLIT = re.compile(r"[^A-Z0-9]+")


def clean_label(value: Any) -> str:
    """Coerce a category or exercise name to a usable string, or ``""``.

    Real Garmin data has sets with no exercise recorded, and pandas represents a
    missing value in an otherwise-text column as ``float('nan')`` — which is
    *truthy*, so ``name or category`` happily passes it on and everything
    downstream explodes on ``.replace``. Anything that isn't a non-empty string
    becomes an empty string here, once, at the edge.
    """
    if value is None or isinstance(value, float):   # float covers NaN
        return ""
    try:
        if value != value:                          # any other NaN-alike
            return ""
    except TypeError:                               # pandas NA: not comparable
        return ""
    if not isinstance(value, str):
        value = str(value)
    text = value.strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def _tokens(text: Any) -> set[str]:
    text = clean_label(text)
    if not text:
        return set()
    return {t for t in _TOKEN_SPLIT.split(text.upper()) if t}


_NAME_KEY_TOKENS: list[tuple[str, set[str], Profile]] = sorted(
    ((key, _tokens(key), prof) for key, prof in NAME_PROFILES.items()),
    key=lambda item: len(item[1]),
    reverse=True,
)


@dataclass(frozen=True)
class Resolved:
    """The outcome of mapping one Garmin exercise to muscles."""

    muscles: dict[str, float] = field(default_factory=dict)
    pattern: str = ISOLATION
    # garmin_name | garmin_category | name | category | unmapped | non_loading
    source: str = "unmapped"
    display_name: str = ""
    equipment: str | None = None

    @property
    def from_garmin(self) -> bool:
        return self.source.startswith("garmin")

    @property
    def is_mapped(self) -> bool:
        return bool(self.muscles)

    def primary_muscles(self) -> list[str]:
        return [m for m, w in self.muscles.items() if w >= 1.0]


def prettify(name: Any) -> str:
    text = clean_label(name)
    if not text:
        return "Unknown exercise"
    return text.replace("_", " ").title()


def detect_equipment(name: Any) -> str | None:
    toks = _tokens(name)
    for eq in EQUIPMENT_TOKENS:
        if _tokens(eq) <= toks:
            return eq.replace("_", " ").title()
    return None


class GarminMuscleMap:
    """Garmin's own exercise → muscle assignments, keyed for fast lookup.

    Built from synced workout definitions and the exercise library. Lookups try
    the exact ``(category, name)`` pair first, then the category alone, so a
    named variant can be more precise than its category default.
    """

    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self._by_name: dict[tuple[str, str], dict[str, float]] = {}
        self._by_category: dict[str, dict[str, float]] = {}
        self.unmatched_names: dict[str, int] = {}
        for record in records or []:
            self.add(record)

    def add(self, record: dict[str, Any]) -> None:
        from .garmin_muscles import build_profile

        profile, unmatched = build_profile(
            record.get("primary_muscles"), record.get("secondary_muscles"))
        for name in unmatched:
            self.unmatched_names[name] = self.unmatched_names.get(name, 0) + 1
        if not profile:
            return

        category = clean_label(record.get("category")).upper()
        exercise = clean_label(record.get("exercise_name")).upper()
        if exercise:
            self._by_name[(category, exercise)] = profile
            # Also key on the name alone: the same exercise can be logged under a
            # different category than the workout declared it in.
            self._by_name[("", exercise)] = profile
        elif category:
            self._by_category[category] = profile

    def lookup(self, category: Any, name: Any) -> tuple[dict[str, float], str] | None:
        category, name = clean_label(category).upper(), clean_label(name).upper()
        for key in ((category, name), ("", name)):
            if key[1] and key in self._by_name:
                return self._by_name[key], "garmin_name"
        if category and category in self._by_category:
            return self._by_category[category], "garmin_category"
        return None

    def __len__(self) -> int:
        """Distinct exercises and categories covered.

        Counted rather than summing the key dicts, because a named exercise is
        stored under two keys so a category mismatch still resolves.
        """
        return len({key[1] for key in self._by_name if key[1]}) + len(self._by_category)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "named_exercises": len({k[1] for k in self._by_name if k[1]}),
            "categories": len(self._by_category),
            "unmatched_muscle_names": len(self.unmatched_names),
        }


# Module-level default so callers that don't thread a map through still benefit
# once one has been installed for the session.
_ACTIVE_GARMIN_MAP: GarminMuscleMap | None = None


def set_garmin_muscle_map(muscle_map: GarminMuscleMap | None) -> None:
    """Install the Garmin-derived map used by :func:`resolve` when none is passed."""
    global _ACTIVE_GARMIN_MAP
    _ACTIVE_GARMIN_MAP = muscle_map


def active_garmin_muscle_map() -> GarminMuscleMap | None:
    return _ACTIVE_GARMIN_MAP


def resolve(category: Any = None, name: Any = None,
            garmin_map: GarminMuscleMap | None = None) -> Resolved:
    """Map a Garmin (category, name) pair to a muscle-activation profile.

    Both inputs are cleaned first: they come straight off a DataFrame column, so
    a set logged without an exercise arrives as NaN rather than None.
    """
    category = clean_label(category)
    name = clean_label(name)
    cat = category.upper()
    display = prettify(name or category)
    equipment = detect_equipment(name)

    if cat in NON_LOADING_CATEGORIES and not name:
        return Resolved(pattern=CONDITIONING, source="non_loading", display_name=display)

    # 0. Garmin's own assignment wins — it knows what the exercise works.
    #    The movement pattern still comes from our tables, since Garmin doesn't
    #    classify movements that way.
    muscle_map = garmin_map if garmin_map is not None else _ACTIVE_GARMIN_MAP
    if muscle_map is not None:
        found = muscle_map.lookup(cat, (name or "").upper().strip())
        if found:
            profile, source = found
            return Resolved(dict(profile), _pattern_for(cat, name), source, display,
                            equipment)

    # 1. Most specific name match. Search the name and, failing that, the
    #    category — some Garmin exports put the detail in the category field.
    haystack = _tokens(name) | _tokens(category)
    for _key, key_tokens, prof in _NAME_KEY_TOKENS:
        if key_tokens <= haystack:
            return Resolved(dict(prof.muscles), prof.pattern, "name", display, equipment)

    # 2. Category default.
    prof = CATEGORY_PROFILES.get(cat)
    if prof:
        return Resolved(dict(prof.muscles), prof.pattern, "category", display, equipment)

    if cat in NON_LOADING_CATEGORIES:
        return Resolved(pattern=CONDITIONING, source="non_loading", display_name=display)

    # 3. Nothing matched — surfaced in the UI as an unmapped exercise.
    return Resolved(source="unmapped", display_name=display, equipment=equipment)


def _pattern_for(category: str, name: str | None) -> str:
    """Movement pattern from our own tables — Garmin doesn't classify movements."""
    haystack = _tokens(name) | _tokens(category)
    for _key, key_tokens, profile in _NAME_KEY_TOKENS:
        if key_tokens <= haystack:
            return profile.pattern
    profile = CATEGORY_PROFILES.get(category)
    return profile.pattern if profile else ISOLATION


def coverage_report(garmin_map: GarminMuscleMap | None = None) -> dict[str, int]:
    """Size of the mapping tables, shown in the app's diagnostics panel."""
    muscle_map = garmin_map if garmin_map is not None else _ACTIVE_GARMIN_MAP
    report = {
        "categories": len(CATEGORY_PROFILES),
        "named_variants": len(NAME_PROFILES),
        "muscles": len({m for p in CATEGORY_PROFILES.values() for m in p.muscles}),
        "garmin_entries": 0,
        "garmin_named": 0,
        "garmin_categories": 0,
    }
    if muscle_map is not None:
        report["garmin_entries"] = len(muscle_map)
        report["garmin_named"] = muscle_map.stats["named_exercises"]
        report["garmin_categories"] = muscle_map.stats["categories"]
    return report
