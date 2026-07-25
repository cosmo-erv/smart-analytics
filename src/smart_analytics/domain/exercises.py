"""Garmin exercise → muscle mapping.

Garmin's strength activities expose each set as an *exercise category* (the FIT
`exercise_category` enum, e.g. ``BENCH_PRESS``) plus an optional, more specific
*exercise name* (e.g. ``INCLINE_DUMBBELL_BENCH_PRESS``). Neither says anything
about muscles, so this module supplies that layer.

Resolution order for a set:

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


def _tokens(text: str | None) -> set[str]:
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
    source: str = "unmapped"  # name | category | unmapped | non_loading
    display_name: str = ""
    equipment: str | None = None

    @property
    def is_mapped(self) -> bool:
        return bool(self.muscles)

    def primary_muscles(self) -> list[str]:
        return [m for m, w in self.muscles.items() if w >= 1.0]


def prettify(name: str | None) -> str:
    if not name:
        return "Unknown exercise"
    return name.replace("_", " ").title()


def detect_equipment(name: str | None) -> str | None:
    toks = _tokens(name)
    for eq in EQUIPMENT_TOKENS:
        if _tokens(eq) <= toks:
            return eq.replace("_", " ").title()
    return None


def resolve(category: str | None, name: str | None = None) -> Resolved:
    """Map a Garmin (category, name) pair to a muscle-activation profile."""
    cat = (category or "").upper().strip()
    display = prettify(name or category)
    equipment = detect_equipment(name)

    if cat in NON_LOADING_CATEGORIES and not name:
        return Resolved(pattern=CONDITIONING, source="non_loading", display_name=display)

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


def coverage_report() -> dict[str, int]:
    """Size of the mapping tables, shown in the app's diagnostics panel."""
    return {
        "categories": len(CATEGORY_PROFILES),
        "named_variants": len(NAME_PROFILES),
        "muscles": len({m for p in CATEGORY_PROFILES.values() for m in p.muscles}),
    }
