"""Muscle taxonomy used by the strength analytics.

Muscles are the unit of analysis: Garmin records *exercises*, and the balance
model needs to know which tissue each exercise actually loads. Regions and
movement patterns give us the coarser views (push/pull balance, upper/lower
split) without a second mapping table.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Muscle ids -------------------------------------------------------------

CHEST = "chest"
FRONT_DELTS = "front_delts"
SIDE_DELTS = "side_delts"
REAR_DELTS = "rear_delts"
LATS = "lats"
UPPER_BACK = "upper_back"
LOWER_BACK = "lower_back"
BICEPS = "biceps"
TRICEPS = "triceps"
FOREARMS = "forearms"
ABS = "abs"
OBLIQUES = "obliques"
GLUTES = "glutes"
QUADS = "quads"
HAMSTRINGS = "hamstrings"
ADDUCTORS = "adductors"
CALVES = "calves"
HIP_FLEXORS = "hip_flexors"


@dataclass(frozen=True)
class Muscle:
    id: str
    label: str
    region: str  # chest | shoulders | back | arms | core | legs
    chain: str  # push | pull | core | lower


MUSCLES: dict[str, Muscle] = {
    m.id: m
    for m in [
        Muscle(CHEST, "Chest", "chest", "push"),
        Muscle(FRONT_DELTS, "Front delts", "shoulders", "push"),
        Muscle(SIDE_DELTS, "Side delts", "shoulders", "push"),
        Muscle(REAR_DELTS, "Rear delts", "shoulders", "pull"),
        Muscle(LATS, "Lats", "back", "pull"),
        Muscle(UPPER_BACK, "Upper back / traps", "back", "pull"),
        Muscle(LOWER_BACK, "Lower back / erectors", "back", "pull"),
        Muscle(BICEPS, "Biceps", "arms", "pull"),
        Muscle(TRICEPS, "Triceps", "arms", "push"),
        Muscle(FOREARMS, "Forearms / grip", "arms", "pull"),
        Muscle(ABS, "Abs", "core", "core"),
        Muscle(OBLIQUES, "Obliques", "core", "core"),
        Muscle(HIP_FLEXORS, "Hip flexors", "core", "core"),
        Muscle(GLUTES, "Glutes", "legs", "lower"),
        Muscle(QUADS, "Quads", "legs", "lower"),
        Muscle(HAMSTRINGS, "Hamstrings", "legs", "lower"),
        Muscle(ADDUCTORS, "Adductors", "legs", "lower"),
        Muscle(CALVES, "Calves", "legs", "lower"),
    ]
}

MUSCLE_IDS: list[str] = list(MUSCLES)

REGIONS = ["chest", "shoulders", "back", "arms", "core", "legs"]

# Antagonist / structural pairs the balance model reports on. Each entry is
# (label, numerator muscles, denominator muscles, healthy ratio range).
BALANCE_PAIRS: list[tuple[str, list[str], list[str], tuple[float, float]]] = [
    ("Push : Pull", [CHEST, FRONT_DELTS, TRICEPS], [LATS, UPPER_BACK, BICEPS], (0.7, 1.3)),
    ("Quads : Hamstrings", [QUADS], [HAMSTRINGS], (0.8, 1.6)),
    ("Chest : Upper back", [CHEST], [UPPER_BACK, LATS], (0.6, 1.2)),
    ("Front : Rear delts", [FRONT_DELTS], [REAR_DELTS], (0.7, 2.0)),
    ("Upper : Lower body", [CHEST, LATS, UPPER_BACK, FRONT_DELTS, SIDE_DELTS, BICEPS, TRICEPS],
     [QUADS, HAMSTRINGS, GLUTES, CALVES], (0.7, 1.8)),
]


def label(muscle_id: str) -> str:
    m = MUSCLES.get(muscle_id)
    return m.label if m else muscle_id.replace("_", " ").title()


def region_of(muscle_id: str) -> str:
    m = MUSCLES.get(muscle_id)
    return m.region if m else "other"
