"""Translate Garmin's muscle names into this app's taxonomy.

Garmin's workout definitions and exercise library label muscles anatomically
(``BICEPS_BRACHII``, ``LATISSIMUS_DORSI``, ``GLUTEUS_MAXIMUS``), at a finer grain
than the 18-muscle model used for volume analysis. This module maps one onto the
other so Garmin's own assignment can take precedence over the curated table in
:mod:`.exercises`.

Matching is by anatomical keyword rather than exact string, because the naming
varies between the workout service, the exercise library and locales
(``DELTOID_ANTERIOR`` / ``FRONT_DELTS`` / ``ANTERIOR_DELTOID`` all appear). Names
that don't match anything are returned as unmatched rather than dropped, so the
UI can report exactly what Garmin sent that we couldn't place.
"""

from __future__ import annotations

import re

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

# (keyword, muscle id). Order matters: the first match wins, so more specific
# keywords must precede the general ones they contain — "deltoid_anterior" before
# "deltoid", "biceps_femoris" (a hamstring) before "biceps".
MUSCLE_KEYWORDS: list[tuple[str, str]] = [
    # Shoulders — the three heads are distinct muscles for balance purposes, so
    # an unqualified "deltoid" is deliberately left unmatched rather than guessed.
    ("deltoid anterior", FRONT_DELTS),
    ("anterior deltoid", FRONT_DELTS),
    ("front delt", FRONT_DELTS),
    ("deltoid lateral", SIDE_DELTS),
    ("lateral deltoid", SIDE_DELTS),
    ("deltoid medial", SIDE_DELTS),
    ("middle delt", SIDE_DELTS),
    ("side delt", SIDE_DELTS),
    ("deltoid posterior", REAR_DELTS),
    ("posterior deltoid", REAR_DELTS),
    ("rear delt", REAR_DELTS),
    # Chest
    ("pectoral", CHEST),
    ("pectoralis", CHEST),
    ("chest", CHEST),
    ("serratus", CHEST),
    # Back
    ("latissimus", LATS),
    ("lats", LATS),
    ("trapezius", UPPER_BACK),
    ("traps", UPPER_BACK),
    ("rhomboid", UPPER_BACK),
    ("teres", UPPER_BACK),
    ("infraspinatus", UPPER_BACK),
    ("supraspinatus", UPPER_BACK),
    ("levator scapulae", UPPER_BACK),
    ("upper back", UPPER_BACK),
    ("erector spinae", LOWER_BACK),
    ("erector", LOWER_BACK),
    ("multifidus", LOWER_BACK),
    ("quadratus lumborum", LOWER_BACK),
    ("lower back", LOWER_BACK),
    # Arms — biceps femoris is a hamstring, so it must be matched first.
    ("biceps femoris", HAMSTRINGS),
    ("biceps brachii", BICEPS),
    ("brachialis", BICEPS),
    ("biceps", BICEPS),
    ("triceps brachii", TRICEPS),
    ("triceps", TRICEPS),
    ("brachioradialis", FOREARMS),
    ("flexor carpi", FOREARMS),
    ("extensor carpi", FOREARMS),
    ("forearm", FOREARMS),
    ("wrist", FOREARMS),
    ("grip", FOREARMS),
    # Core
    ("rectus abdominis", ABS),
    ("transverse abdominis", ABS),
    ("abdominal", ABS),
    ("abs", ABS),
    ("oblique", OBLIQUES),
    ("iliopsoas", HIP_FLEXORS),
    ("psoas", HIP_FLEXORS),
    ("iliacus", HIP_FLEXORS),
    ("hip flexor", HIP_FLEXORS),
    ("tensor fasciae latae", HIP_FLEXORS),
    # Legs
    ("gluteus", GLUTES),
    ("glute", GLUTES),
    ("quadriceps", QUADS),
    ("rectus femoris", QUADS),
    ("vastus", QUADS),
    ("quads", QUADS),
    ("semitendinosus", HAMSTRINGS),
    ("semimembranosus", HAMSTRINGS),
    ("hamstring", HAMSTRINGS),
    ("adductor", ADDUCTORS),
    ("pectineus", ADDUCTORS),
    ("gracilis", ADDUCTORS),
    ("abductor", GLUTES),          # hip abduction is glute-medius work
    ("gastrocnemius", CALVES),
    ("soleus", CALVES),
    ("calf", CALVES),
    ("calves", CALVES),
]

# Garmin names we knowingly can't place, and why. Listed so the diagnostics panel
# can distinguish "we haven't handled this" from "this has no home in the model".
UNMAPPABLE_NOTES = {
    "deltoid": "unqualified deltoid — can't tell front from side from rear",
    "tibialis": "shin; the model has no anterior lower-leg muscle",
    "rotator cuff": "stabiliser group, not a volume target here",
    "neck": "not tracked",
    "sternocleidomastoid": "not tracked",
    "full body": "too diffuse to attribute",
    "cardiovascular": "not a muscle",
}

_SEPARATORS = re.compile(r"[_\-/,]+")


def _normalise(name: str) -> str:
    return _SEPARATORS.sub(" ", str(name).strip().lower())


def translate_one(name: str) -> str | None:
    """Map a single Garmin muscle name onto a muscle id, or None if unmatched."""
    text = _normalise(name)
    if not text:
        return None
    for keyword, muscle in MUSCLE_KEYWORDS:
        if keyword in text:
            return muscle
    return None


def translate(names: list[str] | None) -> tuple[set[str], list[str]]:
    """Translate a list of Garmin muscle names.

    Returns ``(matched muscle ids, unmatched original names)``.
    """
    matched: set[str] = set()
    unmatched: list[str] = []
    for name in names or []:
        muscle = translate_one(name)
        if muscle:
            matched.add(muscle)
        elif name:
            unmatched.append(str(name))
    return matched, unmatched


def build_profile(primary: list[str] | None, secondary: list[str] | None,
                  primary_weight: float = 1.0,
                  secondary_weight: float = 0.5) -> tuple[dict[str, float], list[str]]:
    """Turn Garmin's primary/secondary lists into weighted muscle credit.

    Garmin states which muscles an exercise works but not how much, so the same
    convention as the curated table is applied: primary movers get a full set,
    secondary involvement half of one. A muscle listed in both takes the primary
    weight.
    """
    primary_ids, unmatched_primary = translate(primary)
    secondary_ids, unmatched_secondary = translate(secondary)

    profile: dict[str, float] = {muscle: secondary_weight for muscle in secondary_ids}
    profile.update({muscle: primary_weight for muscle in primary_ids})
    return profile, unmatched_primary + unmatched_secondary


def unmappable_reason(name: str) -> str | None:
    text = _normalise(name)
    for keyword, note in UNMAPPABLE_NOTES.items():
        if keyword in text:
            return note
    return None
