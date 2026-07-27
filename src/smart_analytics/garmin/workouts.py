"""Structured-workout ingest, and Garmin's own exercise → muscle assignments.

If you follow a structured workout, Garmin's workout definition is a richer
source than the recorded sets alone: it names each exercise, carries the planned
reps and load, and — for strength workouts — states the **muscle groups Garmin
assigns to that exercise**. That assignment is authoritative in a way a curated
mapping can't be, so it takes precedence (see
:class:`~smart_analytics.domain.exercises.GarminMuscleMap`).

Two shape problems are handled here:

* **Steps nest.** A workout is segments → steps, and a repeat group is itself a
  step containing more steps, to arbitrary depth. Everything is flattened.
* **Muscle fields vary.** ``primaryMuscles`` may be a list of strings, a list of
  dicts with a name key, or absent entirely; weights may be kilograms, pounds or
  grams. Each is detected rather than assumed, and anything unrecognised is
  reported rather than guessed at.
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

# Endpoints that have been used to serve Garmin's exercise library. None is
# documented or stable, so each is tried in turn and the first usable response
# wins; if they all fail we simply fall back to the curated mapping.
EXERCISE_LIBRARY_PATHS = [
    "/workout-service/exercise/categories",
    "/workout-service/exercises",
    "/web-gateway/rest/exercise-library",
    "/exercise-service/exercises",
]

MUSCLE_KEYS_PRIMARY = ("primaryMuscles", "primaryMuscleGroups", "primaryMuscle",
                       "muscleGroups", "primary")
MUSCLE_KEYS_SECONDARY = ("secondaryMuscles", "secondaryMuscleGroups",
                         "secondaryMuscle", "secondary")

POUNDS_TO_KG = 0.45359237


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out or out == 0 else out


def _muscle_names(payload: Any) -> list[str]:
    """Coerce a muscle field into a flat list of names.

    Accepts ``["PECTORALIS_MAJOR"]``, ``[{"name": "..."}]``, a single string, or a
    comma-separated string — all of which appear across Garmin's services.
    """
    if payload is None:
        return []
    if isinstance(payload, str):
        return [part.strip() for part in payload.split(",") if part.strip()]
    if isinstance(payload, dict):
        for key in ("name", "muscleName", "key", "displayName"):
            if payload.get(key):
                return [str(payload[key])]
        return []
    if isinstance(payload, list):
        names: list[str] = []
        for item in payload:
            names.extend(_muscle_names(item))
        return names
    return []


def _extract_muscles(step: dict[str, Any]) -> tuple[list[str], list[str]]:
    primary: list[str] = []
    secondary: list[str] = []
    for key in MUSCLE_KEYS_PRIMARY:
        if step.get(key) is not None:
            primary = _muscle_names(step[key])
            if primary:
                break
    for key in MUSCLE_KEYS_SECONDARY:
        if step.get(key) is not None:
            secondary = _muscle_names(step[key])
            if secondary:
                break
    return primary, secondary


def _weight_kg(step: dict[str, Any]) -> float | None:
    """Planned load in kilograms, whatever unit Garmin expressed it in."""
    value = _num(step.get("weightValue") or step.get("weight"))
    if value is None:
        return None
    unit = step.get("weightUnit")
    unit_key = ""
    if isinstance(unit, dict):
        unit_key = str(unit.get("unitKey") or unit.get("key") or "").lower()
    elif isinstance(unit, str):
        unit_key = unit.lower()

    if "pound" in unit_key or unit_key == "lb":
        return round(value * POUNDS_TO_KG, 2)
    if "gram" in unit_key and "kilo" not in unit_key:
        return round(value / 1000, 2)
    # No usable unit: values in the thousands are grams, as elsewhere in the API.
    if not unit_key and value > 1000:
        return round(value / 1000, 2)
    return round(value, 2)


def _target_reps(step: dict[str, Any]) -> int | None:
    """Planned repetitions for a step.

    A strength step normally expresses reps as ``endCondition: reps`` plus
    ``endConditionValue``; ``numberOfIterations`` is the fallback, and is what a
    repeat group uses for its set count.
    """
    condition = step.get("endCondition")
    condition_key = ""
    if isinstance(condition, dict):
        condition_key = str(condition.get("conditionTypeKey")
                            or condition.get("key") or "").lower()
    elif isinstance(condition, str):
        condition_key = condition.lower()
    if "rep" in condition_key:
        value = _num(step.get("endConditionValue"))
        if value:
            return int(value)

    for key in ("numberOfIterations", "repeatValue", "reps"):
        value = _num(step.get(key))
        if value:
            return int(value)
    return None


def _flatten_steps(container: Any, out: list[dict[str, Any]], depth: int = 0) -> None:
    """Walk segments and nested repeat groups, collecting executable steps."""
    if depth > 8 or container is None:
        return
    if isinstance(container, list):
        for item in container:
            _flatten_steps(item, out, depth + 1)
        return
    if not isinstance(container, dict):
        return

    # A repeat group carries its children under the same key name.
    children = container.get("workoutSteps")
    if isinstance(children, list) and children:
        _flatten_steps(children, out, depth + 1)

    for key in ("workoutSegments", "segments", "steps"):
        if isinstance(container.get(key), list):
            _flatten_steps(container[key], out, depth + 1)

    # An executable step is one that names an exercise.
    category = container.get("category") or container.get("exerciseCategory")
    exercise = container.get("exerciseName") or container.get("exercise")
    if category or exercise:
        out.append(container)


def normalise_workout(payload: Any) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Split a workout payload into its summary row and its flattened steps."""
    if not isinstance(payload, dict):
        return None, []

    workout_id = payload.get("workoutId") or payload.get("id")
    if workout_id is None:
        return None, []

    sport = payload.get("sportType")
    sport_key = None
    if isinstance(sport, dict):
        sport_key = sport.get("sportTypeKey") or sport.get("key")
    elif isinstance(sport, str):
        sport_key = sport

    raw_steps: list[dict[str, Any]] = []
    _flatten_steps(payload, raw_steps)

    steps: list[dict[str, Any]] = []
    for index, step in enumerate(raw_steps):
        primary, secondary = _extract_muscles(step)
        category = step.get("category") or step.get("exerciseCategory")
        exercise = step.get("exerciseName") or step.get("exercise")
        steps.append({
            "workout_id": str(workout_id),
            "step_index": int(step.get("stepOrder") or index + 1),
            "category": str(category).upper() if category else None,
            "exercise_name": str(exercise).upper() if exercise else None,
            "target_reps": _target_reps(step),
            "target_weight_kg": _weight_kg(step),
            "primary_muscles": json.dumps(primary) if primary else None,
            "secondary_muscles": json.dumps(secondary) if secondary else None,
        })

    summary = {
        "workout_id": str(workout_id),
        "name": payload.get("workoutName") or payload.get("name"),
        "sport": sport_key,
        "updated_at": str(payload.get("updateDate") or payload.get("updatedDate")
                          or payload.get("createDate") or "")[:19] or None,
        "step_count": len(steps),
        "raw_json": json.dumps(payload, default=str),
    }
    return summary, steps


def muscle_records_from_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract reusable (category, name) → muscle entries from workout steps."""
    records: list[dict[str, Any]] = []
    for step in steps:
        primary = json.loads(step["primary_muscles"]) if step.get("primary_muscles") else []
        secondary = (json.loads(step["secondary_muscles"])
                     if step.get("secondary_muscles") else [])
        if not primary and not secondary:
            continue
        records.append({
            "category": step.get("category") or "",
            "exercise_name": step.get("exercise_name") or "",
            "primary_muscles": primary,
            "secondary_muscles": secondary,
            "source": "workout",
        })
    return records


def normalise_exercise_library(payload: Any) -> list[dict[str, Any]]:
    """Pull (category, name) → muscle entries out of an exercise-library payload.

    The library's shape is undocumented and differs between endpoints, so this
    walks the structure looking for objects that carry both an exercise
    identifier and a muscle field, rather than assuming a layout.
    """
    records: list[dict[str, Any]] = []

    def visit(node: Any, category: str | None, depth: int = 0) -> None:
        if depth > 8 or node is None:
            return
        if isinstance(node, list):
            for item in node:
                visit(item, category, depth + 1)
            return
        if not isinstance(node, dict):
            return

        local_category = (node.get("category") or node.get("categoryKey")
                          or node.get("exerciseCategory") or category)
        primary, secondary = _extract_muscles(node)
        name = (node.get("exerciseName") or node.get("name") or node.get("key")
                or node.get("exercise"))

        if (primary or secondary) and (name or local_category):
            records.append({
                "category": str(local_category).upper() if local_category else "",
                "exercise_name": str(name).upper() if name else "",
                "primary_muscles": primary,
                "secondary_muscles": secondary,
                "source": "exercise_library",
            })

        for value in node.values():
            if isinstance(value, (dict, list)):
                visit(value, str(local_category) if local_category else None, depth + 1)

    visit(payload, None)

    # De-duplicate, preferring entries that name a specific exercise.
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        deduped[(record["category"], record["exercise_name"])] = record
    return list(deduped.values())
