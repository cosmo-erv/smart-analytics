"""Fetch and normalise the physiological metrics Garmin already computes.

Garmin's own longitudinal model is better than anything derivable from activity
summaries — it knows your lactate threshold, configured heart-rate zones, VO2max,
FTP, race predictions, training status and daily readiness. This module pulls
those instead of re-deriving them, and reserves local computation for what Garmin
genuinely doesn't do (muscle-level analysis, hybrid interference, split analysis).

**These payloads are the least stable part of the Garmin surface.** Shapes differ
by device generation, account age and which features the watch supports, and keys
move between releases. So rather than indexing fixed paths, the normalisers here
search the payload for candidate keys at any depth (:func:`deep_find`) and treat
every field as optional. A missing metric produces ``None`` and is reported as
unavailable in the UI — never a crash, and never a fabricated number.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date
from typing import Any

log = logging.getLogger(__name__)

# Garmin personal-record type ids → what they actually measure. Values are
# (label, unit); time records are in seconds, distance records in metres.
PR_TYPES: dict[int, tuple[str, str]] = {
    1: ("1 km best time", "s"),
    2: ("1 mile best time", "s"),
    3: ("5 km best time", "s"),
    4: ("10 km best time", "s"),
    5: ("Longest run", "m"),
    6: ("Longest ride", "m"),
    7: ("Longest ride", "m"),
    8: ("Biggest ascent", "m"),
    9: ("Best average power", "w"),
    10: ("Half marathon best time", "s"),
    11: ("Marathon best time", "s"),
    12: ("Most steps in a day", "steps"),
    13: ("Most steps in a week", "steps"),
    14: ("Most steps in a month", "steps"),
    15: ("Longest goal streak", "days"),
    16: ("Most distance in a day", "m"),
}

# Race-prediction keys → distance in metres.
PREDICTION_DISTANCES: dict[str, float] = {
    "time5K": 5_000.0,
    "time10K": 10_000.0,
    "timeHalfMarathon": 21_097.0,
    "timeMarathon": 42_195.0,
}

TIME_UNIT_RECORDS = {"s"}


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out == 0:  # NaN, and Garmin's zero-as-missing convention
        return None
    return out


def deep_find(payload: Any, keys: Iterable[str], _depth: int = 0) -> Any:
    """Return the first value found for any of ``keys``, searched breadth-first.

    Garmin nests the same field at different depths depending on the endpoint and
    device (``trainingStatus`` may sit at the root, under
    ``mostRecentTrainingStatus``, or under a per-device id map). Searching by key
    name is more durable than hard-coding those paths.
    """
    if _depth > 8 or payload is None:
        return None
    wanted = list(keys)

    if isinstance(payload, dict):
        for key in wanted:
            if key in payload and payload[key] is not None:
                return payload[key]
        for value in payload.values():
            found = deep_find(value, wanted, _depth + 1)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = deep_find(item, wanted, _depth + 1)
            if found is not None:
                return found
    return None


# --- normalisers ------------------------------------------------------------

def normalise_max_metrics(payload: Any, day: date) -> dict[str, Any]:
    """VO2max for running and cycling, plus fitness age."""
    running = _num(deep_find(payload, ["vo2MaxPreciseValue", "vo2MaxValue"]))
    cycling = None
    fitness_age = _num(deep_find(payload, ["fitnessAge", "chronologicalAge"]))

    # The cycling estimate lives under its own key when the account has one.
    entries = payload if isinstance(payload, list) else [payload]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cycling_block = entry.get("cycling")
        if isinstance(cycling_block, dict):
            cycling = _num(deep_find(cycling_block, ["vo2MaxPreciseValue", "vo2MaxValue"]))
            break

    return {"local_date": day.isoformat(), "vo2max_running": running,
            "vo2max_cycling": cycling, "fitness_age": fitness_age}


def normalise_lactate_threshold(payload: Any, day: date) -> dict[str, Any]:
    """Threshold heart rate and speed — the anchor for every training zone."""
    hr = _num(deep_find(payload, [
        "lactateThresholdHeartRate", "lactateThresholdHeartRateValue",
        "heartRate", "lactateThresholdBpm",
    ]))
    speed = _num(deep_find(payload, [
        "lactateThresholdSpeed", "lactateThresholdSpeedValue", "speed",
    ]))
    # Garmin sometimes reports threshold pace as seconds per km instead of m/s.
    if speed is not None and speed > 30:
        speed = 1000.0 / speed
    return {"local_date": day.isoformat(), "lt_hr": hr, "lt_speed_mps": speed}


def normalise_training_status(payload: Any, day: date) -> dict[str, Any]:
    """Training status plus Garmin's own acute/chronic load and target range."""
    status = deep_find(payload, ["trainingStatusFeedbackPhrase", "trainingStatus"])
    if isinstance(status, (int, float)):
        status = _STATUS_CODES.get(int(status), f"status {int(status)}")
    if isinstance(status, str):
        # "PRODUCTIVE_1" / "MAINTAINING_2" → "Productive"
        status = status.split("_")[0].replace("-", " ").title()

    note = deep_find(payload, ["trainingStatusFeedbackPhrase", "feedbackLong", "feedbackShort"])
    if isinstance(note, str):
        note = note.replace("_", " ").title()

    return {
        "local_date": day.isoformat(),
        "training_status": status if isinstance(status, str) else None,
        "training_status_note": note if isinstance(note, str) else None,
        "acute_load": _num(deep_find(payload, [
            "acuteTrainingLoad", "dailyTrainingLoadAcute", "acwrAcute"])),
        "chronic_load": _num(deep_find(payload, [
            "dailyTrainingLoadChronic", "chronicTrainingLoad", "acwrChronic"])),
        "load_ratio": _num(deep_find(payload, [
            "dailyAcuteChronicWorkloadRatio", "acwrPercent", "acuteChronicWorkloadRatio"])),
        "load_target_low": _num(deep_find(payload, [
            "minTrainingLoadChronic", "loadTargetLow", "minLoadTarget"])),
        "load_target_high": _num(deep_find(payload, [
            "maxTrainingLoadChronic", "loadTargetHigh", "maxLoadTarget"])),
        "vo2max_running": _num(deep_find(payload, ["vo2MaxPreciseValue", "vo2MaxValue"])),
    }


_STATUS_CODES = {
    0: "No status", 1: "Detraining", 2: "Recovery", 3: "Maintaining",
    4: "Productive", 5: "Peaking", 6: "Overreaching", 7: "Unproductive",
    8: "Strained",
}


def normalise_training_readiness(payload: Any, day: date) -> dict[str, Any]:
    """Daily readiness score, level and remaining recovery time."""
    score = _num(deep_find(payload, ["score", "trainingReadinessScore"]))
    level = deep_find(payload, ["level", "trainingReadinessLevel"])
    if isinstance(level, str):
        level = level.replace("_", " ").title()
    recovery_minutes = _num(deep_find(payload, ["recoveryTime", "recoveryTimeMinutes"]))
    return {
        "local_date": day.isoformat(),
        "readiness_score": score,
        "readiness_level": level if isinstance(level, str) else None,
        "recovery_time_h": round(recovery_minutes / 60, 1) if recovery_minutes else None,
    }


def normalise_scored_metric(payload: Any, day: date, field: str,
                            keys: list[str]) -> dict[str, Any]:
    """Endurance score, hill score and similar single-value daily metrics."""
    return {"local_date": day.isoformat(), field: _num(deep_find(payload, keys))}


def normalise_ftp(payload: Any, day: date) -> dict[str, Any]:
    return {"local_date": day.isoformat(),
            "ftp_watts": _num(deep_find(payload, [
                "functionalThresholdPower", "ftpValue", "ftp", "cyclingFtp"]))}


def normalise_race_predictions(payload: Any) -> list[dict[str, Any]]:
    """Garmin's predicted times, kept per day so improvement is visible."""
    entries = payload if isinstance(payload, list) else [payload]
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        day = (entry.get("calendarDate") or entry.get("date")
               or entry.get("fromCalendarDate") or date.today().isoformat())
        for key, distance in PREDICTION_DISTANCES.items():
            seconds = _num(entry.get(key))
            if seconds:
                rows.append({"local_date": str(day)[:10], "distance_m": distance,
                             "predicted_time_s": seconds})
    return rows


def normalise_personal_records(payload: Any) -> list[dict[str, Any]]:
    entries = payload if isinstance(payload, list) else [payload]
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        type_id = entry.get("typeId")
        value = _num(entry.get("value"))
        if type_id is None or value is None:
            continue
        label, unit = PR_TYPES.get(int(type_id), (f"Record type {type_id}", ""))
        achieved = (entry.get("prStartTimeGmtFormatted") or entry.get("prStartTimeGmt")
                    or entry.get("prTypeLabelKey") or "")
        rows.append({
            "record_id": str(entry.get("id") or f"{type_id}-{achieved}"),
            "label": label,
            "activity_type": entry.get("activityType"),
            "value": value,
            "unit": unit,
            "achieved_on": str(achieved)[:10] or None,
            "raw_json": None,
        })
    return rows


def normalise_hr_zones(payload: Any) -> list[dict[str, Any]]:
    """Zone boundaries from either the profile settings or an activity's zone report.

    Two shapes are handled: the profile's ``zone1Floor``-style fields, and the
    per-activity ``[{zoneNumber, zoneLowBoundary}, …]`` list. The latter is the
    more reliable source, since every recorded activity carries it.
    """
    zones: list[dict[str, Any]] = []

    entries = payload if isinstance(payload, list) else None
    if entries and all(isinstance(e, dict) for e in entries):
        boundaries = []
        for entry in entries:
            number = entry.get("zoneNumber") or entry.get("zone")
            floor = _num(entry.get("zoneLowBoundary") or entry.get("floor")
                         or entry.get("secsInZoneLowBoundary"))
            if number is not None and floor is not None:
                boundaries.append((int(number), floor))
        boundaries.sort()
        for index, (number, floor) in enumerate(boundaries):
            ceiling = boundaries[index + 1][1] if index + 1 < len(boundaries) else None
            zones.append({"zone": number, "floor_bpm": floor, "ceiling_bpm": ceiling})
        if zones:
            return zones

    if isinstance(payload, dict):
        for number in range(1, 6):
            floor = _num(deep_find(payload, [f"zone{number}Floor", f"heartRateZone{number}Floor"]))
            ceiling = _num(deep_find(payload, [f"zone{number}Ceiling",
                                               f"heartRateZone{number}Ceiling"]))
            if floor is not None:
                zones.append({"zone": number, "floor_bpm": floor, "ceiling_bpm": ceiling})
    return zones


def normalise_splits(activity_id: str, activity_date: str,
                     payload: Any) -> list[dict[str, Any]]:
    """Per-lap rows from ``lapDTOs`` (or whichever key this account returns)."""
    laps = None
    if isinstance(payload, dict):
        for key in ("lapDTOs", "splits", "splitSummaries", "typedSplits"):
            candidate = payload.get(key)
            if isinstance(candidate, list) and candidate:
                laps = candidate
                break
    elif isinstance(payload, list):
        laps = payload
    if not laps:
        return []

    rows: list[dict[str, Any]] = []
    for index, lap in enumerate(laps):
        if not isinstance(lap, dict):
            continue
        rows.append({
            "activity_id": str(activity_id),
            "split_index": int(lap.get("lapIndex") or index + 1),
            "split_type": (lap.get("intensityType") or lap.get("type")
                           or lap.get("splitType") or "LAP"),
            "local_date": activity_date,
            "distance_m": _num(lap.get("distance")),
            "duration_s": _num(lap.get("duration")),
            "moving_s": _num(lap.get("movingDuration") or lap.get("duration")),
            "avg_hr": _num(lap.get("averageHR") or lap.get("avgHr")),
            "max_hr": _num(lap.get("maxHR") or lap.get("maxHr")),
            "avg_speed_mps": _num(lap.get("averageSpeed") or lap.get("avgSpeed")),
            "avg_cadence": _num(lap.get("averageRunCadence")
                                or lap.get("averageBikingCadenceInRevPerMinute")
                                or lap.get("averageCadence")),
            "avg_power": _num(lap.get("averagePower") or lap.get("avgPower")),
            "elevation_gain_m": _num(lap.get("elevationGain")),
            "elevation_loss_m": _num(lap.get("elevationLoss")),
        })
    return rows


def merge_metric_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse several partial daily rows into one row per date."""
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        day = row.get("local_date")
        if not day:
            continue
        target = merged.setdefault(day, {"local_date": day})
        for key, value in row.items():
            if key != "local_date" and value is not None:
                target[key] = value
    return list(merged.values())
