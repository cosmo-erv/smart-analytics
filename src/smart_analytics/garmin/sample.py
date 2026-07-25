"""Synthetic Garmin data for demo mode and tests.

This emits payloads shaped like Garmin Connect's own responses and exposes the
same method surface as :class:`~smart_analytics.garmin.client.GarminClient`, so
``sync()`` drives it through the identical normalisation path — demo mode
exercises real code rather than a parallel shortcut.

The generated athlete has deliberate, findable problems, so the analytics and the
coaching layer have something true to say:

* hamstrings and rear delts are trained far less than their antagonists;
* rowing strength has been flat for months while bench and squat progress;
* running volume sits in the moderate "grey zone" instead of polarised easy/hard;
* cadence is low and efficiency factor plateaus in the final block.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np

DEFAULT_DAYS = 400
_SEED = 7


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


class SampleGarminClient:
    """Drop-in stand-in for :class:`GarminClient` backed by generated data."""

    def __init__(self, days: int = DEFAULT_DAYS, end: date | None = None,
                 seed: int = _SEED) -> None:
        self.days = days
        self.end = end or date.today()
        self.rng = np.random.default_rng(seed)
        self._activities: list[dict[str, Any]] = []
        self._sets: dict[str, list[dict[str, Any]]] = {}
        # activity_id -> (distance_m, duration_s, avg_hr, plan), used to build splits
        self._run_params: dict[str, tuple[float, float, float, str]] = {}
        self._build()

    # --- GarminClient interface -------------------------------------------

    def connect(self) -> "SampleGarminClient":
        return self

    def display_name(self) -> str:
        return "Demo Athlete"

    def iter_activities(self, since: date | None = None, page_size: int = 50,
                        max_activities: int | None = None) -> Iterator[dict[str, Any]]:
        count = 0
        for raw in self._activities:  # already newest-first
            started = datetime.strptime(raw["startTimeLocal"], "%Y-%m-%d %H:%M:%S").date()
            if since and started < since:
                return
            yield raw
            count += 1
            if max_activities and count >= max_activities:
                return

    def exercise_sets(self, activity_id: str) -> list[dict[str, Any]]:
        return self._sets.get(str(activity_id), [])

    # --- Garmin's own physiological model (demo equivalents) ----------------

    def physiology_snapshot(self, day: date) -> dict[str, Any]:
        age_days = max((self.end - day).days, 0)
        fitness = 1 - min(age_days / 300, 1.0)
        ramp = max(0.0, 1 - age_days / 90)
        status = ("Productive" if fitness > 0.55 else "Maintaining")
        if ramp > 0.85:
            status = "Overreaching"
        return {
            "local_date": day.isoformat(),
            "vo2max_running": round(46 + 4.5 * fitness, 1),
            "training_status": status,
            "training_status_note": f"{status} — demo data",
            "acute_load": round(380 + 180 * ramp + float(self.rng.normal(0, 30)), 0),
            "chronic_load": round(460 + 40 * fitness, 0),
            "load_ratio": round(0.85 + 0.35 * ramp + float(self.rng.normal(0, 0.06)), 2),
            "load_target_low": 400.0,
            "load_target_high": 650.0,
            "readiness_score": round(float(np.clip(72 - 18 * ramp + self.rng.normal(0, 8),
                                                   10, 99)), 0),
            "readiness_level": "Ready" if ramp < 0.6 else "Low",
            "recovery_time_h": round(float(np.clip(8 + 22 * ramp + self.rng.normal(0, 4),
                                                    0, 60)), 1),
        }

    def latest_thresholds(self, day: date | None = None) -> dict[str, Any]:
        day = day or self.end
        return {
            "local_date": day.isoformat(),
            # 4:52/km threshold pace, LT HR 168 — the anchors the zone model uses.
            "lt_hr": 168.0,
            "lt_speed_mps": 3.42,
            "ftp_watts": 232.0,
            "endurance_score": 6420.0,
            "hill_score": 58.0,
            "fitness_age": 31.0,
            "running_tolerance_km": 52.0,
        }

    def race_predictions(self) -> list[dict[str, Any]]:
        rows = []
        for weeks_ago in range(0, 40, 4):
            day = self.end - timedelta(weeks=weeks_ago)
            decay = 1 + weeks_ago * 0.0035  # predictions were slower further back
            for distance, base in ((5_000.0, 1195.0), (10_000.0, 2490.0),
                                   (21_097.0, 5510.0), (42_195.0, 11_640.0)):
                rows.append({"local_date": day.isoformat(), "distance_m": distance,
                             "predicted_time_s": round(base * decay, 0)})
        return rows

    def personal_records(self) -> list[dict[str, Any]]:
        return [
            {"record_id": "pr-5k", "label": "5 km best time", "activity_type": "running",
             "value": 1204.0, "unit": "s",
             "achieved_on": (self.end - timedelta(days=142)).isoformat(), "raw_json": None},
            {"record_id": "pr-10k", "label": "10 km best time", "activity_type": "running",
             "value": 2571.0, "unit": "s",
             "achieved_on": (self.end - timedelta(days=96)).isoformat(), "raw_json": None},
            {"record_id": "pr-long", "label": "Longest run", "activity_type": "running",
             "value": 24_100.0, "unit": "m",
             "achieved_on": (self.end - timedelta(days=61)).isoformat(), "raw_json": None},
        ]

    def hr_zones(self, reference_activity_id: str | None = None) -> list[dict[str, Any]]:
        floors = [94.0, 121.0, 141.0, 161.0, 175.0]
        return [{"zone": index + 1, "floor_bpm": floor,
                 "ceiling_bpm": floors[index + 1] if index + 1 < len(floors) else None}
                for index, floor in enumerate(floors)]

    def splits(self, activity_id: str) -> dict[str, Any]:
        """Per-km laps, with deliberate heart-rate drift on the long runs."""
        params = self._run_params.get(str(activity_id))
        if not params:
            return {}
        rng = np.random.default_rng(int(activity_id) % 100_000)
        distance_m, duration_s, avg_hr, plan = params

        if plan == "quality":
            return {"lapDTOs": self._interval_laps(distance_m, duration_s, avg_hr, rng)}

        laps = []
        full_km = max(int(distance_m // 1000), 1)
        pace = duration_s / (distance_m / 1000)
        # Long runs drift ~7% on heart rate by the end at the same pace — that is
        # aerobic decoupling, and it only shows up in split data.
        drift = 0.07 if plan == "long" else 0.02
        for km in range(full_km):
            share = km / max(full_km - 1, 1)
            laps.append({
                "lapIndex": km + 1,
                "distance": 1000.0,
                "duration": round(pace * float(rng.normal(1.0, 0.02)), 1),
                "averageHR": round(avg_hr * (1 - drift / 2 + drift * share)
                                   + float(rng.normal(0, 1.5)), 0),
                "maxHR": round(avg_hr * (1 + drift) + 6, 0),
                "averageSpeed": round(1000.0 / pace * float(rng.normal(1.0, 0.015)), 3),
                "averageRunCadence": round(float(rng.normal(163, 2)), 1),
                "elevationGain": round(float(abs(rng.normal(6, 4))), 1),
                "intensityType": "ACTIVE",
            })
        remainder = distance_m - full_km * 1000
        if remainder > 150:
            laps.append({
                "lapIndex": full_km + 1, "distance": round(remainder, 1),
                "duration": round(pace * remainder / 1000, 1),
                "averageHR": round(avg_hr * (1 + drift / 2), 0),
                "averageSpeed": round(1000.0 / pace, 3),
                "averageRunCadence": round(float(rng.normal(163, 2)), 1),
                "intensityType": "ACTIVE",
            })
        return {"lapDTOs": laps}

    def _interval_laps(self, distance_m: float, duration_s: float, avg_hr: float,
                       rng) -> list[dict[str, Any]]:
        """Warm-up, alternating work/recovery reps, cool-down."""
        laps = [{
            "lapIndex": 1, "distance": 1500.0, "duration": 480.0,
            "averageHR": round(avg_hr * 0.82, 0), "averageSpeed": 3.13,
            "averageRunCadence": 160.0, "intensityType": "WARMUP",
        }]
        reps = 6
        rep_distance = max((distance_m - 3000) / reps / 1.6, 400)
        index = 2
        for rep in range(reps):
            # Reps fade slightly through the session — pacing, not fitness.
            fade = 1 + rep * 0.012
            laps.append({
                "lapIndex": index, "distance": round(rep_distance, 0),
                "duration": round(rep_distance / (4.05 / fade), 1),
                "averageHR": round(avg_hr * (1.02 + rep * 0.004), 0),
                "averageSpeed": round(4.05 / fade, 3),
                "averageRunCadence": round(float(rng.normal(178, 2)), 1),
                "intensityType": "INTERVAL",
            })
            index += 1
            laps.append({
                "lapIndex": index, "distance": 200.0, "duration": 90.0,
                "averageHR": round(avg_hr * 0.88, 0), "averageSpeed": 2.22,
                "averageRunCadence": 150.0, "intensityType": "RECOVERY",
            })
            index += 1
        laps.append({
            "lapIndex": index, "distance": 1200.0, "duration": 420.0,
            "averageHR": round(avg_hr * 0.78, 0), "averageSpeed": 2.86,
            "averageRunCadence": 158.0, "intensityType": "COOLDOWN",
        })
        return laps

    def daily_metrics(self, day: date) -> dict[str, Any]:
        # Recovery degrades slightly as training load ramps in the final block.
        age_days = (self.end - day).days
        ramp = max(0.0, 1 - age_days / 90)
        jitter = self.rng.normal
        return {
            "local_date": day.isoformat(),
            "resting_hr": round(48 + 4 * ramp + jitter(0, 1.6), 1),
            "hrv_ms": round(62 - 8 * ramp + jitter(0, 5), 1),
            "sleep_hours": round(float(np.clip(7.2 - 0.5 * ramp + jitter(0, 0.7), 4.5, 9.5)), 2),
            "sleep_score": round(float(np.clip(78 - 6 * ramp + jitter(0, 7), 35, 98)), 0),
            "body_battery_high": round(float(np.clip(82 - 8 * ramp + jitter(0, 6), 30, 100)), 0),
            "body_battery_low": round(float(np.clip(24 - 5 * ramp + jitter(0, 5), 5, 60)), 0),
            "steps": round(float(np.clip(9500 + jitter(0, 2600), 1500, 25000)), 0),
            "weight_kg": round(78.5 - 1.6 * (1 - age_days / max(self.days, 1)) + jitter(0, 0.3), 1),
            "stress_avg": round(float(np.clip(32 + 6 * ramp + jitter(0, 8), 10, 80)), 0),
        }

    def iter_daily_metrics(self, start: date, end: date) -> Iterator[dict[str, Any]]:
        day = start
        while day <= end:
            yield self.daily_metrics(day)
            day += timedelta(days=1)

    # --- generation -------------------------------------------------------

    def _build(self) -> None:
        start = self.end - timedelta(days=self.days)
        activity_id = 9_000_000_000
        rows: list[dict[str, Any]] = []

        for offset in range(self.days + 1):
            day = start + timedelta(days=offset)
            week_index = offset / 7.0
            weekday = day.weekday()

            # Running: Mon easy, Wed tempo/intervals, Fri easy, Sun long.
            run_plan = {0: "easy", 2: "quality", 4: "easy", 6: "long"}.get(weekday)
            if run_plan and self.rng.random() > 0.12:
                activity_id += 1
                summary = self._run(activity_id, day, run_plan, week_index)
                rows.append(summary)
                self._run_params[str(activity_id)] = (
                    summary["distance"], summary["duration"], summary["averageHR"], run_plan)

            # Strength: Tue push, Thu pull, Sat legs.
            lift_plan = {1: "push", 3: "pull", 5: "legs"}.get(weekday)
            if lift_plan and self.rng.random() > 0.15:
                activity_id += 1
                summary, sets = self._lift(activity_id, day, lift_plan, week_index)
                rows.append(summary)
                self._sets[str(activity_id)] = sets

            # Cross-training so the app has more than two activity types.
            if weekday == 5 and self.rng.random() > 0.6:
                activity_id += 1
                rows.append(self._ride(activity_id, day))
            if weekday in (0, 3) and self.rng.random() > 0.5:
                activity_id += 1
                rows.append(self._walk(activity_id, day))

        rows.sort(key=lambda r: r["startTimeLocal"], reverse=True)
        self._activities = rows

    # --- individual activity builders -------------------------------------

    def _run(self, activity_id: int, day: date, plan: str, week: float) -> dict[str, Any]:
        rng = self.rng
        # Fitness improves for ~40 weeks then plateaus — a findable plateau.
        fitness = min(week, 40.0) / 40.0
        base_speed = 2.72 + 0.30 * fitness  # m/s for easy running

        if plan == "easy":
            distance = float(rng.normal(7200, 900))
            speed = base_speed * float(rng.normal(1.0, 0.02))
            # Grey-zone problem: "easy" runs are run too hard.
            hr = 152 + 9 * fitness + rng.normal(0, 4)
            zones = [180, 900, 2100, 300, 0]
        elif plan == "quality":
            distance = float(rng.normal(9500, 1100))
            speed = base_speed * 1.17 * float(rng.normal(1.0, 0.025))
            hr = 170 + rng.normal(0, 4)
            zones = [120, 480, 900, 1500, 240]
        else:  # long
            distance = float(rng.normal(16500, 2400))
            speed = base_speed * 0.96 * float(rng.normal(1.0, 0.02))
            hr = 154 + rng.normal(0, 4)
            zones = [240, 1500, 3600, 600, 0]

        distance = max(distance, 2500.0)
        duration = distance / speed
        scale = duration / max(sum(zones), 1)
        zones = [round(z * scale, 1) for z in zones]
        started = datetime.combine(day, datetime.min.time()) + timedelta(hours=6, minutes=30)

        return {
            "activityId": activity_id,
            "activityName": {"easy": "Easy Run", "quality": "Tempo Run",
                             "long": "Long Run"}[plan],
            "activityType": {"typeKey": "running"},
            "startTimeLocal": _iso(started),
            "startTimeGMT": _iso(started),
            "distance": round(distance, 1),
            "duration": round(duration, 1),
            "movingDuration": round(duration * 0.99, 1),
            "elapsedDuration": round(duration * 1.02, 1),
            "averageSpeed": round(speed, 3),
            "maxSpeed": round(speed * 1.25, 3),
            "averageHR": round(float(hr), 0),
            "maxHR": round(float(hr) + 12, 0),
            "calories": round(distance / 1000 * 68, 0),
            "elevationGain": round(float(abs(rng.normal(60, 40))), 0),
            "elevationLoss": round(float(abs(rng.normal(60, 40))), 0),
            # Cadence is low across the board — an obvious improvement lever.
            "averageRunningCadenceInStepsPerMinute": round(float(rng.normal(163, 2.5)), 1),
            "maxRunningCadenceInStepsPerMinute": round(float(rng.normal(178, 3)), 1),
            "avgStrideLength": round(speed * 60 / 163 * 200, 1),  # cm
            "avgGroundContactTime": round(float(rng.normal(258, 9)), 0),
            "avgVerticalOscillation": round(float(rng.normal(9.4, 0.6)), 1),
            "aerobicTrainingEffect": round(float(np.clip(rng.normal(3.2, 0.5), 1, 5)), 1),
            "anaerobicTrainingEffect": round(
                float(np.clip(rng.normal(1.2 if plan != "quality" else 2.4, 0.4), 0, 5)), 1),
            "activityTrainingLoad": round(duration / 60 * (2.4 if plan == "quality" else 1.7), 0),
            "vO2MaxValue": round(46 + 4.5 * fitness + float(rng.normal(0, 0.3)), 1),
            "hrTimeInZone_1": zones[0], "hrTimeInZone_2": zones[1], "hrTimeInZone_3": zones[2],
            "hrTimeInZone_4": zones[3], "hrTimeInZone_5": zones[4],
        }

    def _lift(self, activity_id: int, day: date, plan: str,
              week: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        rng = self.rng
        started = datetime.combine(day, datetime.min.time()) + timedelta(hours=17, minutes=45)

        # (category, name, sets, reps, base kg, kg gained per week, skip probability)
        plans: dict[str, list[tuple[str, str, int, int, float, float, float]]] = {
            "push": [
                ("BENCH_PRESS", "BARBELL_BENCH_PRESS", 4, 6, 72.5, 0.22, 0.0),
                ("BENCH_PRESS", "INCLINE_DUMBBELL_BENCH_PRESS", 3, 10, 24.0, 0.05, 0.1),
                ("SHOULDER_PRESS", "DUMBBELL_SHOULDER_PRESS", 3, 9, 20.0, 0.05, 0.15),
                ("TRICEPS_EXTENSION", "CABLE_TRICEPS_EXTENSION", 3, 12, 25.0, 0.06, 0.2),
                ("LATERAL_RAISE", "DUMBBELL_LATERAL_RAISE", 3, 14, 8.0, 0.02, 0.3),
            ],
            "pull": [
                # Rowing strength has been stalled for months (gain ≈ 0).
                ("ROW", "BARBELL_ROW", 4, 8, 62.5, 0.01, 0.0),
                ("PULL_UP", "WIDE_GRIP_LAT_PULLDOWN", 3, 10, 55.0, 0.04, 0.1),
                ("CURL", "DUMBBELL_CURL", 3, 11, 14.0, 0.03, 0.15),
                # Rear delts almost never get trained.
                ("SHOULDER_STABILITY", "CABLE_FACE_PULL", 3, 15, 18.0, 0.02, 0.8),
            ],
            "legs": [
                ("SQUAT", "BARBELL_BACK_SQUAT", 5, 5, 95.0, 0.35, 0.0),
                ("SQUAT", "LEG_PRESS", 3, 10, 140.0, 0.3, 0.15),
                ("CALF_RAISE", "STANDING_CALF_RAISE", 3, 15, 60.0, 0.1, 0.35),
                # Hamstrings are the neglected antagonist.
                ("LEG_CURL", "SEATED_LEG_CURL", 3, 12, 35.0, 0.03, 0.75),
            ],
        }

        sets: list[dict[str, Any]] = []
        clock = started
        for category, name, n_sets, reps, base_kg, per_week, skip in plans[plan]:
            if rng.random() < skip:
                continue
            weight = base_kg + per_week * week + float(rng.normal(0, 1.0))
            weight = max(round(weight / 2.5) * 2.5, 5.0)
            for set_number in range(n_sets):
                actual_reps = max(1, int(reps - rng.integers(0, 2)))
                work_s = float(actual_reps * rng.normal(3.2, 0.3))
                sets.append({
                    "setType": "ACTIVE",
                    "startTime": _iso(clock),
                    "duration": round(work_s, 1),
                    "repetitionCount": actual_reps,
                    "weight": round(weight * 1000, 0),  # Garmin reports grams
                    "exercises": [{"category": category, "name": name, "probability": 100.0}],
                })
                clock += timedelta(seconds=work_s)
                rest_s = float(rng.normal(105 if n_sets > 3 else 80, 15))
                sets.append({
                    "setType": "REST",
                    "startTime": _iso(clock),
                    "duration": round(max(rest_s, 20.0), 1),
                    "repetitionCount": None,
                    "weight": None,
                    "exercises": [],
                })
                clock += timedelta(seconds=max(rest_s, 20.0))

        duration = (clock - started).total_seconds()
        working = [s for s in sets if s["setType"] == "ACTIVE"]
        summary = {
            "activityId": activity_id,
            "activityName": f"{plan.title()} Day",
            "activityType": {"typeKey": "strength_training"},
            "startTimeLocal": _iso(started),
            "startTimeGMT": _iso(started),
            "duration": round(duration, 1),
            "movingDuration": round(sum(s["duration"] for s in working), 1),
            "distance": 0.0,
            "calories": round(duration / 60 * 6.5, 0),
            "averageHR": round(float(rng.normal(118, 6)), 0),
            "maxHR": round(float(rng.normal(152, 7)), 0),
            "totalSets": len(working),
            "totalReps": sum(s["repetitionCount"] or 0 for s in working),
            "aerobicTrainingEffect": round(float(np.clip(rng.normal(1.6, 0.3), 0, 5)), 1),
            "anaerobicTrainingEffect": round(float(np.clip(rng.normal(1.9, 0.4), 0, 5)), 1),
            "activityTrainingLoad": round(duration / 60 * 1.1, 0),
        }
        return summary, sets

    def _ride(self, activity_id: int, day: date) -> dict[str, Any]:
        rng = self.rng
        started = datetime.combine(day, datetime.min.time()) + timedelta(hours=9)
        distance = float(rng.normal(38000, 9000))
        speed = float(rng.normal(7.4, 0.6))
        duration = max(distance, 8000) / speed
        return {
            "activityId": activity_id,
            "activityName": "Weekend Ride",
            "activityType": {"typeKey": "cycling"},
            "startTimeLocal": _iso(started),
            "startTimeGMT": _iso(started),
            "distance": round(max(distance, 8000), 1),
            "duration": round(duration, 1),
            "movingDuration": round(duration * 0.95, 1),
            "averageSpeed": round(speed, 3),
            "maxSpeed": round(speed * 1.7, 3),
            "averageHR": round(float(rng.normal(136, 7)), 0),
            "maxHR": round(float(rng.normal(168, 8)), 0),
            "calories": round(duration / 60 * 9.5, 0),
            "elevationGain": round(float(abs(rng.normal(380, 160))), 0),
            "avgPower": round(float(rng.normal(178, 18)), 0),
            "normPower": round(float(rng.normal(192, 18)), 0),
            "averageBikingCadenceInRevPerMinute": round(float(rng.normal(84, 4)), 1),
            "aerobicTrainingEffect": round(float(np.clip(rng.normal(2.8, 0.5), 1, 5)), 1),
            "activityTrainingLoad": round(duration / 60 * 1.5, 0),
        }

    def _walk(self, activity_id: int, day: date) -> dict[str, Any]:
        rng = self.rng
        started = datetime.combine(day, datetime.min.time()) + timedelta(hours=12, minutes=30)
        distance = float(rng.normal(3400, 900))
        speed = float(rng.normal(1.38, 0.1))
        duration = max(distance, 800) / speed
        return {
            "activityId": activity_id,
            "activityName": "Lunch Walk",
            "activityType": {"typeKey": "walking"},
            "startTimeLocal": _iso(started),
            "startTimeGMT": _iso(started),
            "distance": round(max(distance, 800), 1),
            "duration": round(duration, 1),
            "movingDuration": round(duration * 0.97, 1),
            "averageSpeed": round(speed, 3),
            "averageHR": round(float(rng.normal(96, 6)), 0),
            "maxHR": round(float(rng.normal(118, 7)), 0),
            "calories": round(duration / 60 * 4.2, 0),
            "averageRunningCadenceInStepsPerMinute": round(float(rng.normal(112, 5)), 1),
            "activityTrainingLoad": round(duration / 60 * 0.4, 0),
        }
