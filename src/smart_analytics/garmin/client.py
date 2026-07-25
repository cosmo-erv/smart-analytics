"""Garmin Connect client and payload normalisation.

Garmin has no public consumer API — the official Connect Developer programme is
partner-only — so this uses ``garminconnect``/``garth``, which speaks the same
OAuth flow as the mobile app. Practical consequences we design around:

* **Tokens, not passwords.** The password is used once to mint OAuth tokens,
  which are cached in ``settings.token_store``. Later syncs resume from there,
  so MFA is prompted at most once per token lifetime.
* **Rate limits are real.** Callers page through history and stop early; detail
  endpoints (one HTTP call per strength workout) are fetched in bounded batches.
* **Units vary by endpoint.** Set weights come back in *grams*, stride length in
  *centimetres*, speed in m/s. Normalisation happens here so nothing downstream
  has to guess.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from datetime import date, datetime, timedelta
from typing import Any

from ..config import Settings, settings as default_settings
from . import physiology
from . import workouts as workouts_mod

log = logging.getLogger(__name__)

# Garmin typeKey → coarse family used across the app.
TYPE_FAMILIES: dict[str, str] = {
    "running": "running", "treadmill_running": "running", "trail_running": "running",
    "track_running": "running", "indoor_running": "running", "obstacle_run": "running",
    "ultra_run": "running", "virtual_run": "running", "street_running": "running",
    "cycling": "cycling", "road_biking": "cycling", "mountain_biking": "cycling",
    "indoor_cycling": "cycling", "virtual_ride": "cycling", "gravel_cycling": "cycling",
    "cyclocross": "cycling", "track_cycling": "cycling", "bmx": "cycling",
    "lap_swimming": "swimming", "open_water_swimming": "swimming", "swimming": "swimming",
    "strength_training": "strength_training", "indoor_cardio": "cardio",
    "walking": "walking", "casual_walking": "walking", "speed_walking": "walking",
    "hiking": "hiking", "mountaineering": "hiking",
    "yoga": "mobility", "pilates": "mobility", "breathwork": "mobility",
    "stretching": "mobility", "mobility": "mobility",
    "cardio": "cardio", "hiit": "cardio", "elliptical": "cardio",
    "stair_climbing": "cardio", "indoor_rowing": "rowing", "rowing": "rowing",
    "fitness_equipment": "cardio", "bouldering": "climbing", "rock_climbing": "climbing",
    "indoor_climbing": "climbing",
}

# Types whose per-set detail is worth fetching.
SET_BEARING_TYPES = {"strength_training"}

# Types where per-lap splits carry real signal (pace/HR over distance).
SPLIT_BEARING_TYPES = ("running", "cycling", "swimming")


def family_for(type_key: str | None) -> str:
    if not type_key:
        return "other"
    key = type_key.lower()
    if key in TYPE_FAMILIES:
        return TYPE_FAMILIES[key]
    for token, fam in (("run", "running"), ("cycl", "cycling"), ("bik", "cycling"),
                       ("swim", "swimming"), ("walk", "walking"), ("hik", "hiking"),
                       ("row", "rowing"), ("strength", "strength_training"),
                       ("climb", "climbing")):
        if token in key:
            return fam
    return key


def _f(value: Any) -> float | None:
    """Coerce to float, treating Garmin's nulls and sentinels as missing."""
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out  # drop NaN


def _first(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if payload.get(key) is not None:
            return payload[key]
    return None


class GarminAuthError(RuntimeError):
    """Login failed — bad credentials, MFA not satisfied, or expired tokens."""


# How the account's second factor is delivered. Garmin reports this as
# ``mfaLastMethodUsed`` on the MFA challenge, and it decides what the UI should
# tell the user to go and look at — an emailed code and an authenticator code
# arrive in very different places.
MFA_EMAIL = "email"
MFA_SMS = "sms"
MFA_AUTHENTICATOR = "authenticator"
MFA_UNKNOWN = "unknown"


def normalise_mfa_method(raw: Any) -> str:
    """Classify Garmin's MFA method string, which isn't a documented enum."""
    text = str(raw or "").strip().lower()
    if not text:
        return MFA_UNKNOWN
    if "mail" in text:
        return MFA_EMAIL
    if "sms" in text or "text" in text or "phone" in text:
        return MFA_SMS
    if any(token in text for token in ("auth", "totp", "app", "token")):
        return MFA_AUTHENTICATOR
    return MFA_UNKNOWN


class GarminClient:
    """Thin, normalising wrapper around ``garminconnect.Garmin``."""

    def __init__(self, config: Settings | None = None,
                 mfa_prompt: Callable[[], str] | None = None) -> None:
        self.settings = config or default_settings
        self._mfa_prompt = mfa_prompt
        self._api: Any = None
        # Opaque state handed back by garminconnect between the two login steps.
        # It is None in current versions, so a separate flag records that a login
        # is waiting on a code.
        self._pending_mfa_state: Any = None
        self._mfa_pending = False
        # Address used for an interactive login, so the UI can name the account
        # even when nothing was configured in .env.
        self._login_email: str = ""

    # --- auth --------------------------------------------------------------

    def connect(self) -> "GarminClient":
        """Resume from cached tokens, falling back to a credentialed login."""
        from garminconnect import Garmin, GarminConnectAuthenticationError

        store = str(self.settings.token_store)

        if self.settings.has_cached_tokens:
            try:
                api = Garmin()
                api.login(store)
                self._api = api
                log.info("Resumed Garmin session from cached tokens")
                return self
            except Exception as exc:  # token expired/corrupt — fall through to password
                log.warning("Cached Garmin tokens unusable (%s); re-authenticating", exc)

        if not self.settings.has_garmin_credentials:
            raise GarminAuthError(
                "No cached Garmin tokens and no GARMIN_EMAIL/GARMIN_PASSWORD set. "
                "Add them to .env (see .env.example) or use Demo mode."
            )

        try:
            api = Garmin(
                email=self.settings.garmin_email,
                password=self.settings.garmin_password,
                prompt_mfa=self._mfa_prompt,
            )
            api.login()
            self._api = api
            self._save_tokens()
        except GarminAuthError:
            raise
        except GarminConnectAuthenticationError as exc:
            raise GarminAuthError(f"Garmin rejected the login: {exc}") from exc
        except Exception as exc:
            raise GarminAuthError(f"Could not log in to Garmin Connect: {exc}") from exc

        return self

    # --- interactive login (for the GUI) -----------------------------------

    def begin_login(self, email: str, password: str) -> dict[str, Any]:
        """Start a login without blocking on an MFA prompt.

        A desktop script can block on ``input()`` for the emailed code, but a web
        UI can't — so this uses garminconnect's early-return mode: the call comes
        back with ``mfa_required`` plus the opaque client state, which
        :meth:`complete_login` hands back along with the code.

        Returns ``{"status": "ok"}`` when no MFA was needed, or
        ``{"status": "mfa_required", "method": ...}`` — the method being where the
        code will turn up, so the UI can point at the right place.
        """
        from garminconnect import Garmin, GarminConnectAuthenticationError

        if not email or not password:
            raise GarminAuthError("Email and password are both required.")

        self._login_email = email.strip()
        try:
            api = Garmin(email=email, password=password, return_on_mfa=True)
            status, client_state = api.login()
        except GarminConnectAuthenticationError as exc:
            raise GarminAuthError(f"Garmin rejected the login: {exc}") from exc
        except Exception as exc:
            raise GarminAuthError(f"Could not reach Garmin Connect: {exc}") from exc

        self._api = api
        if status == "needs_mfa":
            # The opaque state is None in current garminconnect — the challenge
            # is held on the client itself — so a separate flag tracks that a
            # login is mid-flight rather than testing the state for truthiness.
            self._pending_mfa_state = client_state
            self._mfa_pending = True
            return {"status": "mfa_required", "method": self.mfa_method()}

        self._pending_mfa_state = None
        self._mfa_pending = False
        self._save_tokens()
        return {"status": "ok", "display_name": self.display_name()}

    def mfa_method(self) -> str:
        """Where this account's second factor is delivered, if Garmin said."""
        holder = getattr(self._api, "client", None) or self._api
        return normalise_mfa_method(getattr(holder, "_mfa_method", None))

    def complete_login(self, mfa_code: str) -> dict[str, Any]:
        """Finish a login that needed a multi-factor code."""
        if self._api is None or not self._mfa_pending:
            raise GarminAuthError("No login is waiting for a code — start again.")
        code = (mfa_code or "").strip()
        if not code:
            raise GarminAuthError("Enter the code Garmin sent you.")

        try:
            self._api.resume_login(self._pending_mfa_state, code)
        except Exception as exc:
            raise GarminAuthError(
                f"That code wasn't accepted: {exc}. Codes expire quickly — request a "
                f"new one and try again."
            ) from exc

        self._pending_mfa_state = None
        self._mfa_pending = False
        self._save_tokens()
        return {"status": "ok", "display_name": self.display_name()}

    @staticmethod
    def _token_writer(api: Any) -> Any:
        """The object that can persist OAuth tokens.

        garminconnect exposed the underlying garth session as ``.garth`` in the
        0.2 line and as ``.client`` from 0.3, and both have ``dump(path)``.
        """
        writer = getattr(api, "garth", None) or getattr(api, "client", None)
        if writer is None or not hasattr(writer, "dump"):
            raise GarminAuthError(
                "This version of garminconnect doesn't expose a way to cache tokens; "
                "pin garminconnect>=0.2.19."
            )
        return writer

    def _save_tokens(self) -> None:
        """Persist OAuth tokens so later syncs don't need the password again."""
        self.settings.token_store.mkdir(parents=True, exist_ok=True)
        self._token_writer(self._api).dump(str(self.settings.token_store))
        log.info("Garmin tokens cached in %s", self.settings.token_store)

    def sign_out(self) -> None:
        """Delete the cached tokens. The next sync will need a fresh login."""
        store = self.settings.token_store
        if store.exists():
            for path in store.iterdir():
                if path.is_file():
                    path.unlink()
        self._api = None
        self._pending_mfa_state = None
        self._mfa_pending = False
        self._login_email = ""

    @property
    def api(self) -> Any:
        if self._api is None:
            raise GarminAuthError("Not connected — call connect() first.")
        return self._api

    def display_name(self) -> str:
        fallback = self._login_email or self.settings.garmin_email
        try:
            return self.api.get_full_name() or fallback
        except Exception:
            return fallback

    # --- activities --------------------------------------------------------

    def iter_activities(self, since: date | None = None, page_size: int = 50,
                        max_activities: int | None = None) -> Iterator[dict[str, Any]]:
        """Yield raw activity summaries, newest first, stopping at ``since``."""
        start = 0
        fetched = 0
        while True:
            batch = self.api.get_activities(start, page_size) or []
            if not batch:
                return
            for raw in batch:
                started = _parse_dt(_first(raw, "startTimeLocal", "startTimeGMT"))
                if since and started and started.date() < since:
                    return
                yield raw
                fetched += 1
                if max_activities and fetched >= max_activities:
                    return
            if len(batch) < page_size:
                return
            start += page_size

    def exercise_sets(self, activity_id: str) -> list[dict[str, Any]]:
        payload = self.api.get_activity_exercise_sets(activity_id) or {}
        return payload.get("exerciseSets") or []

    # --- wellness ----------------------------------------------------------

    def daily_metrics(self, day: date) -> dict[str, Any]:
        """Best-effort daily wellness snapshot; missing pieces are left null."""
        iso = day.isoformat()
        out: dict[str, Any] = {"local_date": iso}

        try:
            stats = self.api.get_stats(iso) or {}
            out["resting_hr"] = _f(_first(stats, "restingHeartRate", "restingHeartRateTimestamp"))
            out["steps"] = _f(stats.get("totalSteps"))
            out["stress_avg"] = _f(stats.get("averageStressLevel"))
            out["body_battery_high"] = _f(stats.get("bodyBatteryHighestValue"))
            out["body_battery_low"] = _f(stats.get("bodyBatteryLowestValue"))
        except Exception as exc:
            log.debug("stats unavailable for %s: %s", iso, exc)

        try:
            sleep = (self.api.get_sleep_data(iso) or {}).get("dailySleepDTO") or {}
            seconds = _f(sleep.get("sleepTimeSeconds"))
            out["sleep_hours"] = round(seconds / 3600, 2) if seconds else None
            scores = sleep.get("sleepScores") or {}
            overall = scores.get("overall") or {}
            out["sleep_score"] = _f(overall.get("value"))
        except Exception as exc:
            log.debug("sleep unavailable for %s: %s", iso, exc)

        try:
            hrv = self.api.get_hrv_data(iso) or {}
            summary = hrv.get("hrvSummary") or {}
            out["hrv_ms"] = _f(_first(summary, "lastNightAvg", "weeklyAvg"))
        except Exception as exc:
            log.debug("hrv unavailable for %s: %s", iso, exc)

        try:
            weigh = self.api.get_daily_weigh_ins(iso) or {}
            grams = _f((weigh.get("totalAverage") or {}).get("weight"))
            out["weight_kg"] = round(grams / 1000, 2) if grams else None
        except Exception as exc:
            log.debug("weight unavailable for %s: %s", iso, exc)

        return out

    def iter_daily_metrics(self, start: date, end: date) -> Iterator[dict[str, Any]]:
        day = start
        while day <= end:
            yield self.daily_metrics(day)
            day += timedelta(days=1)

    # --- Garmin's own physiological model ----------------------------------

    def _try(self, label: str, call, *args, **kwargs) -> Any:
        """Call an endpoint, returning None if this account doesn't support it.

        Availability varies by device and account — an older watch has no
        training-readiness endpoint at all. A missing metric must degrade to
        "unavailable", never to a failed sync.
        """
        try:
            return call(*args, **kwargs)
        except Exception as exc:
            log.debug("%s unavailable: %s", label, exc)
            return None

    def physiology_snapshot(self, day: date) -> dict[str, Any]:
        """Per-day metrics: VO2max, training status and load, daily readiness."""
        iso = day.isoformat()
        rows = []

        payload = self._try("max metrics", self.api.get_max_metrics, iso)
        if payload:
            rows.append(physiology.normalise_max_metrics(payload, day))

        payload = self._try("training status", self.api.get_training_status, iso)
        if payload:
            rows.append(physiology.normalise_training_status(payload, day))

        payload = self._try("training readiness", self.api.get_training_readiness, iso)
        if payload:
            rows.append(physiology.normalise_training_readiness(payload, day))

        merged = physiology.merge_metric_rows(rows)
        return merged[0] if merged else {"local_date": iso}

    def latest_thresholds(self, day: date | None = None) -> dict[str, Any]:
        """Threshold and score metrics that only need pulling once per sync."""
        day = day or date.today()
        iso = day.isoformat()
        window_start = (day - timedelta(days=28)).isoformat()
        rows = []

        payload = self._try("lactate threshold", self.api.get_lactate_threshold, latest=True)
        if payload:
            rows.append(physiology.normalise_lactate_threshold(payload, day))

        payload = self._try("cycling FTP", self.api.get_cycling_ftp)
        if payload:
            rows.append(physiology.normalise_ftp(payload, day))

        payload = self._try("endurance score", self.api.get_endurance_score, window_start, iso)
        if payload:
            rows.append(physiology.normalise_scored_metric(
                payload, day, "endurance_score", ["enduranceScore", "overallScore", "avg"]))

        payload = self._try("hill score", self.api.get_hill_score, window_start, iso)
        if payload:
            rows.append(physiology.normalise_scored_metric(
                payload, day, "hill_score", ["overallScore", "hillScore", "strengthScore"]))

        payload = self._try("running tolerance", self.api.get_running_tolerance,
                            window_start, iso, "weekly")
        if payload:
            rows.append(physiology.normalise_scored_metric(
                payload, day, "running_tolerance_km",
                ["runningTolerance", "toleranceDistance", "weeklyDistance"]))

        merged = physiology.merge_metric_rows(rows)
        return merged[0] if merged else {"local_date": iso}

    def race_predictions(self) -> list[dict[str, Any]]:
        payload = self._try("race predictions", self.api.get_race_predictions)
        return physiology.normalise_race_predictions(payload) if payload else []

    def personal_records(self) -> list[dict[str, Any]]:
        payload = self._try("personal records", self.api.get_personal_record)
        return physiology.normalise_personal_records(payload) if payload else []

    def hr_zones(self, reference_activity_id: str | None = None) -> list[dict[str, Any]]:
        """Configured zone boundaries, preferring an activity's own zone report."""
        if reference_activity_id:
            payload = self._try("activity HR zones", self.api.get_activity_hr_in_timezones,
                                reference_activity_id)
            zones = physiology.normalise_hr_zones(payload) if payload else []
            if zones:
                return zones
        payload = self._try("profile settings", self.api.get_userprofile_settings)
        return physiology.normalise_hr_zones(payload) if payload else []

    def splits(self, activity_id: str) -> Any:
        return self._try("activity splits", self.api.get_activity_splits, activity_id)

    # --- structured workouts (the source of Garmin's muscle assignments) ----

    def workout_list(self, limit: int = 100) -> list[dict[str, Any]]:
        payload = self._try("workout list", self.api.get_workouts, 0, limit)
        return payload if isinstance(payload, list) else []

    def workout(self, workout_id: str) -> Any:
        return self._try("workout detail", self.api.get_workout_by_id, workout_id)

    def exercise_library(self) -> list[dict[str, Any]]:
        """Garmin's exercise library, if this account exposes one.

        The library isn't part of garminconnect's documented surface, so each
        known-plausible path is tried through the generic ``connectapi`` escape
        hatch. Failure is expected and harmless — the curated mapping covers it.
        """
        for path in workouts_mod.EXERCISE_LIBRARY_PATHS:
            payload = self._try(f"exercise library ({path})", self.api.connectapi, path)
            if not payload:
                continue
            records = workouts_mod.normalise_exercise_library(payload)
            if records:
                log.info("Exercise library: %d entries from %s", len(records), path)
                return records
        return []


# --- normalisation ----------------------------------------------------------

def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip().replace("Z", "").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def normalise_activity(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Map a Garmin activity summary onto the ``activities`` table schema."""
    activity_id = _first(raw, "activityId", "activityid")
    if activity_id is None:
        return None

    started = _parse_dt(_first(raw, "startTimeLocal", "startTimeGMT"))
    if started is None:
        return None

    type_key = ((raw.get("activityType") or {}).get("typeKey")
                if isinstance(raw.get("activityType"), dict) else raw.get("activityType"))
    stride_cm = _f(raw.get("avgStrideLength"))

    zones = {}
    for zone in range(1, 6):
        seconds = _f(raw.get(f"hrTimeInZone_{zone}"))
        if seconds is not None:
            zones[f"z{zone}"] = round(seconds, 1)

    return {
        "activity_id": str(activity_id),
        "start_time": started.isoformat(sep=" "),
        "local_date": started.date().isoformat(),
        "activity_type": family_for(type_key),
        "activity_subtype": type_key,
        "name": raw.get("activityName"),
        "duration_s": _f(raw.get("duration")),
        "moving_s": _f(_first(raw, "movingDuration", "duration")),
        "distance_m": _f(raw.get("distance")),
        "calories": _f(raw.get("calories")),
        "avg_hr": _f(raw.get("averageHR")),
        "max_hr": _f(raw.get("maxHR")),
        "avg_speed_mps": _f(raw.get("averageSpeed")),
        "max_speed_mps": _f(raw.get("maxSpeed")),
        "elevation_gain_m": _f(raw.get("elevationGain")),
        "elevation_loss_m": _f(raw.get("elevationLoss")),
        "avg_cadence": _f(_first(raw, "averageRunningCadenceInStepsPerMinute",
                                 "averageBikingCadenceInRevPerMinute", "averageSwimCadence")),
        "max_cadence": _f(_first(raw, "maxRunningCadenceInStepsPerMinute",
                                 "maxBikingCadenceInRevPerMinute")),
        "avg_power": _f(raw.get("avgPower")),
        "norm_power": _f(raw.get("normPower")),
        "training_load": _f(_first(raw, "activityTrainingLoad", "trainingLoad")),
        "aerobic_te": _f(raw.get("aerobicTrainingEffect")),
        "anaerobic_te": _f(raw.get("anaerobicTrainingEffect")),
        "vo2max": _f(raw.get("vO2MaxValue")),
        "avg_stride_m": round(stride_cm / 100, 3) if stride_cm else None,
        "avg_gct_ms": _f(raw.get("avgGroundContactTime")),
        "avg_vert_osc_cm": _f(raw.get("avgVerticalOscillation")),
        "total_sets": _f(raw.get("totalSets")),
        "total_reps": _f(raw.get("totalReps")),
        "total_volume_kg": None,  # computed from sets during detail sync
        "hr_zone_json": json.dumps(zones) if zones else None,
        "details_fetched": 0,
        # Links the activity to the structured workout it followed, which is where
        # Garmin's own exercise and muscle detail lives.
        "workout_id": (str(_first(raw, "workoutId", "workout_id"))
                       if _first(raw, "workoutId", "workout_id") else None),
        "raw_json": json.dumps(raw, default=str),
    }


def normalise_splits(activity_id: str, activity_date: str, payload: Any) -> list[dict[str, Any]]:
    """Delegate to the physiology module, keeping one import surface for callers."""
    return physiology.normalise_splits(activity_id, activity_date, payload)


def normalise_sets(activity_id: str, activity_date: str,
                   raw_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map Garmin ``exerciseSets`` onto the ``strength_sets`` schema.

    Garmin reports set weight in grams and includes REST sets, which carry no
    volume but do define the workout's rest structure, so they are kept with a
    ``set_type`` marker rather than filtered out here.
    """
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw_sets):
        exercises = item.get("exercises") or []
        best = max(exercises, key=lambda e: _f(e.get("probability")) or 0.0) if exercises else {}
        grams = _f(item.get("weight"))
        started = _parse_dt(item.get("startTime"))
        rows.append({
            "activity_id": str(activity_id),
            "set_index": index,
            "start_time": started.isoformat(sep=" ") if started else None,
            "local_date": (started.date().isoformat() if started else activity_date),
            "set_type": (item.get("setType") or "ACTIVE").upper(),
            "category": best.get("category"),
            "exercise_name": best.get("name"),
            "reps": int(_f(item.get("repetitionCount")) or 0) or None,
            "weight_kg": round(grams / 1000, 3) if grams else None,
            "duration_s": _f(item.get("duration")),
        })
    return rows
