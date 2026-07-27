"""Running analytics: performance trend, intensity distribution, PBs, predictions.

The central problem with judging running progress from summary data is that pace
alone is confounded — a faster run at a higher heart rate is not necessarily
fitness. So the primary progress metric here is **aerobic efficiency** (metres
covered per heartbeat), which normalises pace by cardiac cost and therefore
improves only when fitness genuinely does. Pace, VO2max, cadence and volume are
reported alongside it as supporting evidence.

Everything is computed from activity summaries, which is all Garmin's list
endpoint returns. Intra-run metrics that need split data (aerobic decoupling,
true interval analysis) are deliberately out of scope rather than faked from
averages.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from .findings import Finding
from .strength import linear_trend

# (label, target metres, tolerance fraction)
DISTANCE_BUCKETS: list[tuple[str, float, float]] = [
    ("5K", 5_000, 0.15),
    ("10K", 10_000, 0.12),
    ("15K", 15_000, 0.12),
    ("Half marathon", 21_097, 0.08),
    ("Marathon", 42_195, 0.05),
]

RIEGEL_EXPONENT = 1.06

# Runs at or below this share of a session's peak HR count as easy for the
# polarisation check when zone data is missing.
EASY_HR_FRACTION = 0.80

DEFAULT_LOOKBACK_DAYS = 180


# --- preparation ------------------------------------------------------------

def prepare_runs(activities: pd.DataFrame) -> pd.DataFrame:
    """Filter to runs and derive pace, efficiency and intensity columns."""
    columns = ["activity_id", "local_date", "start_time", "name", "distance_km", "duration_min",
               "pace_s_per_km", "pace_label", "avg_hr", "max_hr", "avg_cadence", "avg_stride_m",
               "elevation_gain_m", "training_load", "vo2max", "m_per_beat", "avg_gct_ms",
               "avg_vert_osc_cm", "easy_s", "moderate_s", "hard_s", "intensity"]
    if activities is None or activities.empty:
        return pd.DataFrame(columns=columns)

    runs = activities[activities["activity_type"] == "running"].copy()
    runs = runs[(runs["distance_m"].fillna(0) > 400) & (runs["duration_s"].fillna(0) > 120)]
    if runs.empty:
        return pd.DataFrame(columns=columns)

    runs["distance_km"] = runs["distance_m"] / 1000.0
    runs["duration_min"] = runs["duration_s"] / 60.0
    runs["pace_s_per_km"] = runs["duration_s"] / runs["distance_km"]
    runs["pace_label"] = runs["pace_s_per_km"].map(format_pace)
    # Metres per heartbeat: pace normalised by cardiac cost.
    runs["m_per_beat"] = np.where(
        runs["avg_hr"].fillna(0) > 0,
        runs["distance_m"] / (runs["duration_s"] / 60.0 * runs["avg_hr"]),
        np.nan,
    )

    zones = runs["hr_zone_json"].map(_parse_zones)
    runs["easy_s"] = [z.get("easy", np.nan) for z in zones]
    runs["moderate_s"] = [z.get("moderate", np.nan) for z in zones]
    runs["hard_s"] = [z.get("hard", np.nan) for z in zones]
    runs["intensity"] = [
        _classify_intensity(row) for row in runs.itertuples()
    ]

    keep = [c for c in columns if c in runs.columns]
    return runs[keep].sort_values("local_date").reset_index(drop=True)


def _parse_zones(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    try:
        zones = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    easy = float(zones.get("z1", 0) or 0) + float(zones.get("z2", 0) or 0)
    moderate = float(zones.get("z3", 0) or 0)
    hard = float(zones.get("z4", 0) or 0) + float(zones.get("z5", 0) or 0)
    if easy + moderate + hard <= 0:
        return {}
    return {"easy": easy, "moderate": moderate, "hard": hard}


def _classify_intensity(row) -> str:
    """Label a run easy / moderate / hard from where most of its time sat."""
    easy = getattr(row, "easy_s", np.nan)
    moderate = getattr(row, "moderate_s", np.nan)
    hard = getattr(row, "hard_s", np.nan)
    if not any(pd.isna(v) for v in (easy, moderate, hard)) and (easy + moderate + hard) > 0:
        if hard >= 0.2 * (easy + moderate + hard):
            return "hard"
        return "easy" if easy >= moderate else "moderate"
    avg_hr, max_hr = getattr(row, "avg_hr", np.nan), getattr(row, "max_hr", np.nan)
    if pd.notna(avg_hr) and pd.notna(max_hr) and max_hr > 0:
        ratio = avg_hr / max_hr
        if ratio <= EASY_HR_FRACTION:
            return "easy"
        return "moderate" if ratio <= 0.88 else "hard"
    return "unknown"


def format_pace(seconds_per_km: float | None) -> str:
    if seconds_per_km is None or pd.isna(seconds_per_km) or seconds_per_km <= 0:
        return "—"
    minutes, seconds = divmod(int(round(seconds_per_km)), 60)
    return f"{minutes}:{seconds:02d}/km"


def format_duration(seconds: float | None) -> str:
    if seconds is None or pd.isna(seconds) or seconds <= 0:
        return "—"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


# --- aggregation ------------------------------------------------------------

def weekly_running(runs: pd.DataFrame) -> pd.DataFrame:
    """Weekly distance, time, session count, mean pace, efficiency and load."""
    columns = ["week", "distance_km", "duration_min", "runs", "avg_pace_s_per_km",
               "m_per_beat", "training_load", "longest_km", "elevation_gain_m"]
    if runs.empty:
        return pd.DataFrame(columns=columns)
    frame = runs.copy()
    frame["week"] = pd.to_datetime(frame["local_date"]).dt.to_period("W").dt.start_time
    weekly = frame.groupby("week", as_index=False).agg(
        distance_km=("distance_km", "sum"),
        duration_min=("duration_min", "sum"),
        runs=("activity_id", "nunique"),
        m_per_beat=("m_per_beat", "mean"),
        training_load=("training_load", "sum"),
        longest_km=("distance_km", "max"),
        elevation_gain_m=("elevation_gain_m", "sum"),
    )
    # Distance-weighted mean pace, not the mean of paces.
    weekly["avg_pace_s_per_km"] = (weekly["duration_min"] * 60) / weekly["distance_km"]
    return weekly[columns].sort_values("week").reset_index(drop=True)


def intensity_distribution(runs: pd.DataFrame, lookback_days: int = 84,
                           as_of: datetime | None = None) -> dict[str, float]:
    """Share of running *time* spent easy / moderate / hard."""
    as_of = as_of or _latest(runs)
    if runs.empty or as_of is None:
        return {}
    window = runs[pd.to_datetime(runs["local_date"]) >= as_of - pd.Timedelta(days=lookback_days)]
    if window.empty:
        return {}

    zoned = window.dropna(subset=["easy_s", "moderate_s", "hard_s"])
    if not zoned.empty and (zoned["easy_s"].sum() + zoned["moderate_s"].sum()
                            + zoned["hard_s"].sum()) > 0:
        easy = float(zoned["easy_s"].sum())
        moderate = float(zoned["moderate_s"].sum())
        hard = float(zoned["hard_s"].sum())
        source = "hr_zones"
    else:
        by_label = window.groupby("intensity")["duration_min"].sum() * 60
        easy = float(by_label.get("easy", 0.0))
        moderate = float(by_label.get("moderate", 0.0))
        hard = float(by_label.get("hard", 0.0))
        source = "session_labels"

    total = easy + moderate + hard
    if total <= 0:
        return {}
    return {
        "easy_pct": round(easy / total * 100, 1),
        "moderate_pct": round(moderate / total * 100, 1),
        "hard_pct": round(hard / total * 100, 1),
        "total_hours": round(total / 3600, 1),
        "source": source,
    }


# --- performance ------------------------------------------------------------

def personal_bests(runs: pd.DataFrame, recent_days: int = 90,
                   as_of: datetime | None = None) -> pd.DataFrame:
    """Best effort per distance bucket, all-time and recent.

    Efforts within a bucket are compared on Riegel-equivalent time for the exact
    distance, so a 10.8 km run doesn't beat a 10.0 km race by being longer.
    """
    columns = ["bucket", "target_km", "best_time_s", "best_pace_s_per_km", "best_date",
               "best_activity_id", "attempts", "recent_best_time_s", "recent_best_date",
               "pct_off_best", "days_since_best"]
    as_of = as_of or _latest(runs)
    if runs.empty or as_of is None:
        return pd.DataFrame(columns=columns)

    rows = []
    for name, target_m, tolerance in DISTANCE_BUCKETS:
        low, high = target_m * (1 - tolerance), target_m * (1 + tolerance)
        bucket = runs[(runs["distance_km"] * 1000 >= low) & (runs["distance_km"] * 1000 <= high)]
        if bucket.empty:
            continue
        equivalent = (bucket["duration_min"] * 60
                      * (target_m / (bucket["distance_km"] * 1000)) ** RIEGEL_EXPONENT)
        bucket = bucket.assign(equivalent_s=equivalent)
        best = bucket.loc[bucket["equivalent_s"].idxmin()]

        recent = bucket[pd.to_datetime(bucket["local_date"])
                        >= as_of - pd.Timedelta(days=recent_days)]
        recent_best = recent.loc[recent["equivalent_s"].idxmin()] if not recent.empty else None

        pct_off = np.nan
        if recent_best is not None and best["equivalent_s"] > 0:
            pct_off = round((recent_best["equivalent_s"] / best["equivalent_s"] - 1) * 100, 1)

        rows.append({
            "bucket": name,
            "target_km": round(target_m / 1000, 2),
            "best_time_s": round(float(best["equivalent_s"]), 0),
            "best_pace_s_per_km": round(float(best["equivalent_s"]) / (target_m / 1000), 1),
            "best_date": pd.to_datetime(best["local_date"]),
            "best_activity_id": best["activity_id"],
            "attempts": int(len(bucket)),
            "recent_best_time_s": (round(float(recent_best["equivalent_s"]), 0)
                                   if recent_best is not None else np.nan),
            "recent_best_date": (pd.to_datetime(recent_best["local_date"])
                                 if recent_best is not None else pd.NaT),
            "pct_off_best": pct_off,
            "days_since_best": int((as_of - pd.to_datetime(best["local_date"])).days),
        })
    return pd.DataFrame(rows, columns=columns)


def race_predictions(runs: pd.DataFrame, reference_days: int = 120,
                     as_of: datetime | None = None) -> pd.DataFrame:
    """Riegel predictions from the strongest recent effort.

    Riegel over-predicts long distances from short efforts when endurance is the
    limiter, so the reference used is the *longest* qualifying recent effort,
    and the source is reported alongside every prediction.
    """
    columns = ["bucket", "target_km", "predicted_time_s", "predicted_pace_s_per_km",
               "source_distance_km", "source_date"]
    as_of = as_of or _latest(runs)
    if runs.empty or as_of is None:
        return pd.DataFrame(columns=columns)

    window = runs[(pd.to_datetime(runs["local_date"]) >= as_of - pd.Timedelta(days=reference_days))
                  & (runs["distance_km"] >= 5.0)]
    if window.empty:
        return pd.DataFrame(columns=columns)

    # Score efforts by pace relative to the window's median, then prefer longer.
    median_pace = window["pace_s_per_km"].median()
    strong = window[window["pace_s_per_km"] <= median_pace]
    if strong.empty:
        strong = window
    reference = strong.loc[strong["distance_km"].idxmax()]

    ref_m = float(reference["distance_km"]) * 1000
    ref_s = float(reference["duration_min"]) * 60

    rows = []
    for name, target_m, _tolerance in DISTANCE_BUCKETS:
        predicted = ref_s * (target_m / ref_m) ** RIEGEL_EXPONENT
        rows.append({
            "bucket": name,
            "target_km": round(target_m / 1000, 2),
            "predicted_time_s": round(predicted, 0),
            "predicted_pace_s_per_km": round(predicted / (target_m / 1000), 1),
            "source_distance_km": round(ref_m / 1000, 2),
            "source_date": pd.to_datetime(reference["local_date"]),
        })
    return pd.DataFrame(rows, columns=columns)


def performance_trends(runs: pd.DataFrame, lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                       as_of: datetime | None = None) -> dict[str, dict]:
    """Trends for the metrics that actually indicate running progress."""
    as_of = as_of or _latest(runs)
    if runs.empty or as_of is None:
        return {}
    window = runs[pd.to_datetime(runs["local_date"]) >= as_of - pd.Timedelta(days=lookback_days)]
    if window.empty:
        return {}

    easy = window[window["intensity"] == "easy"]
    weekly = weekly_running(window)

    def pack(trend, unit: str, higher_is_better: bool) -> dict:
        return {
            "per_month": None if not trend.reliable else round(trend.slope_per_month, 3),
            "pct_per_month": None if not trend.reliable else round(trend.pct_per_month, 2),
            "r_squared": None if not trend.reliable else round(trend.r_squared, 2),
            "n": trend.n_points,
            "span_days": trend.span_days,
            "reliable": trend.reliable,
            "unit": unit,
            "higher_is_better": higher_is_better,
        }

    trends = {
        "aerobic_efficiency": pack(
            linear_trend(window["local_date"], window["m_per_beat"]), "m/beat", True),
        "easy_pace": pack(
            linear_trend(easy["local_date"], easy["pace_s_per_km"]) if not easy.empty
            else linear_trend(pd.Series(dtype="datetime64[ns]"), pd.Series(dtype=float)),
            "s/km", False),
        "cadence": pack(linear_trend(window["local_date"], window["avg_cadence"]), "spm", True),
        "weekly_distance": pack(
            linear_trend(weekly["week"], weekly["distance_km"]) if not weekly.empty
            else linear_trend(pd.Series(dtype="datetime64[ns]"), pd.Series(dtype=float)),
            "km/week", True),
    }
    if window["vo2max"].notna().any():
        trends["vo2max"] = pack(
            linear_trend(window["local_date"], window["vo2max"]), "ml/kg/min", True)
    return trends


def consistency(runs: pd.DataFrame, lookback_days: int = 84,
                as_of: datetime | None = None) -> dict[str, float]:
    """Volume stability — the strongest predictor of long-run improvement."""
    as_of = as_of or _latest(runs)
    if runs.empty or as_of is None:
        return {}
    weekly = weekly_running(
        runs[pd.to_datetime(runs["local_date"]) >= as_of - pd.Timedelta(days=lookback_days)])
    if weekly.empty:
        return {}
    expected_weeks = max(int(round(lookback_days / 7)), 1)
    distances = weekly["distance_km"]
    # Weeks with no running at all don't appear as rows; count them explicitly.
    zero_weeks = max(expected_weeks - len(weekly), 0)
    mean_km = float(distances.mean())
    return {
        "weeks": expected_weeks,
        "weeks_with_runs": int(len(weekly)),
        "zero_weeks": zero_weeks,
        "mean_weekly_km": round(mean_km, 1),
        "cv_pct": round(float(distances.std(ddof=0)) / mean_km * 100, 1) if mean_km else np.nan,
        "runs_per_week": round(float(weekly["runs"].mean()), 1),
        "longest_run_share_pct": (round(float(weekly["longest_km"].mean()) / mean_km * 100, 1)
                                  if mean_km else np.nan),
    }


# --- findings ---------------------------------------------------------------

def running_findings(runs: pd.DataFrame, trends: dict, distribution: dict,
                     bests: pd.DataFrame, stability: dict) -> list[Finding]:
    """Where the running is going well, and where the improvements are."""
    findings: list[Finding] = []

    if runs is None or runs.empty:
        return [Finding(area="running", title="No runs found",
                        detail="No running activities in the local cache yet.",
                        severity="info")]

    # 1. Aerobic efficiency — the headline progress signal.
    efficiency = trends.get("aerobic_efficiency", {})
    if efficiency.get("reliable"):
        pct = efficiency["pct_per_month"]
        if pct >= 1.0:
            findings.append(Finding(
                area="running", title="Aerobic efficiency is improving",
                detail=(f"Metres per heartbeat up {pct:+.1f}%/month across "
                        f"{efficiency['n']} runs — you're covering more ground per unit of "
                        f"cardiac work, which reflects real fitness rather than just "
                        f"trying harder."),
                severity="good", metric=f"{pct:+.1f}%/mo",
                evidence=efficiency))
        elif pct <= -1.0:
            findings.append(Finding(
                area="running", title="Aerobic efficiency is declining",
                detail=(f"Metres per heartbeat down {pct:+.1f}%/month across "
                        f"{efficiency['n']} runs. Same paces are costing more heartbeats, "
                        f"which usually means accumulated fatigue or lost aerobic base."),
                severity="act", metric=f"{pct:+.1f}%/mo",
                recommendation=("Insert an easier week, then rebuild with a higher share of "
                                "genuinely easy running."),
                evidence=efficiency))
        else:
            findings.append(Finding(
                area="running", title="Aerobic efficiency has plateaued",
                detail=(f"Metres per heartbeat is flat at {pct:+.1f}%/month over "
                        f"{efficiency['span_days']} days. Fitness has stopped responding to "
                        f"the current stimulus."),
                severity="watch", metric=f"{pct:+.1f}%/mo",
                recommendation=("Change one variable for 4–6 weeks: more easy volume, or one "
                                "weekly threshold session if volume is already high."),
                evidence=efficiency))

    # 2. Intensity distribution — the most common fixable error.
    if distribution:
        easy_pct = distribution["easy_pct"]
        moderate_pct = distribution["moderate_pct"]
        if moderate_pct >= 35:
            findings.append(Finding(
                area="running", title="Too much time in the moderate grey zone",
                detail=(f"{moderate_pct:.0f}% of running time is at moderate intensity, with "
                        f"only {easy_pct:.0f}% easy. Grey-zone running is hard enough to "
                        f"accumulate fatigue but not hard enough to drive adaptation."),
                severity="act", metric=f"{moderate_pct:.0f}% moderate",
                recommendation=("Slow the easy runs until easy time is 75–80% of the total, and "
                                "make the hard sessions genuinely hard."),
                evidence=distribution))
        elif easy_pct >= 75:
            findings.append(Finding(
                area="running", title="Intensity distribution is well polarised",
                detail=(f"{easy_pct:.0f}% easy / {moderate_pct:.0f}% moderate / "
                        f"{distribution['hard_pct']:.0f}% hard across "
                        f"{distribution['total_hours']:.0f} hours."),
                severity="good", metric=f"{easy_pct:.0f}% easy", evidence=distribution))
        if distribution["hard_pct"] < 5 and easy_pct > 85:
            findings.append(Finding(
                area="running", title="Almost no hard running",
                detail=(f"Only {distribution['hard_pct']:.0f}% of time is at high intensity. "
                        f"Aerobic base is being built without the top-end stimulus that "
                        f"converts it into race pace."),
                severity="watch", metric=f"{distribution['hard_pct']:.0f}% hard",
                recommendation="Add one interval or threshold session per week."))

    # 3. Cadence.
    cadence = trends.get("cadence", {})
    mean_cadence = float(runs["avg_cadence"].dropna().mean()) if runs["avg_cadence"].notna().any() else np.nan
    if not pd.isna(mean_cadence) and mean_cadence < 168:
        findings.append(Finding(
            area="running", title="Cadence is low",
            detail=(f"Average cadence {mean_cadence:.0f} spm. Low cadence usually means a long "
                    f"overstriding contact phase, which raises impact load and braking force."),
            severity="watch", metric=f"{mean_cadence:.0f} spm",
            recommendation=("Practise 3×1 min at ~5% higher cadence inside easy runs; let stride "
                            "length shorten rather than pushing speed."),
            evidence={"mean_cadence": round(mean_cadence, 1), "trend": cadence}))

    # 4. Volume trend and consistency.
    volume = trends.get("weekly_distance", {})
    if stability:
        if stability.get("zero_weeks", 0) >= max(2, stability.get("weeks", 12) * 0.15):
            findings.append(Finding(
                area="running", title="Training has gaps",
                detail=(f"{stability['zero_weeks']} of the last {stability['weeks']} weeks had "
                        f"no runs at all (mean {stability['mean_weekly_km']:.0f} km/week when "
                        f"running). Consistency drives more improvement than any single session."),
                severity="act", metric=f"{stability['zero_weeks']} blank weeks",
                recommendation="Set a floor of two short runs in weeks you can't train properly.",
                evidence=stability))
        cv = stability.get("cv_pct")
        if cv is not None and not pd.isna(cv) and cv > 45:
            findings.append(Finding(
                area="running", title="Weekly volume swings a lot",
                detail=(f"Week-to-week distance varies by {cv:.0f}% (coefficient of variation). "
                        f"Large swings raise injury risk without adding fitness."),
                severity="watch", metric=f"CV {cv:.0f}%",
                recommendation="Keep week-to-week changes inside ±20%.", evidence=stability))
        share = stability.get("longest_run_share_pct")
        if share is not None and not pd.isna(share) and share > 40:
            findings.append(Finding(
                area="running", title="The long run dominates weekly volume",
                detail=(f"The long run averages {share:.0f}% of weekly distance. Above ~35% the "
                        f"long run tends to drive fatigue that costs the rest of the week."),
                severity="watch", metric=f"{share:.0f}% of volume",
                recommendation="Grow midweek volume before extending the long run further."))

    if volume.get("reliable") and volume["pct_per_month"] is not None:
        pct = volume["pct_per_month"]
        if pct <= -5:
            findings.append(Finding(
                area="running", title="Weekly volume is trending down",
                detail=f"Weekly distance {pct:+.1f}%/month over {volume['span_days']} days.",
                severity="watch", metric=f"{pct:+.1f}%/mo", evidence=volume))
        elif pct >= 15:
            findings.append(Finding(
                area="running", title="Volume is ramping quickly",
                detail=(f"Weekly distance {pct:+.1f}%/month. Fast ramps are the classic "
                        f"precursor to overuse injury."),
                severity="watch", metric=f"{pct:+.1f}%/mo",
                recommendation="Cap increases near 10%/month and take a down week every 4th week.",
                evidence=volume))

    # 5. Form relative to personal bests.
    if bests is not None and not bests.empty:
        rated = bests.dropna(subset=["pct_off_best"])
        for row in rated.itertuples():
            if row.pct_off_best >= 4:
                findings.append(Finding(
                    area="running", subject=row.bucket,
                    title=f"{row.bucket} form is off your best",
                    detail=(f"Recent best {format_duration(row.recent_best_time_s)} versus a PB "
                            f"of {format_duration(row.best_time_s)} set "
                            f"{row.days_since_best} days ago — {row.pct_off_best:+.1f}%."),
                    severity="watch", metric=f"{row.pct_off_best:+.1f}% vs PB",
                    evidence={"best_s": row.best_time_s,
                              "recent_s": row.recent_best_time_s,
                              "days_since_best": row.days_since_best}))
            elif row.pct_off_best <= 0.5:
                findings.append(Finding(
                    area="running", subject=row.bucket,
                    title=f"{row.bucket} is at or near a personal best",
                    detail=(f"Recent best {format_duration(row.recent_best_time_s)} against an "
                            f"all-time {format_duration(row.best_time_s)}."),
                    severity="good", metric=format_duration(row.recent_best_time_s)))

    vo2 = trends.get("vo2max", {})
    if vo2.get("reliable") and vo2["per_month"] is not None and abs(vo2["per_month"]) >= 0.15:
        direction = "rising" if vo2["per_month"] > 0 else "falling"
        findings.append(Finding(
            area="running", title=f"Garmin VO2max estimate is {direction}",
            detail=(f"{vo2['per_month']:+.2f} ml/kg/min per month across {vo2['n']} runs. "
                    f"Treat it as a coarse corroborating signal, not a measurement."),
            severity="good" if vo2["per_month"] > 0 else "watch",
            metric=f"{vo2['per_month']:+.2f}/mo", evidence=vo2))

    return findings


def _latest(runs: pd.DataFrame) -> datetime | None:
    if runs is None or runs.empty:
        return None
    return pd.to_datetime(runs["local_date"]).max()
