"""Split-level running analysis: decoupling, interval quality, pacing.

Activity averages hide the two things that matter most about a session's quality:

* **Aerobic decoupling** — whether heart rate drifts upward at constant pace as a
  long run progresses. A run averaging 150 bpm might hold 145 throughout, or climb
  from 138 to 162; the first is aerobically comfortable, the second means the
  distance is past current endurance. Same average, opposite meaning. Decoupling
  under ~5% is the usual marker of aerobic durability at that pace.
* **Interval execution** — whether reps hold pace or fade. Averages tell you the
  session happened; the rep-by-rep sequence tells you whether it was paced or
  started too hard.

All of this needs per-lap data, which is why the sync fetches splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .findings import Finding
from .running import format_pace
from .strength import linear_trend

# Laps shorter than this are warm-up fragments or auto-lap noise, not signal.
MIN_LAP_DISTANCE_M = 400
MIN_LAPS_FOR_DECOUPLING = 6
DECOUPLING_GOOD = 5.0     # percent
DECOUPLING_HIGH = 10.0

WORK_LAP_TYPES = {"INTERVAL", "ACTIVE", "WORK"}
RECOVERY_LAP_TYPES = {"RECOVERY", "REST", "WARMUP", "COOLDOWN"}


def _efficiency(distance_m: pd.Series, duration_s: pd.Series,
                avg_hr: pd.Series) -> pd.Series:
    """Metres per heartbeat for each lap — pace normalised by cardiac cost."""
    beats = duration_s / 60.0 * avg_hr
    return np.where(beats > 0, distance_m / beats, np.nan)


def decoupling(splits: pd.DataFrame, min_distance_km: float = 8.0) -> pd.DataFrame:
    """Per-run aerobic decoupling: efficiency in the first half versus the second.

    Computed on metres-per-heartbeat rather than raw heart rate, so a run that
    slows down as well as drifting isn't scored as if it held pace.
    """
    columns = ["activity_id", "local_date", "name", "distance_km", "laps",
               "first_half_m_per_beat", "second_half_m_per_beat", "decoupling_pct",
               "hr_drift_bpm", "verdict"]
    if splits is None or splits.empty:
        return pd.DataFrame(columns=columns)

    usable = splits[(splits["distance_m"].fillna(0) >= MIN_LAP_DISTANCE_M)
                    & (splits["avg_hr"].fillna(0) > 0)
                    & (splits["duration_s"].fillna(0) > 0)].copy()
    if usable.empty:
        return pd.DataFrame(columns=columns)
    # Steady-state only: interval sessions decouple by design, not by fatigue.
    usable = usable[~usable["split_type"].fillna("").str.upper().isin(
        {"INTERVAL", "RECOVERY"})]
    if usable.empty:
        return pd.DataFrame(columns=columns)

    usable["m_per_beat"] = _efficiency(usable["distance_m"], usable["duration_s"],
                                       usable["avg_hr"])

    rows = []
    for activity_id, group in usable.groupby("activity_id"):
        group = group.sort_values("split_index")
        total_km = float(group["distance_m"].sum()) / 1000
        if len(group) < MIN_LAPS_FOR_DECOUPLING or total_km < min_distance_km:
            continue
        midpoint = len(group) // 2
        first = group.iloc[:midpoint]["m_per_beat"].mean()
        second = group.iloc[midpoint:]["m_per_beat"].mean()
        if not np.isfinite(first) or not np.isfinite(second) or first <= 0:
            continue
        pct = (first - second) / first * 100
        rows.append({
            "activity_id": activity_id,
            "local_date": group["local_date"].iloc[0],
            "name": group["name"].iloc[0] if "name" in group else None,
            "distance_km": round(total_km, 1),
            "laps": int(len(group)),
            "first_half_m_per_beat": round(float(first), 3),
            "second_half_m_per_beat": round(float(second), 3),
            "decoupling_pct": round(float(pct), 1),
            "hr_drift_bpm": round(float(group.iloc[midpoint:]["avg_hr"].mean()
                                        - group.iloc[:midpoint]["avg_hr"].mean()), 1),
            "verdict": _decoupling_verdict(pct),
        })
    return pd.DataFrame(rows, columns=columns).sort_values("local_date").reset_index(drop=True)


def _decoupling_verdict(pct: float) -> str:
    if pct <= DECOUPLING_GOOD:
        return "aerobically comfortable"
    if pct <= DECOUPLING_HIGH:
        return "moderate drift"
    return "beyond current endurance"


def interval_sessions(splits: pd.DataFrame) -> pd.DataFrame:
    """Detect interval sessions and score how evenly the reps were executed."""
    columns = ["activity_id", "local_date", "name", "reps", "rep_distance_m",
               "mean_rep_pace_s", "first_rep_pace_s", "last_rep_pace_s", "fade_pct",
               "pace_cv_pct", "mean_rep_hr", "verdict"]
    if splits is None or splits.empty:
        return pd.DataFrame(columns=columns)

    frame = splits.copy()
    frame["type_upper"] = frame["split_type"].fillna("").str.upper()

    rows = []
    for activity_id, group in frame.groupby("activity_id"):
        group = group.sort_values("split_index")
        work = group[group["type_upper"] == "INTERVAL"]
        if len(work) < 3:
            # No explicit interval laps — infer them from a bimodal pace pattern.
            work = _infer_work_laps(group)
            if work is None or len(work) < 3:
                continue

        paces = (work["duration_s"] / (work["distance_m"] / 1000)).dropna()
        paces = paces[np.isfinite(paces)]
        if len(paces) < 3:
            continue
        mean_pace = float(paces.mean())
        first, last = float(paces.iloc[0]), float(paces.iloc[-1])
        fade = (last - first) / first * 100
        rows.append({
            "activity_id": activity_id,
            "local_date": group["local_date"].iloc[0],
            "name": group["name"].iloc[0] if "name" in group else None,
            "reps": int(len(paces)),
            "rep_distance_m": round(float(work["distance_m"].median()), 0),
            "mean_rep_pace_s": round(mean_pace, 1),
            "first_rep_pace_s": round(first, 1),
            "last_rep_pace_s": round(last, 1),
            "fade_pct": round(float(fade), 1),
            "pace_cv_pct": round(float(paces.std(ddof=0) / mean_pace * 100), 1),
            "mean_rep_hr": (round(float(work["avg_hr"].mean()), 0)
                            if work["avg_hr"].notna().any() else np.nan),
            "verdict": _interval_verdict(fade),
        })
    return pd.DataFrame(rows, columns=columns).sort_values("local_date").reset_index(drop=True)


def _infer_work_laps(group: pd.DataFrame) -> pd.DataFrame | None:
    """Pick out work reps when the device didn't label lap intensity.

    A session with real reps has a clear fast cluster; a steady run doesn't. The
    split is taken at the midpoint between the fastest and slowest lap speed, and
    only accepted when the fast cluster is meaningfully quicker than the rest.
    """
    laps = group[(group["distance_m"].fillna(0) >= 200)
                 & (group["duration_s"].fillna(0) > 0)].copy()
    if len(laps) < 5:
        return None
    laps["speed"] = laps["distance_m"] / laps["duration_s"]
    fastest, slowest = laps["speed"].max(), laps["speed"].min()
    if fastest <= 0 or (fastest - slowest) / fastest < 0.18:
        return None  # too uniform to be an interval session
    threshold = slowest + (fastest - slowest) * 0.6
    work = laps[laps["speed"] >= threshold]
    return work if 3 <= len(work) <= len(laps) - 1 else None


def _interval_verdict(fade_pct: float) -> str:
    if fade_pct <= 1.5:
        return "well paced"
    if fade_pct <= 4.0:
        return "slight fade"
    return "started too hard"


def negative_split_rate(splits: pd.DataFrame, min_laps: int = 4) -> dict:
    """How often the second half of a run is quicker than the first."""
    if splits is None or splits.empty:
        return {}
    usable = splits[(splits["distance_m"].fillna(0) >= MIN_LAP_DISTANCE_M)
                    & (splits["duration_s"].fillna(0) > 0)]
    if usable.empty:
        return {}

    negatives = total = 0
    for _activity_id, group in usable.groupby("activity_id"):
        group = group.sort_values("split_index")
        if len(group) < min_laps:
            continue
        midpoint = len(group) // 2
        first = (group.iloc[:midpoint]["duration_s"].sum()
                 / (group.iloc[:midpoint]["distance_m"].sum() / 1000))
        second = (group.iloc[midpoint:]["duration_s"].sum()
                  / (group.iloc[midpoint:]["distance_m"].sum() / 1000))
        if not np.isfinite(first) or not np.isfinite(second):
            continue
        total += 1
        negatives += int(second < first)

    if not total:
        return {}
    return {"runs": total, "negative_splits": negatives,
            "negative_pct": round(negatives / total * 100, 0)}


def decoupling_trend(decoupling_table: pd.DataFrame) -> dict:
    """Is aerobic durability improving? Falling decoupling means yes."""
    if decoupling_table is None or len(decoupling_table) < 4:
        return {}
    trend = linear_trend(decoupling_table["local_date"], decoupling_table["decoupling_pct"])
    if not trend.reliable:
        return {}
    return {"per_month": round(trend.slope_per_month, 2),
            "r_squared": round(trend.r_squared, 2),
            "n": trend.n_points, "span_days": trend.span_days,
            "recent_mean": round(float(decoupling_table.tail(5)["decoupling_pct"].mean()), 1)}


# --- findings ---------------------------------------------------------------

def split_findings(decoupling_table: pd.DataFrame, intervals: pd.DataFrame,
                   negatives: dict, trend: dict) -> list[Finding]:
    findings: list[Finding] = []

    if (decoupling_table is None or decoupling_table.empty) and \
            (intervals is None or intervals.empty):
        findings.append(Finding(
            area="running", title="No split data synced yet",
            detail=("Per-lap data drives aerobic decoupling and interval analysis. Enable "
                    "split fetching in Sync & Settings — it costs one request per run."),
            severity="info"))
        return findings

    if decoupling_table is not None and not decoupling_table.empty:
        recent = decoupling_table.tail(6)
        mean_pct = float(recent["decoupling_pct"].mean())
        worst = recent.loc[recent["decoupling_pct"].idxmax()]
        if mean_pct > DECOUPLING_HIGH:
            findings.append(Finding(
                area="running", subject="decoupling",
                title="Heart rate drifts badly on long runs",
                detail=(f"Aerobic decoupling averages {mean_pct:.1f}% across the last "
                        f"{len(recent)} long runs (good is under {DECOUPLING_GOOD:.0f}%). "
                        f"The worst was {worst['distance_km']:.1f} km at "
                        f"{worst['decoupling_pct']:.1f}%, with heart rate "
                        f"{worst['hr_drift_bpm']:+.0f} bpm higher in the second half. That "
                        f"means those distances are beyond your current aerobic endurance, "
                        f"not that you paced them badly."),
                severity="act", metric=f"{mean_pct:.1f}% drift",
                recommendation=("Hold long runs at a pace you can finish with under 5% drift, "
                                "and extend distance only once that holds."),
                evidence={"mean_pct": round(mean_pct, 1),
                          "runs": int(len(recent))}))
        elif mean_pct <= DECOUPLING_GOOD:
            findings.append(Finding(
                area="running", subject="decoupling",
                title="Long runs are aerobically comfortable",
                detail=(f"Decoupling averages {mean_pct:.1f}% over the last {len(recent)} long "
                        f"runs — heart rate holds steady at pace, which means there's room to "
                        f"extend distance or lift the pace."),
                severity="good", metric=f"{mean_pct:.1f}% drift"))
        else:
            findings.append(Finding(
                area="running", subject="decoupling",
                title="Moderate heart-rate drift on long runs",
                detail=(f"Decoupling averages {mean_pct:.1f}% (good is under "
                        f"{DECOUPLING_GOOD:.0f}%). Endurance is close to the demand of these "
                        f"distances but not comfortably ahead of it."),
                severity="watch", metric=f"{mean_pct:.1f}% drift",
                recommendation="Hold the current long-run distance for 3–4 weeks before extending."))

    if trend:
        direction = "improving" if trend["per_month"] < -0.3 else (
            "worsening" if trend["per_month"] > 0.3 else "flat")
        if direction != "flat":
            findings.append(Finding(
                area="running", subject="decoupling trend",
                title=f"Aerobic durability is {direction}",
                detail=(f"Decoupling is changing {trend['per_month']:+.2f} percentage points "
                        f"per month across {trend['n']} long runs "
                        f"(recent mean {trend['recent_mean']:.1f}%). Falling decoupling is one "
                        f"of the clearest signs that base training is working."),
                severity="good" if direction == "improving" else "watch",
                metric=f"{trend['per_month']:+.2f} pp/mo", evidence=trend))

    if intervals is not None and not intervals.empty:
        recent = intervals.tail(5)
        mean_fade = float(recent["fade_pct"].mean())
        if mean_fade > 4.0:
            findings.append(Finding(
                area="running", subject="interval pacing",
                title="Interval sessions start too hard",
                detail=(f"Across the last {len(recent)} interval sessions the final rep is "
                        f"{mean_fade:.1f}% slower than the first "
                        f"({format_pace(recent['first_rep_pace_s'].mean())} → "
                        f"{format_pace(recent['last_rep_pace_s'].mean())}). Fading that much "
                        f"means the early reps were faster than the session's target."),
                severity="watch", metric=f"{mean_fade:.1f}% fade",
                recommendation=("Set the target from the last rep you could hold, and run the "
                                "first rep at that pace — no quicker."),
                evidence={"mean_fade_pct": round(mean_fade, 1),
                          "sessions": int(len(recent))}))
        else:
            findings.append(Finding(
                area="running", subject="interval pacing",
                title="Interval pacing is consistent",
                detail=(f"Reps hold within {mean_fade:+.1f}% from first to last across the "
                        f"last {len(recent)} sessions, averaging "
                        f"{format_pace(recent['mean_rep_pace_s'].mean())}."),
                severity="good", metric=f"{mean_fade:+.1f}% fade"))

    if negatives and negatives.get("runs", 0) >= 8 and negatives["negative_pct"] < 20:
        findings.append(Finding(
            area="running", subject="pacing",
            title="Runs are almost always positive-split",
            detail=(f"Only {negatives['negative_pct']:.0f}% of runs "
                    f"({negatives['negative_splits']} of {negatives['runs']}) are quicker in "
                    f"the second half. Consistently starting faster than you finish is the "
                    f"most common pacing error, and it costs most in races."),
            severity="watch", metric=f"{negatives['negative_pct']:.0f}% negative",
            recommendation="Deliberately run the first half of easy and long runs slower.",
            evidence=negatives))

    return findings
