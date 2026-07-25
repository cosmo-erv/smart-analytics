"""Strength analytics: per-muscle volume, progression, balance and lag detection.

The pipeline is:

``strength_sets`` → :func:`expand_sets` (one row per set × muscle, carrying
fractional credit) → weekly volume per muscle, per-exercise estimated-1RM
trends, antagonist balance ratios → :func:`lagging_muscles`, which combines four
independent signals into one attention score.

Why *effective sets* rather than tonnage as the primary unit: tonnage is
dominated by whichever lift moves the most absolute weight (a set of squats
outweighs ten sets of lateral raises), and it silently drops bodyweight work.
Counting sets with fractional credit for secondary involvement tracks the
stimulus each muscle actually receives, which is what a balance question is
really asking. Tonnage is still reported — it's the right unit for tracking
work done on a single lift over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ..domain import exercises as ex
from ..domain.muscles import BALANCE_PAIRS, MUSCLE_IDS, MUSCLES, label, region_of
from .findings import Finding

# Sets above this rep count get their e1RM estimate flagged as unreliable
# (Epley's formula degrades badly in high-rep territory).
E1RM_REP_CAP = 15

# A muscle needs at least this much history before a trend claim is made.
MIN_SESSIONS_FOR_TREND = 4
MIN_DAYS_FOR_TREND = 21

DEFAULT_LOOKBACK_DAYS = 84  # 12 weeks


# --- set expansion ----------------------------------------------------------

def expand_sets(sets_df: pd.DataFrame) -> pd.DataFrame:
    """Explode working sets into one row per (set, muscle) with fractional credit.

    Adds ``effective_sets`` (the muscle's share of one set) and ``volume_kg``
    (weight × reps × share). REST sets and non-loading categories are dropped;
    unmapped exercises are dropped from muscle rows but counted by
    :func:`unmapped_exercises` so nothing disappears silently.
    """
    columns = ["activity_id", "local_date", "muscle", "exercise", "category", "pattern",
               "equipment", "share", "effective_sets", "reps", "weight_kg", "volume_kg", "e1rm_kg"]
    if sets_df is None or sets_df.empty:
        return pd.DataFrame(columns=columns)

    working = sets_df[sets_df["set_type"].fillna("ACTIVE").str.upper() == "ACTIVE"].copy()
    working = working[working["reps"].fillna(0) > 0]
    if working.empty:
        return pd.DataFrame(columns=columns)

    # Resolve each distinct (category, name) once — set counts run to tens of
    # thousands, distinct exercises to dozens.
    keys = working[["category", "exercise_name"]].drop_duplicates()
    resolutions = {
        (row.category, row.exercise_name): ex.resolve(row.category, row.exercise_name)
        for row in keys.itertuples()
    }

    rows: list[dict] = []
    for row in working.itertuples():
        resolved = resolutions[(row.category, row.exercise_name)]
        if not resolved.is_mapped:
            continue
        reps = float(row.reps or 0)
        weight = float(row.weight_kg) if pd.notna(row.weight_kg) else np.nan
        e1rm = estimate_1rm(weight, reps)
        for muscle, share in resolved.muscles.items():
            rows.append({
                "activity_id": row.activity_id,
                "local_date": row.local_date,
                "muscle": muscle,
                "exercise": resolved.display_name,
                "category": row.category,
                "pattern": resolved.pattern,
                "equipment": resolved.equipment,
                "share": share,
                "effective_sets": share,
                "reps": reps,
                "weight_kg": weight,
                "volume_kg": (weight * reps * share) if not np.isnan(weight) else 0.0,
                "e1rm_kg": e1rm,
            })

    out = pd.DataFrame(rows, columns=columns)
    if not out.empty:
        out["local_date"] = pd.to_datetime(out["local_date"])
        out["region"] = out["muscle"].map(region_of)
        out["muscle_label"] = out["muscle"].map(label)
    return out


def estimate_1rm(weight_kg: float, reps: float) -> float:
    """Epley estimated 1RM. Returns NaN for bodyweight or implausible input."""
    if weight_kg is None or np.isnan(weight_kg) or weight_kg <= 0 or reps <= 0:
        return float("nan")
    capped = min(reps, E1RM_REP_CAP)
    return float(weight_kg * (1 + capped / 30.0))


def unmapped_exercises(sets_df: pd.DataFrame) -> pd.DataFrame:
    """Exercises we couldn't map, so the UI can prompt for mapping additions."""
    if sets_df is None or sets_df.empty:
        return pd.DataFrame(columns=["category", "exercise_name", "sets"])
    working = sets_df[sets_df["set_type"].fillna("ACTIVE").str.upper() == "ACTIVE"]
    if working.empty:
        return pd.DataFrame(columns=["category", "exercise_name", "sets"])
    grouped = (working.groupby(["category", "exercise_name"], dropna=False)
               .size().reset_index(name="sets"))
    mask = [not ex.resolve(row.category, row.exercise_name).is_mapped
            for row in grouped.itertuples()]
    return grouped[mask].sort_values("sets", ascending=False).reset_index(drop=True)


# --- volume -----------------------------------------------------------------

def weekly_muscle_volume(expanded: pd.DataFrame) -> pd.DataFrame:
    """Effective sets and tonnage per muscle per ISO week."""
    if expanded.empty:
        return pd.DataFrame(columns=["week", "muscle", "muscle_label", "region",
                                     "effective_sets", "volume_kg", "sessions"])
    frame = expanded.copy()
    frame["week"] = frame["local_date"].dt.to_period("W").dt.start_time
    grouped = (frame.groupby(["week", "muscle"], as_index=False)
               .agg(effective_sets=("effective_sets", "sum"),
                    volume_kg=("volume_kg", "sum"),
                    sessions=("activity_id", "nunique")))
    grouped["muscle_label"] = grouped["muscle"].map(label)
    grouped["region"] = grouped["muscle"].map(region_of)
    return grouped.sort_values(["week", "muscle"]).reset_index(drop=True)


def muscle_volume_summary(expanded: pd.DataFrame, lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                          as_of: datetime | None = None) -> pd.DataFrame:
    """Per-muscle weekly averages over the lookback window, plus recency.

    Includes every muscle in the taxonomy — a muscle with zero volume is the
    single most important row on the page, so it must not be a missing row.
    """
    as_of = as_of or _latest(expanded)
    columns = ["muscle", "muscle_label", "region", "weekly_sets", "weekly_volume_kg",
               "total_sets", "total_volume_kg", "sessions", "days_since", "top_exercise"]
    if expanded.empty or as_of is None:
        return pd.DataFrame([{**{c: 0 for c in columns[3:]}, "muscle": m,
                              "muscle_label": label(m), "region": region_of(m),
                              "days_since": np.nan, "top_exercise": None}
                             for m in MUSCLE_IDS], columns=columns)

    start = as_of - pd.Timedelta(days=lookback_days)
    window = expanded[expanded["local_date"] >= start]
    weeks = max(lookback_days / 7.0, 1.0)

    rows = []
    for muscle in MUSCLE_IDS:
        subset = window[window["muscle"] == muscle]
        all_time = expanded[expanded["muscle"] == muscle]
        last_date = all_time["local_date"].max() if not all_time.empty else pd.NaT
        top = None
        if not subset.empty:
            by_exercise = subset.groupby("exercise")["effective_sets"].sum()
            top = by_exercise.idxmax() if not by_exercise.empty else None
        rows.append({
            "muscle": muscle,
            "muscle_label": label(muscle),
            "region": region_of(muscle),
            "weekly_sets": round(subset["effective_sets"].sum() / weeks, 2),
            "weekly_volume_kg": round(subset["volume_kg"].sum() / weeks, 1),
            "total_sets": round(subset["effective_sets"].sum(), 1),
            "total_volume_kg": round(subset["volume_kg"].sum(), 1),
            "sessions": int(subset["activity_id"].nunique()),
            "days_since": ((as_of - last_date).days if pd.notna(last_date) else np.nan),
            "top_exercise": top,
        })
    return pd.DataFrame(rows, columns=columns)


# --- progression ------------------------------------------------------------

@dataclass
class Trend:
    slope_per_month: float
    pct_per_month: float
    r_squared: float
    n_points: int
    span_days: int
    reliable: bool

    @classmethod
    def empty(cls) -> "Trend":
        return cls(float("nan"), float("nan"), float("nan"), 0, 0, False)


def linear_trend(dates: pd.Series, values: pd.Series,
                 min_points: int = MIN_SESSIONS_FOR_TREND,
                 min_days: int = MIN_DAYS_FOR_TREND) -> Trend:
    """Least-squares trend of ``values`` over time, expressed per 30 days."""
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "value": values}).dropna()
    if len(frame) < min_points:
        return Trend.empty()
    days = (frame["date"] - frame["date"].min()).dt.total_seconds() / 86400.0
    span = float(days.max())
    if span < min_days:
        return Trend.empty()

    x = days.to_numpy(dtype=float)
    y = frame["value"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mean = float(np.mean(y))
    per_month = float(slope * 30.0)
    return Trend(
        slope_per_month=per_month,
        pct_per_month=(per_month / mean * 100.0) if mean else float("nan"),
        r_squared=float(r2),
        n_points=len(frame),
        span_days=int(span),
        reliable=True,
    )


def exercise_sessions(expanded: pd.DataFrame) -> pd.DataFrame:
    """Best set per exercise per session — the series progression is measured on."""
    if expanded.empty:
        return pd.DataFrame(columns=["exercise", "local_date", "best_e1rm", "top_weight",
                                     "sets", "volume_kg", "reps"])
    # De-duplicate the muscle fan-out: one row per set per exercise.
    per_set = (expanded.groupby(["exercise", "activity_id", "local_date",
                                 "reps", "weight_kg", "e1rm_kg"], dropna=False)
               .size().reset_index(name="_n"))
    grouped = (per_set.groupby(["exercise", "activity_id", "local_date"], as_index=False)
               .agg(best_e1rm=("e1rm_kg", "max"),
                    top_weight=("weight_kg", "max"),
                    sets=("_n", "size"),
                    reps=("reps", "sum")))
    volume = (expanded[expanded["share"] >= 1.0]
              .groupby(["exercise", "activity_id"], as_index=False)["volume_kg"].sum())
    grouped = grouped.merge(volume, on=["exercise", "activity_id"], how="left")
    return grouped.sort_values(["exercise", "local_date"]).reset_index(drop=True)


def exercise_progress(expanded: pd.DataFrame, lookback_days: int = 180,
                      as_of: datetime | None = None) -> pd.DataFrame:
    """Per-exercise estimated-1RM trend over the lookback window."""
    columns = ["exercise", "sessions", "first_date", "last_date", "current_e1rm",
               "best_e1rm", "kg_per_month", "pct_per_month", "r_squared", "reliable",
               "days_since", "status"]
    sessions = exercise_sessions(expanded)
    as_of = as_of or _latest(expanded)
    if sessions.empty or as_of is None:
        return pd.DataFrame(columns=columns)

    start = as_of - pd.Timedelta(days=lookback_days)
    window = sessions[sessions["local_date"] >= start]

    rows = []
    for exercise, group in window.groupby("exercise"):
        group = group.sort_values("local_date")
        trend = linear_trend(group["local_date"], group["best_e1rm"])
        last_date = group["local_date"].max()
        current = group["best_e1rm"].dropna()
        rows.append({
            "exercise": exercise,
            "sessions": int(len(group)),
            "first_date": group["local_date"].min(),
            "last_date": last_date,
            "current_e1rm": round(float(current.iloc[-1]), 1) if not current.empty else np.nan,
            "best_e1rm": round(float(current.max()), 1) if not current.empty else np.nan,
            "kg_per_month": round(trend.slope_per_month, 2) if trend.reliable else np.nan,
            "pct_per_month": round(trend.pct_per_month, 2) if trend.reliable else np.nan,
            "r_squared": round(trend.r_squared, 2) if trend.reliable else np.nan,
            "reliable": trend.reliable,
            "days_since": int((as_of - last_date).days),
            "status": _trend_status(trend),
        })
    return (pd.DataFrame(rows, columns=columns)
            .sort_values("pct_per_month", na_position="last").reset_index(drop=True))


def _trend_status(trend: Trend) -> str:
    """Bands are calibrated for a trained lifter, not a novice.

    Beginners add several percent a month; past the first year, ~1%/month on
    estimated 1RM is genuine progress and 0.3–1% is slow but real, so those are
    the thresholds rather than the much higher novice rates.
    """
    if not trend.reliable:
        return "insufficient data"
    if trend.pct_per_month >= 1.0:
        return "progressing"
    if trend.pct_per_month >= 0.3:
        return "slow progress"
    if trend.pct_per_month > -0.5:
        return "stalled"
    return "regressing"


def muscle_strength_trend(expanded: pd.DataFrame, progress: pd.DataFrame | None = None,
                          lookback_days: int = 180,
                          as_of: datetime | None = None) -> pd.DataFrame:
    """Attribute exercise e1RM trends to muscles, weighted by involvement.

    A muscle's trend is the involvement-weighted mean of the trends of the
    exercises that load it, so bench-press progress credits chest more than
    triceps, and a stalled row shows up on upper back and biceps alike.
    """
    columns = ["muscle", "muscle_label", "pct_per_month", "kg_per_month",
               "exercises", "reliable"]
    if expanded.empty:
        return pd.DataFrame(columns=columns)
    progress = exercise_progress(expanded, lookback_days, as_of) if progress is None else progress
    if progress.empty:
        return pd.DataFrame(columns=columns)

    trends = progress.set_index("exercise")
    # Involvement weight = total effective sets that exercise gave the muscle.
    involvement = (expanded.groupby(["muscle", "exercise"], as_index=False)["effective_sets"].sum())

    rows = []
    for muscle, group in involvement.groupby("muscle"):
        weights, pcts, kgs, names = [], [], [], []
        for row in group.itertuples():
            if row.exercise not in trends.index:
                continue
            entry = trends.loc[row.exercise]
            if not bool(entry["reliable"]) or pd.isna(entry["pct_per_month"]):
                continue
            weights.append(float(row.effective_sets))
            pcts.append(float(entry["pct_per_month"]))
            kgs.append(float(entry["kg_per_month"]))
            names.append(row.exercise)
        if not weights:
            rows.append({"muscle": muscle, "muscle_label": label(muscle),
                         "pct_per_month": np.nan, "kg_per_month": np.nan,
                         "exercises": 0, "reliable": False})
            continue
        total = float(sum(weights))
        rows.append({
            "muscle": muscle,
            "muscle_label": label(muscle),
            "pct_per_month": round(float(np.dot(weights, pcts) / total), 2),
            "kg_per_month": round(float(np.dot(weights, kgs) / total), 2),
            "exercises": len(names),
            "reliable": True,
        })
    return pd.DataFrame(rows, columns=columns)


# --- balance ----------------------------------------------------------------

def balance_ratios(volume_summary: pd.DataFrame) -> pd.DataFrame:
    """Antagonist and structural ratios with a healthy-range verdict."""
    columns = ["pair", "ratio", "low", "high", "status", "numerator_sets",
               "denominator_sets", "numerator", "denominator"]
    if volume_summary.empty:
        return pd.DataFrame(columns=columns)
    sets_by_muscle = volume_summary.set_index("muscle")["weekly_sets"].to_dict()

    rows = []
    for pair_label, numerator, denominator, (low, high) in BALANCE_PAIRS:
        num = sum(sets_by_muscle.get(m, 0.0) for m in numerator)
        den = sum(sets_by_muscle.get(m, 0.0) for m in denominator)
        if num <= 0 and den <= 0:
            continue
        ratio = num / den if den > 0 else float("inf")
        if ratio == float("inf"):
            status = "no antagonist work"
        elif ratio < low:
            status = "numerator behind"
        elif ratio > high:
            status = "denominator behind"
        else:
            status = "balanced"
        rows.append({
            "pair": pair_label,
            "ratio": round(ratio, 2) if np.isfinite(ratio) else np.nan,
            "low": low, "high": high, "status": status,
            "numerator_sets": round(num, 1), "denominator_sets": round(den, 1),
            "numerator": ", ".join(label(m) for m in numerator),
            "denominator": ", ".join(label(m) for m in denominator),
        })
    return pd.DataFrame(rows, columns=columns)


def pattern_coverage(expanded: pd.DataFrame, lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                     as_of: datetime | None = None) -> pd.DataFrame:
    """Weekly effective sets per movement pattern (catches missing hinge work)."""
    columns = ["pattern", "weekly_sets", "sessions"]
    as_of = as_of or _latest(expanded)
    if expanded.empty or as_of is None:
        return pd.DataFrame([{"pattern": p, "weekly_sets": 0.0, "sessions": 0}
                             for p in ex.PATTERNS], columns=columns)
    window = expanded[expanded["local_date"] >= as_of - pd.Timedelta(days=lookback_days)]
    weeks = max(lookback_days / 7.0, 1.0)
    # A pattern is credited once per set, not once per muscle.
    per_set = window.drop_duplicates(subset=["activity_id", "local_date", "exercise",
                                             "reps", "weight_kg", "pattern"])
    counts = per_set.groupby("pattern").agg(sets=("pattern", "size"),
                                            sessions=("activity_id", "nunique"))
    rows = [{"pattern": p,
             "weekly_sets": round(float(counts["sets"].get(p, 0)) / weeks, 2),
             "sessions": int(counts["sessions"].get(p, 0))}
            for p in ex.PATTERNS]
    return pd.DataFrame(rows, columns=columns)


# --- the headline: which muscles are falling behind -------------------------

LAG_WEIGHTS = {"volume": 0.40, "trend": 0.25, "recency": 0.20, "balance": 0.15}


def lagging_muscles(volume_summary: pd.DataFrame, trends: pd.DataFrame,
                    weekly_sets_min: int = 10, weekly_sets_max: int = 20) -> pd.DataFrame:
    """Score every muscle 0–100 on how much attention it needs.

    Four independent signals are combined, so a muscle flagged here is not just
    "low volume" — the reasons column says which of them fired:

    ``volume``   weekly effective sets below the target minimum
    ``trend``    estimated-1RM trend flat or negative
    ``recency``  long gap since the muscle was last loaded
    ``balance``  trained much less than its structural counterpart
    """
    columns = ["muscle", "muscle_label", "region", "weekly_sets", "target_min",
               "pct_of_target", "pct_per_month", "days_since", "attention_score",
               "volume_component", "trend_component", "recency_component",
               "balance_component", "reasons", "verdict"]
    if volume_summary.empty:
        return pd.DataFrame(columns=columns)

    trend_map = {}
    if trends is not None and not trends.empty:
        trend_map = trends.set_index("muscle")["pct_per_month"].to_dict()

    balance_penalty = _balance_penalties(volume_summary)

    rows = []
    for row in volume_summary.itertuples():
        weekly = float(row.weekly_sets or 0.0)
        coverage = weekly / weekly_sets_min if weekly_sets_min else 1.0
        volume_component = float(np.clip(1.0 - coverage, 0.0, 1.0))

        pct = trend_map.get(row.muscle, np.nan)
        if pct is None or pd.isna(pct):
            # No trend evidence: stay neutral rather than punishing or excusing.
            trend_component = 0.35 if weekly > 0 else 0.5
        else:
            trend_component = float(np.clip((0.8 - pct) / 1.8, 0.0, 1.0))

        days_since = row.days_since
        if days_since is None or pd.isna(days_since):
            recency_component = 1.0
        else:
            recency_component = float(np.clip((float(days_since) - 7.0) / 21.0, 0.0, 1.0))

        balance_component = float(balance_penalty.get(row.muscle, 0.0))

        score = 100.0 * (
            LAG_WEIGHTS["volume"] * volume_component
            + LAG_WEIGHTS["trend"] * trend_component
            + LAG_WEIGHTS["recency"] * recency_component
            + LAG_WEIGHTS["balance"] * balance_component
        )

        reasons = []
        if volume_component >= 0.5:
            reasons.append(f"{weekly:.1f} sets/wk vs {weekly_sets_min} target")
        elif weekly > weekly_sets_max:
            reasons.append(f"{weekly:.1f} sets/wk is above the {weekly_sets_max} ceiling")
        if pct is not None and not pd.isna(pct) and pct < 0.4:
            reasons.append(f"strength trend {pct:+.1f}%/month")
        if recency_component >= 0.5 and pd.notna(days_since):
            reasons.append(f"not trained for {int(days_since)} days")
        if balance_component >= 0.4:
            reasons.append("trained much less than its counterpart")
        if not reasons:
            reasons.append("on track")

        rows.append({
            "muscle": row.muscle,
            "muscle_label": row.muscle_label,
            "region": row.region,
            "weekly_sets": round(weekly, 2),
            "target_min": weekly_sets_min,
            "pct_of_target": round(coverage * 100, 0),
            "pct_per_month": pct if pct is None or pd.isna(pct) else round(float(pct), 2),
            "days_since": days_since,
            "attention_score": round(score, 1),
            "volume_component": round(volume_component, 2),
            "trend_component": round(trend_component, 2),
            "recency_component": round(recency_component, 2),
            "balance_component": round(balance_component, 2),
            "reasons": "; ".join(reasons),
            "verdict": _verdict(score),
        })

    return (pd.DataFrame(rows, columns=columns)
            .sort_values("attention_score", ascending=False).reset_index(drop=True))


def _verdict(score: float) -> str:
    if score >= 60:
        return "falling behind"
    if score >= 40:
        return "watch"
    if score >= 22:
        return "adequate"
    return "well trained"


def _balance_penalties(volume_summary: pd.DataFrame) -> dict[str, float]:
    """How far below its counterpart each muscle sits, as a 0–1 penalty."""
    sets_by_muscle = volume_summary.set_index("muscle")["weekly_sets"].to_dict()
    penalties: dict[str, float] = {}
    for _pair, numerator, denominator, (low, high) in BALANCE_PAIRS:
        num = sum(sets_by_muscle.get(m, 0.0) for m in numerator)
        den = sum(sets_by_muscle.get(m, 0.0) for m in denominator)
        if num <= 0 and den <= 0:
            continue
        ratio = num / den if den > 0 else float("inf")
        if ratio > high:  # denominator side is the weak one
            shortfall = min((ratio - high) / max(high, 0.1), 1.0)
            behind = denominator
        elif ratio < low:
            shortfall = min((low - ratio) / max(low, 0.1), 1.0)
            behind = numerator
        else:
            continue
        for muscle in behind:
            penalties[muscle] = max(penalties.get(muscle, 0.0), float(shortfall))
    return penalties


# --- findings ---------------------------------------------------------------

def strength_findings(lagging: pd.DataFrame, balance: pd.DataFrame, progress: pd.DataFrame,
                      patterns: pd.DataFrame, unmapped: pd.DataFrame,
                      weekly_sets_max: int = 20) -> list[Finding]:
    """Turn the strength tables into ranked, evidence-backed observations."""
    findings: list[Finding] = []

    if lagging is None or lagging.empty:
        return [Finding(area="strength", title="No strength data yet",
                        detail="No mapped strength sets were found, so muscle balance can't "
                               "be assessed. Sync a strength workout logged on the watch.",
                        severity="info")]

    behind = lagging[lagging["attention_score"] >= 40]
    for row in behind.head(6).itertuples():
        severity = "act" if row.attention_score >= 60 else "watch"
        trend_text = ("no reliable trend" if pd.isna(row.pct_per_month)
                      else f"{row.pct_per_month:+.1f}%/month")
        findings.append(Finding(
            area="strength",
            subject=row.muscle,
            title=f"{row.muscle_label} is {row.verdict}",
            detail=(f"{row.weekly_sets:.1f} effective sets/week "
                    f"({row.pct_of_target:.0f}% of the {row.target_min}-set target), "
                    f"strength trend {trend_text}. {row.reasons}."),
            severity=severity,
            metric=f"score {row.attention_score:.0f}/100",
            recommendation=_lag_recommendation(row),
            evidence={
                "weekly_sets": row.weekly_sets,
                "target_min": row.target_min,
                "pct_per_month": None if pd.isna(row.pct_per_month) else row.pct_per_month,
                "days_since": None if pd.isna(row.days_since) else int(row.days_since),
                "components": {
                    "volume": row.volume_component, "trend": row.trend_component,
                    "recency": row.recency_component, "balance": row.balance_component,
                },
            },
        ))

    strongest = lagging[lagging["attention_score"] < 22]
    if not strongest.empty:
        names = ", ".join(strongest.head(4)["muscle_label"])
        findings.append(Finding(
            area="strength", title="Well-covered muscles",
            detail=f"{names} are hitting volume targets with positive strength trends.",
            severity="good", metric=f"{len(strongest)} muscles"))

    overloaded = lagging[lagging["weekly_sets"] > weekly_sets_max]
    if not overloaded.empty:
        row = overloaded.sort_values("weekly_sets", ascending=False).iloc[0]
        findings.append(Finding(
            area="strength", subject=row["muscle"],
            title=f"{row['muscle_label']} volume is above the productive ceiling",
            detail=(f"{row['weekly_sets']:.1f} effective sets/week versus a "
                    f"{weekly_sets_max}-set ceiling. Extra sets here are competing for "
                    f"recovery with muscles that need the work."),
            severity="watch", metric=f"{row['weekly_sets']:.1f} sets/wk",
            recommendation="Redirect 3–5 sets/week from here to the lagging muscles above."))

    if balance is not None and not balance.empty:
        for row in balance[balance["status"] != "balanced"].itertuples():
            if row.status == "no antagonist work":
                detail = (f"{row.numerator_sets:.1f} sets/week on {row.numerator} against "
                          f"none on {row.denominator}.")
            else:
                behind_side = row.denominator if row.status == "denominator behind" else row.numerator
                detail = (f"Ratio {row.ratio:.2f} sits outside the {row.low}–{row.high} "
                          f"healthy band; {behind_side} is the side behind.")
            findings.append(Finding(
                area="strength", subject=row.pair,
                title=f"{row.pair} balance is off",
                detail=detail, severity="watch",
                metric=f"{row.ratio:.2f}" if pd.notna(row.ratio) else "no data",
                recommendation="Even this out before adding load to the dominant side.",
                evidence={"ratio": None if pd.isna(row.ratio) else row.ratio,
                          "healthy_range": [row.low, row.high],
                          "numerator_sets": row.numerator_sets,
                          "denominator_sets": row.denominator_sets}))

    if progress is not None and not progress.empty:
        stalled = progress[(progress["reliable"]) & (progress["status"].isin(
            ["stalled", "regressing"])) & (progress["sessions"] >= MIN_SESSIONS_FOR_TREND)]
        for row in stalled.head(4).itertuples():
            findings.append(Finding(
                area="strength", subject=row.exercise,
                title=f"{row.exercise} has {row.status}",
                detail=(f"Estimated 1RM {row.pct_per_month:+.1f}%/month "
                        f"({row.kg_per_month:+.1f} kg/month) across {row.sessions} sessions; "
                        f"currently {row.current_e1rm:.0f} kg with a best of {row.best_e1rm:.0f} kg."),
                severity="act" if row.status == "regressing" else "watch",
                metric=f"{row.pct_per_month:+.1f}%/mo",
                recommendation=("Change the stimulus: add a set, drop reps and add load, or "
                                "swap in a close variation for 4–6 weeks."),
                evidence={"pct_per_month": row.pct_per_month, "sessions": row.sessions,
                          "current_e1rm": row.current_e1rm, "best_e1rm": row.best_e1rm}))

        climbing = progress[(progress["reliable"]) & (progress["status"] == "progressing")]
        if not climbing.empty:
            best = climbing.sort_values("pct_per_month", ascending=False).iloc[0]
            findings.append(Finding(
                area="strength", subject=best["exercise"],
                title=f"{best['exercise']} is progressing well",
                detail=(f"Estimated 1RM up {best['pct_per_month']:+.1f}%/month "
                        f"({best['kg_per_month']:+.1f} kg/month) over "
                        f"{best['sessions']} sessions."),
                severity="good", metric=f"{best['pct_per_month']:+.1f}%/mo"))

    if patterns is not None and not patterns.empty:
        missing = patterns[(patterns["weekly_sets"] < 1.0)
                           & (~patterns["pattern"].isin([ex.CARRY_P, ex.ISOLATION]))]
        if not missing.empty:
            names = ", ".join(p.replace("_", " ") for p in missing["pattern"])
            findings.append(Finding(
                area="strength", title="Movement patterns barely trained",
                detail=f"Under one set per week of: {names}. Whole patterns missing usually "
                       f"means whole muscle groups under-stimulated.",
                severity="watch", metric=f"{len(missing)} patterns",
                recommendation="Add one exercise per missing pattern to the weakest session."))

    if unmapped is not None and not unmapped.empty:
        total = int(unmapped["sets"].sum())
        top = ", ".join(str(n) for n in unmapped["exercise_name"].head(3) if n)
        findings.append(Finding(
            area="strength", title="Some exercises aren't mapped to muscles",
            detail=(f"{total} sets across {len(unmapped)} exercises were excluded from the "
                    f"muscle model{f' (e.g. {top})' if top else ''}."),
            severity="info", metric=f"{total} sets",
            recommendation="Add these to NAME_PROFILES in domain/exercises.py to include them."))

    return findings


def _lag_recommendation(row) -> str:
    deficit = max(row.target_min - row.weekly_sets, 0)
    if pd.notna(row.days_since) and row.days_since >= 21:
        return (f"{row.muscle_label} hasn't been loaded in {int(row.days_since)} days — get it "
                f"back into the weekly rotation, starting at 2–3 sets per session.")
    if deficit >= 4:
        return (f"Add about {deficit:.0f} effective sets/week for {row.muscle_label} — roughly "
                f"{max(1, round(deficit / 3))} extra exercise slot(s) per week.")
    if row.trend_component >= 0.6:
        return (f"Volume is adequate but strength isn't moving: progress load on the main "
                f"{row.muscle_label} lift, or reduce reps and add weight.")
    return f"Nudge {row.muscle_label} volume up by 2–3 sets/week and re-check in a month."


def _latest(expanded: pd.DataFrame) -> datetime | None:
    if expanded is None or expanded.empty:
        return None
    return expanded["local_date"].max()
