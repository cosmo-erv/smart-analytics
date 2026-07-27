"""Concurrent-training analysis for athletes doing strength and running together.

Training both at once isn't just two programmes side by side — they compete. The
specific costs are well established and all of them are *scheduling* problems
rather than volume problems:

* **Heavy lower-body work degrades running quality for 24–48 hours.** A hard leg
  session the day before an interval session means the reps are run on tired legs,
  so the session under-delivers even though both were "completed".
* **Running before lifting blunts strength adaptation** more than the reverse. In
  a same-day double, order matters — put the priority session first.
* **Hard days should cluster.** Spreading hard sessions across every day of the
  week means never being fully recovered *or* fully loaded; stacking them leaves
  genuinely easy days between.

So this module looks at the calendar, not the totals: which sessions collided,
which order they ran in, and how the week is actually structured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .findings import Finding
from ..domain.muscles import ADDUCTORS, CALVES, GLUTES, HAMSTRINGS, QUADS

LOWER_BODY = {QUADS, HAMSTRINGS, GLUTES, CALVES, ADDUCTORS}

# A strength session counts as leg-dominant above this share of its effective sets.
LEG_SHARE_THRESHOLD = 0.35
# Hours after heavy legs during which running quality is measurably compromised.
INTERFERENCE_WINDOW_H = 30
# Hard sessions closer together than this aren't separated by real recovery.
MIN_HARD_GAP_H = 40


def leg_sessions(expanded: pd.DataFrame) -> pd.DataFrame:
    """Strength sessions dominated by lower-body work, with their leg share."""
    columns = ["activity_id", "local_date", "leg_sets", "total_sets", "leg_share"]
    if expanded is None or expanded.empty:
        return pd.DataFrame(columns=columns)

    frame = expanded.copy()
    frame["is_leg"] = frame["muscle"].isin(LOWER_BODY)
    grouped = frame.groupby(["activity_id", "local_date"], as_index=False).agg(
        leg_sets=("effective_sets", lambda s: float(s[frame.loc[s.index, "is_leg"]].sum())),
        total_sets=("effective_sets", "sum"),
    )
    grouped["leg_share"] = np.where(grouped["total_sets"] > 0,
                                    grouped["leg_sets"] / grouped["total_sets"], 0.0)
    legs = grouped[grouped["leg_share"] >= LEG_SHARE_THRESHOLD].copy()
    return legs[columns].sort_values("local_date").reset_index(drop=True)


def hard_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Runs that carry a real quality stimulus."""
    if runs is None or runs.empty:
        return pd.DataFrame(columns=["activity_id", "local_date", "start_time", "name",
                                     "distance_km", "intensity"])
    hard = runs[runs["intensity"] == "hard"].copy()
    if hard.empty:
        # Fall back to the longest 20% of runs — for many athletes the long run is
        # the week's real stress even when heart rate stays moderate.
        threshold = runs["distance_km"].quantile(0.8)
        hard = runs[runs["distance_km"] >= threshold].copy()
    keep = [c for c in ["activity_id", "local_date", "start_time", "name", "distance_km",
                        "intensity"] if c in hard.columns]
    return hard[keep].sort_values("local_date").reset_index(drop=True)


def interference_events(activities: pd.DataFrame, expanded: pd.DataFrame,
                        runs: pd.DataFrame, lookback_days: int = 84) -> pd.DataFrame:
    """Find leg sessions and quality runs that landed too close together."""
    columns = ["date", "kind", "gap_hours", "first", "second", "detail"]
    legs = leg_sessions(expanded)
    quality = hard_runs(runs)
    if legs.empty or quality.empty or activities is None or activities.empty:
        return pd.DataFrame(columns=columns)

    times = activities.set_index("activity_id")["start_time"].to_dict()
    as_of = pd.to_datetime(activities["local_date"]).max()
    cutoff = as_of - pd.Timedelta(days=lookback_days)

    leg_times = [(aid, pd.to_datetime(times.get(aid)), share)
                 for aid, share in zip(legs["activity_id"], legs["leg_share"])
                 if times.get(aid) is not None]
    run_times = [(aid, pd.to_datetime(times.get(aid)), name)
                 for aid, name in zip(quality["activity_id"],
                                      quality.get("name", quality["activity_id"]))
                 if times.get(aid) is not None]

    rows = []
    for run_id, run_at, run_name in run_times:
        if pd.isna(run_at) or run_at < cutoff:
            continue
        for leg_id, leg_at, share in leg_times:
            if pd.isna(leg_at):
                continue
            gap_h = (run_at - leg_at).total_seconds() / 3600.0
            if 0 <= gap_h <= INTERFERENCE_WINDOW_H:
                same_day = run_at.date() == leg_at.date()
                rows.append({
                    "date": run_at.date().isoformat(),
                    "kind": "run after legs (same day)" if same_day else "run after legs",
                    "gap_hours": round(gap_h, 1),
                    "first": f"Leg session ({share * 100:.0f}% lower body)",
                    "second": str(run_name or "Quality run"),
                    "detail": (f"Quality run {gap_h:.0f} h after a leg-dominant lift — "
                               f"legs were unlikely to be fresh."),
                })
            elif -INTERFERENCE_WINDOW_H <= gap_h < 0 and run_at.date() == leg_at.date():
                rows.append({
                    "date": run_at.date().isoformat(),
                    "kind": "legs after run (same day)",
                    "gap_hours": round(abs(gap_h), 1),
                    "first": str(run_name or "Quality run"),
                    "second": f"Leg session ({share * 100:.0f}% lower body)",
                    "detail": (f"Lifted {abs(gap_h):.0f} h after a quality run — running first "
                               f"costs more strength adaptation than the reverse."),
                })
    return (pd.DataFrame(rows, columns=columns)
            .drop_duplicates(subset=["date", "kind"]).sort_values("date", ascending=False)
            .reset_index(drop=True))


def discipline_split(activities: pd.DataFrame, lookback_days: int = 84) -> pd.DataFrame:
    """Weekly load and hours split between strength, running and everything else."""
    columns = ["week", "strength_load", "running_load", "other_load", "strength_pct",
               "running_pct", "strength_hours", "running_hours"]
    if activities is None or activities.empty:
        return pd.DataFrame(columns=columns)

    frame = activities.copy()
    as_of = pd.to_datetime(frame["local_date"]).max()
    frame = frame[pd.to_datetime(frame["local_date"]) >= as_of - pd.Timedelta(days=lookback_days)]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame["week"] = pd.to_datetime(frame["local_date"]).dt.to_period("W").dt.start_time
    frame["bucket"] = np.where(frame["activity_type"] == "strength_training", "strength",
                               np.where(frame["activity_type"] == "running", "running", "other"))
    frame["load"] = frame["training_load"].fillna(0)
    frame["hours"] = frame["duration_s"].fillna(0) / 3600

    pivot = frame.pivot_table(index="week", columns="bucket", values="load",
                              aggfunc="sum").fillna(0)
    hours = frame.pivot_table(index="week", columns="bucket", values="hours",
                              aggfunc="sum").fillna(0)
    out = pd.DataFrame({"week": pivot.index})
    for bucket in ("strength", "running", "other"):
        out[f"{bucket}_load"] = pivot[bucket].to_numpy() if bucket in pivot else 0.0
    for bucket in ("strength", "running"):
        out[f"{bucket}_hours"] = hours[bucket].to_numpy() if bucket in hours else 0.0

    total = out[["strength_load", "running_load", "other_load"]].sum(axis=1).replace(0, np.nan)
    out["strength_pct"] = (out["strength_load"] / total * 100).round(0).fillna(0)
    out["running_pct"] = (out["running_load"] / total * 100).round(0).fillna(0)
    return out[columns].reset_index(drop=True)


def hard_day_structure(activities: pd.DataFrame, expanded: pd.DataFrame,
                       runs: pd.DataFrame, lookback_days: int = 84) -> dict:
    """Are hard sessions clustered onto shared days, or spread across the week?"""
    if activities is None or activities.empty:
        return {}

    legs = leg_sessions(expanded)
    quality = hard_runs(runs)
    hard_dates = pd.concat([
        pd.to_datetime(legs["local_date"]) if not legs.empty else pd.Series(dtype="datetime64[ns]"),
        pd.to_datetime(quality["local_date"]) if not quality.empty
        else pd.Series(dtype="datetime64[ns]"),
    ]).dropna()
    if hard_dates.empty:
        return {}

    as_of = pd.to_datetime(activities["local_date"]).max()
    cutoff = as_of - pd.Timedelta(days=lookback_days)
    hard_dates = hard_dates[hard_dates >= cutoff]
    if hard_dates.empty:
        return {}

    unique_days = sorted(set(hard_dates.dt.date))
    shared_days = int(len(hard_dates) - len(unique_days))  # two hard sessions, one day
    weeks = max(lookback_days / 7.0, 1)

    gaps = [(unique_days[i + 1] - unique_days[i]).days for i in range(len(unique_days) - 1)]
    back_to_back = sum(1 for gap in gaps if gap == 1)

    # A day with no hard session and no session at all is a true rest day.
    all_dates = set(pd.to_datetime(activities["local_date"]).dt.date)
    span_days = (as_of.date() - max(cutoff.date(), min(all_dates))).days + 1
    rest_days = span_days - len(all_dates & set(
        d for d in all_dates if d >= cutoff.date()))

    return {
        "hard_sessions": int(len(hard_dates)),
        "hard_days": len(unique_days),
        "hard_sessions_per_week": round(len(hard_dates) / weeks, 1),
        "shared_hard_days": shared_days,
        "back_to_back_hard_days": back_to_back,
        "mean_gap_days": round(float(np.mean(gaps)), 1) if gaps else np.nan,
        "rest_days_per_week": round(max(rest_days, 0) / weeks, 1),
    }


# --- findings ---------------------------------------------------------------

def hybrid_findings(events: pd.DataFrame, split: pd.DataFrame, structure: dict,
                    lookback_days: int = 84) -> list[Finding]:
    findings: list[Finding] = []
    weeks = max(lookback_days / 7.0, 1)

    if events is not None and not events.empty:
        per_week = len(events) / weeks
        same_day_runs_first = events[events["kind"] == "legs after run (same day)"]
        if per_week >= 0.6:
            findings.append(Finding(
                area="hybrid", subject="interference",
                title="Leg sessions and quality runs are colliding",
                detail=(f"{len(events)} collisions in the last {lookback_days} days "
                        f"({per_week:.1f} per week) where a quality run fell within "
                        f"{INTERFERENCE_WINDOW_H} hours of a leg-dominant lift. Both sessions "
                        f"happen, but the second one runs on compromised legs, so it delivers "
                        f"less than the calendar suggests."),
                severity="act" if per_week >= 1 else "watch",
                metric=f"{per_week:.1f}/week",
                recommendation=("Put at least 36 hours between heavy legs and any quality run — "
                                "in practice that means legs the day *after* the hard run, not "
                                "before."),
                evidence={"collisions": int(len(events)),
                          "per_week": round(per_week, 2)}))
        if len(same_day_runs_first) >= 2:
            findings.append(Finding(
                area="hybrid", subject="session order",
                title="Lifting after running on shared days",
                detail=(f"{len(same_day_runs_first)} sessions where the lift followed a quality "
                        f"run the same day. Running first costs more strength adaptation than "
                        f"lifting first costs running, so on a double day the priority session "
                        f"should lead."),
                severity="watch", metric=f"{len(same_day_runs_first)} sessions",
                recommendation=("If strength is the day's priority, lift first — or split the "
                                "two by at least six hours.")))
    elif split is not None and not split.empty:
        findings.append(Finding(
            area="hybrid", subject="interference",
            title="Strength and running are well separated",
            detail=("No leg sessions landed inside the interference window before a quality run "
                    "in this period — the two are scheduled so each gets fresh legs."),
            severity="good", metric="no collisions"))

    if split is not None and not split.empty:
        recent = split.tail(4)
        strength_pct = float(recent["strength_pct"].mean())
        running_pct = float(recent["running_pct"].mean())
        strength_hours = float(recent["strength_hours"].mean())
        running_hours = float(recent["running_hours"].mean())
        if strength_pct < 15 and running_pct > 70:
            findings.append(Finding(
                area="hybrid", subject="balance",
                title="The mix has drifted toward running",
                detail=(f"Over the last four weeks strength is {strength_pct:.0f}% of training "
                        f"load ({strength_hours:.1f} h/week) against running's "
                        f"{running_pct:.0f}% ({running_hours:.1f} h/week). For a hybrid goal "
                        f"that's running with some lifting attached, not concurrent training."),
                severity="watch", metric=f"{strength_pct:.0f}% strength",
                recommendation="Protect two full strength sessions a week before adding run volume.",
                evidence={"strength_pct": strength_pct, "running_pct": running_pct}))
        elif running_pct < 25 and strength_pct > 60:
            findings.append(Finding(
                area="hybrid", subject="balance",
                title="The mix has drifted toward strength",
                detail=(f"Strength is {strength_pct:.0f}% of load ({strength_hours:.1f} h/week) "
                        f"versus running's {running_pct:.0f}% ({running_hours:.1f} h/week). "
                        f"Aerobic fitness will drift down at this ratio."),
                severity="watch", metric=f"{running_pct:.0f}% running",
                recommendation="Add one easy aerobic run per week before adding lifting volume."))
        else:
            findings.append(Finding(
                area="hybrid", subject="balance",
                title="Strength and running load are reasonably balanced",
                detail=(f"Last four weeks: {strength_pct:.0f}% of load from strength "
                        f"({strength_hours:.1f} h/week), {running_pct:.0f}% from running "
                        f"({running_hours:.1f} h/week)."),
                severity="good", metric=f"{strength_pct:.0f}/{running_pct:.0f} split"))

    if structure:
        if structure.get("back_to_back_hard_days", 0) >= max(2, weeks * 0.4):
            findings.append(Finding(
                area="hybrid", subject="week structure",
                title="Hard days are running back to back",
                detail=(f"{structure['back_to_back_hard_days']} instances of consecutive hard "
                        f"days, with {structure['hard_sessions_per_week']:.1f} hard sessions per "
                        f"week across {structure['hard_days']} separate days. Consecutive hard "
                        f"days leave neither session fully supported."),
                severity="watch", metric=f"{structure['back_to_back_hard_days']} pairs",
                recommendation=("Stack hard work onto the same day where you can — two hard "
                                "sessions on one day buys a genuinely easy day after it."),
                evidence=structure))
        elif structure.get("shared_hard_days", 0) >= 2:
            findings.append(Finding(
                area="hybrid", subject="week structure",
                title="Hard sessions are sensibly clustered",
                detail=(f"{structure['shared_hard_days']} days carry two hard sessions, which "
                        f"concentrates stress and leaves the days between genuinely easy. "
                        f"Mean gap between hard days is {structure['mean_gap_days']:.1f} days."),
                severity="good", metric=f"{structure['hard_days']} hard days"))

        if structure.get("rest_days_per_week", 0) < 0.5:
            findings.append(Finding(
                area="hybrid", subject="rest",
                title="Almost no complete rest days",
                detail=(f"Only {structure['rest_days_per_week']:.1f} fully unloaded days per "
                        f"week. Concurrent training needs more recovery than either discipline "
                        f"alone, not less."),
                severity="watch", metric=f"{structure['rest_days_per_week']:.1f}/week",
                recommendation="Schedule one full rest day a week and hold it."))

    return findings
