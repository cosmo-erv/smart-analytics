"""Turns the diagnosis into a decision: what to train next, and what this week needs.

Everything else in this package describes the past. This module answers the
question actually asked at 6am: *what should I do today?*

It is deliberately rule-based and transparent rather than clever. Every
recommendation carries the reasons that produced it, so it can be overruled by
someone who knows something the data doesn't. The ordering of gates matters and
reflects how a coach would triage:

1. **Safety first** — a severe open niggle, or a load ratio in the danger band,
   overrides everything. No amount of muscle imbalance justifies training into an
   injury.
2. **Recovery state** — Garmin's readiness and remaining recovery time gate hard
   sessions; if the body isn't ready, a quality session just makes fatigue.
3. **Interference** — a hard session yesterday constrains what's useful today
   (heavy legs after a quality run, or the reverse).
4. **Then the biggest gap** — the lagging muscle, the missing easy volume, or the
   absent quality session, whichever the data says matters most.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .findings import Finding
from .running import format_pace

# Readiness below this means no hard session today.
READINESS_HARD_FLOOR = 50
READINESS_EASY_FLOOR = 30
# Remaining recovery time (hours) above which a quality session is premature.
RECOVERY_HOURS_BLOCK = 20
ACWR_CEILING = 1.5
# A muscle needs this many effective sets short of target to earn a dedicated slot.
DEFICIT_FOR_SLOT = 3.0


@dataclass
class Recommendation:
    kind: str                     # rest | easy_run | long_run | quality_run | strength | mobility
    title: str
    detail: str
    targets: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    alternative: str | None = None
    confidence: str = "medium"    # low | medium | high

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "title": self.title, "detail": self.detail,
            "targets": self.targets, "reasons": self.reasons,
            "alternative": self.alternative, "confidence": self.confidence,
        }


def _latest_metric(athlete_metrics: pd.DataFrame, column: str) -> float | None:
    if athlete_metrics is None or athlete_metrics.empty or column not in athlete_metrics:
        return None
    series = athlete_metrics[column].dropna()
    return float(series.iloc[-1]) if not series.empty else None


def _days_since_type(activities: pd.DataFrame, activity_type: str,
                     as_of: date) -> int | None:
    if activities is None or activities.empty:
        return None
    subset = activities[activities["activity_type"] == activity_type]
    if subset.empty:
        return None
    last = pd.to_datetime(subset["local_date"]).max()
    return int((pd.Timestamp(as_of) - last).days)


def _days_since_hard_run(runs: pd.DataFrame, as_of: date) -> int | None:
    if runs is None or runs.empty:
        return None
    hard = runs[runs["intensity"] == "hard"]
    if hard.empty:
        return None
    return int((pd.Timestamp(as_of) - pd.to_datetime(hard["local_date"]).max()).days)


def _leg_session_yesterday(expanded: pd.DataFrame, as_of: date) -> bool:
    from .hybrid import leg_sessions
    legs = leg_sessions(expanded)
    if legs.empty:
        return False
    recent = pd.to_datetime(legs["local_date"]).max()
    return bool((pd.Timestamp(as_of) - recent).days <= 1)


def next_session(report, as_of: date | None = None) -> Recommendation:
    """The single most useful thing to do today, with its reasoning."""
    as_of = as_of or date.today()
    reasons: list[str] = []

    readiness = _latest_metric(report.athlete_metrics, "readiness_score")
    recovery_h = _latest_metric(report.athlete_metrics, "recovery_time_h")
    zone_model = report.zone_model
    easy_zone = zone_model.get("easy") if zone_model and zone_model.has_pace_zones else None
    threshold_zone = zone_model.get("threshold") if zone_model and zone_model.has_pace_zones \
        else None
    interval_zone = zone_model.get("interval") if zone_model and zone_model.has_pace_zones \
        else None

    acwr = None
    if not report.load_series.empty:
        latest = report.load_series.dropna(subset=["acwr"]).tail(1)
        if not latest.empty:
            acwr = float(latest.iloc[0]["acwr"])

    # --- gate 1: open niggles ---------------------------------------------
    severe = pd.DataFrame()
    if getattr(report, "niggle_context", None) is not None \
            and not report.niggle_context.empty:
        open_niggles = report.niggle_context[report.niggle_context["status"] == "open"]
        severe = open_niggles[open_niggles["severity"].fillna(0) >= 4]
        if not severe.empty:
            areas = ", ".join(severe["area"].unique())
            return Recommendation(
                kind="rest",
                title="Rest or cross-train — an open niggle is limiting you",
                detail=(f"You have an open niggle ({areas}) logged at severity 4 or above, "
                        f"which means it's already cutting sessions short. Training through "
                        f"that is how a niggle becomes an injury."),
                reasons=[f"{areas} logged at severity ≥4"],
                alternative=("If it's pain-free, swimming or easy cycling keeps aerobic fitness "
                             "without loading the area."),
                confidence="high",
            )
        moderate = open_niggles[open_niggles["severity"].fillna(0) == 3]
        if not moderate.empty:
            reasons.append(f"open niggle ({', '.join(moderate['area'].unique())}) "
                           f"— avoid loading it")

    # --- gate 2: load and recovery ----------------------------------------
    hard_blocked = False
    if acwr is not None and acwr >= ACWR_CEILING:
        hard_blocked = True
        reasons.append(f"load ratio {acwr:.2f} is in the high-risk band")
    if readiness is not None and readiness < READINESS_EASY_FLOOR:
        return Recommendation(
            kind="rest",
            title="Take a rest day",
            detail=(f"Garmin readiness is {readiness:.0f}/100"
                    + (f" with {recovery_h:.0f} hours of recovery still showing"
                       if recovery_h else "")
                    + ". At this level even easy training adds fatigue faster than fitness."),
            reasons=[f"readiness {readiness:.0f}/100"]
                    + ([f"{recovery_h:.0f} h recovery remaining"] if recovery_h else []),
            alternative="Walking or mobility work is fine.",
            confidence="high",
        )
    if readiness is not None and readiness < READINESS_HARD_FLOOR:
        hard_blocked = True
        reasons.append(f"readiness {readiness:.0f}/100 is too low for quality work")
    if recovery_h is not None and recovery_h >= RECOVERY_HOURS_BLOCK:
        hard_blocked = True
        reasons.append(f"{recovery_h:.0f} h of recovery still remaining")

    # --- gate 3: interference from yesterday ------------------------------
    legs_yesterday = _leg_session_yesterday(report.expanded, as_of)
    days_since_hard_run = _days_since_hard_run(report.runs, as_of)
    if legs_yesterday:
        reasons.append("heavy legs in the last 24 h — quality running would be on tired legs")
    if days_since_hard_run is not None and days_since_hard_run <= 1:
        reasons.append("hard run in the last 24 h — heavy legs today would compound it")

    # --- the biggest gap --------------------------------------------------
    lagging = report.lagging
    deficits = []
    if lagging is not None and not lagging.empty:
        target = report.settings.weekly_sets_min
        candidates = lagging[(lagging["attention_score"] >= 40)].copy()
        candidates["deficit"] = (target - candidates["weekly_sets"]).clip(lower=0)
        deficits = [
            (row.muscle_label, float(row.deficit), row.days_since)
            for row in candidates.itertuples() if row.deficit >= DEFICIT_FOR_SLOT
        ]

    days_since_strength = _days_since_type(report.activities, "strength_training", as_of)
    easy_share = report.intensity.get("easy_pct") if report.intensity else None

    # A. Strength slot: real deficits, legs not just trained, and it's been a day.
    if deficits and (days_since_strength is None or days_since_strength >= 1) \
            and not (legs_yesterday and _is_lower_body(deficits)):
        top = deficits[:3]
        targets = [f"{name}: {deficit:.0f} effective sets" for name, deficit, _ in top]
        reasons.append(f"{len(deficits)} muscle groups are short of the weekly target")
        if days_since_strength is not None:
            reasons.append(f"last strength session {days_since_strength} days ago")
        return Recommendation(
            kind="strength",
            title=f"Strength session — prioritise {top[0][0].lower()}",
            detail=(f"The biggest gaps in your training are muscular, not aerobic. "
                    f"{top[0][0]} is {top[0][1]:.0f} effective sets short of the weekly target"
                    + (f", and hasn't been loaded in {int(top[0][2])} days"
                       if pd.notna(top[0][2]) and top[0][2] > 7 else "")
                    + ". Build the session around these and treat everything else as optional."),
            targets=targets,
            reasons=reasons,
            alternative=("If you're short on time, three hard sets of one movement per listed "
                         "muscle beats a full session you skip."),
            confidence="high",
        )

    # B. Quality run: none recently, body is ready, legs are fresh.
    if not hard_blocked and not legs_yesterday and (
            days_since_hard_run is None or days_since_hard_run >= 4):
        pace_text = ""
        if threshold_zone and interval_zone:
            pace_text = (f" Target {threshold_zone.range_label} for tempo work, or "
                         f"{interval_zone.range_label} for shorter reps.")
        reasons.append("no quality run in the last four days"
                       if days_since_hard_run else "no hard run on record")
        if readiness:
            reasons.append(f"readiness {readiness:.0f}/100 supports it")
        return Recommendation(
            kind="quality_run",
            title="Quality run — tempo or intervals",
            detail=("You're recovered and it's been long enough since the last hard effort."
                    + pace_text
                    + " Make it genuinely hard; the analysis shows the risk here is drifting "
                      "into moderate rather than going too deep."),
            targets=([f"Tempo: 20–30 min at {threshold_zone.range_label}"]
                     if threshold_zone else [])
                    + ([f"Or 5–6 × 3 min at {interval_zone.range_label}"]
                       if interval_zone else []),
            reasons=reasons,
            alternative="If legs feel flat in the warm-up, convert it to an easy run.",
            confidence="high" if readiness else "medium",
        )

    # C. Easy volume: the default, and the fix for a grey-zone distribution.
    if easy_zone:
        detail = (f"Easy running at {easy_zone.range_label} is what your distribution is short "
                  f"of.")
        if easy_share is not None and easy_share < 70:
            detail += (f" Only {easy_share:.0f}% of your running time is genuinely easy — the "
                       f"discipline to run this slowly is the single biggest available gain.")
            reasons.append(f"easy share is {easy_share:.0f}% (target 75–80%)")
        if hard_blocked:
            detail += " Today is not a day for quality work anyway."
        return Recommendation(
            kind="easy_run",
            title=f"Easy run at {easy_zone.range_label}",
            detail=detail,
            targets=[f"40–70 minutes at {easy_zone.range_label}",
                     "Heart rate in zone 2, conversational throughout"],
            reasons=reasons or ["no higher-priority gap today"],
            alternative="Swap for a rest day if yesterday was hard and legs feel heavy.",
            confidence="medium",
        )

    return Recommendation(
        kind="easy_run",
        title="Easy aerobic session",
        detail=("Nothing in the data points to a specific priority today, so default to easy "
                "aerobic work. Sync Garmin's threshold data to get target paces here."),
        reasons=reasons or ["insufficient data for a specific recommendation"],
        confidence="low",
    )


def _is_lower_body(deficits: list[tuple[str, float, Any]]) -> bool:
    lower = {"quads", "hamstrings", "glutes", "calves", "adductors"}
    return any(name.lower().split(" /")[0] in lower for name, _, _ in deficits[:2])


def weekly_targets(report) -> dict[str, Any]:
    """What this week needs, in countable terms rather than advice."""
    out: dict[str, Any] = {"strength": [], "running": [], "notes": []}

    lagging = report.lagging
    if lagging is not None and not lagging.empty:
        target = report.settings.weekly_sets_min
        for row in lagging.itertuples():
            deficit = target - float(row.weekly_sets)
            if deficit >= 2:
                out["strength"].append({
                    "muscle": row.muscle_label,
                    "current": round(float(row.weekly_sets), 1),
                    "target": target,
                    "add_sets": round(deficit, 0),
                    "days_since": (None if pd.isna(row.days_since) else int(row.days_since)),
                })
        out["strength"] = sorted(out["strength"], key=lambda item: -item["add_sets"])[:6]

    weekly = report.weekly_runs
    if weekly is not None and not weekly.empty:
        recent = float(weekly.tail(4)["distance_km"].mean())
        this_week = float(weekly.tail(1)["distance_km"].iloc[0])
        # 10% above the four-week mean is the conventional safe ceiling.
        ceiling = recent * 1.1
        out["running"].append({
            "metric": "Weekly distance",
            "current": round(this_week, 1),
            "target": round(ceiling, 1),
            "unit": "km",
            "note": (f"{max(ceiling - this_week, 0):.1f} km remaining before you'd exceed a "
                     f"10% step up on your four-week average of {recent:.0f} km."),
        })

    if report.intensity:
        easy_pct = report.intensity["easy_pct"]
        if easy_pct < 75:
            out["running"].append({
                "metric": "Easy share of running time",
                "current": round(easy_pct, 0),
                "target": 78,
                "unit": "%",
                "note": "Slow the easy runs rather than cutting the hard ones.",
            })

    zone_model = report.zone_model
    if zone_model and zone_model.has_pace_zones:
        easy = zone_model.get("easy")
        threshold = zone_model.get("threshold")
        out["notes"].append(f"Easy pace {easy.range_label}; tempo {threshold.range_label}.")
    else:
        out["notes"].append("No threshold data from Garmin yet, so no target paces available.")

    if getattr(report, "hybrid_events", None) is not None and not report.hybrid_events.empty:
        out["notes"].append(
            "Keep 36 hours between heavy legs and any quality run — recent weeks have had "
            "collisions.")

    return out


def prescription_findings(recommendation: Recommendation, targets: dict[str, Any]) -> list[Finding]:
    """Surface the recommendation as a finding so it reaches the coach briefing."""
    findings = [Finding(
        area="plan", title=f"Next session: {recommendation.title}",
        detail=recommendation.detail,
        severity="info",
        metric=recommendation.kind.replace("_", " "),
        recommendation="; ".join(recommendation.targets) if recommendation.targets else None,
        evidence={"reasons": recommendation.reasons, "confidence": recommendation.confidence},
    )]

    if targets.get("strength"):
        top = targets["strength"][:3]
        listing = ", ".join(f"{item['muscle']} +{item['add_sets']:.0f}" for item in top)
        findings.append(Finding(
            area="plan", title="Strength volume to add this week",
            detail=f"Effective sets short of target: {listing}.",
            severity="info", metric=f"{len(targets['strength'])} muscles",
            evidence={"targets": targets["strength"]}))
    return findings
