"""Training load across every activity type, plus recovery context.

Strength and running can't be compared in their own units, so everything is
reduced to a single daily load number. Garmin's own ``activityTrainingLoad`` is
used when present; when it isn't (older devices, manual entries, most strength
sessions), load is estimated with Banister's TRIMP from heart rate, and the
estimate is flagged so the UI can say which is which.

From the daily series we derive:

* **ACWR** — 7-day load against the 28-day weekly average. The widely used
  injury-risk heuristic: below 0.8 is detraining, 0.8–1.3 productive, above 1.5
  is the danger zone.
* **CTL / ATL / TSB** — exponentially weighted fitness (42 d), fatigue (7 d) and
  the form balance between them.
* **Monotony and strain** (Foster) — training that never varies is harder to
  absorb than the same load with hard/easy contrast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .findings import Finding

CTL_DAYS = 42
ATL_DAYS = 7
ACUTE_DAYS = 7
CHRONIC_DAYS = 28

ACWR_BANDS = [
    (0.0, 0.80, "detraining", "watch"),
    (0.80, 1.30, "productive", "good"),
    (1.30, 1.50, "caution", "watch"),
    (1.50, float("inf"), "high risk", "act"),
]

DEFAULT_MAX_HR = 190.0
DEFAULT_RESTING_HR = 60.0


def infer_hr_bounds(activities: pd.DataFrame, daily: pd.DataFrame | None = None,
                    max_hr: int | None = None,
                    resting_hr: int | None = None) -> tuple[float, float]:
    """Best available max/resting HR: configured value, else inferred from data."""
    inferred_max = DEFAULT_MAX_HR
    if activities is not None and not activities.empty and activities["max_hr"].notna().any():
        # 99th percentile guards against a single spurious spike.
        inferred_max = float(np.nanpercentile(activities["max_hr"].dropna(), 99))
    inferred_rest = DEFAULT_RESTING_HR
    if daily is not None and not daily.empty and daily["resting_hr"].notna().any():
        inferred_rest = float(daily["resting_hr"].dropna().median())

    final_max = float(max_hr) if max_hr else inferred_max
    final_rest = float(resting_hr) if resting_hr else inferred_rest
    if final_max - final_rest < 40:  # implausible spread — fall back
        final_max, final_rest = DEFAULT_MAX_HR, DEFAULT_RESTING_HR
    return final_max, final_rest


def trimp(duration_s: float, avg_hr: float, max_hr: float, resting_hr: float) -> float:
    """Banister TRIMP — duration weighted exponentially by heart-rate reserve."""
    if not duration_s or duration_s <= 0 or not avg_hr or avg_hr <= 0:
        return 0.0
    reserve = (avg_hr - resting_hr) / max(max_hr - resting_hr, 1.0)
    reserve = float(np.clip(reserve, 0.0, 1.0))
    minutes = duration_s / 60.0
    return float(minutes * reserve * 0.64 * np.exp(1.92 * reserve))


def daily_load(activities: pd.DataFrame, max_hr: float = DEFAULT_MAX_HR,
               resting_hr: float = DEFAULT_RESTING_HR) -> pd.DataFrame:
    """One row per calendar day with total load, on a gap-free date index."""
    columns = ["date", "load", "estimated_share", "activities", "duration_min"]
    if activities is None or activities.empty:
        return pd.DataFrame(columns=columns)

    frame = activities.copy()
    reported = frame["training_load"].fillna(0)
    estimated = frame.apply(
        lambda row: trimp(row.get("duration_s") or 0, row.get("avg_hr") or 0, max_hr, resting_hr),
        axis=1,
    )
    frame["load"] = np.where(reported > 0, reported, estimated)
    frame["was_estimated"] = reported <= 0
    frame["date"] = pd.to_datetime(frame["local_date"]).dt.normalize()

    grouped = frame.groupby("date", as_index=False).agg(
        load=("load", "sum"),
        estimated_share=("was_estimated", "mean"),
        activities=("activity_id", "nunique"),
        duration_min=("duration_s", lambda s: float(s.fillna(0).sum()) / 60.0),
    )
    # Rest days must exist as zero-load rows or every rolling window is wrong.
    full_index = pd.date_range(grouped["date"].min(), grouped["date"].max(), freq="D")
    grouped = (grouped.set_index("date").reindex(full_index)
               .rename_axis("date").reset_index())
    grouped["load"] = grouped["load"].fillna(0.0)
    grouped["activities"] = grouped["activities"].fillna(0).astype(int)
    grouped["duration_min"] = grouped["duration_min"].fillna(0.0)
    grouped["estimated_share"] = grouped["estimated_share"].fillna(0.0)
    return grouped[columns]


def load_series(daily: pd.DataFrame) -> pd.DataFrame:
    """Add acute/chronic loads, ACWR, CTL/ATL/TSB and Foster monotony/strain."""
    columns = ["date", "load", "acute_7d", "chronic_28d", "chronic_weekly", "acwr",
               "ctl", "atl", "tsb", "monotony", "strain", "acwr_status"]
    if daily is None or daily.empty:
        return pd.DataFrame(columns=columns)

    frame = daily.copy().sort_values("date").reset_index(drop=True)
    load = frame["load"]

    frame["acute_7d"] = load.rolling(ACUTE_DAYS, min_periods=ACUTE_DAYS).sum()
    frame["chronic_28d"] = load.rolling(CHRONIC_DAYS, min_periods=CHRONIC_DAYS).sum()
    frame["chronic_weekly"] = frame["chronic_28d"] / (CHRONIC_DAYS / ACUTE_DAYS)
    frame["acwr"] = np.where(frame["chronic_weekly"] > 0,
                             frame["acute_7d"] / frame["chronic_weekly"], np.nan)

    # Exponentially weighted fitness / fatigue, seeded on the first observation.
    frame["ctl"] = load.ewm(alpha=1 / CTL_DAYS, adjust=False).mean()
    frame["atl"] = load.ewm(alpha=1 / ATL_DAYS, adjust=False).mean()
    frame["tsb"] = frame["ctl"] - frame["atl"]

    weekly_mean = load.rolling(ACUTE_DAYS, min_periods=ACUTE_DAYS).mean()
    weekly_std = load.rolling(ACUTE_DAYS, min_periods=ACUTE_DAYS).std(ddof=0)
    frame["monotony"] = np.where(weekly_std > 0, weekly_mean / weekly_std, np.nan)
    frame["strain"] = frame["acute_7d"] * frame["monotony"]
    frame["acwr_status"] = frame["acwr"].map(acwr_status)
    return frame[columns]


def acwr_status(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "insufficient history"
    for low, high, label, _severity in ACWR_BANDS:
        if low <= value < high:
            return label
    return "unknown"


def _acwr_severity(value: float) -> str:
    for low, high, _label, severity in ACWR_BANDS:
        if low <= value < high:
            return severity
    return "info"


def activity_mix(activities: pd.DataFrame, lookback_days: int = 84) -> pd.DataFrame:
    """Where the training time and load actually go, by activity type."""
    columns = ["activity_type", "sessions", "hours", "load", "load_share_pct", "distance_km"]
    if activities is None or activities.empty:
        return pd.DataFrame(columns=columns)
    as_of = pd.to_datetime(activities["local_date"]).max()
    window = activities[pd.to_datetime(activities["local_date"])
                        >= as_of - pd.Timedelta(days=lookback_days)]
    if window.empty:
        return pd.DataFrame(columns=columns)

    grouped = window.groupby("activity_type", as_index=False).agg(
        sessions=("activity_id", "nunique"),
        hours=("duration_s", lambda s: round(float(s.fillna(0).sum()) / 3600, 1)),
        load=("training_load", lambda s: round(float(s.fillna(0).sum()), 0)),
        distance_km=("distance_m", lambda s: round(float(s.fillna(0).sum()) / 1000, 1)),
    )
    total = grouped["load"].sum()
    grouped["load_share_pct"] = (grouped["load"] / total * 100).round(1) if total > 0 else 0.0
    return grouped.sort_values("hours", ascending=False).reset_index(drop=True)


def recovery_trends(daily_metrics: pd.DataFrame, lookback_days: int = 28) -> dict[str, float]:
    """Recent recovery markers against their longer-term baseline."""
    if daily_metrics is None or daily_metrics.empty:
        return {}
    frame = daily_metrics.copy()
    frame["local_date"] = pd.to_datetime(frame["local_date"])
    as_of = frame["local_date"].max()
    recent = frame[frame["local_date"] >= as_of - pd.Timedelta(days=lookback_days)]
    baseline = frame[(frame["local_date"] < as_of - pd.Timedelta(days=lookback_days))
                     & (frame["local_date"] >= as_of - pd.Timedelta(days=lookback_days * 4))]

    out: dict[str, float] = {}
    for column, key in [("resting_hr", "resting_hr"), ("hrv_ms", "hrv"),
                        ("sleep_hours", "sleep_hours"), ("sleep_score", "sleep_score"),
                        ("body_battery_high", "body_battery")]:
        if column not in frame.columns or not recent[column].notna().any():
            continue
        recent_mean = float(recent[column].dropna().mean())
        out[f"{key}_recent"] = round(recent_mean, 1)
        if not baseline.empty and baseline[column].notna().any():
            base_mean = float(baseline[column].dropna().mean())
            out[f"{key}_baseline"] = round(base_mean, 1)
            out[f"{key}_delta"] = round(recent_mean - base_mean, 1)
    return out


# --- findings ---------------------------------------------------------------

def load_findings(series: pd.DataFrame, mix: pd.DataFrame,
                  recovery: dict[str, float]) -> list[Finding]:
    findings: list[Finding] = []

    if series is None or series.empty:
        return [Finding(area="load", title="Not enough history for load analysis",
                        detail="At least 28 days of activities are needed for the acute:chronic "
                               "ratio.", severity="info")]

    latest = series.dropna(subset=["acwr"]).tail(1)
    if not latest.empty:
        row = latest.iloc[0]
        acwr = float(row["acwr"])
        severity = _acwr_severity(acwr)
        detail = (f"7-day load {row['acute_7d']:.0f} against a 28-day weekly average of "
                  f"{row['chronic_weekly']:.0f} (ratio {acwr:.2f}, {row['acwr_status']}).")
        recommendation = None
        if severity == "act":
            recommendation = ("Pull the next 7–10 days back toward the chronic average before "
                              "adding any more load.")
        elif row["acwr_status"] == "detraining":
            recommendation = "There's room to build — add roughly 10% load per week."
        findings.append(Finding(
            area="load", title=f"Training load is {row['acwr_status']}",
            detail=detail, severity=severity, metric=f"ACWR {acwr:.2f}",
            recommendation=recommendation,
            evidence={"acwr": round(acwr, 2), "acute_7d": round(float(row["acute_7d"]), 0),
                      "chronic_weekly": round(float(row["chronic_weekly"]), 0)}))

        tsb = float(row["tsb"]) if pd.notna(row["tsb"]) else None
        if tsb is not None:
            if tsb < -12:
                findings.append(Finding(
                    area="load", title="Fatigue is running well ahead of fitness",
                    detail=(f"Form balance {tsb:.0f} (fitness {row['ctl']:.0f}, fatigue "
                            f"{row['atl']:.0f}). Sustained deep-negative balance is where "
                            f"performance and motivation both dip."),
                    severity="watch", metric=f"TSB {tsb:.0f}",
                    recommendation="Schedule 3–5 easier days.",
                    evidence={"tsb": round(tsb, 1), "ctl": round(float(row["ctl"]), 1),
                              "atl": round(float(row["atl"]), 1)}))
            elif tsb > 15 and float(row["ctl"]) > 0:
                findings.append(Finding(
                    area="load", title="You're well recovered — and slightly detrained",
                    detail=(f"Form balance {tsb:+.0f} with fitness at {row['ctl']:.0f}. Good "
                            f"for racing; if you're not racing, there's capacity to build."),
                    severity="info", metric=f"TSB {tsb:+.0f}"))

        monotony = float(row["monotony"]) if pd.notna(row["monotony"]) else None
        if monotony is not None and monotony > 2.0:
            findings.append(Finding(
                area="load", title="Training is too monotonous",
                detail=(f"Monotony {monotony:.1f} (weekly strain {row['strain']:.0f}). Every day "
                        f"looks like every other day, which blunts adaptation and raises "
                        f"illness risk."),
                severity="watch", metric=f"monotony {monotony:.1f}",
                recommendation="Make the hard days harder and the easy days genuinely easy."))

    if mix is not None and not mix.empty:
        strength_hours = float(mix.loc[mix["activity_type"] == "strength_training", "hours"].sum())
        total_hours = float(mix["hours"].sum())
        if total_hours > 0 and strength_hours / total_hours < 0.1:
            findings.append(Finding(
                area="load", title="Very little strength work in the mix",
                detail=(f"Strength training is {strength_hours / total_hours * 100:.0f}% of "
                        f"training time ({strength_hours:.0f} of {total_hours:.0f} hours). Two "
                        f"sessions a week is the usual injury-resilience floor for runners."),
                severity="watch", metric=f"{strength_hours:.0f} h",
                evidence={"strength_hours": strength_hours, "total_hours": total_hours}))

    if recovery:
        rhr_delta = recovery.get("resting_hr_delta")
        if rhr_delta is not None and rhr_delta >= 3:
            findings.append(Finding(
                area="recovery", title="Resting heart rate is elevated",
                detail=(f"Recent average {recovery['resting_hr_recent']:.0f} bpm versus a "
                        f"baseline of {recovery['resting_hr_baseline']:.0f} bpm "
                        f"({rhr_delta:+.0f}). A sustained rise usually means incomplete "
                        f"recovery, illness, or life stress."),
                severity="act" if rhr_delta >= 5 else "watch", metric=f"{rhr_delta:+.0f} bpm",
                recommendation="Take 2–3 genuinely easy days and re-check.",
                evidence={k: v for k, v in recovery.items() if k.startswith("resting_hr")}))
        hrv_delta = recovery.get("hrv_delta")
        if hrv_delta is not None and hrv_delta <= -4:
            findings.append(Finding(
                area="recovery", title="HRV is below baseline",
                detail=(f"Recent average {recovery['hrv_recent']:.0f} ms versus "
                        f"{recovery['hrv_baseline']:.0f} ms baseline ({hrv_delta:+.0f} ms)."),
                severity="watch", metric=f"{hrv_delta:+.0f} ms",
                evidence={k: v for k, v in recovery.items() if k.startswith("hrv")}))
        sleep = recovery.get("sleep_hours_recent")
        if sleep is not None and sleep < 7.0:
            findings.append(Finding(
                area="recovery", title="Sleep is short for the training load",
                detail=(f"Averaging {sleep:.1f} hours a night. Sleep debt limits the adaptation "
                        f"you're paying for in training."),
                severity="watch", metric=f"{sleep:.1f} h",
                recommendation="Target 7.5–8.5 hours, especially the night after hard sessions."))

    return findings
