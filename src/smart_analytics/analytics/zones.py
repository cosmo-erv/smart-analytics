"""Training zones and target paces, anchored on Garmin's lactate threshold.

The app detects that too much running sits in the moderate grey zone. That's only
half useful — the actionable half is *what pace easy should actually be*, in
minutes per kilometre, for this athlete. Garmin already stores the two anchors
needed: lactate-threshold speed and threshold heart rate, plus the heart-rate
zones configured on the device.

Pace zones are expressed as fractions of threshold speed. That's the standard
construction (threshold pace is the most reproducible field anchor there is —
more stable than a max-HR percentage, which depends on a max you probably haven't
measured). If Garmin has no threshold for the account, everything here degrades
to "unavailable" rather than substituting a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .findings import Finding
from .running import format_pace

# (key, label, purpose, low fraction of threshold speed, high fraction)
ZONE_MODEL: list[tuple[str, str, str, float, float]] = [
    ("recovery", "Recovery", "Shake-out days and the jog between reps", 0.60, 0.71),
    ("easy", "Easy", "The bulk of weekly volume — should feel conversational", 0.71, 0.82),
    ("steady", "Steady / marathon", "Long-run and race-pace work for longer events", 0.82, 0.91),
    ("threshold", "Threshold / tempo", "Comfortably hard, ~1 hour race effort", 0.92, 1.01),
    ("interval", "Interval (VO2max)", "3–5 minute reps developing top-end aerobic power",
     1.02, 1.11),
    ("repetition", "Repetition / speed", "Short, fast reps for economy and mechanics",
     1.11, 1.25),
]

# Garmin zone number → what that zone is for, so the table reads as guidance.
HR_ZONE_PURPOSE = {
    1: "Warm-up and recovery",
    2: "Aerobic base — most easy running lives here",
    3: "Aerobic capacity — the grey zone if overused",
    4: "Threshold — comfortably hard",
    5: "Maximal — short intervals only",
}


@dataclass(frozen=True)
class PaceZone:
    key: str
    label: str
    purpose: str
    fast_pace_s: float  # the quick end of the range (lower number)
    slow_pace_s: float
    fast_speed_mps: float
    slow_speed_mps: float

    @property
    def range_label(self) -> str:
        # Fast end first: "5:57/km – 6:52/km" is how a pace range is normally written.
        return f"{format_pace(self.fast_pace_s)} – {format_pace(self.slow_pace_s)}"

    @property
    def compact_label(self) -> str:
        """Same range with one unit suffix — fits a stat tile without truncating."""
        return (f"{format_pace(self.fast_pace_s).replace('/km', '')}–"
                f"{format_pace(self.slow_pace_s)}")

    def contains(self, pace_s_per_km: float) -> bool:
        return self.fast_pace_s <= pace_s_per_km <= self.slow_pace_s


@dataclass
class ZoneModel:
    """The athlete's zones, or an explicit statement that they're unavailable."""

    lt_speed_mps: float | None = None
    lt_hr: float | None = None
    lt_pace_s: float | None = None
    pace_zones: list[PaceZone] = None  # type: ignore[assignment]
    hr_zones: pd.DataFrame = None  # type: ignore[assignment]
    source: str = "unavailable"

    def __post_init__(self) -> None:
        if self.pace_zones is None:
            self.pace_zones = []
        if self.hr_zones is None:
            self.hr_zones = pd.DataFrame()

    @property
    def has_pace_zones(self) -> bool:
        return bool(self.pace_zones)

    @property
    def has_hr_zones(self) -> bool:
        return self.hr_zones is not None and not self.hr_zones.empty

    def zone_for_pace(self, pace_s_per_km: float | None) -> PaceZone | None:
        if pace_s_per_km is None or pd.isna(pace_s_per_km):
            return None
        for zone in self.pace_zones:
            if zone.contains(pace_s_per_km):
                return zone
        # Outside every band: faster than repetition, or slower than recovery.
        if self.pace_zones and pace_s_per_km < self.pace_zones[-1].fast_pace_s:
            return self.pace_zones[-1]
        return self.pace_zones[0] if self.pace_zones else None

    def get(self, key: str) -> PaceZone | None:
        return next((z for z in self.pace_zones if z.key == key), None)

    def summary_table(self) -> pd.DataFrame:
        if not self.pace_zones:
            return pd.DataFrame()
        return pd.DataFrame([{
            "Zone": zone.label,
            "Pace range": zone.range_label,
            "% of threshold speed": f"{zone.fast_speed_mps / self.lt_speed_mps * 100:.0f}"
                                    f"–{zone.slow_speed_mps / self.lt_speed_mps * 100:.0f}%",
            "What it's for": zone.purpose,
        } for zone in reversed(self.pace_zones)])


def build_zone_model(athlete_metrics: pd.DataFrame,
                     hr_zones: pd.DataFrame | None = None) -> ZoneModel:
    """Build the zone model from Garmin's stored threshold values."""
    lt_speed = lt_hr = None
    if athlete_metrics is not None and not athlete_metrics.empty:
        speeds = athlete_metrics["lt_speed_mps"].dropna() \
            if "lt_speed_mps" in athlete_metrics else pd.Series(dtype=float)
        heart_rates = athlete_metrics["lt_hr"].dropna() \
            if "lt_hr" in athlete_metrics else pd.Series(dtype=float)
        lt_speed = float(speeds.iloc[-1]) if not speeds.empty else None
        lt_hr = float(heart_rates.iloc[-1]) if not heart_rates.empty else None

    zones: list[PaceZone] = []
    if lt_speed and lt_speed > 0.5:
        for key, label, purpose, low, high in ZONE_MODEL:
            fast_speed, slow_speed = lt_speed * high, lt_speed * low
            zones.append(PaceZone(
                key=key, label=label, purpose=purpose,
                fast_pace_s=1000.0 / fast_speed, slow_pace_s=1000.0 / slow_speed,
                fast_speed_mps=fast_speed, slow_speed_mps=slow_speed,
            ))

    hr_table = pd.DataFrame()
    if hr_zones is not None and not hr_zones.empty:
        hr_table = hr_zones.copy()
        hr_table["purpose"] = hr_table["zone"].map(HR_ZONE_PURPOSE)
        if lt_hr:
            hr_table["pct_of_threshold"] = (hr_table["floor_bpm"] / lt_hr * 100).round(0)

    source = "garmin" if (zones or not hr_table.empty) else "unavailable"
    return ZoneModel(
        lt_speed_mps=lt_speed, lt_hr=lt_hr,
        lt_pace_s=(1000.0 / lt_speed) if lt_speed else None,
        pace_zones=zones, hr_zones=hr_table, source=source,
    )


def classify_runs(runs: pd.DataFrame, model: ZoneModel) -> pd.DataFrame:
    """Tag each run with the pace zone its average pace falls into.

    Average pace mislabels interval sessions — the mean of hard reps and jogged
    recoveries lands in tempo — so this is used for distribution summaries, while
    :mod:`.splits` handles per-rep analysis where it matters.
    """
    if runs is None or runs.empty or not model.has_pace_zones:
        return runs.assign(pace_zone=None) if runs is not None else pd.DataFrame()
    frame = runs.copy()
    frame["pace_zone"] = [
        (model.zone_for_pace(pace).label if model.zone_for_pace(pace) else None)
        for pace in frame["pace_s_per_km"]
    ]
    return frame


def easy_run_discipline(runs: pd.DataFrame, model: ZoneModel, lookback_days: int = 84,
                        as_of: pd.Timestamp | None = None) -> dict:
    """How much faster than their own easy range the athlete's easy runs actually are.

    This is the concrete version of the grey-zone finding: not "run easier" but
    "your easy runs average 5:20/km; your easy range is 5:55–6:45".
    """
    easy_zone = model.get("easy")
    recovery_zone = model.get("recovery")
    if runs is None or runs.empty or easy_zone is None:
        return {}

    as_of = as_of or pd.to_datetime(runs["local_date"]).max()
    window = runs[pd.to_datetime(runs["local_date"]) >= as_of - pd.Timedelta(days=lookback_days)]
    if window.empty:
        return {}

    # "Intended easy" = everything that isn't a hard session. Judged by the
    # device's own intensity labels, not by pace, to avoid circularity.
    intended = window[window["intensity"].isin(["easy", "moderate"])]
    if intended.empty:
        return {}

    slowest_easy = recovery_zone.slow_pace_s if recovery_zone else easy_zone.slow_pace_s
    too_fast = intended[intended["pace_s_per_km"] < easy_zone.fast_pace_s]
    mean_pace = float(intended["pace_s_per_km"].mean())

    return {
        "runs_considered": int(len(intended)),
        "mean_pace_s": mean_pace,
        "mean_pace_label": format_pace(mean_pace),
        "easy_range_label": easy_zone.range_label,
        "easy_fast_pace_s": easy_zone.fast_pace_s,
        "easy_slow_pace_s": easy_zone.slow_pace_s,
        "recovery_slow_pace_s": slowest_easy,
        "too_fast_count": int(len(too_fast)),
        "too_fast_pct": round(len(too_fast) / len(intended) * 100, 0),
        "seconds_too_fast": round(easy_zone.fast_pace_s - mean_pace, 0),
    }


def zone_distribution(runs: pd.DataFrame, model: ZoneModel, lookback_days: int = 84,
                      as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Share of running distance spent in each pace zone."""
    if runs is None or runs.empty or not model.has_pace_zones:
        return pd.DataFrame(columns=["zone", "runs", "distance_km", "share_pct"])
    tagged = classify_runs(runs, model)
    as_of = as_of or pd.to_datetime(tagged["local_date"]).max()
    window = tagged[pd.to_datetime(tagged["local_date"])
                    >= as_of - pd.Timedelta(days=lookback_days)]
    if window.empty:
        return pd.DataFrame(columns=["zone", "runs", "distance_km", "share_pct"])

    grouped = window.groupby("pace_zone", as_index=False).agg(
        runs=("activity_id", "nunique"), distance_km=("distance_km", "sum"))
    total = grouped["distance_km"].sum()
    grouped["share_pct"] = (grouped["distance_km"] / total * 100).round(1) if total else 0.0
    order = {zone.label: index for index, zone in enumerate(model.pace_zones)}
    grouped["_order"] = grouped["pace_zone"].map(order).fillna(99)
    return (grouped.sort_values("_order", ascending=False).drop(columns="_order")
            .rename(columns={"pace_zone": "zone"}).reset_index(drop=True))


def target_paces(model: ZoneModel) -> pd.DataFrame:
    """A session-prescription table: what pace to run for each session type."""
    if not model.has_pace_zones:
        return pd.DataFrame()
    return model.summary_table()


# --- findings ---------------------------------------------------------------

def zone_findings(model: ZoneModel, discipline: dict) -> list[Finding]:
    findings: list[Finding] = []

    if not model.has_pace_zones:
        findings.append(Finding(
            area="running", title="No threshold data from Garmin yet",
            detail=("Garmin hasn't produced a lactate-threshold estimate for this account, so "
                    "personal pace zones can't be built. It usually appears after a few hard "
                    "runs with a chest strap or a compatible optical sensor."),
            severity="info",
            recommendation="Run a hard 20–30 minute effort with heart rate recorded."))
        return findings

    if discipline and discipline.get("too_fast_pct", 0) >= 40:
        findings.append(Finding(
            area="running", subject="easy pace",
            title="Easy runs are being run too fast",
            detail=(f"{discipline['too_fast_pct']:.0f}% of your easy and moderate runs "
                    f"({discipline['too_fast_count']} of {discipline['runs_considered']}) are "
                    f"quicker than your easy range of {discipline['easy_range_label']}. "
                    f"They average {discipline['mean_pace_label']}, about "
                    f"{abs(discipline['seconds_too_fast']):.0f} s/km too quick. That is what "
                    f"pushes aerobic work into the grey zone."),
            severity="act",
            metric=f"{discipline['too_fast_pct']:.0f}% too fast",
            recommendation=(f"Cap easy runs at {format_pace(discipline['easy_fast_pace_s'])} "
                            f"or slower — ideally nearer "
                            f"{format_pace(discipline['easy_slow_pace_s'])}. Slower easy "
                            f"running is what makes the hard sessions possible."),
            evidence=discipline))
    elif discipline:
        findings.append(Finding(
            area="running", subject="easy pace",
            title="Easy-run pace discipline is good",
            detail=(f"Easy and moderate runs average {discipline['mean_pace_label']}, inside "
                    f"the {discipline['easy_range_label']} easy range."),
            severity="good", metric=discipline["mean_pace_label"], evidence=discipline))

    threshold = model.get("threshold")
    interval = model.get("interval")
    if threshold and interval:
        findings.append(Finding(
            area="running", title="Your personal training paces",
            detail=(f"From Garmin's threshold estimate "
                    f"({format_pace(model.lt_pace_s)}"
                    + (f", LT heart rate {model.lt_hr:.0f} bpm" if model.lt_hr else "")
                    + f"): easy {model.get('easy').range_label}, "
                    f"tempo {threshold.range_label}, intervals {interval.range_label}."),
            severity="info", metric=format_pace(model.lt_pace_s),
            evidence={"lt_pace_s": model.lt_pace_s, "lt_hr": model.lt_hr}))

    return findings
