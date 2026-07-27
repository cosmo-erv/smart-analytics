"""Niggle and injury log — the one input Garmin can't provide.

A watch records what you did, never how it felt. Yet the useful question about a
load spike is whether anything hurt afterwards, and the useful question about a
niggle is what the training looked like in the ten days before it appeared.

This module keeps that correlation honest: it reports what the load was doing
around each logged niggle, and flags the recurring ones. It deliberately stops
short of claiming causation — n=1 with confounders everywhere — and it never
diagnoses. Persistent or severe entries get pointed at a professional.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .findings import Finding

# Body areas offered in the UI. Kept coarse — finer detail invites false precision.
AREAS = [
    "Achilles", "Ankle", "Calf", "Knee", "Shin", "Hamstring", "Quad", "Groin",
    "Hip", "Glute", "Lower back", "Upper back", "Shoulder", "Elbow", "Wrist",
    "Neck", "Foot / plantar", "Chest", "Other",
]

SEVERITY_LABELS = {
    1: "Aware of it, no effect",
    2: "Noticeable during training",
    3: "Changes how I train",
    4: "Cutting sessions short",
    5: "Stopping training",
}

# Window before a niggle in which training is plausibly relevant.
LOOKBACK_DAYS = 10
RECURRENCE_DAYS = 120


def _lower_first(text: str) -> str:
    """Lowercase only the first character — "Changes how I train" keeps its "I"."""
    return text[:1].lower() + text[1:] if text else text


def context_for_niggles(niggles: pd.DataFrame, load_series: pd.DataFrame,
                        runs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach what the training load was doing in the days before each niggle."""
    columns = ["id", "noted_on", "area", "severity", "severity_label", "note", "status",
               "days_open", "acwr_at_onset", "load_7d_at_onset", "load_vs_chronic_pct",
               "km_prior_10d"]
    if niggles is None or niggles.empty:
        return pd.DataFrame(columns=columns)

    load = load_series.copy() if load_series is not None and not load_series.empty \
        else pd.DataFrame()
    if not load.empty:
        load["date"] = pd.to_datetime(load["date"]).dt.normalize()
        load = load.set_index("date")

    today = pd.Timestamp.today().normalize()
    rows = []
    for entry in niggles.itertuples():
        noted = pd.to_datetime(entry.noted_on).normalize()
        resolved = pd.to_datetime(entry.resolved_on) if pd.notna(entry.resolved_on) else None

        acwr = load_7d = load_vs_chronic = np.nan
        if not load.empty:
            # Nearest load row at or before onset — loads only exist on dated rows.
            window = load[load.index <= noted]
            if not window.empty:
                row = window.iloc[-1]
                acwr = float(row["acwr"]) if pd.notna(row["acwr"]) else np.nan
                load_7d = float(row["acute_7d"]) if pd.notna(row["acute_7d"]) else np.nan
                if pd.notna(row["chronic_weekly"]) and row["chronic_weekly"] > 0:
                    load_vs_chronic = round(
                        (row["acute_7d"] / row["chronic_weekly"] - 1) * 100, 0)

        km_prior = np.nan
        if runs is not None and not runs.empty:
            prior = runs[(pd.to_datetime(runs["local_date"]) <= noted)
                         & (pd.to_datetime(runs["local_date"])
                            >= noted - pd.Timedelta(days=LOOKBACK_DAYS))]
            if not prior.empty:
                km_prior = round(float(prior["distance_km"].sum()), 1)

        rows.append({
            "id": int(entry.id),
            "noted_on": noted,
            "area": entry.area,
            "severity": int(entry.severity) if pd.notna(entry.severity) else None,
            "severity_label": SEVERITY_LABELS.get(
                int(entry.severity) if pd.notna(entry.severity) else 0, "—"),
            "note": entry.note,
            "status": "resolved" if resolved is not None else "open",
            "days_open": int(((resolved or today) - noted).days),
            "acwr_at_onset": round(acwr, 2) if not np.isnan(acwr) else np.nan,
            "load_7d_at_onset": round(load_7d, 0) if not np.isnan(load_7d) else np.nan,
            "load_vs_chronic_pct": load_vs_chronic,
            "km_prior_10d": km_prior,
        })
    return pd.DataFrame(rows, columns=columns).sort_values("noted_on", ascending=False)


def recurrence(context: pd.DataFrame) -> pd.DataFrame:
    """Areas logged more than once — the pattern that matters most."""
    columns = ["area", "entries", "open_entries", "worst_severity", "first_noted",
               "last_noted", "total_days_open"]
    if context is None or context.empty:
        return pd.DataFrame(columns=columns)
    recent = context[context["noted_on"]
                     >= pd.Timestamp.today().normalize() - pd.Timedelta(days=RECURRENCE_DAYS)]
    if recent.empty:
        return pd.DataFrame(columns=columns)
    grouped = recent.groupby("area", as_index=False).agg(
        entries=("id", "count"),
        open_entries=("status", lambda s: int((s == "open").sum())),
        worst_severity=("severity", "max"),
        first_noted=("noted_on", "min"),
        last_noted=("noted_on", "max"),
        total_days_open=("days_open", "sum"),
    )
    return grouped.sort_values(["entries", "worst_severity"], ascending=False).reset_index(
        drop=True)


def niggle_findings(context: pd.DataFrame, repeats: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    if context is None or context.empty:
        return findings

    open_entries = context[context["status"] == "open"]

    for entry in open_entries.sort_values("severity", ascending=False).head(3).itertuples():
        load_text = ""
        if not np.isnan(entry.load_vs_chronic_pct):
            direction = "above" if entry.load_vs_chronic_pct > 0 else "below"
            load_text = (f" At onset the 7-day load was "
                         f"{abs(entry.load_vs_chronic_pct):.0f}% {direction} your 28-day "
                         f"average")
            if not np.isnan(entry.acwr_at_onset):
                load_text += f" (ACWR {entry.acwr_at_onset:.2f})"
            load_text += "."
        if not np.isnan(entry.km_prior_10d):
            load_text += (f" You ran {entry.km_prior_10d:.0f} km in the 10 days before it.")

        severity = "act" if (entry.severity or 0) >= 3 else "watch"
        findings.append(Finding(
            area="niggles", subject=entry.area,
            title=f"Open niggle: {entry.area}",
            detail=(f"Logged {entry.noted_on.strftime('%d %b')} "
                    f"({entry.days_open} days ago) — {_lower_first(entry.severity_label)}."
                    + (f' "{entry.note}"' if entry.note else "") + load_text),
            severity=severity,
            metric=f"severity {entry.severity}/5" if entry.severity else None,
            recommendation=(
                "Something that changes how you train for more than a couple of weeks is worth "
                "a physio's opinion — this tool can show the load pattern around it, but it "
                "can't tell you what the tissue is doing."
                if (entry.severity or 0) >= 3 or entry.days_open > 14
                else "Keep logging it; if it starts changing sessions, back the load off."),
            evidence={"days_open": entry.days_open,
                      "acwr_at_onset": (None if np.isnan(entry.acwr_at_onset)
                                        else entry.acwr_at_onset),
                      "load_vs_chronic_pct": (None if np.isnan(entry.load_vs_chronic_pct)
                                              else entry.load_vs_chronic_pct)}))

    if repeats is not None and not repeats.empty:
        for entry in repeats[repeats["entries"] >= 2].head(2).itertuples():
            findings.append(Finding(
                area="niggles", subject=entry.area,
                title=f"{entry.area} keeps coming back",
                detail=(f"Logged {entry.entries} times since "
                        f"{entry.first_noted.strftime('%d %b')}, most recently "
                        f"{entry.last_noted.strftime('%d %b')}, worst severity "
                        f"{entry.worst_severity}/5. A recurring site is usually a capacity or "
                        f"mechanics issue rather than bad luck."),
                severity="act",
                metric=f"{entry.entries} episodes",
                recommendation=("Worth a professional assessment of the area, and worth checking "
                                "whether the muscles around it are among the under-trained ones "
                                "on the Strength page."),
                evidence={"entries": int(entry.entries),
                          "worst_severity": int(entry.worst_severity or 0)}))

    high_load_onsets = context[(context["acwr_at_onset"] > 1.3) & context["acwr_at_onset"].notna()]
    if len(high_load_onsets) >= 2:
        findings.append(Finding(
            area="niggles", title="Niggles cluster after load spikes",
            detail=(f"{len(high_load_onsets)} of {len(context)} logged niggles began when the "
                    f"acute:chronic load ratio was above 1.3. That's an association in your own "
                    f"data, not proof of cause — but it's the pattern injury-risk research "
                    f"would predict."),
            severity="watch", metric=f"{len(high_load_onsets)} of {len(context)}",
            recommendation="Treat 1.3 as your personal ceiling rather than a general guideline.",
            evidence={"spike_onsets": int(len(high_load_onsets)),
                      "total": int(len(context))}))

    return findings
