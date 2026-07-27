"""Weekly digest export — a shareable summary of what changed and what's next.

Produces Markdown (for pasting into a note, a message to a coach, or a training
log) and a self-contained HTML version for saving or printing. Both are generated
from the same :class:`TrainingReport`, so they can't drift from what the GUI shows.
"""

from __future__ import annotations

import html
from datetime import date
from typing import Any

import pandas as pd

from .analytics.findings import SEVERITY_LABEL, sort_findings
from .analytics.snapshots import format_unit
from .analytics.running import format_duration, format_pace


def _fmt(value: Any, spec: str = "{:.1f}") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def weekly_digest_markdown(report, week_ending: date | None = None) -> str:
    """The digest as Markdown."""
    week_ending = week_ending or date.today()
    lines: list[str] = []

    lines.append(f"# Training digest — week ending {week_ending:%d %b %Y}")
    lines.append("")
    if report.meta.get("first_date"):
        lines.append(f"_Data {report.meta['first_date']} → {report.meta['last_date']} · "
                     f"{report.meta.get('activity_count', 0)} activities._")
        lines.append("")

    # --- next session ------------------------------------------------------
    if report.recommendation is not None:
        rec = report.recommendation
        lines.append("## Next session")
        lines.append("")
        lines.append(f"**{rec.title}**")
        lines.append("")
        lines.append(rec.detail)
        if rec.targets:
            lines.append("")
            for target in rec.targets:
                lines.append(f"- {target}")
        if rec.reasons:
            lines.append("")
            lines.append("_Why: " + "; ".join(rec.reasons) + "._")
        lines.append("")

    # --- headline numbers --------------------------------------------------
    lines.append("## Where things stand")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    for label, value in _headline_rows(report):
        lines.append(f"| {label} | {value} |")
    lines.append("")

    # --- movement since last check ----------------------------------------
    deltas = report.progress_deltas
    if deltas and deltas.get("metrics"):
        lines.append(f"## Changed in the last {deltas['days_between']} days")
        lines.append("")
        lines.append("| Metric | Then | Now | Direction |")
        lines.append("| --- | --- | --- | --- |")
        for metric in deltas["metrics"]:
            lines.append(
                f"| {metric['label']} | {format_unit(metric['then_label'], metric['unit'])} "
                f"| {format_unit(metric['now_label'], metric['unit'])} "
                f"| {metric['direction']} |")
        lines.append("")
        improved = [m for m in deltas["muscles"] if m["score_change"] <= -8][:3]
        worsened = [m for m in deltas["muscles"] if m["score_change"] >= 8][:3]
        if improved:
            lines.append("**Catching up:** " + ", ".join(
                f"{m['muscle'].replace('_', ' ')} ({m['then_score']:.0f} → "
                f"{m['now_score']:.0f})" for m in improved))
            lines.append("")
        if worsened:
            lines.append("**Slipping:** " + ", ".join(
                f"{m['muscle'].replace('_', ' ')} ({m['then_score']:.0f} → "
                f"{m['now_score']:.0f})" for m in worsened))
            lines.append("")

    # --- priorities --------------------------------------------------------
    act = [f for f in sort_findings(report.findings) if f.severity == "act"]
    if act:
        lines.append("## Needs action")
        lines.append("")
        for finding in act[:6]:
            metric = f" *({finding.metric})*" if finding.metric else ""
            lines.append(f"### {finding.title}{metric}")
            lines.append("")
            lines.append(finding.detail)
            if finding.recommendation:
                lines.append("")
                lines.append(f"→ **{finding.recommendation}**")
            lines.append("")

    watch = [f for f in sort_findings(report.findings) if f.severity == "watch"]
    if watch:
        lines.append("## Worth watching")
        lines.append("")
        for finding in watch[:8]:
            metric = f" ({finding.metric})" if finding.metric else ""
            lines.append(f"- **{finding.title}**{metric} — {finding.detail}")
        lines.append("")

    # --- this week's targets ----------------------------------------------
    targets = report.weekly_targets or {}
    if targets.get("strength") or targets.get("running"):
        lines.append("## This week's targets")
        lines.append("")
        if targets.get("strength"):
            lines.append("**Strength — effective sets to add:**")
            lines.append("")
            for item in targets["strength"]:
                lines.append(f"- {item['muscle']}: {item['current']:.1f} → "
                             f"{item['target']} sets/week (add {item['add_sets']:.0f})")
            lines.append("")
        if targets.get("running"):
            lines.append("**Running:**")
            lines.append("")
            for item in targets["running"]:
                lines.append(f"- {item['metric']}: {item['current']} "
                             f"(target {item['target']}) — {item['note']}")
            lines.append("")
        for note in targets.get("notes", []):
            lines.append(f"_{note}_")
            lines.append("")

    good = [f for f in sort_findings(report.findings) if f.severity == "good"]
    if good:
        lines.append("## Going well")
        lines.append("")
        for finding in good[:6]:
            metric = f" ({finding.metric})" if finding.metric else ""
            lines.append(f"- **{finding.title}**{metric}")
        lines.append("")

    if report.niggle_context is not None and not report.niggle_context.empty:
        open_niggles = report.niggle_context[report.niggle_context["status"] == "open"]
        if not open_niggles.empty:
            lines.append("## Open niggles")
            lines.append("")
            for entry in open_niggles.itertuples():
                note = f' — "{entry.note}"' if entry.note else ""
                lines.append(f"- **{entry.area}** (severity {entry.severity}/5, "
                             f"{entry.days_open} days){note}")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("_Generated by Smart Analytics from Garmin Connect data. "
                 "Not medical advice._")
    return "\n".join(lines)


def _headline_rows(report) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    if not report.weekly_runs.empty:
        recent = float(report.weekly_runs.tail(4)["distance_km"].mean())
        rows.append(("Running volume (4-week mean)", f"{recent:.0f} km/week"))
        this_week = float(report.weekly_runs.tail(1)["distance_km"].iloc[0])
        rows.append(("This week's distance", f"{this_week:.1f} km"))

    if report.intensity:
        rows.append(("Easy / moderate / hard share",
                     f"{report.intensity['easy_pct']:.0f}% / "
                     f"{report.intensity['moderate_pct']:.0f}% / "
                     f"{report.intensity['hard_pct']:.0f}%"))

    if report.zone_model is not None and report.zone_model.has_pace_zones:
        easy = report.zone_model.get("easy")
        rows.append(("Easy pace target", easy.range_label))
        rows.append(("Threshold pace", format_pace(report.zone_model.lt_pace_s)))

    if not report.decoupling.empty:
        rows.append(("Aerobic decoupling (recent long runs)",
                     f"{float(report.decoupling.tail(6)['decoupling_pct'].mean()):.1f}%"))

    if not report.lagging.empty:
        behind = int((report.lagging["attention_score"] >= 60).sum())
        rows.append(("Weekly effective sets",
                     f"{float(report.lagging['weekly_sets'].sum()):.0f}"))
        rows.append(("Muscles falling behind", str(behind)))

    if not report.load_series.empty:
        latest = report.load_series.dropna(subset=["acwr"]).tail(1)
        if not latest.empty:
            row = latest.iloc[0]
            rows.append(("Load balance (ACWR)", f"{row['acwr']:.2f} ({row['acwr_status']})"))
        last = report.load_series.tail(1).iloc[0]
        if pd.notna(last["ctl"]):
            rows.append(("Fitness / fatigue / form",
                         f"{last['ctl']:.0f} / {last['atl']:.0f} / {last['tsb']:+.0f}"))

    if not report.athlete_metrics.empty:
        latest = report.athlete_metrics.tail(1).iloc[0]
        if pd.notna(latest.get("training_status")):
            rows.append(("Garmin training status", str(latest["training_status"])))
        if pd.notna(latest.get("readiness_score")):
            rows.append(("Garmin readiness", f"{latest['readiness_score']:.0f}/100"))
        if pd.notna(latest.get("vo2max_running")):
            rows.append(("VO2max (Garmin)", f"{latest['vo2max_running']:.1f}"))

    if not report.garmin_predictions.empty:
        latest_day = report.garmin_predictions["local_date"].max()
        current = report.garmin_predictions[
            report.garmin_predictions["local_date"] == latest_day]
        parts = [f"{int(row.distance_m / 1000)}k {format_duration(row.predicted_time_s)}"
                 for row in current.sort_values("distance_m").itertuples()]
        if parts:
            rows.append(("Garmin race predictions", " · ".join(parts)))

    return rows


def weekly_digest_html(report, week_ending: date | None = None) -> str:
    """Self-contained HTML version — no external assets, prints cleanly."""
    markdown = weekly_digest_markdown(report, week_ending)
    body = _markdown_to_html(markdown)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training digest — {(week_ending or date.today()):%d %b %Y}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
         max-width: 46rem; margin: 2.5rem auto; padding: 0 1.25rem;
         line-height: 1.6; color: #0b0b0b; background: #fcfcfb; }}
  h1 {{ font-size: 1.6rem; margin-bottom: .2rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2rem; padding-bottom: .3rem;
        border-bottom: 1px solid #e1e0d9; }}
  h3 {{ font-size: 1rem; margin-top: 1.4rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: .8rem 0; font-size: .93rem; }}
  th, td {{ text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #e1e0d9; }}
  th {{ color: #52514e; font-weight: 600; }}
  td:last-child {{ font-variant-numeric: tabular-nums; }}
  em {{ color: #52514e; }}
  hr {{ border: none; border-top: 1px solid #e1e0d9; margin: 2rem 0 1rem; }}
  li {{ margin: .25rem 0; }}
  @media (prefers-color-scheme: dark) {{
    body {{ color: #ffffff; background: #1a1a19; }}
    h2, th, td {{ border-color: #2c2c2a; }}
    th, em {{ color: #c3c2b7; }}
  }}
</style></head>
<body>
{body}
</body></html>
"""


def _markdown_to_html(markdown: str) -> str:
    """Minimal Markdown rendering for the digest's own subset.

    Deliberately not a general Markdown implementation — it handles exactly the
    constructs :func:`weekly_digest_markdown` emits (headings, tables, lists,
    bold, italics, rules), which keeps the export dependency-free.
    """
    out: list[str] = []
    in_table = in_list = False
    header_done = False

    def close_blocks() -> None:
        nonlocal in_table, in_list, header_done
        if in_table:
            out.append("</tbody></table>")
            in_table = False
            header_done = False
        if in_list:
            out.append("</ul>")
            in_list = False

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if not line:
            close_blocks()
            continue

        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= {"-", ":", " "} and c for c in cells):
                continue  # the |---| separator row
            if not in_table:
                out.append("<table><thead>")
                in_table, header_done = True, False
            tag = "td" if header_done else "th"
            row = "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells)
            out.append(f"<tr>{row}</tr>")
            if not header_done:
                out.append("</thead><tbody>")
                header_done = True
            continue

        close_blocks() if not line.startswith("- ") else None

        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(line[2:])}</li>")
        elif line.startswith("### "):
            out.append(f"<h3>{_inline(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.startswith("---"):
            out.append("<hr>")
        else:
            out.append(f"<p>{_inline(line)}</p>")

    close_blocks()
    return "\n".join(out)


def _inline(text: str) -> str:
    """Escape, then apply bold/italic — escaping first so content can't inject HTML."""
    escaped = html.escape(text)
    parts = escaped.split("**")
    rebuilt = "".join(f"<strong>{part}</strong>" if index % 2 else part
                      for index, part in enumerate(parts))
    parts = rebuilt.split("_")
    if len(parts) > 2:
        rebuilt = "".join(f"<em>{part}</em>" if index % 2 else part
                          for index, part in enumerate(parts))
    return rebuilt
