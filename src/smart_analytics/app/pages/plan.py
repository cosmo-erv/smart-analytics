"""Plan: what to do next, what this week needs, and the digest export."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from ...analytics.niggles import AREAS, SEVERITY_LABELS
from ...reporting import weekly_digest_html, weekly_digest_markdown
from ...viz import niggle_timeline
from ...viz.theme import severity_color
from .. import components as ui
from ..state import bump_data_version, conn, current_report, palette
from ... import db

KIND_ICONS = {
    "rest": "🛌", "easy_run": "🏃", "long_run": "🏔", "quality_run": "⚡",
    "strength": "🏋", "mobility": "🧘",
}


def render() -> None:
    report = current_report()
    colors = palette()

    st.title("Plan")

    if not report.has_data:
        ui.no_data_notice("activities")
        return

    _next_session(report, colors)
    st.divider()
    _weekly_targets(report, colors)
    st.divider()
    _niggle_log(report, colors)
    st.divider()
    _digest(report, colors)


def _next_session(report, colors) -> None:
    rec = report.recommendation
    if rec is None:
        st.info("Not enough data to recommend a session yet.")
        return

    icon = KIND_ICONS.get(rec.kind, "•")
    confidence_color = {"high": colors.status["good"], "medium": colors.status["warning"],
                        "low": colors.ink_muted}.get(rec.confidence, colors.ink_muted)

    with st.container(border=True):
        st.markdown(
            f'<div style="font-size:0.78rem;font-weight:600;letter-spacing:0.05em;'
            f'text-transform:uppercase;color:{colors.ink_muted};">Today\'s session</div>'
            f'<div style="font-size:1.75rem;line-height:1.2;font-weight:680;'
            f'color:{colors.ink};margin:4px 0 2px;">{icon} {rec.title}</div>'
            f'<div style="font-size:0.78rem;color:{confidence_color};font-weight:600;">'
            f'{rec.confidence} confidence</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="color:{colors.ink_secondary};font-size:0.96rem;line-height:1.55;'
            f'margin-top:10px;">{rec.detail}</div>',
            unsafe_allow_html=True,
        )

        if rec.targets:
            st.markdown(
                f'<div style="margin-top:12px;font-size:0.8rem;font-weight:600;'
                f'color:{colors.ink_muted};text-transform:uppercase;letter-spacing:0.04em;">'
                f'Session targets</div>', unsafe_allow_html=True)
            for target in rec.targets:
                st.markdown(f"- {target}")

        if rec.reasons:
            with st.expander("Why this, and not something else"):
                for reason in rec.reasons:
                    st.markdown(f"- {reason}")
                st.caption(
                    "The recommendation is rule-based and applied in order: open niggles first, "
                    "then recovery state, then interference from yesterday's session, then "
                    "whichever gap in your training is largest. Override it freely — it can't "
                    "see how you actually feel."
                )
        if rec.alternative:
            st.markdown(
                f'<div style="margin-top:8px;padding-left:10px;'
                f'border-left:2px solid {colors.axis};color:{colors.ink_secondary};'
                f'font-size:0.9rem;"><strong>If that doesn\'t fit:</strong> '
                f'{rec.alternative}</div>', unsafe_allow_html=True)


def _weekly_targets(report, colors) -> None:
    targets = report.weekly_targets or {}
    ui.section("This week", "Countable targets rather than advice.", colors)

    columns = st.columns([1.15, 1])
    with columns[0]:
        st.markdown("**Strength — effective sets to add**")
        if targets.get("strength"):
            frame = pd.DataFrame(targets["strength"])
            # A muscle with no history at all reads "never", not a blank cell.
            frame["days_since"] = frame["days_since"].map(
                lambda value: "never" if value is None or pd.isna(value) else f"{int(value)}")
            frame = frame.rename(columns={
                "muscle": "Muscle", "current": "Now", "target": "Target",
                "add_sets": "Add", "days_since": "Last trained"})
            st.dataframe(frame, use_container_width=True, hide_index=True)
        else:
            st.success("Every muscle is at or above its weekly target.")

    with columns[1]:
        st.markdown("**Running**")
        if targets.get("running"):
            for item in targets["running"]:
                unit = item.get("unit", "")
                st.metric(item["metric"], f"{item['current']:g}{unit}")
                st.caption(f"Target {item['target']:g}{unit} — {item['note']}")
        else:
            st.caption("No running targets — sync some runs first.")

    for note in targets.get("notes", []):
        st.caption(note)


def _niggle_log(report, colors) -> None:
    ui.section(
        "Niggle log",
        "The one thing Garmin can't record. Logging niggles lets the app show what the "
        "load was doing when each one appeared, and feeds the AI coach context it "
        "otherwise has no way to know.",
        colors,
    )

    connection = conn()
    context = report.niggle_context

    with st.form("add_niggle", clear_on_submit=True):
        row = st.columns([1.1, 1.4, 1.6, 0.9])
        with row[0]:
            noted_on = st.date_input("Date noticed", value=date.today(),
                                     max_value=date.today())
        with row[1]:
            area = st.selectbox("Area", AREAS, index=AREAS.index("Knee"))
        with row[2]:
            severity = st.select_slider(
                "Severity", options=list(SEVERITY_LABELS), value=2,
                format_func=lambda value: f"{value} — {SEVERITY_LABELS[value]}")
        with row[3]:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Log it", use_container_width=True)
        note = st.text_input("Note (optional)",
                             placeholder="e.g. tight for the first 2 km, eased off after")

    if submitted:
        db.add_niggle(connection, noted_on.isoformat(), area, int(severity), note or None)
        bump_data_version()
        st.success(f"Logged {area} at severity {severity}.")
        st.rerun()

    if context is None or context.empty:
        st.caption("Nothing logged yet.")
        return

    open_entries = context[context["status"] == "open"]
    if not open_entries.empty:
        st.markdown("**Open**")
        for entry in open_entries.itertuples():
            with st.container(border=True):
                head, action = st.columns([5, 1])
                with head:
                    color = severity_color(
                        "act" if (entry.severity or 0) >= 4 else
                        "watch" if (entry.severity or 0) == 3 else "info", colors)
                    st.markdown(
                        f'<span style="color:{color};font-weight:700;">■</span> '
                        f'<strong>{entry.area}</strong> — severity {entry.severity}/5 '
                        f'({entry.severity_label})<br>'
                        f'<span style="color:{colors.ink_muted};font-size:0.85rem;">'
                        f'{entry.noted_on:%d %b %Y} · {entry.days_open} days'
                        + (f' · ACWR {entry.acwr_at_onset:.2f} at onset'
                           if pd.notna(entry.acwr_at_onset) else "")
                        + (f' · {entry.km_prior_10d:.0f} km in the prior 10 days'
                           if pd.notna(entry.km_prior_10d) else "")
                        + '</span>'
                        + (f'<br><em style="color:{0};">"{1}"</em>'.format(
                            colors.ink_secondary, entry.note) if entry.note else ""),
                        unsafe_allow_html=True)
                with action:
                    if st.button("Resolved", key=f"resolve_{entry.id}",
                                 use_container_width=True):
                        db.resolve_niggle(connection, int(entry.id), date.today().isoformat())
                        bump_data_version()
                        st.rerun()

    ui.chart(niggle_timeline(context, report.load_series, colors), colors,
             key="plan_niggle_timeline")

    with st.expander(f"Full log ({len(context)} entries)"):
        display = context[["noted_on", "area", "severity", "status", "days_open",
                           "acwr_at_onset", "km_prior_10d", "note"]].copy()
        display["noted_on"] = display["noted_on"].dt.strftime("%d %b %Y")
        st.dataframe(display.rename(columns={
            "noted_on": "Date", "area": "Area", "severity": "Severity", "status": "Status",
            "days_open": "Days", "acwr_at_onset": "ACWR at onset",
            "km_prior_10d": "km prior 10d", "note": "Note"}),
            use_container_width=True, hide_index=True)
        remove = st.selectbox("Delete an entry", ["—"] + [
            f"{row.id}: {row.area} ({row.noted_on:%d %b %Y})" for row in context.itertuples()])
        if remove != "—" and st.button("Delete", type="secondary"):
            db.delete_niggle(connection, int(remove.split(":")[0]))
            bump_data_version()
            st.rerun()


def _digest(report, colors) -> None:
    ui.section(
        "Weekly digest",
        "A shareable summary of where things stand, what changed, and what's next — "
        "for your own log or to send to a coach.",
        colors,
    )
    markdown = weekly_digest_markdown(report)
    stamp = date.today().isoformat()

    row = st.columns(2)
    with row[0]:
        st.download_button(
            "Download Markdown", markdown, file_name=f"training-digest-{stamp}.md",
            mime="text/markdown", use_container_width=True)
    with row[1]:
        st.download_button(
            "Download HTML", weekly_digest_html(report),
            file_name=f"training-digest-{stamp}.html", mime="text/html",
            use_container_width=True)

    with st.expander("Preview"):
        st.markdown(markdown)
