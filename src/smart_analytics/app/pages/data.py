"""Sync & Settings: pull data from Garmin, tune the analysis, inspect the cache."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ... import db
from ...config import settings
from ...domain.exercises import coverage_report
from ...garmin.sync import last_sync_at
from .. import components as ui
from ..state import bump_data_version, conn, palette, run_sync


def render() -> None:
    colors = palette()
    connection = conn()
    counts = db.counts(connection)

    st.title("Sync & settings")

    ui.stat_tiles([
        {"label": "Activities", "value": f"{counts['activities']:,}"},
        {"label": "Strength workouts", "value": f"{counts['strength_activities']:,}"},
        {"label": "Sets", "value": f"{counts['sets']:,}"},
        {"label": "Splits", "value": f"{counts['splits']:,}"},
        {"label": "Garmin metric days", "value": f"{counts['athlete_metrics']:,}"},
    ], palette=colors)
    ui.stat_tiles([
        {"label": "Wellness days", "value": f"{counts['daily_metrics']:,}"},
        {"label": "Personal records", "value": f"{counts['personal_records']:,}"},
        {"label": "Open niggles", "value": f"{counts['niggles']:,}"},
        {"label": "Snapshots", "value": f"{counts['snapshots']:,}"},
        {"label": "Activity types", "value": f"{counts['types']:,}"},
    ], palette=colors)
    if counts["first_date"]:
        st.caption(f"Cached range {counts['first_date']} → {counts['last_date']} · "
                   f"last sync {last_sync_at(connection) or 'never'} · "
                   f"database `{settings.db_path}`")

    st.divider()
    _sync_panel(colors)

    st.divider()
    _analysis_settings(colors)

    st.divider()
    _connection_status(colors)

    st.divider()
    _maintenance(connection, colors)


def _sync_panel(colors) -> None:
    ui.section("Pull data", None, colors)

    have_garmin = settings.has_garmin_credentials or settings.has_cached_tokens
    default_demo = not have_garmin
    demo = st.toggle(
        "Demo mode (generated data, no Garmin account needed)",
        value=st.session_state.get("demo_mode", default_demo),
        help=("Generates a realistic 400-day history with deliberate weaknesses, so you can "
              "see what the analytics find. It runs through the same normalisation and "
              "storage path as a real sync."),
    )
    if not demo and not have_garmin:
        st.warning(
            "No Garmin credentials found. Copy `.env.example` to `.env` and set "
            "`GARMIN_EMAIL` and `GARMIN_PASSWORD`, or stay in demo mode."
        )

    with st.form("sync_form"):
        row = st.columns(2)
        with row[0]:
            history_days = st.number_input(
                "History to fetch (days)", min_value=30, max_value=2000, value=400, step=30,
                help="How far back to pull activity summaries on a full sync.")
            wellness_days = st.number_input(
                "Wellness days", min_value=0, max_value=365, value=90, step=15,
                help=("Resting HR, HRV, sleep, body battery and weight. One request per day, "
                      "so this dominates sync time — set 0 to skip."))
        with row[1]:
            detail_batch = st.number_input(
                "Strength workouts to detail per sync", min_value=10, max_value=1000,
                value=150, step=25,
                help=("Per-set detail costs one request per workout. Large histories are "
                      "spread over several syncs so Garmin's rate limiter stays happy."))
            split_batch = st.number_input(
                "Runs to fetch splits for per sync", min_value=0, max_value=1000,
                value=120, step=20,
                help=("Per-lap data, which is what makes aerobic decoupling and interval "
                      "analysis possible. One request per activity."))
            physiology_days = st.number_input(
                "Days of Garmin training metrics", min_value=0, max_value=120, value=21,
                step=7,
                help=("Training status, readiness and load from Garmin's own model, plus "
                      "lactate threshold, zones, race predictions and personal records. "
                      "Roughly three requests per day."))

        row2 = st.columns(3)
        with row2[0]:
            incremental = st.checkbox(
                "Incremental (resume from the last synced activity)", value=True,
                help="Overlaps three days so edited activities get picked up.")
        with row2[1]:
            fetch_details = st.checkbox("Fetch per-set strength detail", value=True)
        with row2[2]:
            fetch_splits = st.checkbox("Fetch per-lap splits", value=True)

        submitted = st.form_submit_button("Run sync", type="primary")

    if submitted:
        result = run_sync(demo=demo, history_days=int(history_days),
                          fetch_details=fetch_details, detail_batch=int(detail_batch),
                          wellness_days=int(wellness_days), incremental=incremental,
                          fetch_splits=fetch_splits, split_batch=int(split_batch),
                          physiology_days=int(physiology_days))
        if result["ok"]:
            st.success(f"Done — {result['summary']}")
            if result.get("errors"):
                with st.expander(f"{len(result['errors'])} warnings"):
                    for message in result["errors"][:40]:
                        st.text(message)
        else:
            st.error(result["error"])


def _analysis_settings(colors) -> None:
    ui.section("Analysis settings", "These change how the metrics are computed.", colors)
    row = st.columns(3)
    with row[0]:
        st.slider(
            "Volume window (days)", min_value=28, max_value=180,
            value=st.session_state.get("lookback_days", 84), step=7, key="lookback_days",
            help="Window for per-muscle weekly volume, intensity distribution and mix.")
        st.slider(
            "Progression window (days)", min_value=60, max_value=540,
            value=st.session_state.get("progress_lookback_days", 180), step=30,
            key="progress_lookback_days",
            help="Window for estimated-1RM and running performance trends.")
    with row[1]:
        st.number_input(
            "Weekly set target — minimum", min_value=2, max_value=40,
            value=st.session_state.get("weekly_sets_min", settings.weekly_sets_min),
            key="weekly_sets_min",
            help="Effective sets per muscle per week below which a muscle is under-trained.")
        st.number_input(
            "Weekly set target — maximum", min_value=4, max_value=60,
            value=st.session_state.get("weekly_sets_max", settings.weekly_sets_max),
            key="weekly_sets_max",
            help="Above this, extra sets compete for recovery with muscles that need work.")
    with row[2]:
        st.radio("Chart theme", ["Auto", "Light", "Dark"],
                 index=["Auto", "Light", "Dark"].index(
                     st.session_state.get("theme_override", "Auto")),
                 key="theme_override",
                 help="Auto follows your Streamlit theme.")

    st.caption(
        "Set `MAX_HR` and `RESTING_HR` in `.env` to pin the heart-rate bounds used to estimate "
        "load for activities the device didn't score. Left blank, they're inferred from your "
        "own data — see the note on the Load & recovery page for the values in use."
    )


def _connection_status(colors) -> None:
    ui.section("Connections", None, colors)
    rows = [
        {"Service": "Garmin Connect",
         "Status": ("cached tokens" if settings.has_cached_tokens else
                    "credentials set" if settings.has_garmin_credentials else "not configured"),
         "Detail": (f"token store `{settings.token_store.name}`"
                    if settings.has_cached_tokens
                    else settings.garmin_email or "set GARMIN_EMAIL / GARMIN_PASSWORD in .env")},
        {"Service": "Claude (AI coach)",
         "Status": "key set" if settings.has_anthropic_key else "not configured",
         "Detail": (f"model `{settings.model}`" if settings.has_anthropic_key
                    else "set ANTHROPIC_API_KEY in .env for the coaching layer")},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    coverage = coverage_report()
    st.caption(
        f"Exercise map: {coverage['categories']} Garmin categories and "
        f"{coverage['named_variants']} named variants across {coverage['muscles']} muscles. "
        f"Garmin has no muscle data of its own — this mapping is what makes the strength "
        f"analysis possible, and it lives in `src/smart_analytics/domain/exercises.py`."
    )


def _maintenance(connection, colors) -> None:
    ui.section("Maintenance", None, colors)
    st.caption(
        "Your data never leaves this machine: everything is in the local SQLite file, which "
        "is gitignored. The only outbound calls are to Garmin (to fetch) and, if you enable "
        "the coach, to the Claude API (the computed briefing only — never raw activity data)."
    )
    with st.expander("Danger zone"):
        st.write("Deleting the cache is irreversible. A fresh full sync will re-pull it.")
        confirm = st.text_input("Type DELETE to confirm", key="confirm_delete")
        if st.button("Delete all cached data", type="secondary"):
            if confirm == "DELETE":
                db.clear_all(connection)
                bump_data_version()
                st.success("Local cache cleared.")
            else:
                st.error("Type DELETE in the box first.")
