"""Sync & Settings: pull data from Garmin, tune the analysis, inspect the cache."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ... import db
from ...config import settings
from ...domain.exercises import coverage_report
from ...domain.garmin_muscles import unmappable_reason
from ...garmin.client import MFA_AUTHENTICATOR, MFA_EMAIL, MFA_SMS, MFA_UNKNOWN
from ...garmin.sync import last_sync_at
from .. import components as ui
from ..state import (
    begin_garmin_login,
    bump_data_version,
    complete_garmin_login,
    conn,
    current_report,
    garmin_account_name,
    garmin_login_stage,
    garmin_mfa_flow,
    garmin_mfa_method,
    garmin_sign_out,
    garmin_web_flow_note,
    palette,
    run_sync,
)


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
        {"label": "Workouts", "value": f"{counts['workouts']:,}"},
        {"label": "Personal records", "value": f"{counts['personal_records']:,}"},
        {"label": "Open niggles", "value": f"{counts['niggles']:,}"},
        {"label": "Snapshots", "value": f"{counts['snapshots']:,}"},
    ], palette=colors)
    st.caption(f"{counts['types']} activity types · "
               f"{counts['workout_steps']:,} workout steps cached.")
    if counts["first_date"]:
        st.caption(f"Cached range {counts['first_date']} → {counts['last_date']} · "
                   f"last sync {last_sync_at(connection) or 'never'} · "
                   f"database `{settings.db_path}`")

    st.divider()
    _garmin_account(colors)

    st.divider()
    _sync_panel(colors)

    st.divider()
    _analysis_settings(colors)

    st.divider()
    _connection_status(colors)

    st.divider()
    _maintenance(connection, colors)


_MFA_PROMPTS: dict[str, str] = {
    MFA_EMAIL: "Garmin emailed you a verification code. Enter it to finish signing in.",
    MFA_SMS: "Garmin texted you a verification code. Enter it to finish signing in.",
    MFA_AUTHENTICATOR: ("Your account uses an authenticator app — no email or text is "
                        "sent. Open the app and enter the current code for Garmin."),
    MFA_UNKNOWN: ("Garmin wants a verification code, but didn't say how it sent it — check "
                  "your email, your texts, and your authenticator app if you use one."),
}


def _no_code_help() -> None:
    """What to try when the code never turns up.

    Worth spelling out, because most of the causes aren't visible from here: the
    wrong address on the account, a throttled burst of attempts, or a delivery
    problem at the mail provider all look identical to the user.
    """
    with st.expander("The code hasn't arrived"):
        st.markdown(
            "1. **Check spam**, and search for `noreply@garmin.com`. Codes expire quickly, "
            "so an older one won't work — use **Start over** for a fresh one.\n"
            "2. **Check the address on the Garmin account.** The code goes there, which "
            "isn't necessarily the address you typed here.\n"
            "3. **Confirm the method** at Garmin Connect → Account settings → Sign-in & "
            "security → Multi-factor authentication. An authenticator app sends nothing.\n"
            "4. **Wait, then try once.** Garmin throttles repeated MFA attempts, so a burst "
            "of retries can stop codes arriving entirely. Give it 30 minutes and make a "
            "single attempt.\n"
            "5. **Check whether it's us or Garmin:** sign in at connect.garmin.com in a "
            "browser. If no code arrives there either, the problem is on Garmin's side or "
            "at your mail provider, and nothing in this app can route around it."
        )
        flow, web_flow = garmin_mfa_flow(), garmin_web_flow_note()
        if flow or web_flow:
            lines = []
            if web_flow:
                lines.append(f"- Browser sign-in: {web_flow}")
            if flow:
                lines.append(f"- Challenge raised over Garmin's `{flow}` flow")
            st.caption("**Diagnostics**\n" + "\n".join(lines))


def _garmin_account(colors) -> None:
    ui.section("Garmin Connect account", None, colors)
    stage = garmin_login_stage()

    if stage == "connected":
        name = garmin_account_name()
        st.success(f"Connected to Garmin Connect{f' as **{name}**' if name else ''}.")
        st.caption(
            "Your password isn't stored anywhere. The login exchanged it for OAuth tokens, "
            f"which are cached in `{settings.token_store.name}/` (gitignored) and are what "
            "later syncs resume from — so you shouldn't be asked for a code again until they "
            "expire."
        )
        if st.button("Sign out of Garmin", type="secondary"):
            garmin_sign_out()
            st.rerun()
        return

    if stage == "mfa":
        method = garmin_mfa_method()
        st.info(_MFA_PROMPTS.get(method, _MFA_PROMPTS[MFA_UNKNOWN]))
        if method != MFA_AUTHENTICATOR:
            _no_code_help()
        with st.form("garmin_mfa"):
            code = st.text_input("Verification code", max_chars=12,
                                 help="Codes expire quickly — request a new one if it's stale.")
            confirmed = st.form_submit_button("Verify and connect", type="primary")
        if confirmed:
            result = complete_garmin_login(code)
            if result["ok"]:
                st.rerun()
            else:
                st.error(result["error"])
        # Garmin has no resend endpoint — a new code means a new login attempt.
        if st.button("Start over (requests a new code)", type="secondary"):
            garmin_sign_out()
            st.rerun()
        return

    st.write(
        "Sign in to sync your own activities, structured workouts and Garmin's own "
        "training metrics. Multi-factor authentication is supported."
    )
    with st.form("garmin_login"):
        email = st.text_input("Garmin Connect email", value=settings.garmin_email,
                              autocomplete="username")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Connect to Garmin", type="primary")
    if submitted:
        with st.spinner("Signing in to Garmin Connect…"):
            result = begin_garmin_login(email, password)
        if not result["ok"]:
            st.error(result["error"])
        else:
            st.rerun()

    st.caption(
        "This uses the same OAuth flow as the Garmin Connect mobile app — Garmin has no "
        "public consumer API. The password is sent to Garmin only, used once to mint tokens, "
        "and never written to the database. You can also set `GARMIN_EMAIL` / "
        "`GARMIN_PASSWORD` in `.env` instead of signing in here."
    )


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

        row2 = st.columns(4)
        with row2[0]:
            incremental = st.checkbox(
                "Incremental (resume from the last synced activity)", value=True,
                help="Overlaps three days so edited activities get picked up.")
        with row2[1]:
            fetch_details = st.checkbox("Fetch per-set strength detail", value=True)
        with row2[2]:
            fetch_splits = st.checkbox("Fetch per-lap splits", value=True)
        with row2[3]:
            fetch_workouts = st.checkbox(
                "Fetch structured workouts", value=True,
                help=("Workout definitions carry Garmin's own muscle assignments for each "
                      "exercise, which take precedence over the built-in mapping."))

        submitted = st.form_submit_button("Run sync", type="primary")

    if submitted:
        result = run_sync(demo=demo, history_days=int(history_days),
                          fetch_details=fetch_details, detail_batch=int(detail_batch),
                          wellness_days=int(wellness_days), incremental=incremental,
                          fetch_splits=fetch_splits, split_batch=int(split_batch),
                          fetch_workouts=fetch_workouts,
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
         "Status": ("signed in" if settings.has_cached_tokens else
                    "credentials set" if settings.has_garmin_credentials else "not configured"),
         "Detail": (f"tokens cached in `{settings.token_store.name}`"
                    if settings.has_cached_tokens
                    else settings.garmin_email or "sign in above, or set GARMIN_EMAIL in .env")},
        {"Service": "AI coach",
         "Status": (f"{settings.provider} key set" if settings.has_ai_key
                    else "not configured"),
         "Detail": (f"model `{settings.active_model}`" if settings.has_ai_key
                    else "set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    _muscle_mapping(colors)


def _muscle_mapping(colors) -> None:
    """Where each exercise's muscle attribution came from.

    Garmin's structured workouts state which muscles it assigns to an exercise,
    and that assignment wins. The curated table only fills the gaps, so it's
    worth showing which of the two is actually doing the work.
    """
    ui.section("Muscle mapping", None, colors)
    sources = current_report().muscle_sources
    coverage = sources or coverage_report()

    ui.stat_tiles([
        {"label": "From Garmin", "value": f"{coverage.get('garmin_entries', 0):,}",
         "note": "exercises and categories Garmin assigned muscles to"},
        {"label": "Named exercises", "value": f"{coverage.get('garmin_named', 0):,}",
         "note": "from your structured workouts"},
        {"label": "Built-in fallback", "value": f"{coverage.get('named_variants', 0):,}",
         "note": f"named variants, plus {coverage.get('categories', 0)} categories"},
    ], palette=colors)

    if coverage.get("garmin_entries"):
        st.caption(
            "Garmin's own assignment is used wherever it exists — matched on the exercise "
            "name first, then its category. The built-in table in "
            "`src/smart_analytics/domain/exercises.py` covers anything Garmin didn't label, "
            "and movement patterns (push/pull/hinge/squat) always come from it, since Garmin "
            "doesn't classify movements that way."
        )
    else:
        st.caption(
            "No Garmin muscle data has been synced yet, so the built-in mapping is doing all "
            "the work. Sign in and run a sync with **Fetch structured workouts** enabled to "
            "use Garmin's own assignments instead."
        )

    unmatched = sources.get("unmatched_names") or {}
    if unmatched:
        label = ("1 Garmin muscle name couldn't be placed" if len(unmatched) == 1
                 else f"{len(unmatched)} Garmin muscle names couldn't be placed")
        with st.expander(label):
            st.caption(
                "These appeared in your workouts but have no home in the 18-muscle model. "
                "The exercise is still credited using the muscles that did match."
            )
            st.dataframe(
                pd.DataFrame([
                    {"Garmin name": name, "Times seen": count,
                     "Why": unmappable_reason(name) or "not recognised — worth mapping"}
                    for name, count in unmatched.items()
                ]),
                use_container_width=True, hide_index=True)


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
