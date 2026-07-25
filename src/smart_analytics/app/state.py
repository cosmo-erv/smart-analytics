"""Shared Streamlit state: the database handle, the cached report, the palette.

Analytics run over the whole local cache, which is too slow to redo on every
widget interaction, so the report is cached and keyed on a ``data_version``
counter that only a sync bumps.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from typing import Any

import streamlit as st

from .. import db
from ..analytics import TrainingReport, build_report
from ..analytics import snapshots
from ..config import Settings, settings
from ..garmin import GarminAuthError, GarminClient, SampleGarminClient, sync
from ..garmin.client import MFA_UNKNOWN
from ..garmin.sync import incremental_since
from ..viz.theme import Palette, palette_for

DATA_VERSION_KEY = "data_version"
DEMO_KEY = "demo_mode"
LOGIN_CLIENT_KEY = "garmin_login_client"
LOGIN_STAGE_KEY = "garmin_login_stage"   # "" | "mfa" | "connected"
LOGIN_NAME_KEY = "garmin_account_name"
LOGIN_MFA_METHOD_KEY = "garmin_mfa_method"
LOGIN_MFA_FLOW_KEY = "garmin_mfa_flow"


@st.cache_resource(show_spinner=False)
def get_connection(db_path: str) -> sqlite3.Connection:
    return db.connect(db_path)


def conn() -> sqlite3.Connection:
    return get_connection(str(settings.db_path))


def data_version() -> int:
    return st.session_state.setdefault(DATA_VERSION_KEY, 0)


def bump_data_version() -> None:
    st.session_state[DATA_VERSION_KEY] = data_version() + 1
    load_report.clear()


@st.cache_data(show_spinner="Running analytics…")
def load_report(version: int, lookback_days: int, progress_lookback_days: int,
                weekly_sets_min: int, weekly_sets_max: int) -> TrainingReport:
    """Build the full report. ``version`` exists purely to invalidate the cache."""
    config = Settings(
        garmin_email=settings.garmin_email,
        garmin_password=settings.garmin_password,
        token_store=settings.token_store,
        db_path=settings.db_path,
        anthropic_api_key=settings.anthropic_api_key,
        model=settings.model,
        max_hr=settings.max_hr,
        resting_hr=settings.resting_hr,
        weekly_sets_min=weekly_sets_min,
        weekly_sets_max=weekly_sets_max,
    )
    return build_report(conn(), config, lookback_days=lookback_days,
                        progress_lookback_days=progress_lookback_days)


def current_report() -> TrainingReport:
    return load_report(
        data_version(),
        st.session_state.get("lookback_days", 84),
        st.session_state.get("progress_lookback_days", 180),
        st.session_state.get("weekly_sets_min", settings.weekly_sets_min),
        st.session_state.get("weekly_sets_max", settings.weekly_sets_max),
    )


def palette() -> Palette:
    """Follow the viewer's Streamlit theme unless they override it in the sidebar."""
    override = st.session_state.get("theme_override", "Auto")
    if override in {"Light", "Dark"}:
        return palette_for(override.lower())
    detected = "light"
    try:  # newer Streamlit exposes the resolved theme on st.context
        detected = st.context.theme.type or "light"
    except Exception:
        try:
            detected = st.get_option("theme.base") or "light"
        except Exception:
            detected = "light"
    return palette_for(detected)


# --- Garmin Connect login ---------------------------------------------------
# The password is used once, to mint OAuth tokens; only those tokens are written
# to disk (in settings.token_store, gitignored). Nothing is stored in the
# database, and the password is never held past the login call.

def garmin_login_stage() -> str:
    if st.session_state.get(LOGIN_STAGE_KEY) == "mfa":
        return "mfa"
    return "connected" if settings.has_cached_tokens else ""


def garmin_account_name() -> str | None:
    """The signed-in account, if this session has established one."""
    return st.session_state.get(LOGIN_NAME_KEY) or None


def garmin_mfa_method() -> str:
    """How Garmin said it would deliver the code for the pending login."""
    return st.session_state.get(LOGIN_MFA_METHOD_KEY) or MFA_UNKNOWN


def garmin_mfa_flow() -> str | None:
    """Which Garmin login flow raised the challenge, for the no-code diagnostics."""
    return st.session_state.get(LOGIN_MFA_FLOW_KEY) or None


def begin_garmin_login(email: str, password: str) -> dict[str, Any]:
    """Step one: exchange credentials for tokens, or ask for an MFA code.

    The half-finished client is kept in session state because Garmin's MFA
    resume needs the same client object, and a Streamlit rerun happens between
    the two steps.
    """
    client = GarminClient()
    try:
        result = client.begin_login(email, password)
    except GarminAuthError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 — network, unexpected payload
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if result["status"] == "mfa_required":
        st.session_state[LOGIN_CLIENT_KEY] = client
        st.session_state[LOGIN_STAGE_KEY] = "mfa"
        st.session_state[LOGIN_MFA_METHOD_KEY] = result.get("method")
        st.session_state[LOGIN_MFA_FLOW_KEY] = result.get("flow")
        return {"ok": True, "mfa_required": True, "method": result.get("method")}

    _finish_login(client, result.get("display_name"))
    return {"ok": True, "mfa_required": False, "name": garmin_account_name()}


def complete_garmin_login(code: str) -> dict[str, Any]:
    """Step two: hand Garmin the emailed code and cache the resulting tokens."""
    client = st.session_state.get(LOGIN_CLIENT_KEY)
    if client is None:
        st.session_state[LOGIN_STAGE_KEY] = ""
        return {"ok": False, "error": "That login attempt expired — start again."}
    try:
        result = client.complete_login(code)
    except GarminAuthError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    _finish_login(client, result.get("display_name"))
    return {"ok": True, "name": garmin_account_name()}


def _finish_login(client: GarminClient, name: str | None) -> None:
    st.session_state[LOGIN_STAGE_KEY] = "connected"
    st.session_state[LOGIN_NAME_KEY] = name or "your Garmin account"
    st.session_state[DEMO_KEY] = False
    # Drop the client: the tokens on disk are what later syncs resume from, so
    # there's no reason to keep a live session (or its credentials) around.
    st.session_state.pop(LOGIN_CLIENT_KEY, None)


def garmin_sign_out() -> None:
    """Forget the cached tokens. Nothing already synced is affected."""
    GarminClient().sign_out()
    for key in (LOGIN_CLIENT_KEY, LOGIN_STAGE_KEY, LOGIN_NAME_KEY,
                LOGIN_MFA_METHOD_KEY, LOGIN_MFA_FLOW_KEY):
        st.session_state.pop(key, None)


# --- sync -------------------------------------------------------------------

def run_sync(*, demo: bool, history_days: int, fetch_details: bool, detail_batch: int,
             wellness_days: int, incremental: bool, fetch_splits: bool = True,
             split_batch: int = 120, physiology_days: int = 21,
             fetch_workouts: bool = True) -> dict[str, Any]:
    """Run a sync with live progress, returning a small result summary."""
    connection = conn()
    since: date | None = incremental_since(connection) if incremental else None

    status = st.status("Connecting…", expanded=True)
    bar = status.progress(0.0)

    def report_progress(message: str, fraction: float) -> None:
        status.update(label=message)
        bar.progress(min(max(fraction, 0.0), 1.0))

    try:
        if demo:
            client = SampleGarminClient(days=max(history_days, 120))
            status.update(label="Generating demo data…")
        else:
            client = GarminClient().connect()
            status.update(label=f"Connected as {client.display_name()}")

        result = sync(
            connection, client,
            since=since,
            history_days=history_days,
            fetch_details=fetch_details,
            detail_batch=detail_batch,
            wellness_days=wellness_days,
            fetch_splits=fetch_splits,
            split_batch=split_batch,
            fetch_workouts=fetch_workouts,
            physiology_days=physiology_days,
            throttle_s=0.0 if demo else 0.25,
            progress=report_progress,
        )
    except GarminAuthError as exc:
        status.update(label="Garmin login failed", state="error")
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # network, rate limit, unexpected payload
        status.update(label="Sync failed", state="error")
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    status.update(label=f"Synced — {result.summary()}", state="complete")
    bump_data_version()
    # Snapshot the freshly computed report so the Progress page has a series to
    # trend. Failing here must not fail the sync that already succeeded.
    try:
        snapshots.save_snapshot(connection, current_report())
    except Exception as exc:  # noqa: BLE001 - snapshotting is best-effort
        st.caption(f"Sync completed; snapshot skipped ({type(exc).__name__}).")
    st.session_state[DEMO_KEY] = demo
    return {"ok": True, "summary": result.summary(), "errors": result.errors}
