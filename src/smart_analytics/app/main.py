"""Streamlit entry point. Run with ``streamlit run app.py`` from the repo root."""

from __future__ import annotations

import streamlit as st

from ..config import settings
from .. import db
from .pages import coach, data, load, overview, plan, progress, running, strength
from .state import conn, garmin_account_name, palette
from ..garmin.sync import last_sync_at

PAGE_ICON = "🏃"
PAGES_KEY = "nav_pages"


def main() -> None:
    st.set_page_config(
        page_title="Smart Analytics — Garmin training insight",
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _sidebar()

    pages = {
        "overview": st.Page(overview.render, title="Overview", icon=":material/dashboard:",
                            default=True),
        "strength": st.Page(strength.render, title="Strength",
                            icon=":material/fitness_center:", url_path="strength"),
        "running": st.Page(running.render, title="Running",
                           icon=":material/directions_run:", url_path="running"),
        "load": st.Page(load.render, title="Load & recovery",
                        icon=":material/monitor_heart:", url_path="load"),
        "plan": st.Page(plan.render, title="Plan", icon=":material/event_available:",
                        url_path="plan"),
        "progress": st.Page(progress.render, title="Progress",
                            icon=":material/trending_up:", url_path="progress"),
        "coach": st.Page(coach.render, title="AI coach", icon=":material/psychology:",
                         url_path="coach"),
        "settings": st.Page(data.render, title="Sync & settings", icon=":material/settings:",
                            url_path="settings"),
    }
    # Pages need to reference each other for cross-links, and st.page_link only
    # accepts the StreamlitPage object rather than a path string.
    st.session_state[PAGES_KEY] = pages
    st.navigation(list(pages.values())).run()


def _sidebar() -> None:
    colors = palette()
    with st.sidebar:
        st.markdown(
            f'<div style="font-size:1.06rem;font-weight:680;color:{colors.ink};">'
            f"Smart Analytics</div>"
            f'<div style="font-size:0.82rem;color:{colors.ink_muted};margin-bottom:10px;">'
            f"Garmin data, local-first, AI-assisted</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        try:
            counts = db.counts(conn())
            synced = last_sync_at(conn())
        except Exception as exc:
            st.error(f"Could not open the local database: {exc}")
            return

        if counts["activities"]:
            st.caption(
                f"**{counts['activities']:,}** activities · **{counts['sets']:,}** sets\n\n"
                f"{counts['first_date']} → {counts['last_date']}"
            )
            if synced:
                st.caption(f"Last sync {str(synced)[:16]}")
        else:
            st.info("No data yet. Open **Sync & settings** to load some.")

        st.divider()
        st.caption(
            ("AI coach: ready" if settings.has_ai_key else "AI coach: no API key")
            + "  \n"
            + _garmin_status()
        )


def _garmin_status() -> str:
    if settings.has_cached_tokens:
        name = garmin_account_name()
        return f"Garmin: signed in{f' as {name}' if name else ''}"
    if settings.has_garmin_credentials:
        return "Garmin: credentials set"
    return "Garmin: not signed in"
