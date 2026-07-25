"""Reusable UI pieces: stat tiles, finding cards, chart+table pairs.

Two conventions are enforced here so no page has to remember them:

* A status colour never carries meaning alone — every finding card pairs its
  colour with an icon and a written severity label.
* Every chart ships a table view alongside it. Three light-mode palette slots sit
  under 3:1 contrast on the light surface, and the table is the documented relief
  for that; it also makes the underlying numbers checkable.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..analytics.findings import SEVERITY_LABEL, Finding
from ..viz.theme import SEVERITY_ICON, Palette, severity_color

PLOTLY_CONFIG = {"displayModeBar": False, "scrollZoom": False, "responsive": True}


def chart(figure: go.Figure, palette: Palette, *, table: pd.DataFrame | None = None,
          table_label: str = "View as table", key: str | None = None) -> None:
    """Render a figure, with the table view that the contrast rule obliges."""
    st.plotly_chart(figure, use_container_width=True, config=PLOTLY_CONFIG, key=key)
    if table is not None and not table.empty:
        with st.expander(table_label):
            st.dataframe(table, use_container_width=True, hide_index=True)


def stat_tiles(tiles: Sequence[dict[str, Any]], columns: int | None = None,
               palette: Palette | None = None) -> None:
    """A KPI row.

    Each tile is ``{label, value, delta?, note?, help?, delta_color?}``. ``delta``
    is for signed change only — Streamlit draws a direction arrow on it, so
    non-numeric context ("of 18 muscles") goes in ``note`` instead, which renders
    as plain caption text. An arrow that points nowhere is worse than no arrow.
    """
    if not tiles:
        return
    cols = st.columns(columns or len(tiles))
    for col, tile in zip(cols, tiles):
        with col:
            st.metric(
                label=tile["label"],
                value=tile["value"],
                delta=tile.get("delta"),
                delta_color=tile.get("delta_color", "normal"),
                help=tile.get("help"),
            )
            if tile.get("note"):
                ink = palette.ink_muted if palette else "#898781"
                st.markdown(
                    f'<div style="margin-top:-8px;font-size:0.78rem;color:{ink};">'
                    f'{tile["note"]}</div>',
                    unsafe_allow_html=True,
                )


def severity_badge(severity: str, palette: Palette) -> str:
    """Icon + colour + written label, so colour is never the only signal."""
    color = severity_color(severity, palette)
    icon = SEVERITY_ICON.get(severity, "•")
    label = SEVERITY_LABEL.get(severity, severity.title())
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'font-size:0.78rem;font-weight:600;color:{palette.ink_secondary};'
        f'letter-spacing:0.01em;">'
        f'<span style="color:{color};font-size:0.9rem;line-height:1;">{icon}</span>'
        f'{label}</span>'
    )


def finding_card(finding: Finding, palette: Palette) -> None:
    with st.container(border=True):
        header, metric = st.columns([5, 1.4])
        with header:
            st.markdown(
                severity_badge(finding.severity, palette)
                + f'<div style="font-weight:600;font-size:1rem;margin-top:2px;'
                  f'color:{palette.ink};">{finding.title}</div>',
                unsafe_allow_html=True,
            )
        with metric:
            if finding.metric:
                st.markdown(
                    f'<div style="text-align:right;font-size:1.05rem;font-weight:600;'
                    f'color:{palette.ink};font-variant-numeric:tabular-nums;">'
                    f'{finding.metric}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown(
            f'<div style="color:{palette.ink_secondary};font-size:0.92rem;'
            f'line-height:1.5;">{finding.detail}</div>',
            unsafe_allow_html=True,
        )
        if finding.recommendation:
            st.markdown(
                f'<div style="margin-top:8px;padding-left:10px;'
                f'border-left:2px solid {severity_color(finding.severity, palette)};'
                f'color:{palette.ink};font-size:0.92rem;">'
                f'<strong>Do this:</strong> {finding.recommendation}</div>',
                unsafe_allow_html=True,
            )


def findings_list(findings: Iterable[Finding], palette: Palette, *, limit: int | None = None,
                  empty_message: str = "Nothing flagged in this area.") -> None:
    items = list(findings)
    if limit:
        items = items[:limit]
    if not items:
        st.info(empty_message)
        return
    for finding in items:
        finding_card(finding, palette)


def section(title: str, description: str | None, palette: Palette) -> None:
    st.markdown(
        f'<div style="margin:6px 0 2px;font-size:1.12rem;font-weight:650;'
        f'color:{palette.ink};">{title}</div>',
        unsafe_allow_html=True,
    )
    if description:
        st.markdown(
            f'<div style="color:{palette.ink_muted};font-size:0.88rem;'
            f'margin-bottom:10px;">{description}</div>',
            unsafe_allow_html=True,
        )


def hero(value: str, label: str, palette: Palette, sub: str | None = None) -> None:
    """The one number a page leads with."""
    st.markdown(
        f'<div style="margin:2px 0 14px;">'
        f'<div style="font-size:0.82rem;text-transform:uppercase;letter-spacing:0.06em;'
        f'color:{palette.ink_muted};font-weight:600;">{label}</div>'
        f'<div style="font-size:2.6rem;line-height:1.1;font-weight:680;'
        f'color:{palette.ink};">{value}</div>'
        + (f'<div style="color:{palette.ink_secondary};font-size:0.92rem;">{sub}</div>'
           if sub else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def no_data_notice(what: str = "data") -> None:
    st.warning(
        f"No {what} in the local cache yet. Open **Sync & Settings** in the sidebar and "
        f"either connect your Garmin account or load demo data to explore the app."
    )
