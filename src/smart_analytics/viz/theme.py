"""Chart theme and palette.

Colour is assigned by the *job* it does, not picked per chart:

* **categorical** (telling series apart) — fixed slot order, never cycled. The
  first three slots are validated for all-pairs forms (scatter, small multiples);
  adjacent forms (lines, bars, stacks) may use up to six, with direct labels
  mandatory past three.
* **sequential** (magnitude — heatmaps) — one blue hue, light→dark.
* **ordinal** (ordered categories such as easy/moderate/hard intensity) — the
  same blue ramp, clipped so the lightest step still clears the surface.
* **status** (good/watch/act) — a reserved four-step palette, never reused for a
  series, and always shipped with a text label rather than colour alone.

Every hex here is validated against both the light and dark chart surfaces
(lightness band, chroma floor, colour-vision-deficiency separation, normal-vision
separation, contrast). Three light-mode slots sit under 3:1 on the light surface,
which is why every chart in this app also ships a table view — that's the
documented relief for the contrast warning, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

FONT_STACK = 'system-ui, -apple-system, "Segoe UI", sans-serif'


@dataclass(frozen=True)
class Palette:
    mode: str
    surface: str
    page: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    categorical: list[str]
    sequential: list[str]
    ordinal: list[str]
    status: dict[str, str] = field(default_factory=dict)

    @property
    def safe_categorical(self) -> list[str]:
        """The subset validated for all-pairs forms (scatter, small multiples)."""
        return self.categorical[:3]

    def series(self, index: int) -> str:
        """Colour for slot ``index``; folds past the ceiling instead of cycling."""
        if index < len(self.categorical):
            return self.categorical[index]
        return self.ink_muted


LIGHT = Palette(
    mode="light",
    surface="#fcfcfb",
    page="#f9f9f7",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    categorical=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
    sequential=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"],
    ordinal=["#86b6ef", "#2a78d6", "#104281"],
    status={"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a",
            "critical": "#d03b3b"},
)

DARK = Palette(
    mode="dark",
    surface="#1a1a19",
    page="#0d0d0d",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    categorical=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
    sequential=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    ordinal=["#cde2fb", "#3987e5", "#184f95"],
    status={"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a",
            "critical": "#d03b3b"},
)

# Finding severity → status role. Status colours always ship beside a text label.
SEVERITY_STATUS = {"good": "good", "info": "warning", "watch": "serious", "act": "critical"}

SEVERITY_ICON = {"good": "✓", "info": "•", "watch": "▲", "act": "■"}


def palette_for(mode: str | None) -> Palette:
    return DARK if (mode or "light").lower() == "dark" else LIGHT


def severity_color(severity: str, palette: Palette) -> str:
    return palette.status.get(SEVERITY_STATUS.get(severity, "warning"), palette.ink_muted)


def register_template(palette: Palette) -> str:
    """Register a Plotly template for this palette and return its name."""
    name = f"smart_analytics_{palette.mode}"
    template = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=palette.surface,
            plot_bgcolor=palette.surface,
            colorway=palette.categorical,
            font=dict(family=FONT_STACK, size=13, color=palette.ink_secondary),
            title=dict(font=dict(size=15, color=palette.ink), x=0, xanchor="left", pad=dict(b=12)),
            margin=dict(l=8, r=8, t=44, b=8),
            hoverlabel=dict(
                bgcolor=palette.surface, bordercolor=palette.axis,
                font=dict(family=FONT_STACK, size=12, color=palette.ink),
            ),
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                        font=dict(size=12, color=palette.ink_secondary),
                        bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(
                gridcolor=palette.grid, gridwidth=1, zeroline=False,
                linecolor=palette.axis, ticks="outside", ticklen=4,
                tickcolor=palette.axis, tickfont=dict(size=11, color=palette.ink_muted),
                title=dict(font=dict(size=12, color=palette.ink_muted)),
            ),
            yaxis=dict(
                gridcolor=palette.grid, gridwidth=1, zeroline=False,
                linecolor=palette.axis, ticks="outside", ticklen=4,
                tickcolor=palette.axis, tickfont=dict(size=11, color=palette.ink_muted),
                title=dict(font=dict(size=12, color=palette.ink_muted)),
            ),
        )
    )
    pio.templates[name] = template
    return name


def style(fig: go.Figure, palette: Palette, *, title: str | None = None,
         height: int = 340, y_title: str | None = None, x_title: str | None = None,
         show_legend: bool | None = None) -> go.Figure:
    """Apply the template and the recessive-chrome conventions to a figure."""
    fig.update_layout(
        template=register_template(palette),
        height=height,
        showlegend=show_legend if show_legend is not None else _needs_legend(fig),
    )
    if title:
        fig.update_layout(title_text=title)
    if y_title:
        fig.update_yaxes(title_text=y_title)
    if x_title:
        fig.update_xaxes(title_text=x_title)
    return fig


def _needs_legend(fig: go.Figure) -> bool:
    """A legend is present for two or more series, absent for one."""
    named = [t for t in fig.data if getattr(t, "name", None) and getattr(t, "showlegend", None)
             is not False]
    return len(named) >= 2


# --- mark specs -------------------------------------------------------------

LINE_WIDTH = 2
MARKER_SIZE = 8
BAR_RADIUS = 4
FILL_GAP = 2  # surface-coloured gap between adjacent/stacked fills


def line_spec(color: str, dash: str | None = None) -> dict[str, Any]:
    spec: dict[str, Any] = {"color": color, "width": LINE_WIDTH}
    if dash:
        spec["dash"] = dash
    return spec


def bar_spec(color: str, palette: Palette) -> dict[str, Any]:
    """Bars get rounded data-ends and a surface gap so fills never touch."""
    return {"color": color, "line": {"color": palette.surface, "width": FILL_GAP}}


def marker_spec(color: str, palette: Palette) -> dict[str, Any]:
    """Overlapping marks carry a surface ring so they stay separable."""
    return {"color": color, "size": MARKER_SIZE,
            "line": {"color": palette.surface, "width": FILL_GAP}}
