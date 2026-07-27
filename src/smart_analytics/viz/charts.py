"""Chart builders.

Each function picks the form from the data's job (magnitude → bar; trend → line;
ordered part-to-whole → diverging-free stacked bar; grid of magnitudes → heatmap)
and returns a styled Plotly figure. Conventions enforced here rather than at
call sites:

* one y-axis per chart, never two scales;
* a legend whenever there are two or more series, none for one;
* direct labels on multi-line charts, so identity never rests on colour alone;
* hover on every mark;
* reference lines carry their meaning as text, so a bar being "short" is legible
  without reading a colour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..analytics.running import format_duration, format_pace
from .theme import Palette, bar_spec, line_spec, marker_spec, style

EMPTY_NOTE = "Not enough data yet"


def _empty(palette: Palette, message: str = EMPTY_NOTE, height: int = 340) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, xref="paper", yref="paper",
                       x=0.5, y=0.5, font=dict(size=13, color=palette.ink_muted))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, palette, height=height, show_legend=False)


# --- strength ---------------------------------------------------------------

def muscle_volume_vs_target(lagging: pd.DataFrame, palette: Palette,
                            target_min: int, target_max: int) -> go.Figure:
    """Weekly effective sets per muscle against the target band.

    One hue plus a reference line: a muscle short of target is *geometrically*
    short, so the reading doesn't depend on colour at all.
    """
    if lagging is None or lagging.empty:
        return _empty(palette, "No mapped strength sets yet", height=460)

    frame = lagging.sort_values("weekly_sets")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=frame["muscle_label"],
        x=frame["weekly_sets"],
        orientation="h",
        marker=bar_spec(palette.sequential[4], palette),
        customdata=np.stack([frame["verdict"], frame["pct_of_target"],
                             frame["attention_score"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>%{x:.1f} effective sets/week"
                       "<br>%{customdata[1]:.0f}% of target"
                       "<br>Verdict: %{customdata[0]}"
                       "<br>Attention score: %{customdata[2]:.0f}/100<extra></extra>"),
        name="Effective sets/week",
    ))

    # Two hairlines rather than a filled band: when every muscle sits below target
    # a filled region becomes the largest shape on the chart and outweighs the data.
    fig.add_vline(x=target_min, line=dict(color=palette.axis, width=1, dash="dot"),
                  annotation_text=f"target {target_min}", annotation_position="top",
                  annotation_font=dict(size=11, color=palette.ink_muted))
    fig.add_vline(x=target_max, line=dict(color=palette.axis, width=1, dash="dot"),
                  annotation_text=f"{target_max}", annotation_position="top",
                  annotation_font=dict(size=11, color=palette.ink_muted))

    fig.update_layout(bargap=0.28)
    headroom = max(float(frame["weekly_sets"].max()) * 1.1, target_max * 1.08)
    fig.update_xaxes(range=[0, headroom])
    return style(fig, palette, title="Weekly volume per muscle", height=460,
                 x_title="effective sets per week", show_legend=False)


def attention_scores(lagging: pd.DataFrame, palette: Palette, limit: int = 10) -> go.Figure:
    """Which muscles need attention — magnitude, so a single sequential hue."""
    if lagging is None or lagging.empty:
        return _empty(palette, "No mapped strength sets yet")

    frame = lagging.head(limit).sort_values("attention_score")
    # Sequential ramp keyed to the score itself, clipped to steps that clear the surface.
    ramp = palette.sequential[1:]
    scores = frame["attention_score"].to_numpy(dtype=float)
    span = max(scores.max() - scores.min(), 1e-6)
    indices = ((scores - scores.min()) / span * (len(ramp) - 1)).round().astype(int)
    colors = [ramp[i] for i in indices]

    fig = go.Figure(go.Bar(
        y=frame["muscle_label"], x=frame["attention_score"], orientation="h",
        marker={"color": colors, "line": {"color": palette.surface, "width": 2}},
        text=[f"{v:.0f}" for v in frame["attention_score"]],
        textposition="outside",
        textfont=dict(size=11, color=palette.ink_secondary),
        customdata=np.stack([frame["verdict"], frame["reasons"]], axis=-1),
        hovertemplate=("<b>%{y}</b> — %{customdata[0]}<br>score %{x:.0f}/100"
                       "<br>%{customdata[1]}<extra></extra>"),
    ))
    fig.update_layout(bargap=0.3)
    fig.update_xaxes(range=[0, 108])
    return style(fig, palette, title="Attention score (higher = more neglected)",
                 height=400, x_title="score", show_legend=False)


def muscle_week_heatmap(weekly_muscle: pd.DataFrame, palette: Palette,
                        weeks: int = 16) -> go.Figure:
    """Muscle × week grid of effective sets — a magnitude grid, so sequential."""
    if weekly_muscle is None or weekly_muscle.empty:
        return _empty(palette, "No mapped strength sets yet", height=520)

    grid = weekly_muscle.pivot_table(index="muscle_label", columns="week",
                                     values="effective_sets", aggfunc="sum").fillna(0)
    if grid.empty:
        return _empty(palette, "No mapped strength sets yet", height=520)
    grid = grid.iloc[:, -weeks:]
    grid = grid.loc[grid.sum(axis=1).sort_values(ascending=True).index]

    colorscale = [[i / (len(palette.sequential) - 1), c]
                  for i, c in enumerate(palette.sequential)]

    fig = go.Figure(go.Heatmap(
        z=grid.to_numpy(),
        x=[pd.Timestamp(c).strftime("%d %b") for c in grid.columns],
        y=grid.index.tolist(),
        colorscale=colorscale,
        # A 2px surface gap between cells keeps adjacent fills from touching.
        xgap=2, ygap=2,
        colorbar=dict(title=dict(text="sets", font=dict(size=11, color=palette.ink_muted)),
                      tickfont=dict(size=10, color=palette.ink_muted),
                      outlinewidth=0, thickness=10, len=0.7),
        hovertemplate="<b>%{y}</b><br>week of %{x}<br>%{z:.1f} effective sets<extra></extra>",
    ))
    return style(fig, palette, title="Effective sets per muscle per week", height=520,
                 show_legend=False)


def exercise_progression(sessions: pd.DataFrame, exercises: list[str],
                         palette: Palette) -> go.Figure:
    """Estimated 1RM over time — multi-line, direct-labelled at the last point."""
    if sessions is None or sessions.empty or not exercises:
        return _empty(palette, "Pick an exercise to see its progression")

    fig = go.Figure()
    for slot, exercise in enumerate(exercises[:6]):
        subset = sessions[sessions["exercise"] == exercise].sort_values("local_date")
        subset = subset.dropna(subset=["best_e1rm"])
        if subset.empty:
            continue
        color = palette.series(slot)
        fig.add_trace(go.Scatter(
            x=subset["local_date"], y=subset["best_e1rm"], mode="lines+markers",
            name=exercise, line=line_spec(color), marker=marker_spec(color, palette),
            customdata=np.stack([subset["top_weight"].fillna(0),
                                 subset["sets"].fillna(0)], axis=-1),
            hovertemplate=("<b>" + exercise + "</b><br>%{x|%d %b %Y}"
                           "<br>e1RM %{y:.1f} kg<br>top set %{customdata[0]:.1f} kg"
                           "<br>%{customdata[1]:.0f} sets<extra></extra>"),
        ))
        # Direct label: identity is never colour-alone.
        last = subset.iloc[-1]
        fig.add_annotation(x=last["local_date"], y=last["best_e1rm"], text=f" {exercise}",
                           showarrow=False, xanchor="left", yanchor="middle",
                           font=dict(size=11, color=palette.ink_secondary))

    if not fig.data:
        return _empty(palette, "No estimated 1RM history for these exercises")
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(domain=[0, 0.78])
    return style(fig, palette, title="Estimated 1RM over time", height=400,
                 y_title="estimated 1RM (kg)", show_legend=False)


# --- running ----------------------------------------------------------------

def efficiency_trend(runs: pd.DataFrame, palette: Palette) -> go.Figure:
    """Metres per heartbeat per run, with a rolling mean to carry the trend."""
    if runs is None or runs.empty or not runs["m_per_beat"].notna().any():
        return _empty(palette, "No heart-rate data on these runs")

    frame = runs.dropna(subset=["m_per_beat"]).sort_values("local_date")
    rolling = frame["m_per_beat"].rolling(8, min_periods=3).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame["local_date"], y=frame["m_per_beat"], mode="markers",
        name="Individual runs",
        marker={**marker_spec(palette.series(0), palette), "size": 7, "opacity": 0.75},
        customdata=np.stack([frame["distance_km"], frame["pace_label"],
                             frame["avg_hr"].fillna(0), frame["intensity"]], axis=-1),
        hovertemplate=("%{x|%d %b %Y}<br>%{y:.2f} m/beat"
                       "<br>%{customdata[0]:.1f} km at %{customdata[1]}"
                       "<br>avg HR %{customdata[2]:.0f} · %{customdata[3]}<extra></extra>"),
    ))
    fig.add_trace(go.Scatter(
        x=frame["local_date"], y=rolling, mode="lines", name="8-run rolling mean",
        line=line_spec(palette.series(1)),
        hovertemplate="%{x|%d %b %Y}<br>rolling mean %{y:.2f} m/beat<extra></extra>",
    ))
    return style(fig, palette, title="Aerobic efficiency (higher is better)", height=360,
                 y_title="metres per heartbeat")


def weekly_running_volume(weekly: pd.DataFrame, palette: Palette) -> go.Figure:
    """Weekly distance — magnitude over time, so bars in a single hue."""
    if weekly is None or weekly.empty:
        return _empty(palette, "No runs yet")

    fig = go.Figure(go.Bar(
        x=weekly["week"], y=weekly["distance_km"],
        marker=bar_spec(palette.sequential[4], palette),
        customdata=np.stack([weekly["runs"],
                             weekly["avg_pace_s_per_km"].map(format_pace),
                             weekly["longest_km"]], axis=-1),
        hovertemplate=("week of %{x|%d %b %Y}<br>%{y:.1f} km"
                       "<br>%{customdata[0]:.0f} runs at %{customdata[1]}"
                       "<br>longest %{customdata[2]:.1f} km<extra></extra>"),
        name="Distance",
    ))
    return style(fig, palette, title="Weekly running volume", height=320,
                 y_title="km per week", show_legend=False)


def intensity_split(distribution: dict, palette: Palette) -> go.Figure:
    """Easy / moderate / hard share of time — an ordered scale, so an ordinal ramp."""
    if not distribution:
        return _empty(palette, "No intensity data on these runs", height=190)

    bands = [("Easy", distribution["easy_pct"], palette.ordinal[0]),
             ("Moderate", distribution["moderate_pct"], palette.ordinal[1]),
             ("Hard", distribution["hard_pct"], palette.ordinal[2])]

    fig = go.Figure()
    for label, value, color in bands:
        fig.add_trace(go.Bar(
            y=["Share of running time"], x=[value], orientation="h", name=label,
            marker={"color": color, "line": {"color": palette.surface, "width": 2}},
            text=[f"{label} {value:.0f}%"], textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=11, color=palette.surface if color != palette.ordinal[0]
                          else palette.ink),
            hovertemplate=f"<b>{label}</b><br>%{{x:.1f}}% of time<extra></extra>",
        ))
    fig.update_layout(barmode="stack", bargap=0.5, legend_traceorder="normal")
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(showticklabels=False)
    return style(fig, palette, title="Intensity distribution", height=190, show_legend=True)


def pace_by_distance(bests: pd.DataFrame, palette: Palette) -> go.Figure:
    """Best equivalent time per distance bucket, all-time versus recent."""
    if bests is None or bests.empty:
        return _empty(palette, "No efforts long enough to rank yet", height=300)

    frame = bests.copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=frame["bucket"], y=frame["best_pace_s_per_km"] / 60,
        name="All-time best pace",
        marker=bar_spec(palette.series(0), palette),
        customdata=np.stack([frame["best_time_s"].map(format_duration),
                             frame["best_date"].dt.strftime("%d %b %Y"),
                             frame["attempts"]], axis=-1),
        hovertemplate=("<b>%{x}</b><br>%{customdata[0]} · %{y:.2f} min/km"
                       "<br>set %{customdata[1]} · %{customdata[2]:.0f} attempts"
                       "<extra></extra>"),
    ))
    recent = frame.dropna(subset=["recent_best_time_s"])
    if not recent.empty:
        fig.add_trace(go.Bar(
            x=recent["bucket"],
            y=recent["recent_best_time_s"] / recent["target_km"] / 60,
            name="Recent best (90 days)",
            marker=bar_spec(palette.series(1), palette),
            customdata=np.stack([recent["recent_best_time_s"].map(format_duration),
                                 recent["pct_off_best"]], axis=-1),
            hovertemplate=("<b>%{x}</b><br>%{customdata[0]} · %{y:.2f} min/km"
                           "<br>%{customdata[1]:+.1f}% vs personal best<extra></extra>"),
        ))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    return style(fig, palette, title="Best pace by distance", height=320,
                 y_title="min per km")


# --- load -------------------------------------------------------------------

def acwr_chart(series: pd.DataFrame, palette: Palette, days: int = 180) -> go.Figure:
    """Acute:chronic ratio with its risk bands drawn as labelled regions."""
    if series is None or series.empty or not series["acwr"].notna().any():
        return _empty(palette, "Needs 28 days of activity history")

    frame = series.dropna(subset=["acwr"]).tail(days)
    fig = go.Figure()
    # Bands first, so the line sits above them.
    fig.add_hrect(y0=0.8, y1=1.3, fillcolor=palette.grid, opacity=0.6, line_width=0,
                  layer="below", annotation_text="productive 0.8–1.3",
                  annotation_position="top left",
                  annotation_font=dict(size=11, color=palette.ink_muted))
    fig.add_hline(y=1.5, line=dict(color=palette.axis, width=1, dash="dot"),
                  annotation_text="high risk above 1.5", annotation_position="bottom right",
                  annotation_font=dict(size=11, color=palette.ink_muted))
    fig.add_trace(go.Scatter(
        x=frame["date"], y=frame["acwr"], mode="lines", name="ACWR",
        line=line_spec(palette.series(0)),
        customdata=np.stack([frame["acute_7d"], frame["chronic_weekly"],
                             frame["acwr_status"]], axis=-1),
        hovertemplate=("%{x|%d %b %Y}<br>ACWR %{y:.2f} (%{customdata[2]})"
                       "<br>7-day load %{customdata[0]:.0f}"
                       "<br>28-day weekly %{customdata[1]:.0f}<extra></extra>"),
    ))
    return style(fig, palette, title="Training load balance (acute : chronic)", height=340,
                 y_title="ratio", show_legend=False)


def fitness_fatigue(series: pd.DataFrame, palette: Palette, days: int = 240) -> go.Figure:
    """Fitness and fatigue — same unit, so one axis and two series."""
    if series is None or series.empty:
        return _empty(palette, "No training load history yet")

    frame = series.tail(days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame["date"], y=frame["ctl"], mode="lines", name="Fitness (42-day)",
        line=line_spec(palette.series(0)),
        hovertemplate="%{x|%d %b %Y}<br>fitness %{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=frame["date"], y=frame["atl"], mode="lines", name="Fatigue (7-day)",
        line=line_spec(palette.series(1), dash="dash"),
        hovertemplate="%{x|%d %b %Y}<br>fatigue %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(hovermode="x unified")
    return style(fig, palette, title="Fitness and fatigue", height=340,
                 y_title="training load (exponentially weighted)")


def activity_mix(mix: pd.DataFrame, palette: Palette) -> go.Figure:
    """Where training time goes — magnitude by category, one hue."""
    if mix is None or mix.empty:
        return _empty(palette, "No activities in this window", height=280)

    frame = mix.sort_values("hours")
    labels = [t.replace("_", " ").title() for t in frame["activity_type"]]
    fig = go.Figure(go.Bar(
        y=labels, x=frame["hours"], orientation="h",
        marker=bar_spec(palette.sequential[4], palette),
        text=[f"{h:.0f} h" for h in frame["hours"]], textposition="outside",
        textfont=dict(size=11, color=palette.ink_secondary),
        customdata=np.stack([frame["sessions"], frame["distance_km"],
                             frame["load_share_pct"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>%{x:.1f} hours"
                       "<br>%{customdata[0]:.0f} sessions"
                       "<br>%{customdata[1]:.0f} km · %{customdata[2]:.0f}% of load"
                       "<extra></extra>"),
    ))
    fig.update_layout(bargap=0.3)
    fig.update_xaxes(range=[0, float(frame["hours"].max()) * 1.18])
    return style(fig, palette, title="Training time by activity type", height=280,
                 x_title="hours", show_legend=False)


def recovery_trend(daily: pd.DataFrame, palette: Palette, column: str, label: str,
                   palette_slot: int = 0) -> go.Figure:
    """A single wellness metric over time with a 7-day mean."""
    if daily is None or daily.empty or column not in daily or not daily[column].notna().any():
        return _empty(palette, f"No {label.lower()} data synced", height=260)

    frame = daily.dropna(subset=[column]).sort_values("local_date")
    rolling = frame[column].rolling(7, min_periods=3).mean()
    color = palette.series(palette_slot)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame["local_date"], y=frame[column], mode="markers", name=f"Daily {label.lower()}",
        marker={**marker_spec(color, palette), "size": 6, "opacity": 0.55},
        hovertemplate="%{x|%d %b %Y}<br>%{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=frame["local_date"], y=rolling, mode="lines", name="7-day mean",
        line=line_spec(palette.series(palette_slot + 1)),
        hovertemplate="%{x|%d %b %Y}<br>7-day mean %{y:.1f}<extra></extra>",
    ))
    return style(fig, palette, title=label, height=260)


# --- zones, splits, hybrid, progress ----------------------------------------

def zone_distribution_chart(distribution: pd.DataFrame, palette: Palette) -> go.Figure:
    """Share of running distance by pace zone — an ordered scale, so one ramp."""
    if distribution is None or distribution.empty:
        return _empty(palette, "No threshold data from Garmin yet", height=300)

    # The table lists fastest first; a horizontal bar draws its first row at the
    # bottom, so reverse the frame to put the fastest zone at the top. Colour then
    # steps light (slowest) to dark (fastest) with the row order.
    frame = distribution.iloc[::-1].copy()
    ramp = palette.sequential[1:]
    count = max(len(frame) - 1, 1)
    colors = [ramp[min(int(round(i / count * (len(ramp) - 1))), len(ramp) - 1)]
              for i in range(len(frame))]

    fig = go.Figure(go.Bar(
        y=frame["zone"], x=frame["share_pct"], orientation="h",
        marker={"color": colors, "line": {"color": palette.surface, "width": 2}},
        text=[f"{v:.0f}%" for v in frame["share_pct"]], textposition="outside",
        textfont=dict(size=11, color=palette.ink_secondary),
        customdata=np.stack([frame["distance_km"], frame["runs"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>%{x:.1f}% of distance"
                       "<br>%{customdata[0]:.1f} km across %{customdata[1]:.0f} runs"
                       "<extra></extra>"),
    ))
    fig.update_layout(bargap=0.3)
    fig.update_xaxes(range=[0, min(float(frame["share_pct"].max()) * 1.25, 105)], ticksuffix="%")
    return style(fig, palette, title="Distance by pace zone", height=320, show_legend=False)


def decoupling_chart(decoupling: pd.DataFrame, palette: Palette) -> go.Figure:
    """Aerobic decoupling per long run, against the 5% and 10% thresholds."""
    if decoupling is None or decoupling.empty:
        return _empty(palette, "No long runs with split data yet")

    frame = decoupling.sort_values("local_date")
    rolling = frame["decoupling_pct"].rolling(5, min_periods=3).mean()

    fig = go.Figure()
    fig.add_hrect(y0=-5, y1=5, fillcolor=palette.grid, opacity=0.5, line_width=0, layer="below",
                  annotation_text="aerobically comfortable (under 5%)",
                  annotation_position="bottom left",
                  annotation_font=dict(size=11, color=palette.ink_muted))
    fig.add_hline(y=10, line=dict(color=palette.axis, width=1, dash="dot"),
                  annotation_text="beyond current endurance above 10%",
                  annotation_position="top left",
                  annotation_font=dict(size=11, color=palette.ink_muted))
    fig.add_trace(go.Scatter(
        x=frame["local_date"], y=frame["decoupling_pct"], mode="markers",
        name="Long runs",
        marker={**marker_spec(palette.series(0), palette), "size": 7, "opacity": 0.8},
        customdata=np.stack([frame["distance_km"], frame["hr_drift_bpm"],
                             frame["verdict"]], axis=-1),
        hovertemplate=("%{x|%d %b %Y}<br>%{y:.1f}% decoupling"
                       "<br>%{customdata[0]:.1f} km · HR drift %{customdata[1]:+.0f} bpm"
                       "<br>%{customdata[2]}<extra></extra>"),
    ))
    fig.add_trace(go.Scatter(
        x=frame["local_date"], y=rolling, mode="lines", name="5-run rolling mean",
        line=line_spec(palette.series(1)),
        hovertemplate="%{x|%d %b %Y}<br>rolling mean %{y:.1f}%<extra></extra>",
    ))
    return style(fig, palette, title="Aerobic decoupling (lower is better)", height=360,
                 y_title="efficiency lost, first half → second (%)")


def discipline_split_chart(split: pd.DataFrame, palette: Palette) -> go.Figure:
    """Weekly training load split between strength and running."""
    if split is None or split.empty:
        return _empty(palette, "Needs both strength and running data", height=320)

    fig = go.Figure()
    for slot, (column, label) in enumerate([("running_load", "Running"),
                                            ("strength_load", "Strength"),
                                            ("other_load", "Other")]):
        if column not in split or float(split[column].sum()) <= 0:
            continue
        color = palette.series(slot)
        fig.add_trace(go.Bar(
            x=split["week"], y=split[column], name=label,
            marker={"color": color, "line": {"color": palette.surface, "width": 2}},
            hovertemplate=f"<b>{label}</b><br>week of %{{x|%d %b}}"
                          f"<br>load %{{y:.0f}}<extra></extra>",
        ))
    if not fig.data:
        return _empty(palette, "No training load recorded", height=320)
    fig.update_layout(barmode="stack", bargap=0.25)
    return style(fig, palette, title="Weekly load: strength versus running", height=320,
                 y_title="training load")


def readiness_chart(athlete_metrics: pd.DataFrame, palette: Palette) -> go.Figure:
    """Garmin's daily training-readiness score."""
    if athlete_metrics is None or athlete_metrics.empty \
            or "readiness_score" not in athlete_metrics \
            or not athlete_metrics["readiness_score"].notna().any():
        return _empty(palette, "No readiness data from Garmin", height=280)

    frame = athlete_metrics.dropna(subset=["readiness_score"]).sort_values("local_date")
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=50, fillcolor=palette.grid, opacity=0.5, line_width=0, layer="below",
                  annotation_text="below 50: no quality work",
                  annotation_position="bottom right",
                  annotation_font=dict(size=11, color=palette.ink_muted))
    fig.add_trace(go.Scatter(
        x=frame["local_date"], y=frame["readiness_score"], mode="lines+markers",
        name="Readiness", line=line_spec(palette.series(0)),
        marker={**marker_spec(palette.series(0), palette), "size": 6},
        customdata=frame[["readiness_level", "recovery_time_h"]].to_numpy(),
        hovertemplate=("%{x|%d %b %Y}<br>readiness %{y:.0f}/100"
                       "<br>%{customdata[0]} · %{customdata[1]:.0f} h recovery left"
                       "<extra></extra>"),
    ))
    fig.update_yaxes(range=[0, 100])
    return style(fig, palette, title="Training readiness (Garmin)", height=280,
                 show_legend=False)


def muscle_score_history_chart(history: pd.DataFrame, muscles: list[str],
                              labels: dict[str, str], palette: Palette) -> go.Figure:
    """Attention score per muscle over time — is the gap closing?"""
    if history is None or history.empty or not muscles:
        return _empty(palette, "Snapshots accumulate as you sync — check back in a week or two")

    fig = go.Figure()
    for slot, muscle in enumerate(muscles[:6]):
        subset = history[history["muscle"] == muscle].sort_values("taken_on")
        if subset.empty:
            continue
        color = palette.series(slot)
        name = labels.get(muscle, muscle)
        fig.add_trace(go.Scatter(
            x=subset["taken_on"], y=subset["score"], mode="lines+markers", name=name,
            line=line_spec(color), marker={**marker_spec(color, palette), "size": 7},
            hovertemplate=f"<b>{name}</b><br>%{{x|%d %b %Y}}"
                          f"<br>attention score %{{y:.0f}}<extra></extra>",
        ))
        last = subset.iloc[-1]
        fig.add_annotation(x=last["taken_on"], y=last["score"], text=f" {name}",
                           showarrow=False, xanchor="left", yanchor="middle",
                           font=dict(size=11, color=palette.ink_secondary))
    if not fig.data:
        return _empty(palette, "Not enough snapshots yet")
    fig.add_hline(y=60, line=dict(color=palette.axis, width=1, dash="dot"),
                  annotation_text="falling behind above 60", annotation_position="top left",
                  annotation_font=dict(size=11, color=palette.ink_muted))
    fig.update_xaxes(domain=[0, 0.8])
    return style(fig, palette, title="Muscle attention score over time", height=380,
                 y_title="attention score", show_legend=False)


def snapshot_metric_chart(history: pd.DataFrame, label: str, unit: str,
                          palette: Palette) -> go.Figure:
    """A single tracked metric across snapshots."""
    if history is None or history.empty or len(history) < 2:
        return _empty(palette, "Needs at least two snapshots", height=260)
    fig = go.Figure(go.Scatter(
        x=history["taken_on"], y=history["value"], mode="lines+markers",
        line=line_spec(palette.series(0)),
        marker={**marker_spec(palette.series(0), palette), "size": 7},
        hovertemplate=f"%{{x|%d %b %Y}}<br>%{{y:.2f}} {unit}<extra></extra>",
    ))
    return style(fig, palette, title=label, height=260, show_legend=False)


def niggle_timeline(niggles: pd.DataFrame, load_series: pd.DataFrame,
                    palette: Palette) -> go.Figure:
    """Training load with logged niggles marked, so onsets sit against the load."""
    if load_series is None or load_series.empty:
        return _empty(palette, "No load history yet", height=320)

    frame = load_series.dropna(subset=["acute_7d"]).tail(240)
    if frame.empty:
        return _empty(palette, "Needs at least a week of load history", height=320)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame["date"], y=frame["acute_7d"], mode="lines", name="7-day load",
        line=line_spec(palette.series(0)),
        hovertemplate="%{x|%d %b %Y}<br>7-day load %{y:.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=frame["date"], y=frame["chronic_weekly"], mode="lines", name="28-day weekly average",
        line=line_spec(palette.ink_muted, dash="dash"),
        hovertemplate="%{x|%d %b %Y}<br>28-day weekly %{y:.0f}<extra></extra>",
    ))

    if niggles is not None and not niggles.empty:
        window_start = frame["date"].min()
        marks = niggles[pd.to_datetime(niggles["noted_on"]) >= window_start]
        for entry in marks.itertuples():
            # Status colour + a written label: the marker never relies on colour.
            color = (palette.status["critical"] if (entry.severity or 0) >= 4
                     else palette.status["serious"] if (entry.severity or 0) == 3
                     else palette.status["warning"])
            fig.add_vline(x=entry.noted_on, line=dict(color=color, width=1, dash="dot"),
                          annotation_text=f"{entry.area} ({entry.severity}/5)",
                          annotation_position="top",
                          annotation_font=dict(size=10, color=palette.ink_secondary))
    fig.update_layout(hovermode="x unified")
    return style(fig, palette, title="Training load with logged niggles", height=340,
                 y_title="training load")
