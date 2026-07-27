"""Claude-powered coaching layer over the computed analytics."""

from .insights import CoachReport, CoachUnavailable, ask, coach_report, fallback_report

__all__ = ["CoachReport", "CoachUnavailable", "ask", "coach_report", "fallback_report"]
