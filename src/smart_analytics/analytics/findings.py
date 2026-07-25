"""The common currency between the analytics engines, the GUI and the AI layer.

Every engine emits :class:`Finding` objects. The GUI renders them, and the AI
layer receives them as grounded facts to reason about — so the coaching
narrative is explaining computed numbers rather than inventing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

Severity = Literal["good", "info", "watch", "act"]

SEVERITY_ORDER = {"act": 0, "watch": 1, "info": 2, "good": 3}

SEVERITY_LABEL = {
    "act": "Needs action",
    "watch": "Worth watching",
    "info": "Context",
    "good": "Going well",
}


@dataclass
class Finding:
    """One evidence-backed observation about training."""

    area: str  # strength | running | load | recovery
    title: str
    detail: str
    severity: Severity = "info"
    metric: str | None = None  # the headline number, pre-formatted
    recommendation: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    subject: str | None = None  # muscle id, exercise name, …

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sort_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.area, f.title))


def to_records(findings: list[Finding]) -> list[dict[str, Any]]:
    return [f.to_dict() for f in sort_findings(findings)]
