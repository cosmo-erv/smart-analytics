"""Runtime configuration, loaded from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env")


def _int_or_none(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _path(name: str, default: str) -> Path:
    raw = os.getenv(name, "").strip() or default
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass(frozen=True)
class Settings:
    garmin_email: str = ""
    garmin_password: str = ""
    token_store: Path = field(default_factory=lambda: REPO_ROOT / ".garmin_tokens")
    db_path: Path = field(default_factory=lambda: REPO_ROOT / "data" / "smart_analytics.db")
    anthropic_api_key: str = ""
    model: str = "claude-opus-5"
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1"
    # "auto" picks whichever provider has a key; set explicitly to force one.
    ai_provider: str = "auto"
    max_hr: int | None = None
    resting_hr: int | None = None
    weekly_sets_min: int = 10
    weekly_sets_max: int = 20

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            garmin_email=os.getenv("GARMIN_EMAIL", "").strip(),
            garmin_password=os.getenv("GARMIN_PASSWORD", "").strip(),
            token_store=_path("GARMIN_TOKEN_STORE", ".garmin_tokens"),
            db_path=_path("SMART_ANALYTICS_DB", "data/smart_analytics.db"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
            model=os.getenv("SMART_ANALYTICS_MODEL", "").strip() or "claude-opus-5",
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_model=os.getenv("OPENAI_MODEL", "").strip() or "gpt-4.1",
            ai_provider=os.getenv("AI_PROVIDER", "").strip().lower() or "auto",
            max_hr=_int_or_none("MAX_HR"),
            resting_hr=_int_or_none("RESTING_HR"),
            weekly_sets_min=_int_or_none("WEEKLY_SETS_MIN") or 10,
            weekly_sets_max=_int_or_none("WEEKLY_SETS_MAX") or 20,
        )

    @property
    def has_garmin_credentials(self) -> bool:
        return bool(self.garmin_email and self.garmin_password)

    @property
    def has_cached_tokens(self) -> bool:
        return self.token_store.exists() and any(self.token_store.iterdir())

    @property
    def has_anthropic_key(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def provider(self) -> str:
        """Which AI backend to use: ``anthropic``, ``openai`` or ``""`` for none.

        ``AI_PROVIDER`` forces a choice; otherwise whichever key is present wins,
        and if both are set Anthropic goes first purely as a stable tie-break.
        """
        if self.ai_provider == "anthropic":
            return "anthropic" if self.has_anthropic_key else ""
        if self.ai_provider == "openai":
            return "openai" if self.has_openai_key else ""
        if self.has_anthropic_key:
            return "anthropic"
        return "openai" if self.has_openai_key else ""

    @property
    def has_ai_key(self) -> bool:
        return bool(self.provider)

    @property
    def active_model(self) -> str:
        return self.openai_model if self.provider == "openai" else self.model


settings = Settings.load()
