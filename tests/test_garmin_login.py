"""The interactive Garmin Connect login used by the GUI.

A web UI can't block on ``input()`` for a multi-factor code, so the login is a
two-step state machine: credentials in, then the code. These tests drive that
machine against a stand-in for ``garminconnect.Garmin`` — the real one would need
a live account — and check the parts that matter: that tokens (and only tokens)
are written, that a bad code is recoverable, and that signing out forgets them.
"""

from __future__ import annotations

from typing import Any

import garminconnect
import pytest

from smart_analytics.config import Settings
from smart_analytics.garmin.client import GarminAuthError, GarminClient

MFA_PASSWORD = "needs-mfa"
GOOD_CODE = "123456"


class FakeTokenStore:
    """Stands in for the garth session that persists OAuth tokens."""

    def __init__(self) -> None:
        self.dumped_to: str | None = None

    def dump(self, path: str) -> None:
        self.dumped_to = path
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "oauth2_token.json").write_text('{"access_token": "fake"}')


class FakeGarmin:
    """Mimics garminconnect's early-return MFA contract."""

    instances: list["FakeGarmin"] = []

    def __init__(self, email: str | None = None, password: str | None = None,
                 return_on_mfa: bool = False, **_: Any) -> None:
        self.username = email
        self.password = password
        self.return_on_mfa = return_on_mfa
        self.client = FakeTokenStore()
        self.resumed_with: tuple[Any, str] | None = None
        FakeGarmin.instances.append(self)

    def login(self, tokenstore: str | None = None) -> tuple[str | None, Any]:
        if self.password == "wrong":
            raise garminconnect.GarminConnectAuthenticationError("bad credentials")
        if self.password == MFA_PASSWORD:
            return "needs_mfa", {"opaque": "client-state"}
        return None, None

    def resume_login(self, client_state: Any, mfa_code: str) -> tuple[Any, Any]:
        if mfa_code != GOOD_CODE:
            raise RuntimeError("invalid verification code")
        self.resumed_with = (client_state, mfa_code)
        return None, None

    def get_full_name(self) -> str:
        return "Test Athlete"


@pytest.fixture
def client(tmp_path, monkeypatch):
    FakeGarmin.instances.clear()
    monkeypatch.setattr(garminconnect, "Garmin", FakeGarmin)
    config = Settings(token_store=tmp_path / "tokens", db_path=tmp_path / "db.sqlite")
    return GarminClient(config)


def _cached_files(client: GarminClient) -> list[str]:
    store = client.settings.token_store
    return sorted(p.name for p in store.iterdir()) if store.exists() else []


def test_login_without_mfa_caches_tokens_immediately(client):
    result = client.begin_login("athlete@example.com", "correct-horse")

    assert result == {"status": "ok", "display_name": "Test Athlete"}
    assert client.settings.has_cached_tokens
    assert _cached_files(client) == ["oauth2_token.json"]


def test_mfa_login_needs_a_code_before_anything_is_written(client):
    result = client.begin_login("athlete@example.com", MFA_PASSWORD)

    assert result == {"status": "mfa_required"}
    # Nothing is cached until the second factor is satisfied.
    assert not client.settings.has_cached_tokens

    finished = client.complete_login(GOOD_CODE)
    assert finished["status"] == "ok"
    assert client.settings.has_cached_tokens
    assert FakeGarmin.instances[-1].resumed_with == ({"opaque": "client-state"}, GOOD_CODE)


def test_a_rejected_code_is_recoverable_rather_than_fatal(client):
    client.begin_login("athlete@example.com", MFA_PASSWORD)

    with pytest.raises(GarminAuthError, match="wasn't accepted"):
        client.complete_login("000000")
    assert not client.settings.has_cached_tokens

    # The pending login is still usable, so the user can just retype the code.
    assert client.complete_login(GOOD_CODE)["status"] == "ok"


def test_completing_a_login_that_was_never_started_is_an_error(client):
    with pytest.raises(GarminAuthError, match="No login is waiting"):
        client.complete_login(GOOD_CODE)


def test_an_empty_code_is_caught_before_calling_garmin(client):
    client.begin_login("athlete@example.com", MFA_PASSWORD)
    with pytest.raises(GarminAuthError, match="Enter the code"):
        client.complete_login("   ")


def test_missing_credentials_are_rejected_locally(client):
    with pytest.raises(GarminAuthError, match="both required"):
        client.begin_login("athlete@example.com", "")
    assert FakeGarmin.instances == []


def test_bad_credentials_surface_as_an_auth_error(client):
    with pytest.raises(GarminAuthError, match="rejected the login"):
        client.begin_login("athlete@example.com", "wrong")


def test_the_account_name_falls_back_to_the_address_used_to_sign_in(client, monkeypatch):
    monkeypatch.setattr(FakeGarmin, "get_full_name",
                        lambda self: (_ for _ in ()).throw(RuntimeError("profile down")))
    client.begin_login("athlete@example.com", "correct-horse")
    assert client.display_name() == "athlete@example.com"


def test_signing_out_deletes_the_cached_tokens(client):
    client.begin_login("athlete@example.com", "correct-horse")
    assert client.settings.has_cached_tokens

    client.sign_out()

    assert not client.settings.has_cached_tokens
    assert _cached_files(client) == []
    with pytest.raises(GarminAuthError, match="Not connected"):
        _ = client.api


def test_tokens_go_to_the_configured_store_and_nowhere_else(client):
    """The password is used once; only OAuth tokens are ever persisted."""
    client.begin_login("athlete@example.com", "correct-horse")

    written = FakeGarmin.instances[-1].client.dumped_to
    assert written == str(client.settings.token_store)
    contents = (client.settings.token_store / "oauth2_token.json").read_text()
    assert "correct-horse" not in contents
