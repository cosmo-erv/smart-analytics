"""Shared fixtures. The demo client is the fixture backbone: it exercises the same
normalisation and storage path as a real Garmin sync, so tests cover production code
rather than a parallel test-only shortcut."""

from __future__ import annotations

import pytest

from smart_analytics import db
from smart_analytics.analytics import build_report
from smart_analytics.garmin import SampleGarminClient, sync


@pytest.fixture(scope="session")
def synced_db(tmp_path_factory):
    """A populated database, synced once for the whole session (it isn't mutated)."""
    path = tmp_path_factory.mktemp("data") / "test.db"
    conn = db.connect(path)
    sync(conn, SampleGarminClient(days=220, seed=11), history_days=220,
         detail_batch=400, split_batch=300, wellness_days=40, physiology_days=10,
         throttle_s=0.0)
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def report(synced_db):
    return build_report(synced_db)


@pytest.fixture
def empty_db(tmp_path):
    conn = db.connect(tmp_path / "empty.db")
    yield conn
    conn.close()
