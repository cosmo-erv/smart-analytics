"""Garmin Connect ingest: client, normalisation, sync and demo data."""

from .client import (
    SPLIT_BEARING_TYPES,
    GarminAuthError,
    GarminClient,
    family_for,
    normalise_activity,
    normalise_sets,
    normalise_splits,
)
from .sample import SampleGarminClient
from .sync import SyncReport, incremental_since, last_sync_at, sync

__all__ = [
    "GarminAuthError",
    "GarminClient",
    "SPLIT_BEARING_TYPES",
    "SampleGarminClient",
    "SyncReport",
    "family_for",
    "incremental_since",
    "last_sync_at",
    "normalise_activity",
    "normalise_sets",
    "normalise_splits",
    "sync",
]
