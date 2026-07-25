"""Incremental sync: Garmin Connect → local SQLite.

The sync is deliberately conservative about HTTP calls. Activity summaries come
in pages and stop as soon as we reach data we already have; per-set detail is
only fetched for strength workouts that don't have it yet, in a bounded batch,
so a first run on a long history can be spread over several syncs instead of
tripping Garmin's rate limiter.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from .. import db
from .client import (
    SET_BEARING_TYPES,
    SPLIT_BEARING_TYPES,
    GarminClient,
    normalise_activity,
    normalise_sets,
    normalise_splits,
)

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]

LAST_SYNC_KEY = "last_sync_at"
LAST_ACTIVITY_DATE_KEY = "last_activity_date"


@dataclass
class SyncReport:
    activities_seen: int = 0
    activities_written: int = 0
    strength_details: int = 0
    sets_written: int = 0
    wellness_days: int = 0
    splits_activities: int = 0
    splits_written: int = 0
    physiology_days: int = 0
    predictions_written: int = 0
    records_written: int = 0
    zones_written: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None

    @property
    def duration_s(self) -> float:
        end = self.finished_at or datetime.now()
        return (end - self.started_at).total_seconds()

    def summary(self) -> str:
        parts = [
            f"{self.activities_written} activities",
            f"{self.strength_details} strength workouts detailed",
            f"{self.sets_written} sets",
        ]
        if self.splits_written:
            parts.append(f"{self.splits_written} splits from {self.splits_activities} runs")
        if self.wellness_days:
            parts.append(f"{self.wellness_days} wellness days")
        if self.physiology_days:
            parts.append(f"{self.physiology_days} days of Garmin metrics")
        text = ", ".join(parts) + f" in {self.duration_s:.0f}s"
        if self.errors:
            text += f" ({len(self.errors)} warnings)"
        return text


def _noop(_message: str, _fraction: float) -> None:
    return None


def sync(
    conn: sqlite3.Connection,
    client: GarminClient,
    *,
    since: date | None = None,
    history_days: int = 365,
    fetch_details: bool = True,
    detail_batch: int = 150,
    wellness_days: int = 90,
    fetch_splits: bool = True,
    split_batch: int = 120,
    physiology_days: int = 21,
    max_activities: int | None = None,
    throttle_s: float = 0.25,
    progress: ProgressFn = _noop,
) -> SyncReport:
    """Pull activities, detail, wellness and Garmin's own physiology into ``conn``.

    Steps are ordered cheapest-first and each is separately bounded, so a sync
    interrupted by a rate limit still leaves the cache more complete than it was.
    """
    report = SyncReport()
    cutoff = since or (date.today() - timedelta(days=history_days))

    # --- 1. activity summaries ---------------------------------------------
    progress("Fetching activity summaries…", 0.05)
    batch: list[dict] = []
    try:
        for raw in client.iter_activities(since=cutoff, max_activities=max_activities):
            report.activities_seen += 1
            row = normalise_activity(raw)
            if row:
                batch.append(row)
            if len(batch) >= 100:
                report.activities_written += db.upsert_activities(conn, batch)
                batch.clear()
                progress(f"Fetched {report.activities_seen} activities…", 0.2)
    except Exception as exc:
        report.errors.append(f"Activity list stopped early: {exc}")
        log.exception("activity paging failed")
    if batch:
        report.activities_written += db.upsert_activities(conn, batch)

    # --- 2. strength set detail --------------------------------------------
    if fetch_details:
        pending: list[str] = []
        for activity_type in SET_BEARING_TYPES:
            pending += db.activities_missing_details(conn, activity_type, limit=detail_batch)

        for position, activity_id in enumerate(pending, start=1):
            fraction = 0.2 + 0.25 * (position / max(len(pending), 1))
            progress(f"Reading sets {position}/{len(pending)}…", fraction)
            try:
                raw_sets = client.exercise_sets(activity_id)
            except Exception as exc:
                report.errors.append(f"Sets for {activity_id}: {exc}")
                continue

            activity_date = _activity_date(conn, activity_id)
            rows = normalise_sets(activity_id, activity_date, raw_sets)
            if rows:
                report.sets_written += db.replace_strength_sets(conn, activity_id, rows)
                _store_volume(conn, activity_id, rows)
            else:
                # No per-set data (e.g. logged without a watch) — mark it so we
                # don't re-request the same empty payload on every sync.
                db.mark_details_fetched(conn, activity_id)
            report.strength_details += 1
            if throttle_s:
                time.sleep(throttle_s)

    # --- 3. per-lap splits --------------------------------------------------
    # Splits are what make aerobic decoupling and interval quality measurable;
    # averages alone can't show either.
    if fetch_splits:
        pending_splits = db.activities_missing_splits(
            conn, SPLIT_BEARING_TYPES, limit=split_batch)
        for position, activity_id in enumerate(pending_splits, start=1):
            fraction = 0.45 + 0.2 * (position / max(len(pending_splits), 1))
            progress(f"Reading splits {position}/{len(pending_splits)}…", fraction)
            try:
                payload = client.splits(activity_id)
            except Exception as exc:
                report.errors.append(f"Splits for {activity_id}: {exc}")
                continue
            rows = normalise_splits(activity_id, _activity_date(conn, activity_id), payload)
            report.splits_written += db.replace_splits(conn, activity_id, rows)
            report.splits_activities += 1
            if throttle_s:
                time.sleep(throttle_s)

    # --- 4. Garmin's own physiological metrics ------------------------------
    if physiology_days > 0:
        progress("Fetching Garmin's training metrics…", 0.68)
        metric_rows: list[dict] = []
        try:
            metric_rows.append(client.latest_thresholds())
        except Exception as exc:
            report.errors.append(f"Threshold metrics: {exc}")

        end = date.today()
        for offset in range(physiology_days):
            day = end - timedelta(days=offset)
            try:
                metric_rows.append(client.physiology_snapshot(day))
            except Exception as exc:
                report.errors.append(f"Garmin metrics for {day}: {exc}")
                break
            if throttle_s:
                time.sleep(throttle_s)
        if metric_rows:
            report.physiology_days = db.upsert_athlete_metrics(conn, metric_rows)

        progress("Fetching zones, predictions and records…", 0.8)
        try:
            reference = _recent_run_id(conn)
            zones = client.hr_zones(reference)
            if zones:
                report.zones_written = db.replace_hr_zones(conn, "running", zones)
        except Exception as exc:
            report.errors.append(f"Heart-rate zones: {exc}")

        try:
            predictions = client.race_predictions()
            if predictions:
                report.predictions_written = db.upsert_race_predictions(conn, predictions)
        except Exception as exc:
            report.errors.append(f"Race predictions: {exc}")

        try:
            records = client.personal_records()
            if records:
                report.records_written = db.upsert_personal_records(conn, records)
        except Exception as exc:
            report.errors.append(f"Personal records: {exc}")

    # --- 5. wellness context -----------------------------------------------
    if wellness_days > 0:
        progress("Fetching wellness metrics…", 0.9)
        end = date.today()
        start = end - timedelta(days=wellness_days - 1)
        existing = {
            row["local_date"]
            for row in conn.execute(
                "SELECT local_date FROM daily_metrics WHERE local_date >= ?", (start.isoformat(),)
            ).fetchall()
        }
        # Always refresh the most recent few days; older gaps are backfilled once.
        refresh_from = (end - timedelta(days=2)).isoformat()
        days = [
            day for day in _date_range(start, end)
            if day.isoformat() not in existing or day.isoformat() >= refresh_from
        ]
        wellness_rows = []
        for day in days:
            try:
                wellness_rows.append(client.daily_metrics(day))
            except Exception as exc:
                report.errors.append(f"Wellness for {day}: {exc}")
            if throttle_s:
                time.sleep(throttle_s)
        if wellness_rows:
            report.wellness_days = db.upsert_daily_metrics(conn, wellness_rows)

    # --- 6. bookkeeping ----------------------------------------------------
    report.finished_at = datetime.now()
    db.set_state(conn, LAST_SYNC_KEY, report.finished_at.isoformat(sep=" "))
    last_date = conn.execute("SELECT MAX(local_date) FROM activities").fetchone()[0]
    if last_date:
        db.set_state(conn, LAST_ACTIVITY_DATE_KEY, last_date)
    progress(f"Done — {report.summary()}", 1.0)
    return report


def _date_range(start: date, end: date) -> list[date]:
    span = (end - start).days
    return [start + timedelta(days=offset) for offset in range(span + 1)]


def _recent_run_id(conn: sqlite3.Connection) -> str | None:
    """A recent run, used as the source of the device's configured HR zones."""
    row = conn.execute(
        "SELECT activity_id FROM activities WHERE activity_type = 'running' "
        "ORDER BY start_time DESC LIMIT 1"
    ).fetchone()
    return row["activity_id"] if row else None


def _activity_date(conn: sqlite3.Connection, activity_id: str) -> str:
    row = conn.execute(
        "SELECT local_date FROM activities WHERE activity_id = ?", (activity_id,)
    ).fetchone()
    return row["local_date"] if row else date.today().isoformat()


def _store_volume(conn: sqlite3.Connection, activity_id: str, rows: list[dict]) -> None:
    volume = sum(
        (row.get("weight_kg") or 0) * (row.get("reps") or 0)
        for row in rows
        if row.get("set_type") == "ACTIVE"
    )
    working = [r for r in rows if r.get("set_type") == "ACTIVE"]
    conn.execute(
        "UPDATE activities SET total_volume_kg = ?, total_sets = COALESCE(?, total_sets), "
        "total_reps = COALESCE(?, total_reps) WHERE activity_id = ?",
        (round(volume, 1) or None, len(working) or None,
         sum(r.get("reps") or 0 for r in working) or None, activity_id),
    )
    conn.commit()


def last_sync_at(conn: sqlite3.Connection) -> str | None:
    return db.get_state(conn, LAST_SYNC_KEY)


def incremental_since(conn: sqlite3.Connection, overlap_days: int = 3) -> date | None:
    """Resume point for a follow-up sync, with overlap for edited activities."""
    last = db.get_state(conn, LAST_ACTIVITY_DATE_KEY)
    if not last:
        return None
    try:
        return date.fromisoformat(last) - timedelta(days=overlap_days)
    except ValueError:
        return None
