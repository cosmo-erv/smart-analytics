"""Local SQLite cache for Garmin data.

Everything is stored on disk so the analytics and the GUI never depend on
Garmin being reachable, and so syncs stay incremental (Garmin's endpoints are
rate-limited — re-pulling years of history on every page load is not an
option). The database file lives under ``data/`` and is gitignored.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS activities (
    activity_id      TEXT PRIMARY KEY,
    start_time       TEXT NOT NULL,          -- ISO 8601, local to the device
    local_date       TEXT NOT NULL,          -- YYYY-MM-DD
    activity_type    TEXT,                   -- normalised: running, strength_training, ...
    activity_subtype TEXT,
    name             TEXT,
    duration_s       REAL,
    moving_s         REAL,
    distance_m       REAL,
    calories         REAL,
    avg_hr           REAL,
    max_hr           REAL,
    avg_speed_mps    REAL,
    max_speed_mps    REAL,
    elevation_gain_m REAL,
    elevation_loss_m REAL,
    avg_cadence      REAL,
    max_cadence      REAL,
    avg_power        REAL,
    norm_power       REAL,
    training_load    REAL,
    aerobic_te       REAL,
    anaerobic_te     REAL,
    vo2max           REAL,
    avg_stride_m     REAL,
    avg_gct_ms       REAL,
    avg_vert_osc_cm  REAL,
    total_sets       INTEGER,
    total_reps       INTEGER,
    total_volume_kg  REAL,
    hr_zone_json     TEXT,
    details_fetched  INTEGER DEFAULT 0,
    raw_json         TEXT
);

CREATE INDEX IF NOT EXISTS idx_activities_date ON activities(local_date);
CREATE INDEX IF NOT EXISTS idx_activities_type ON activities(activity_type);

CREATE TABLE IF NOT EXISTS strength_sets (
    activity_id   TEXT NOT NULL,
    set_index     INTEGER NOT NULL,
    start_time    TEXT,
    local_date    TEXT,
    set_type      TEXT,                      -- ACTIVE / REST
    category      TEXT,
    exercise_name TEXT,
    reps          INTEGER,
    weight_kg     REAL,
    duration_s    REAL,
    PRIMARY KEY (activity_id, set_index),
    FOREIGN KEY (activity_id) REFERENCES activities(activity_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sets_date ON strength_sets(local_date);

CREATE TABLE IF NOT EXISTS daily_metrics (
    local_date        TEXT PRIMARY KEY,
    resting_hr        REAL,
    hrv_ms            REAL,
    sleep_hours       REAL,
    sleep_score       REAL,
    body_battery_high REAL,
    body_battery_low  REAL,
    steps             REAL,
    weight_kg         REAL,
    stress_avg        REAL
);

CREATE TABLE IF NOT EXISTS sync_state (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Garmin's own physiological estimates. Preferred over anything we could infer:
-- these come off the device's own longitudinal model.
CREATE TABLE IF NOT EXISTS athlete_metrics (
    local_date              TEXT PRIMARY KEY,
    vo2max_running          REAL,
    vo2max_cycling          REAL,
    lt_hr                   REAL,   -- lactate threshold heart rate (bpm)
    lt_speed_mps            REAL,   -- lactate threshold running speed
    ftp_watts               REAL,
    endurance_score         REAL,
    hill_score              REAL,
    fitness_age             REAL,
    training_status         TEXT,   -- productive / maintaining / overreaching / …
    training_status_note    TEXT,
    acute_load              REAL,   -- Garmin's own 7-day load
    chronic_load            REAL,
    load_ratio              REAL,
    load_target_low         REAL,
    load_target_high        REAL,
    readiness_score         REAL,
    readiness_level         TEXT,
    recovery_time_h         REAL,
    running_tolerance_km    REAL,
    raw_json                TEXT
);

-- Heart-rate zones as configured on the device, per sport.
CREATE TABLE IF NOT EXISTS hr_zones (
    sport       TEXT NOT NULL,
    zone        INTEGER NOT NULL,
    floor_bpm   REAL,
    ceiling_bpm REAL,
    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (sport, zone)
);

-- Garmin's own race predictions, kept as a time series so they can be trended.
CREATE TABLE IF NOT EXISTS race_predictions (
    local_date       TEXT NOT NULL,
    distance_m       REAL NOT NULL,
    predicted_time_s REAL,
    PRIMARY KEY (local_date, distance_m)
);

CREATE TABLE IF NOT EXISTS personal_records (
    record_id     TEXT PRIMARY KEY,
    label         TEXT,
    activity_type TEXT,
    value         REAL,
    unit          TEXT,
    achieved_on   TEXT,
    raw_json      TEXT
);

-- Per-lap detail, which is what makes decoupling and interval analysis possible.
CREATE TABLE IF NOT EXISTS splits (
    activity_id      TEXT NOT NULL,
    split_index      INTEGER NOT NULL,
    split_type       TEXT,
    local_date       TEXT,
    distance_m       REAL,
    duration_s       REAL,
    moving_s         REAL,
    avg_hr           REAL,
    max_hr           REAL,
    avg_speed_mps    REAL,
    avg_cadence      REAL,
    avg_power        REAL,
    elevation_gain_m REAL,
    elevation_loss_m REAL,
    PRIMARY KEY (activity_id, split_index),
    FOREIGN KEY (activity_id) REFERENCES activities(activity_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_splits_date ON splits(local_date);

-- The one thing Garmin cannot know: how the body actually feels.
CREATE TABLE IF NOT EXISTS niggles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    noted_on    TEXT NOT NULL,
    area        TEXT NOT NULL,
    severity    INTEGER,           -- 1 (aware of it) … 5 (stopping training)
    note        TEXT,
    resolved_on TEXT,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_niggles_date ON niggles(noted_on);

-- Snapshots of the computed report, so the diagnostics themselves can be trended:
-- "hamstrings went from 68 to 41 over six weeks" needs history of the score.
CREATE TABLE IF NOT EXISTS report_snapshots (
    taken_on     TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Structured workouts you followed. Garmin's workout definitions carry the
-- exercise list and, for strength, the muscle groups it assigns each exercise —
-- which is better than any mapping we could infer.
CREATE TABLE IF NOT EXISTS workouts (
    workout_id  TEXT PRIMARY KEY,
    name        TEXT,
    sport       TEXT,
    updated_at  TEXT,
    step_count  INTEGER,
    raw_json    TEXT
);

CREATE TABLE IF NOT EXISTS workout_steps (
    workout_id        TEXT NOT NULL,
    step_index        INTEGER NOT NULL,
    category          TEXT,
    exercise_name     TEXT,
    target_reps       INTEGER,
    target_weight_kg  REAL,
    primary_muscles   TEXT,   -- JSON list of Garmin muscle names
    secondary_muscles TEXT,
    PRIMARY KEY (workout_id, step_index),
    FOREIGN KEY (workout_id) REFERENCES workouts(workout_id) ON DELETE CASCADE
);

-- Garmin's own exercise → muscle mapping, cached per (category, name). This takes
-- precedence over the curated table in domain/exercises.py wherever it exists.
CREATE TABLE IF NOT EXISTS exercise_muscles (
    category          TEXT NOT NULL,
    exercise_name     TEXT NOT NULL DEFAULT '',
    primary_muscles   TEXT,   -- JSON list of Garmin muscle names
    secondary_muscles TEXT,
    source            TEXT,   -- workout | exercise_library
    updated_at        TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (category, exercise_name)
);
"""

# Columns added after the first release; applied to existing databases on connect.
MIGRATIONS: dict[str, dict[str, str]] = {
    "activities": {
        "splits_fetched": "INTEGER DEFAULT 0",
        # The structured workout this activity was run against, when there was one.
        "workout_id": "TEXT",
    },
}

ACTIVITY_COLUMNS = [
    "activity_id", "start_time", "local_date", "activity_type", "activity_subtype",
    "name", "duration_s", "moving_s", "distance_m", "calories", "avg_hr", "max_hr",
    "avg_speed_mps", "max_speed_mps", "elevation_gain_m", "elevation_loss_m",
    "avg_cadence", "max_cadence", "avg_power", "norm_power", "training_load",
    "aerobic_te", "anaerobic_te", "vo2max", "avg_stride_m", "avg_gct_ms",
    "avg_vert_osc_cm", "total_sets", "total_reps", "total_volume_kg",
    "hr_zone_json", "details_fetched", "workout_id", "raw_json",
]

# Written on insert, never clobbered by a later summary re-sync.
PRESERVED_ACTIVITY_COLUMNS = ["details_fetched", "splits_fetched", "total_volume_kg"]

SET_COLUMNS = [
    "activity_id", "set_index", "start_time", "local_date", "set_type",
    "category", "exercise_name", "reps", "weight_kg", "duration_s",
]

DAILY_COLUMNS = [
    "local_date", "resting_hr", "hrv_ms", "sleep_hours", "sleep_score",
    "body_battery_high", "body_battery_low", "steps", "weight_kg", "stress_avg",
]


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _apply_migrations(conn)
    conn.commit()


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created."""
    for table, columns in MIGRATIONS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column, definition in columns.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


@contextmanager
def session(db_path: Path | str | None = None):
    conn = connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def _rows_for(columns: Sequence[str], records: Iterable[dict[str, Any]]) -> list[tuple]:
    out = []
    for rec in records:
        out.append(tuple(rec.get(col) for col in columns))
    return out


def _upsert(conn: sqlite3.Connection, table: str, columns: Sequence[str],
            records: Iterable[dict[str, Any]], key_columns: Sequence[str],
            preserve_columns: Sequence[str] = ()) -> int:
    """Insert-or-update ``records``.

    ``preserve_columns`` are written on insert but never overwritten on conflict —
    for bookkeeping flags that the incoming payload knows nothing about.
    """
    rows = _rows_for(columns, records)
    if not rows:
        return 0
    placeholders = ",".join("?" * len(columns))
    updatable = [c for c in columns
                 if c not in key_columns and c not in preserve_columns]
    set_clause = ",".join(f"{c}=excluded.{c}" for c in updatable)
    sql = (
        f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT({','.join(key_columns)}) DO UPDATE SET {set_clause}"
    )
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


def upsert_activities(conn: sqlite3.Connection, records: Iterable[dict[str, Any]]) -> int:
    """Upsert activity summaries, keeping the detail-fetched flags intact.

    A re-sync re-reads summaries, and ``normalise_activity`` always reports
    ``details_fetched = 0`` because it can't know. Overwriting the stored flag
    would make every sync re-request per-set and per-lap detail it already has.
    """
    return _upsert(conn, "activities", ACTIVITY_COLUMNS, records, ["activity_id"],
                   preserve_columns=PRESERVED_ACTIVITY_COLUMNS)


def upsert_daily_metrics(conn: sqlite3.Connection, records: Iterable[dict[str, Any]]) -> int:
    return _upsert(conn, "daily_metrics", DAILY_COLUMNS, records, ["local_date"])


def replace_strength_sets(conn: sqlite3.Connection, activity_id: str,
                          records: Iterable[dict[str, Any]]) -> int:
    """Sets are rewritten wholesale — an edited workout changes set indices."""
    records = list(records)
    conn.execute("DELETE FROM strength_sets WHERE activity_id = ?", (activity_id,))
    n = _upsert(conn, "strength_sets", SET_COLUMNS, records,
                ["activity_id", "set_index"]) if records else 0
    conn.execute("UPDATE activities SET details_fetched = 1 WHERE activity_id = ?", (activity_id,))
    conn.commit()
    return n


def mark_details_fetched(conn: sqlite3.Connection, activity_id: str) -> None:
    conn.execute("UPDATE activities SET details_fetched = 1 WHERE activity_id = ?", (activity_id,))
    conn.commit()


# --- sync bookkeeping -------------------------------------------------------

def set_state(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT INTO sync_state(key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP",
        (key, json.dumps(value)),
    )
    conn.commit()


def get_state(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute("SELECT value FROM sync_state WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except (json.JSONDecodeError, TypeError):
        return default


def state_updated_at(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT updated_at FROM sync_state WHERE key = ?", (key,)).fetchone()
    return row["updated_at"] if row else None


# --- reads -----------------------------------------------------------------

def load_activities(conn: sqlite3.Connection, activity_type: str | None = None) -> pd.DataFrame:
    sql = (
        "SELECT activity_id, start_time, local_date, activity_type, activity_subtype, name, "
        "duration_s, moving_s, distance_m, calories, avg_hr, max_hr, avg_speed_mps, "
        "max_speed_mps, elevation_gain_m, elevation_loss_m, avg_cadence, max_cadence, "
        "avg_power, norm_power, training_load, aerobic_te, anaerobic_te, vo2max, "
        "avg_stride_m, avg_gct_ms, avg_vert_osc_cm, total_sets, total_reps, "
        "total_volume_kg, hr_zone_json, details_fetched FROM activities"
    )
    params: tuple = ()
    if activity_type:
        sql += " WHERE activity_type = ?"
        params = (activity_type,)
    sql += " ORDER BY start_time"
    df = pd.read_sql_query(sql, conn, params=params)
    if not df.empty:
        df["start_time"] = pd.to_datetime(df["start_time"], format="mixed", errors="coerce")
        df["local_date"] = pd.to_datetime(df["local_date"], errors="coerce")
    return df


def load_strength_sets(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT s.* FROM strength_sets s ORDER BY s.local_date, s.activity_id, s.set_index", conn
    )
    if not df.empty:
        df["local_date"] = pd.to_datetime(df["local_date"], errors="coerce")
        df["start_time"] = pd.to_datetime(df["start_time"], format="mixed", errors="coerce")
    return df


def load_daily_metrics(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM daily_metrics ORDER BY local_date", conn)
    if not df.empty:
        df["local_date"] = pd.to_datetime(df["local_date"], errors="coerce")
    return df


def activities_missing_details(conn: sqlite3.Connection, activity_type: str = "strength_training",
                               limit: int | None = None) -> list[str]:
    sql = ("SELECT activity_id FROM activities WHERE activity_type = ? "
           "AND COALESCE(details_fetched, 0) = 0 ORDER BY start_time DESC")
    params: list[Any] = [activity_type]
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return [r["activity_id"] for r in conn.execute(sql, params).fetchall()]


# --- Garmin's own metrics ---------------------------------------------------

ATHLETE_COLUMNS = [
    "local_date", "vo2max_running", "vo2max_cycling", "lt_hr", "lt_speed_mps", "ftp_watts",
    "endurance_score", "hill_score", "fitness_age", "training_status", "training_status_note",
    "acute_load", "chronic_load", "load_ratio", "load_target_low", "load_target_high",
    "readiness_score", "readiness_level", "recovery_time_h", "running_tolerance_km", "raw_json",
]

SPLIT_COLUMNS = [
    "activity_id", "split_index", "split_type", "local_date", "distance_m", "duration_s",
    "moving_s", "avg_hr", "max_hr", "avg_speed_mps", "avg_cadence", "avg_power",
    "elevation_gain_m", "elevation_loss_m",
]

PR_COLUMNS = ["record_id", "label", "activity_type", "value", "unit", "achieved_on", "raw_json"]


def upsert_athlete_metrics(conn: sqlite3.Connection,
                           records: Iterable[dict[str, Any]]) -> int:
    """Merge daily physiological snapshots, preserving fields a call didn't return.

    The metrics come from several endpoints with different availability, so a row
    is built up over multiple calls; a plain upsert would null out whatever the
    current call didn't include.
    """
    records = [r for r in records if r.get("local_date")]
    if not records:
        return 0
    written = 0
    for record in records:
        row = conn.execute(
            "SELECT * FROM athlete_metrics WHERE local_date = ?", (record["local_date"],)
        ).fetchone()
        merged = dict(row) if row else {}
        merged.update({k: v for k, v in record.items() if v is not None})
        merged["local_date"] = record["local_date"]
        written += _upsert(conn, "athlete_metrics", ATHLETE_COLUMNS, [merged], ["local_date"])
    return written


def replace_hr_zones(conn: sqlite3.Connection, sport: str,
                     zones: Iterable[dict[str, Any]]) -> int:
    zones = list(zones)
    conn.execute("DELETE FROM hr_zones WHERE sport = ?", (sport,))
    for zone in zones:
        conn.execute(
            "INSERT INTO hr_zones(sport, zone, floor_bpm, ceiling_bpm) VALUES (?, ?, ?, ?)",
            (sport, zone.get("zone"), zone.get("floor_bpm"), zone.get("ceiling_bpm")),
        )
    conn.commit()
    return len(zones)


def upsert_race_predictions(conn: sqlite3.Connection,
                            records: Iterable[dict[str, Any]]) -> int:
    return _upsert(conn, "race_predictions",
                   ["local_date", "distance_m", "predicted_time_s"], records,
                   ["local_date", "distance_m"])


def upsert_personal_records(conn: sqlite3.Connection,
                            records: Iterable[dict[str, Any]]) -> int:
    return _upsert(conn, "personal_records", PR_COLUMNS, records, ["record_id"])


def replace_splits(conn: sqlite3.Connection, activity_id: str,
                   records: Iterable[dict[str, Any]]) -> int:
    records = list(records)
    conn.execute("DELETE FROM splits WHERE activity_id = ?", (activity_id,))
    written = _upsert(conn, "splits", SPLIT_COLUMNS, records,
                      ["activity_id", "split_index"]) if records else 0
    conn.execute("UPDATE activities SET splits_fetched = 1 WHERE activity_id = ?", (activity_id,))
    conn.commit()
    return written


def activities_missing_splits(conn: sqlite3.Connection, activity_types: Sequence[str],
                              limit: int | None = None) -> list[str]:
    placeholders = ",".join("?" * len(activity_types))
    sql = (f"SELECT activity_id FROM activities WHERE activity_type IN ({placeholders}) "
           f"AND COALESCE(splits_fetched, 0) = 0 AND COALESCE(distance_m, 0) > 1000 "
           f"ORDER BY start_time DESC")
    params: list[Any] = list(activity_types)
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return [r["activity_id"] for r in conn.execute(sql, params).fetchall()]


# --- workouts and Garmin's own muscle mapping -------------------------------

WORKOUT_COLUMNS = ["workout_id", "name", "sport", "updated_at", "step_count", "raw_json"]

WORKOUT_STEP_COLUMNS = [
    "workout_id", "step_index", "category", "exercise_name", "target_reps",
    "target_weight_kg", "primary_muscles", "secondary_muscles",
]

EXERCISE_MUSCLE_COLUMNS = [
    "category", "exercise_name", "primary_muscles", "secondary_muscles", "source",
]


def upsert_workouts(conn: sqlite3.Connection, records: Iterable[dict[str, Any]]) -> int:
    return _upsert(conn, "workouts", WORKOUT_COLUMNS, records, ["workout_id"])


def replace_workout_steps(conn: sqlite3.Connection, workout_id: str,
                          records: Iterable[dict[str, Any]]) -> int:
    records = list(records)
    conn.execute("DELETE FROM workout_steps WHERE workout_id = ?", (workout_id,))
    written = _upsert(conn, "workout_steps", WORKOUT_STEP_COLUMNS, records,
                      ["workout_id", "step_index"]) if records else 0
    conn.commit()
    return written


def upsert_exercise_muscles(conn: sqlite3.Connection,
                            records: Iterable[dict[str, Any]]) -> int:
    """Store Garmin's muscle lists, keyed by (category, name).

    Entries with a specific exercise name are kept alongside the category-level
    entry, so a named variant can be more precise than its category.
    """
    rows = []
    for record in records:
        if not record.get("category") and not record.get("exercise_name"):
            continue
        rows.append({
            "category": (record.get("category") or "").upper(),
            "exercise_name": (record.get("exercise_name") or "").upper(),
            "primary_muscles": json.dumps(record.get("primary_muscles") or []),
            "secondary_muscles": json.dumps(record.get("secondary_muscles") or []),
            "source": record.get("source") or "workout",
        })
    return _upsert(conn, "exercise_muscles", EXERCISE_MUSCLE_COLUMNS, rows,
                   ["category", "exercise_name"])


def load_exercise_muscles(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Garmin's muscle mapping as plain records, ready for the resolver."""
    rows = conn.execute(
        "SELECT category, exercise_name, primary_muscles, secondary_muscles, source "
        "FROM exercise_muscles"
    ).fetchall()
    out = []
    for row in rows:
        try:
            primary = json.loads(row["primary_muscles"] or "[]")
            secondary = json.loads(row["secondary_muscles"] or "[]")
        except (json.JSONDecodeError, TypeError):
            continue
        if not primary and not secondary:
            continue
        out.append({
            "category": row["category"],
            "exercise_name": row["exercise_name"],
            "primary_muscles": primary,
            "secondary_muscles": secondary,
            "source": row["source"],
        })
    return out


def load_workouts(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT workout_id, name, sport, updated_at, step_count FROM workouts "
        "ORDER BY updated_at DESC", conn)


def load_workout_steps(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT s.*, w.name AS workout_name, w.sport FROM workout_steps s "
        "JOIN workouts w ON w.workout_id = s.workout_id "
        "ORDER BY s.workout_id, s.step_index", conn)


def workout_ids_with_steps(conn: sqlite3.Connection) -> set[str]:
    """Workouts whose step detail has already been fetched.

    A workout row on its own proves nothing — the list endpoint gives summaries
    with no steps — so the step table is what says a detail fetch happened.
    """
    return {row[0] for row in conn.execute(
        "SELECT DISTINCT workout_id FROM workout_steps").fetchall()}


def workout_ids_needing_fetch(conn: sqlite3.Connection,
                              limit: int | None = None) -> list[str]:
    """Workouts an activity was run against whose steps aren't stored yet.

    Newest first, so a bounded batch works through the most relevant definitions.
    """
    sql = ("SELECT DISTINCT a.workout_id FROM activities a "
           "LEFT JOIN workout_steps s ON s.workout_id = a.workout_id "
           "WHERE a.workout_id IS NOT NULL AND a.workout_id != '' "
           "AND s.workout_id IS NULL ORDER BY a.start_time DESC")
    params: list[Any] = []
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return [row["workout_id"] for row in conn.execute(sql, params).fetchall()]


def load_athlete_metrics(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        f"SELECT {','.join(c for c in ATHLETE_COLUMNS if c != 'raw_json')} "
        f"FROM athlete_metrics ORDER BY local_date", conn)
    if not df.empty:
        df["local_date"] = pd.to_datetime(df["local_date"], errors="coerce")
    return df


def load_hr_zones(conn: sqlite3.Connection, sport: str | None = None) -> pd.DataFrame:
    sql = "SELECT sport, zone, floor_bpm, ceiling_bpm FROM hr_zones"
    params: tuple = ()
    if sport:
        sql += " WHERE sport = ?"
        params = (sport,)
    sql += " ORDER BY sport, zone"
    return pd.read_sql_query(sql, conn, params=params)


def load_race_predictions(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT local_date, distance_m, predicted_time_s FROM race_predictions "
        "ORDER BY local_date, distance_m", conn)
    if not df.empty:
        df["local_date"] = pd.to_datetime(df["local_date"], errors="coerce")
    return df


def load_personal_records(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT record_id, label, activity_type, value, unit, achieved_on "
        "FROM personal_records ORDER BY achieved_on DESC", conn)
    if not df.empty:
        df["achieved_on"] = pd.to_datetime(df["achieved_on"], errors="coerce")
    return df


def load_splits(conn: sqlite3.Connection, activity_type: str | None = None) -> pd.DataFrame:
    sql = ("SELECT s.*, a.activity_type, a.name FROM splits s "
           "JOIN activities a ON a.activity_id = s.activity_id")
    params: tuple = ()
    if activity_type:
        sql += " WHERE a.activity_type = ?"
        params = (activity_type,)
    sql += " ORDER BY s.local_date, s.activity_id, s.split_index"
    df = pd.read_sql_query(sql, conn, params=params)
    if not df.empty:
        df["local_date"] = pd.to_datetime(df["local_date"], errors="coerce")
    return df


# --- niggles (manual) -------------------------------------------------------

def add_niggle(conn: sqlite3.Connection, noted_on: str, area: str, severity: int,
               note: str | None = None) -> int:
    cursor = conn.execute(
        "INSERT INTO niggles(noted_on, area, severity, note) VALUES (?, ?, ?, ?)",
        (noted_on, area, int(severity), note),
    )
    conn.commit()
    return int(cursor.lastrowid)


def resolve_niggle(conn: sqlite3.Connection, niggle_id: int, resolved_on: str) -> None:
    conn.execute("UPDATE niggles SET resolved_on = ? WHERE id = ?", (resolved_on, niggle_id))
    conn.commit()


def delete_niggle(conn: sqlite3.Connection, niggle_id: int) -> None:
    conn.execute("DELETE FROM niggles WHERE id = ?", (niggle_id,))
    conn.commit()


def load_niggles(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT id, noted_on, area, severity, note, resolved_on FROM niggles "
        "ORDER BY noted_on DESC", conn)
    if not df.empty:
        df["noted_on"] = pd.to_datetime(df["noted_on"], errors="coerce")
        df["resolved_on"] = pd.to_datetime(df["resolved_on"], errors="coerce")
    return df


# --- report snapshots -------------------------------------------------------

def save_snapshot(conn: sqlite3.Connection, taken_on: str, payload: dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO report_snapshots(taken_on, payload_json) VALUES (?, ?) "
        "ON CONFLICT(taken_on) DO UPDATE SET payload_json = excluded.payload_json, "
        "created_at = CURRENT_TIMESTAMP",
        (taken_on, json.dumps(payload, default=str)),
    )
    conn.commit()


def load_snapshots(conn: sqlite3.Connection, limit: int = 60) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT taken_on, payload_json FROM report_snapshots ORDER BY taken_on DESC LIMIT ?",
        (limit,),
    ).fetchall()
    out = []
    for row in rows:
        try:
            out.append({"taken_on": row["taken_on"], **json.loads(row["payload_json"])})
        except (json.JSONDecodeError, TypeError):
            continue
    return out


def counts(conn: sqlite3.Connection) -> dict[str, Any]:
    def scalar(sql: str) -> Any:
        return conn.execute(sql).fetchone()[0]

    return {
        "activities": scalar("SELECT COUNT(*) FROM activities"),
        "strength_activities":
            scalar("SELECT COUNT(*) FROM activities WHERE activity_type='strength_training'"),
        "sets": scalar("SELECT COUNT(*) FROM strength_sets"),
        "daily_metrics": scalar("SELECT COUNT(*) FROM daily_metrics"),
        "splits": scalar("SELECT COUNT(*) FROM splits"),
        "athlete_metrics": scalar("SELECT COUNT(*) FROM athlete_metrics"),
        "personal_records": scalar("SELECT COUNT(*) FROM personal_records"),
        "workouts": scalar("SELECT COUNT(*) FROM workouts"),
        "workout_steps": scalar("SELECT COUNT(*) FROM workout_steps"),
        "niggles": scalar("SELECT COUNT(*) FROM niggles WHERE resolved_on IS NULL"),
        "snapshots": scalar("SELECT COUNT(*) FROM report_snapshots"),
        "first_date": scalar("SELECT MIN(local_date) FROM activities"),
        "last_date": scalar("SELECT MAX(local_date) FROM activities"),
        "types": scalar("SELECT COUNT(DISTINCT activity_type) FROM activities"),
    }


def clear_all(conn: sqlite3.Connection) -> None:
    for table in ("strength_sets", "splits", "activities", "daily_metrics",
                  "athlete_metrics", "hr_zones", "race_predictions", "personal_records",
                  "workout_steps", "workouts", "exercise_muscles",
                  "report_snapshots", "sync_state"):
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
