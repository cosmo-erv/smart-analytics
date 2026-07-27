"""Command line interface — useful for scheduled syncs and quick checks.

    smart-analytics sync --demo          # load generated data
    smart-analytics sync                 # pull from Garmin Connect
    smart-analytics report               # print the findings to the terminal
    smart-analytics ui                   # launch the GUI
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from . import db
from .analytics import build_report
from .analytics.findings import SEVERITY_LABEL
from .config import REPO_ROOT, settings
from .garmin import GarminAuthError, GarminClient, SampleGarminClient, sync
from .garmin.sync import incremental_since


def _sync(args: argparse.Namespace) -> int:
    connection = db.connect()
    try:
        client = (SampleGarminClient(days=args.days) if args.demo
                  else GarminClient().connect())
    except GarminAuthError as exc:
        print(f"Garmin login failed: {exc}", file=sys.stderr)
        return 2

    def progress(message: str, fraction: float) -> None:
        print(f"[{fraction * 100:5.1f}%] {message}")

    report = sync(
        connection, client,
        since=incremental_since(connection) if args.incremental else None,
        history_days=args.days,
        fetch_details=not args.no_details,
        detail_batch=args.detail_batch,
        wellness_days=args.wellness_days,
        fetch_workouts=not args.no_workouts,
        throttle_s=0.0 if args.demo else 0.25,
        progress=progress,
    )
    print(f"\n{report.summary()}")
    for message in report.errors[:10]:
        print(f"  warning: {message}", file=sys.stderr)
    return 0


def _report(args: argparse.Namespace) -> int:
    connection = db.connect()
    report = build_report(connection, lookback_days=args.lookback)
    if not report.has_data:
        print("No data in the local cache. Run `smart-analytics sync --demo` first.")
        return 1

    counts = report.meta
    print(f"\nSmart Analytics — {counts['activity_count']} activities "
          f"({counts['first_date']} to {counts['last_date']})")
    print("=" * 78)

    for finding in report.findings:
        label = SEVERITY_LABEL.get(finding.severity, finding.severity)
        metric = f"  [{finding.metric}]" if finding.metric else ""
        print(f"\n{label.upper():14} {finding.title}{metric}")
        print(f"               {finding.detail}")
        if finding.recommendation:
            print(f"               → {finding.recommendation}")

    if args.json:
        import json
        Path(args.json).write_text(json.dumps(report.briefing(), indent=2, default=str))
        print(f"\nBriefing written to {args.json}")
    return 0


def _ui(args: argparse.Namespace) -> int:
    launcher = REPO_ROOT / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(launcher)]
    if args.port:
        command += ["--server.port", str(args.port)]
    return subprocess.call(command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smart-analytics", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="pull data into the local cache")
    sync_parser.add_argument("--demo", action="store_true",
                             help="use generated data instead of Garmin Connect")
    sync_parser.add_argument("--days", type=int, default=400, help="history window (default 400)")
    sync_parser.add_argument("--incremental", action="store_true",
                             help="resume from the last synced activity")
    sync_parser.add_argument("--no-details", action="store_true",
                             help="skip per-set strength detail")
    sync_parser.add_argument("--detail-batch", type=int, default=150,
                             help="strength workouts to detail this run (default 150)")
    sync_parser.add_argument("--wellness-days", type=int, default=90,
                             help="days of wellness data to pull (0 to skip)")
    sync_parser.add_argument("--no-workouts", action="store_true",
                             help="skip structured workouts and Garmin's muscle assignments")
    sync_parser.set_defaults(func=_sync)

    report_parser = subparsers.add_parser("report", help="print findings from the local cache")
    report_parser.add_argument("--lookback", type=int, default=84,
                               help="volume window in days (default 84)")
    report_parser.add_argument("--json", help="also write the AI briefing JSON to this path")
    report_parser.set_defaults(func=_report)

    ui_parser = subparsers.add_parser("ui", help="launch the Streamlit GUI")
    ui_parser.add_argument("--port", type=int, default=None)
    ui_parser.set_defaults(func=_ui)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    print(f"database: {settings.db_path}", file=sys.stderr)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
