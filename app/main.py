"""
app/main.py
-----------
Application entry point.

Supports two modes:
1. `run-once`     — Process one archive now (either from Telegram or --archive).
2. `watch`        — Run as a daemon, polling Telegram at a configurable interval.

Usage:
    python -m app.main run-once --archive /path/to/2024-06-10.tar.gz
    python -m app.main watch
    python -m app.main run-once                   # polls Telegram first
"""

from __future__ import annotations

import argparse
import datetime
import signal
import time
from pathlib import Path

from app.config import load_settings
from app.jobs.daily_job import run_daily_job
from app.utils.log_setup import configure_logging, get_logger

logger = get_logger(__name__)

_SHUTDOWN_REQUESTED = False


def _signal_handler(signum: int, frame: object) -> None:
    global _SHUTDOWN_REQUESTED
    logger.info("shutdown_signal_received", signum=signum)
    _SHUTDOWN_REQUESTED = True


def _run_watch(settings) -> None:
    """
    Daemon mode: poll Telegram every N seconds and run the job when
    a new archive is found.
    """
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    poll_interval = settings.telegram.poll_interval_seconds
    logger.info("watch_mode_started", poll_interval_s=poll_interval)

    while not _SHUTDOWN_REQUESTED:
        logger.info("polling_for_archive")
        success = run_daily_job(settings=settings)
        if success:
            logger.info("daily_job_succeeded")

        for _ in range(poll_interval):
            if _SHUTDOWN_REQUESTED:
                break
            time.sleep(1)

    logger.info("watch_mode_stopped")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Market Microstructure Research Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.main run-once --archive data/archives/2024-06-10.tar.gz
  python -m app.main run-once --archive data/archives/2024-06-10.tar.gz --date 2024-06-10
  python -m app.main watch
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # run-once subcommand
    once_parser = subparsers.add_parser("run-once", help="Process one archive")
    once_parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Path to .tar.gz archive (if omitted, polls Telegram)",
    )
    once_parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Session date override YYYY-MM-DD",
    )

    # watch subcommand
    subparsers.add_parser("watch", help="Run as daemon, poll Telegram continuously")

    args = parser.parse_args()

    settings = load_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    settings.ensure_directories()

    logger.info(
        "engine_starting",
        mode=args.command,
        log_level=settings.log_level,
    )

    if args.command == "run-once":
        session_date = None
        if args.date:
            session_date = datetime.date.fromisoformat(args.date)

        success = run_daily_job(
            settings=settings,
            archive_path=args.archive,
            session_date=session_date,
        )
        raise SystemExit(0 if success else 1)

    elif args.command == "watch":
        _run_watch(settings)
        raise SystemExit(0)


if __name__ == "__main__":
    main()
