"""
app/jobs/deep_analysis.py
-------------------------
Weekly deep analysis orchestrator.
Finds all weekly dates at 7-day intervals starting from 2026-05-18 up to the latest date.
Resolves the archive for each date (copying from user downloads or polling Telegram),
runs the full microstructure research and validation pipeline, and sends the reports.
"""

from __future__ import annotations

import asyncio
import datetime
import os
import shutil
import sys
from pathlib import Path
from telegram import Bot

from app.config import load_settings, Settings
from app.jobs.daily_job import run_daily_job
from app.utils.log_setup import configure_logging, get_logger

logger = get_logger(__name__)


async def get_archive_for_date(settings: Settings, target_date: datetime.date) -> Path | None:
    """
    Resolve the archive file path for a specific date.
    Checks:
      1. Destination folder (data/data/archives)
      2. User Downloads / Market tf
      3. User Downloads
      4. Telegram channel (allowed updates buffer)
    """
    filename = f"{target_date.isoformat()}.tar.gz"
    dest_path = settings.archives_path / filename

    # 1. Check if already exists in destination archives dir
    if dest_path.exists():
        logger.info("archive_already_exists_locally", date=str(target_date), path=str(dest_path))
        return dest_path

    # Ensure destination directory exists
    settings.archives_path.mkdir(parents=True, exist_ok=True)

    # 2. Check local source directories (Downloads / Market tf and Downloads)
    sources = [
        Path("C:/Users/Vinay/Downloads/Market tf") / filename,
        Path("C:/Users/Vinay/Downloads") / filename,
    ]
    for src in sources:
        if src.exists():
            logger.info("archive_found_in_local_source", src=str(src), date=str(target_date))
            try:
                shutil.copy2(src, dest_path)
                logger.info("archive_copied_successfully", src=src, dest=dest_path)
                return dest_path
            except Exception as e:
                logger.error("failed_to_copy_archive", src=str(src), error=str(e))

    # 3. Poll Telegram channel
    if settings.telegram_bot_token and settings.telegram_channel_id:
        logger.info("polling_telegram_for_archive", date=str(target_date))
        try:
            bot = Bot(token=settings.telegram_bot_token)
            # Fetch recent channel posts
            updates = await bot.get_updates(offset=0, timeout=10, allowed_updates=["channel_post"])
            for update in updates:
                msg = update.channel_post
                if msg and msg.chat.id == settings.telegram_channel_id and msg.document:
                    fname = msg.document.file_name or ""
                    if fname == filename:
                        logger.info("archive_found_in_telegram", filename=fname, file_id=msg.document.file_id)
                        try:
                            tg_file = await bot.get_file(msg.document.file_id)
                            await tg_file.download_to_drive(
                                custom_path=str(dest_path),
                                read_timeout=settings.telegram.download_timeout_seconds,
                                write_timeout=settings.telegram.download_timeout_seconds,
                            )
                            logger.info("archive_downloaded_from_telegram", path=str(dest_path))
                            return dest_path
                        except Exception as e:
                            logger.error("telegram_download_failed", error=str(e))
        except Exception as e:
            logger.error("telegram_polling_failed", error=str(e))

    return None


def run_deep_analysis() -> bool:
    """
    Run sequential deep microstructure backtests at 7-day intervals
    starting from 2026-05-18 to the latest available data.
    """
    settings = load_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    settings.ensure_directories()

    start_date = datetime.date(2026, 5, 18)
    # Today is May 29, 2026
    today = datetime.date(2026, 5, 29)

    target_dates: list[datetime.date] = []
    curr = start_date
    while curr <= today:
        target_dates.append(curr)
        curr += datetime.timedelta(days=7)

    logger.info("deep_analysis_weekly_schedule", target_dates=[d.isoformat() for d in target_dates])

    success_dates = []
    failed_dates = []
    skipped_dates = []

    for target_date in target_dates:
        logger.info("processing_weekly_slice_start", date=target_date.isoformat())
        
        # Resolve the archive file (copying from download folders or fetching from Telegram)
        archive_path = asyncio.run(get_archive_for_date(settings, target_date))
        
        if not archive_path:
            logger.warning("archive_not_found_skipping_weekly_slice", date=target_date.isoformat())
            skipped_dates.append(target_date)
            continue

        try:
            logger.info("running_microstructure_research_pipeline", date=target_date.isoformat(), archive=str(archive_path))
            success = run_daily_job(
                settings=settings,
                archive_path=archive_path,
                session_date=target_date,
            )
            if success:
                logger.info("deep_analysis_weekly_slice_succeeded", date=target_date.isoformat())
                success_dates.append(target_date)
            else:
                logger.error("deep_analysis_weekly_slice_failed", date=target_date.isoformat())
                failed_dates.append(target_date)
        except Exception as e:
            logger.exception("deep_analysis_weekly_slice_exception", date=target_date.isoformat(), error=str(e))
            failed_dates.append(target_date)

    logger.info(
        "deep_analysis_completed_summary",
        success=[d.isoformat() for d in success_dates],
        failed=[d.isoformat() for d in failed_dates],
        skipped=[d.isoformat() for d in skipped_dates],
    )
    
    # Return True if we successfully ran at least one slice and had no failures
    return len(success_dates) > 0 and len(failed_dates) == 0


if __name__ == "__main__":
    success = run_deep_analysis()
    sys.exit(0 if success else 1)
