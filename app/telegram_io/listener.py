"""
app/telegram_io/listener.py
----------------------------
Listens to a Telegram channel for new .tar.gz archive messages.

Uses python-telegram-bot in polling mode (no webhook required).
The listener checks for new file messages on a schedule and
hands off downloaded archives to the ingestion pipeline.

Design:
- We do NOT use PTB's Application/Handler framework for this use case.
  Instead we use the Bot object directly with getUpdates polling.
  This avoids a persistent event loop conflict with the daily scheduler.
- Each call to poll() checks for unseen messages and returns file paths.
- State (last processed message ID) is persisted to disk.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from telegram import Bot, Document, Message, Update
from telegram.error import TelegramError

from app.config import Settings
from app.utils.log_setup import get_logger

logger = get_logger(__name__)

_STATE_FILE = "data/temp/telegram_state.json"
_ARCHIVE_EXTENSION = ".tar.gz"


class TelegramListener:
    """
    Polls a Telegram channel for new archive file messages.

    Parameters
    ----------
    settings: Application settings (token, channel ID, timeouts).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._state_path = Path(settings.data_root) / _STATE_FILE
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_update_id: int = self._load_state()

    # ──────────────────────────────────────────────────────────────────────────
    # State persistence (survives restarts)
    # ──────────────────────────────────────────────────────────────────────────

    def _load_state(self) -> int:
        if self._settings.mongodb_uri:
            try:
                import pymongo
                client = pymongo.MongoClient(self._settings.mongodb_uri)
                try:
                    db = client.get_default_database()
                    if db is None or db.name == "test":
                        db = client["market_research"]
                except Exception:
                    db = client["market_research"]
                doc = db["system_state"].find_one({"key": "telegram_last_update_id"})
                client.close()
                if doc:
                    logger.info("loaded_telegram_state_from_mongodb", last_update_id=doc.get("value"))
                    return int(doc.get("value", 0))
            except Exception as e:
                logger.error("failed_to_load_telegram_state_from_mongodb", error=str(e))

        if self._state_path.exists():
            try:
                data = json.loads(self._state_path.read_text())
                return int(data.get("last_update_id", 0))
            except (json.JSONDecodeError, ValueError):
                pass
        return 0

    def _save_state(self) -> None:
        if self._settings.mongodb_uri:
            try:
                import pymongo
                import datetime
                client = pymongo.MongoClient(self._settings.mongodb_uri)
                try:
                    db = client.get_default_database()
                    if db is None or db.name == "test":
                        db = client["market_research"]
                except Exception:
                    db = client["market_research"]
                db["system_state"].replace_one(
                    {"key": "telegram_last_update_id"},
                    {
                        "key": "telegram_last_update_id",
                        "value": self._last_update_id,
                        "updated_at": datetime.datetime.utcnow()
                    },
                    upsert=True
                )
                client.close()
                logger.info("saved_telegram_state_to_mongodb", last_update_id=self._last_update_id)
            except Exception as e:
                logger.error("failed_to_save_telegram_state_to_mongodb", error=str(e))

        self._state_path.write_text(
            json.dumps({"last_update_id": self._last_update_id})
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    async def poll_for_archive(self, download_dir: Path) -> Optional[Path]:
        """
        Poll Telegram for one new archive message. Download it if found.

        Returns the local path of the downloaded archive, or None.
        Marks the update as processed to avoid re-downloading.
        """
        bot = Bot(token=self._settings.telegram_bot_token)
        channel_id = self._settings.telegram_channel_id

        logger.info(
            "polling_telegram",
            channel_id=channel_id,
            last_update_id=self._last_update_id,
        )

        try:
            updates: list[Update] = await bot.get_updates(
                offset=self._last_update_id + 1,
                timeout=30,
                allowed_updates=["channel_post"],
            )
        except TelegramError as exc:
            logger.error("telegram_get_updates_failed", error=str(exc))
            return None

        if not updates:
            logger.debug("no_new_telegram_updates")
            return None

        archive_path: Optional[Path] = None

        for update in updates:
            self._last_update_id = update.update_id
            msg: Optional[Message] = update.channel_post

            if msg is None:
                continue

            # Confirm it's from the right channel
            if msg.chat.id != channel_id:
                logger.debug(
                    "update_from_different_chat",
                    chat_id=msg.chat.id,
                    expected=channel_id,
                )
                continue

            doc: Optional[Document] = msg.document
            if doc is None:
                logger.debug("channel_post_has_no_document", update_id=update.update_id)
                continue

            fname = doc.file_name or ""
            if not fname.endswith(_ARCHIVE_EXTENSION):
                logger.debug(
                    "document_not_archive",
                    filename=fname,
                    update_id=update.update_id,
                )
                continue

            # Found an archive — download it
            logger.info(
                "archive_found",
                filename=fname,
                file_id=doc.file_id,
                size_bytes=doc.file_size,
            )

            local_path = download_dir / fname
            if local_path.exists():
                logger.info(
                    "archive_already_downloaded",
                    path=str(local_path),
                )
                archive_path = local_path
                break

            try:
                tg_file = await bot.get_file(doc.file_id)
                await tg_file.download_to_drive(
                    custom_path=str(local_path),
                    read_timeout=self._settings.telegram.download_timeout_seconds,
                    write_timeout=self._settings.telegram.download_timeout_seconds,
                )
                logger.info("archive_downloaded", path=str(local_path))
                archive_path = local_path
                break  # Process one archive per poll cycle
            except TelegramError as exc:
                logger.error(
                    "archive_download_failed",
                    filename=fname,
                    error=str(exc),
                )
                continue

        self._save_state()
        return archive_path

    def poll_for_archive_sync(self, download_dir: Path) -> Optional[Path]:
        """Synchronous wrapper for poll_for_archive."""
        return asyncio.run(self.poll_for_archive(download_dir))
