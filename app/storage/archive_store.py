"""
app/storage/archive_store.py
------------------------------
Lightweight file-system store for raw archives and processing state.

Responsibilities:
- Track which archives have already been processed (prevents re-runs).
- Move archives from temp → permanent storage after successful processing.
- Provide a list of available archives for replay.
- Clean up old temporary files.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from app.utils.log_setup import get_logger

logger = get_logger(__name__)

_STATE_FILENAME = "processed.json"
_MAX_TEMP_AGE_HOURS = 24


class ArchiveStore:
    """
    Manages raw archive files on disk.

    Archives are stored in:
        archives_dir/
            YYYY-MM-DD.tar.gz       ← permanent, validated copies
            processed.json          ← set of already-processed session dates
    """

    def __init__(self, archives_dir: Path, temp_dir: Path) -> None:
        self._archives_dir = archives_dir
        self._temp_dir = temp_dir
        self._state_path = archives_dir / _STATE_FILENAME
        self._archives_dir.mkdir(parents=True, exist_ok=True)
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._processed: set[str] = self._load_processed()

    # ──────────────────────────────────────────────────────────────────────────
    # Processed state tracking
    # ──────────────────────────────────────────────────────────────────────────

    def _load_processed(self) -> set[str]:
        if self._state_path.exists():
            try:
                return set(json.loads(self._state_path.read_text()))
            except (json.JSONDecodeError, Exception):
                logger.warning("corrupt_processed_state", path=str(self._state_path))
        return set()

    def _save_processed(self) -> None:
        self._state_path.write_text(json.dumps(sorted(self._processed)))

    def is_processed(self, session_date: date) -> bool:
        return session_date.isoformat() in self._processed

    def mark_processed(self, session_date: date) -> None:
        self._processed.add(session_date.isoformat())
        self._save_processed()
        logger.info("archive_marked_processed", date=str(session_date))

    # ──────────────────────────────────────────────────────────────────────────
    # Archive file management
    # ──────────────────────────────────────────────────────────────────────────

    def permanent_path(self, session_date: date) -> Path:
        return self._archives_dir / f"{session_date.isoformat()}.tar.gz"

    def store_archive(self, source_path: Path, session_date: date) -> Path:
        """
        Move (or copy) an archive into permanent storage.
        Returns the permanent path.
        """
        dest = self.permanent_path(session_date)
        if dest.exists():
            logger.info("archive_already_stored", dest=str(dest))
            return dest

        if source_path.parent == self._archives_dir:
            # Already in the right place
            return source_path

        shutil.copy2(str(source_path), str(dest))
        logger.info("archive_stored", src=str(source_path), dest=str(dest))
        return dest

    def list_available(self) -> list[tuple[date, Path]]:
        """Return all (date, path) pairs for archives on disk, sorted by date."""
        results = []
        for p in sorted(self._archives_dir.glob("*.tar.gz")):
            try:
                d = date.fromisoformat(p.stem.replace(".tar", ""))
                results.append((d, p))
            except ValueError:
                pass
        return results

    def list_unprocessed(self) -> list[tuple[date, Path]]:
        """Return archives that have not yet been processed."""
        return [
            (d, p) for d, p in self.list_available()
            if not self.is_processed(d)
        ]

    def md5_checksum(self, path: Path) -> str:
        """Compute MD5 of a file for integrity verification."""
        h = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    # ──────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def cleanup_temp_files(self, max_age_hours: int = _MAX_TEMP_AGE_HOURS) -> int:
        """
        Remove files from temp_dir older than max_age_hours.
        Returns number of files deleted.
        """
        cutoff = datetime.utcnow().timestamp() - max_age_hours * 3600
        deleted = 0
        for p in self._temp_dir.iterdir():
            if p.is_file() and p.stat().st_mtime < cutoff:
                try:
                    p.unlink()
                    deleted += 1
                    logger.debug("temp_file_deleted", path=str(p))
                except OSError as exc:
                    logger.warning("temp_file_delete_failed", path=str(p), error=str(exc))
        if deleted:
            logger.info("temp_cleanup_complete", deleted=deleted)
        return deleted
