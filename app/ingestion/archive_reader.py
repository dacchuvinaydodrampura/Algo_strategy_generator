"""
app/ingestion/archive_reader.py
--------------------------------
Memory-efficient streaming reader for .tar.gz archives.

Design principles:
- Never load the entire archive into RAM.
- Stream-extract using tarfile + gzip.
- Yield one line at a time per symbol file.
- Cleanly separate SYSTEM events from symbol ticks.
- Report file-level and line-level errors without crashing the pipeline.
"""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterator, Optional

from app.utils.log_setup import get_logger

logger = get_logger(__name__)

# File names that contain system/feed events rather than tick data
_SYSTEM_FILENAME = "SYSTEM.ndjson.gz"
_TICK_EXTENSION = ".ndjson.gz"


@dataclass
class RawLine:
    """
    A single unparsed line from an NDJSON file inside the archive.
    We keep it raw here; parsing and validation happen downstream.
    """

    symbol: str       # derived from filename (without extension)
    filename: str     # original filename inside archive
    line_no: int      # 1-based line number within the file
    raw: str          # the raw JSON string (stripped)
    is_system: bool   # True if from SYSTEM.ndjson.gz


class ArchiveReadError(Exception):
    """Raised when an archive cannot be opened or is structurally invalid."""


def _symbol_from_filename(name: str) -> str:
    """
    Extract the symbol name from a filename.
    'BANKNIFTY26JUNFUT.ndjson.gz' -> 'BANKNIFTY26JUNFUT'
    """
    # Strip path prefix if present
    basename = Path(name).name
    # Remove known extensions in order
    for suffix in (".ndjson.gz", ".ndjson"):
        if basename.endswith(suffix):
            return basename[: -len(suffix)].upper()
    return basename.upper()


def _is_tick_file(name: str) -> bool:
    """Return True if the file is a symbol tick file (not SYSTEM)."""
    basename = Path(name).name
    return basename.endswith(_TICK_EXTENSION) and basename != _SYSTEM_FILENAME


def _is_system_file(name: str) -> bool:
    basename = Path(name).name
    return basename == _SYSTEM_FILENAME


def _stream_gz_lines(fileobj: io.BufferedIOBase) -> Iterator[bytes]:
    """Decompress a gzip stream and yield raw lines."""
    with gzip.GzipFile(fileobj=fileobj, mode="rb") as gz:
        for raw_line in gz:
            yield raw_line


def iter_archive_lines(
    archive_path: Path,
    target_symbols: Optional[list[str]] = None,
) -> Generator[RawLine, None, None]:
    """
    Stream all lines from symbol and SYSTEM files inside a .tar.gz archive.

    Parameters
    ----------
    archive_path:
        Path to the .tar.gz file on disk.
    target_symbols:
        If provided, only yield lines from these symbols (uppercase).
        SYSTEM file is always included.

    Yields
    ------
    RawLine  for each valid, non-empty line encountered.

    Raises
    ------
    ArchiveReadError  if the archive cannot be opened.
    """
    if not archive_path.exists():
        raise ArchiveReadError(f"Archive not found: {archive_path}")

    filter_set = (
        {s.upper() for s in target_symbols} if target_symbols else None
    )

    logger.info("opening_archive", path=str(archive_path))

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            logger.info("archive_members", count=len(members))

            for member in members:
                if member.isdir():
                    continue

                name = member.name
                is_sys = _is_system_file(name)
                is_tick = _is_tick_file(name)

                if not (is_sys or is_tick):
                    logger.debug("skipping_non_tick_file", filename=name)
                    continue

                symbol = _symbol_from_filename(name)

                # Apply symbol filter (SYSTEM always passes)
                if filter_set and not is_sys and symbol not in filter_set:
                    logger.debug("symbol_filtered_out", symbol=symbol)
                    continue

                logger.info("reading_file", filename=name, symbol=symbol, is_system=is_sys)

                fileobj = tar.extractfile(member)
                if fileobj is None:
                    logger.warning("extractfile_returned_none", filename=name)
                    continue

                line_no = 0
                error_count = 0

                try:
                    for raw_bytes in _stream_gz_lines(fileobj):
                        line_no += 1
                        raw_str = raw_bytes.decode("utf-8", errors="replace").strip()

                        if not raw_str:
                            continue  # blank line – skip silently

                        # Quick JSON syntax pre-check to avoid expensive parsing errors
                        if not (raw_str.startswith("{") and raw_str.endswith("}")):
                            logger.warning(
                                "malformed_json_line",
                                filename=name,
                                line_no=line_no,
                                preview=raw_str[:80],
                            )
                            error_count += 1
                            continue

                        yield RawLine(
                            symbol=symbol,
                            filename=name,
                            line_no=line_no,
                            raw=raw_str,
                            is_system=is_sys,
                        )

                except (gzip.BadGzipFile, OSError) as exc:
                    logger.error(
                        "gz_read_error",
                        filename=name,
                        error=str(exc),
                        lines_read=line_no,
                    )
                    continue  # Do not crash the whole archive

                finally:
                    fileobj.close()

                logger.info(
                    "file_done",
                    filename=name,
                    lines_yielded=line_no,
                    errors=error_count,
                )

    except tarfile.TarError as exc:
        raise ArchiveReadError(f"Cannot open archive {archive_path}: {exc}") from exc


def list_archive_contents(archive_path: Path) -> list[dict[str, object]]:
    """
    Return a manifest of files inside the archive without extracting data.
    Useful for validation and UI display.
    """
    if not archive_path.exists():
        raise ArchiveReadError(f"Archive not found: {archive_path}")

    contents: list[dict[str, object]] = []
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                contents.append(
                    {
                        "name": member.name,
                        "size_bytes": member.size,
                        "is_dir": member.isdir(),
                        "is_tick": _is_tick_file(member.name),
                        "is_system": _is_system_file(member.name),
                        "symbol": _symbol_from_filename(member.name)
                        if not member.isdir()
                        else None,
                    }
                )
    except tarfile.TarError as exc:
        raise ArchiveReadError(f"Cannot list archive {archive_path}: {exc}") from exc

    return contents


def validate_archive_integrity(archive_path: Path) -> tuple[bool, str]:
    """
    Quick structural check: can we open it, are the expected files present?
    Returns (ok: bool, message: str).
    Does NOT parse every line.
    """
    try:
        contents = list_archive_contents(archive_path)
    except ArchiveReadError as exc:
        return False, str(exc)

    tick_files = [c for c in contents if c["is_tick"]]
    system_files = [c for c in contents if c["is_system"]]

    if not tick_files:
        return False, "Archive contains no .ndjson.gz tick files"
    if not system_files:
        return False, "Archive missing SYSTEM.ndjson.gz"

    return True, f"OK: {len(tick_files)} symbol files + SYSTEM file"
