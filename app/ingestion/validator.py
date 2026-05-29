"""
app/ingestion/validator.py
--------------------------
Strict schema validation and parsing for raw NDJSON lines.

Responsibility:
- Parse raw JSON strings into dicts.
- Validate required fields exist and have acceptable types.
- Route records to TickRecord or SystemEvent Pydantic models.
- Count and log every rejection — never silently discard.
- Track sequence monotonicity per symbol.
- Track timestamp sanity across the session.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, Union

from pydantic import ValidationError

from app.ingestion.archive_reader import RawLine
from app.models.system_event import SystemEvent, SystemEventType
from app.models.tick import TickRecord
from app.utils.log_setup import get_logger

logger = get_logger(__name__)

# ── Required fields for a tick record ─────────────────────────────────────────
_TICK_REQUIRED_FIELDS: set[str] = {
    "t", "seq", "s", "bid", "ask", "bq", "aq", "spread", "imbalance",
}

# ── Required fields for a system event record ─────────────────────────────────
_SYSTEM_REQUIRED_FIELDS: set[str] = {"event", "t"}

# ── Numeric fields that must be floats/ints ───────────────────────────────────
_NUMERIC_TICK_FIELDS: set[str] = {
    "bid", "ask", "bq", "aq", "spread", "imbalance",
    "bp1", "bp2", "bp3", "bp4", "bp5",
    "bq1", "bq2", "bq3", "bq4", "bq5",
    "ap1", "ap2", "ap3", "ap4", "ap5",
    "aq1", "aq2", "aq3", "aq4", "aq5",
    "db", "da",
}


# ──────────────────────────────────────────────────────────────────────────────
# Rejection tracking
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ValidationStats:
    """Accumulates validation statistics for a single symbol or session."""

    symbol: str
    total_seen: int = 0
    accepted: int = 0
    rejected_json_parse: int = 0
    rejected_missing_fields: int = 0
    rejected_type_error: int = 0
    rejected_model_validation: int = 0
    rejected_sequence_gap: int = 0
    rejected_timestamp_regression: int = 0
    rejected_other: int = 0

    _last_seq: Optional[int] = field(default=None, repr=False)
    _last_t: Optional[int] = field(default=None, repr=False)
    _last_total_bid_depth: Optional[float] = field(default=None, repr=False)
    _last_total_ask_depth: Optional[float] = field(default=None, repr=False)

    @property
    def total_rejected(self) -> int:
        return self.total_seen - self.accepted

    @property
    def rejection_rate(self) -> float:
        if self.total_seen == 0:
            return 0.0
        return self.total_rejected / self.total_seen

    def summary(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "total_seen": self.total_seen,
            "accepted": self.accepted,
            "total_rejected": self.total_rejected,
            "rejection_rate": round(self.rejection_rate, 4),
            "by_reason": {
                "json_parse": self.rejected_json_parse,
                "missing_fields": self.rejected_missing_fields,
                "type_error": self.rejected_type_error,
                "model_validation": self.rejected_model_validation,
                "sequence_gap": self.rejected_sequence_gap,
                "timestamp_regression": self.rejected_timestamp_regression,
                "other": self.rejected_other,
            },
        }


# ──────────────────────────────────────────────────────────────────────────────
# Parser / validator
# ──────────────────────────────────────────────────────────────────────────────


ParsedRecord = Union[TickRecord, SystemEvent]


class RecordValidator:
    """
    Parses and validates raw NDJSON lines into typed model instances.

    Maintains per-symbol state for sequence and timestamp monotonicity.
    Stateful — one instance should be used per session/archive.
    """

    def __init__(self, strict_sequence: bool = False) -> None:
        """
        Parameters
        ----------
        strict_sequence:
            If True, reject any tick where seq is not exactly last_seq + 1.
            If False (default), only reject backwards seq.
            Most feeds have small non-sequential jumps; strict=False is safer.
        """
        self._stats: dict[str, ValidationStats] = {}
        self._strict_sequence = strict_sequence
        self._system_stats = ValidationStats(symbol="SYSTEM")

    def _get_stats(self, symbol: str) -> ValidationStats:
        if symbol not in self._stats:
            self._stats[symbol] = ValidationStats(symbol=symbol)
        return self._stats[symbol]

    def parse(self, raw_line: RawLine) -> Optional[ParsedRecord]:
        """
        Parse one RawLine into a typed record.

        Returns None if the record should be rejected.
        Never raises — all errors are counted and logged.
        """
        if raw_line.is_system:
            return self._parse_system(raw_line)
        return self._parse_tick(raw_line)

    def _parse_system(self, raw_line: RawLine) -> Optional[SystemEvent]:
        stats = self._system_stats
        stats.total_seen += 1

        data = self._parse_json(raw_line, stats)
        if data is None:
            return None

        missing = _SYSTEM_REQUIRED_FIELDS - data.keys()
        if missing:
            stats.rejected_missing_fields += 1
            logger.warning(
                "system_event_missing_fields",
                missing=sorted(missing),
                line_no=raw_line.line_no,
            )
            return None

        try:
            record = SystemEvent(**data)
            stats.accepted += 1
            return record
        except (ValidationError, Exception) as exc:
            stats.rejected_model_validation += 1
            logger.warning(
                "system_event_validation_failed",
                error=str(exc),
                line_no=raw_line.line_no,
            )
            return None

    def _parse_tick(self, raw_line: RawLine) -> Optional[TickRecord]:
        symbol = raw_line.symbol
        stats = self._get_stats(symbol)
        stats.total_seen += 1

        # ── JSON parse ────────────────────────────────────────────────────────
        data = self._parse_json(raw_line, stats)
        if data is None:
            return None

        # Convert db/da from depth list to delta float if they are list representations
        if "db" in data and isinstance(data["db"], list):
            current_depth = sum(float(level[1]) for level in data["db"] if isinstance(level, list) and len(level) > 1)
            last_depth = stats._last_total_bid_depth
            if last_depth is None:
                data["db"] = 0.0
            else:
                data["db"] = current_depth - last_depth
            stats._last_total_bid_depth = current_depth

        if "da" in data and isinstance(data["da"], list):
            current_depth = sum(float(level[1]) for level in data["da"] if isinstance(level, list) and len(level) > 1)
            last_depth = stats._last_total_ask_depth
            if last_depth is None:
                data["da"] = 0.0
            else:
                data["da"] = current_depth - last_depth
            stats._last_total_ask_depth = current_depth

        # Normalize imbalance if it is outside [-1.0, 1.0] (e.g. depth ratio B/A)
        imbalance = data.get("imbalance")
        if isinstance(imbalance, (int, float)):
            if not (-1.0 <= imbalance <= 1.0):
                if imbalance > 1.0:
                    data["imbalance"] = (imbalance - 1.0) / (imbalance + 1.0)
                else:
                    bq = data.get("bq", 0.0)
                    aq = data.get("aq", 0.0)
                    if isinstance(bq, (int, float)) and isinstance(aq, (int, float)) and (bq + aq) > 0:
                        data["imbalance"] = (bq - aq) / (bq + aq)
                    else:
                        data["imbalance"] = 0.0

        # ── Required field presence ───────────────────────────────────────────
        missing = _TICK_REQUIRED_FIELDS - data.keys()
        if missing:
            stats.rejected_missing_fields += 1
            logger.debug(
                "tick_missing_required_fields",
                symbol=symbol,
                missing=sorted(missing),
                line_no=raw_line.line_no,
            )
            return None

        # ── Numeric type check ────────────────────────────────────────────────
        for nf in _NUMERIC_TICK_FIELDS:
            if nf in data and not isinstance(data[nf], (int, float, type(None))):
                stats.rejected_type_error += 1
                logger.debug(
                    "tick_non_numeric_field",
                    symbol=symbol,
                    field=nf,
                    value=data[nf],
                    line_no=raw_line.line_no,
                )
                return None

        # ── Sequence monotonicity ─────────────────────────────────────────────
        seq = data.get("seq")
        t = data.get("t")

        if isinstance(seq, int) and stats._last_seq is not None:
            if self._strict_sequence and seq != stats._last_seq + 1:
                stats.rejected_sequence_gap += 1
                logger.debug(
                    "tick_sequence_gap",
                    symbol=symbol,
                    last_seq=stats._last_seq,
                    current_seq=seq,
                    line_no=raw_line.line_no,
                )
                return None
            elif not self._strict_sequence and seq < stats._last_seq:
                stats.rejected_sequence_gap += 1
                logger.warning(
                    "tick_sequence_regression",
                    symbol=symbol,
                    last_seq=stats._last_seq,
                    current_seq=seq,
                    line_no=raw_line.line_no,
                )
                return None

        # ── Timestamp regression ──────────────────────────────────────────────
        if isinstance(t, int) and stats._last_t is not None:
            if t < stats._last_t:
                stats.rejected_timestamp_regression += 1
                logger.warning(
                    "tick_timestamp_regression",
                    symbol=symbol,
                    last_t=stats._last_t,
                    current_t=t,
                    line_no=raw_line.line_no,
                )
                return None

        # ── Pydantic model validation ─────────────────────────────────────────
        try:
            record = TickRecord(**data)
        except ValidationError as exc:
            stats.rejected_model_validation += 1
            first_error = exc.errors()[0]
            logger.debug(
                "tick_validation_error",
                symbol=symbol,
                error_loc=first_error.get("loc"),
                error_msg=first_error.get("msg"),
                line_no=raw_line.line_no,
            )
            return None
        except Exception as exc:
            stats.rejected_other += 1
            logger.warning(
                "tick_unexpected_error",
                symbol=symbol,
                error=str(exc),
                line_no=raw_line.line_no,
            )
            return None

        # ── Update state ──────────────────────────────────────────────────────
        stats._last_seq = record.seq
        stats._last_t = record.t
        stats.accepted += 1
        return record

    @staticmethod
    def _parse_json(
        raw_line: RawLine, stats: ValidationStats
    ) -> Optional[dict[str, object]]:
        try:
            return json.loads(raw_line.raw)
        except json.JSONDecodeError as exc:
            stats.rejected_json_parse += 1
            logger.debug(
                "json_parse_error",
                filename=raw_line.filename,
                line_no=raw_line.line_no,
                error=str(exc),
                preview=raw_line.raw[:60],
            )
            return None

    def all_stats(self) -> list[ValidationStats]:
        return list(self._stats.values())

    def system_stats(self) -> ValidationStats:
        return self._system_stats

    def total_accepted(self) -> int:
        return sum(s.accepted for s in self._stats.values())

    def total_rejected(self) -> int:
        return sum(s.total_rejected for s in self._stats.values())
