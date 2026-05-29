"""
tests/test_ingestion.py
-----------------------
Tests for the ingestion pipeline: archive reading, schema validation,
tick parsing, system event parsing, and manifest building.
"""

from __future__ import annotations

import gzip
import io
import json
import tarfile
import tempfile
from pathlib import Path

import pytest

from app.ingestion.archive_reader import (
    ArchiveReadError,
    iter_archive_lines,
    list_archive_contents,
    validate_archive_integrity,
)
from app.ingestion.validator import RecordValidator, ValidationStats
from app.models.system_event import SystemEvent, SystemEventType
from app.models.tick import TickRecord
from tests.conftest import make_ndjson_gz_content, make_test_archive


# ──────────────────────────────────────────────────────────────────────────────
# archive_reader tests
# ──────────────────────────────────────────────────────────────────────────────


class TestArchiveReader:
    def test_list_contents_returns_symbol_and_system_files(self, tmp_path, sample_archive):
        contents = list_archive_contents(sample_archive)
        names = [c["name"] for c in contents]
        assert any("NIFTY26JUNFUT.ndjson.gz" in n for n in names)
        assert any("SYSTEM.ndjson.gz" in n for n in names)

    def test_list_contents_classifies_files_correctly(self, tmp_path, sample_archive):
        contents = list_archive_contents(sample_archive)
        tick_files = [c for c in contents if c["is_tick"]]
        sys_files = [c for c in contents if c["is_system"]]
        assert len(tick_files) >= 1
        assert len(sys_files) == 1

    def test_validate_integrity_passes_valid_archive(self, sample_archive):
        ok, msg = validate_archive_integrity(sample_archive)
        assert ok is True
        assert "OK" in msg

    def test_validate_integrity_fails_missing_system(self, tmp_path):
        archive = make_test_archive(tmp_path, include_system=False)
        ok, msg = validate_archive_integrity(archive)
        assert ok is False
        assert "SYSTEM" in msg

    def test_validate_integrity_fails_nonexistent_file(self, tmp_path):
        fake = tmp_path / "nonexistent.tar.gz"
        ok, msg = validate_archive_integrity(fake)
        assert ok is False

    def test_iter_lines_yields_raw_lines_for_tick_file(self, sample_archive):
        lines = list(iter_archive_lines(sample_archive))
        # Should have some tick lines and some system lines
        tick_lines = [l for l in lines if not l.is_system]
        sys_lines = [l for l in lines if l.is_system]
        assert len(tick_lines) > 0
        assert len(sys_lines) > 0

    def test_iter_lines_filters_by_symbol(self, tmp_path):
        archive = make_test_archive(
            tmp_path, symbols=["NIFTY26JUNFUT", "BANKNIFTY26JUNFUT"]
        )
        lines = list(iter_archive_lines(archive, target_symbols=["NIFTY26JUNFUT"]))
        tick_symbols = {l.symbol for l in lines if not l.is_system}
        assert "NIFTY26JUNFUT" in tick_symbols
        assert "BANKNIFTY26JUNFUT" not in tick_symbols

    def test_iter_lines_skips_blank_and_invalid_json(self, tmp_path):
        bad_records = [{"valid": True}, {"t": 1}, {}]  # will be invalid on parse
        raw = b"\n" + b'not-json\n' + json.dumps({"valid": True}).encode() + b"\n"
        gz_buf = io.BytesIO()
        with gzip.GzipFile(fileobj=gz_buf, mode="wb") as gz:
            gz.write(raw)
        gz_bytes = gz_buf.getvalue()

        archive = tmp_path / "test.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            sym_data = make_ndjson_gz_content([
                {"t": 1_700_000_000_000, "seq": 1, "s": "NIFTY26JUNFUT",
                 "bid": 22000.0, "ask": 22000.5, "bq": 100.0, "aq": 80.0,
                 "spread": 0.5, "imbalance": 0.11,
                 "bp1": 22000.0, "bq1": 100.0, "ap1": 22000.5, "aq1": 80.0}
            ])
            info = tarfile.TarInfo("NIFTY26JUNFUT.ndjson.gz")
            info.size = len(sym_data)
            tar.addfile(info, io.BytesIO(sym_data))
            sys_data = make_ndjson_gz_content([{"event": "SESSION_START", "t": 1_700_000_000_000}])
            info2 = tarfile.TarInfo("SYSTEM.ndjson.gz")
            info2.size = len(sys_data)
            tar.addfile(info2, io.BytesIO(sys_data))

        lines = list(iter_archive_lines(archive))
        assert len(lines) >= 1

    def test_raises_on_missing_archive(self, tmp_path):
        with pytest.raises(ArchiveReadError):
            list(iter_archive_lines(tmp_path / "missing.tar.gz"))


# ──────────────────────────────────────────────────────────────────────────────
# Validator tests
# ──────────────────────────────────────────────────────────────────────────────


class TestRecordValidator:
    def _make_raw_tick_line(self, overrides: dict = None):
        """Helper to create a RawLine from conftest make_tick data."""
        from app.ingestion.archive_reader import RawLine
        data = {
            "t": 1_700_000_000_000,
            "seq": 1,
            "s": "NIFTY26JUNFUT",
            "bid": 22000.0,
            "ask": 22000.5,
            "bq": 100.0,
            "aq": 80.0,
            "spread": 0.5,
            "imbalance": 0.11,
            "bp1": 22000.0,
            "bq1": 100.0,
            "ap1": 22000.5,
            "aq1": 80.0,
        }
        if overrides:
            data.update(overrides)
        return RawLine(
            symbol="NIFTY26JUNFUT",
            filename="NIFTY26JUNFUT.ndjson.gz",
            line_no=1,
            raw=json.dumps(data),
            is_system=False,
        )

    def test_valid_tick_parses_successfully(self):
        v = RecordValidator()
        line = self._make_raw_tick_line()
        record = v.parse(line)
        assert isinstance(record, TickRecord)
        assert record.s == "NIFTY26JUNFUT"

    def test_rejects_missing_required_field(self):
        v = RecordValidator()
        data = {"t": 1_700_000_000_000, "seq": 1, "s": "NIFTY26JUNFUT"}
        # missing bid, ask, etc.
        from app.ingestion.archive_reader import RawLine
        line = RawLine(
            symbol="NIFTY26JUNFUT",
            filename="test.ndjson.gz",
            line_no=1,
            raw=json.dumps(data),
            is_system=False,
        )
        result = v.parse(line)
        assert result is None
        assert v.all_stats()[0].rejected_missing_fields > 0

    def test_rejects_crossed_market(self):
        v = RecordValidator()
        line = self._make_raw_tick_line({"bid": 22001.0, "ask": 22000.0, "spread": -1.0})
        result = v.parse(line)
        assert result is None

    def test_rejects_invalid_imbalance(self):
        v = RecordValidator()
        line = self._make_raw_tick_line({"imbalance": "invalid-string"})
        result = v.parse(line)
        assert result is None

    def test_rejects_timestamp_regression(self):
        v = RecordValidator()
        line1 = self._make_raw_tick_line({"t": 1_700_000_100_000, "seq": 1})
        line2 = self._make_raw_tick_line({"t": 1_700_000_000_000, "seq": 2})
        v.parse(line1)
        result = v.parse(line2)
        assert result is None

    def test_rejects_sequence_regression(self):
        v = RecordValidator()
        line1 = self._make_raw_tick_line({"seq": 100, "t": 1_700_000_000_000})
        line2 = self._make_raw_tick_line({"seq": 50, "t": 1_700_000_000_200})
        v.parse(line1)
        result = v.parse(line2)
        assert result is None

    def test_rejects_malformed_json(self):
        from app.ingestion.archive_reader import RawLine
        v = RecordValidator()
        line = RawLine("NIFTY26JUNFUT", "test.ndjson.gz", 1, "{bad json", False)
        result = v.parse(line)
        assert result is None

    def test_parses_system_event(self):
        from app.ingestion.archive_reader import RawLine
        v = RecordValidator()
        line = RawLine(
            symbol="SYSTEM",
            filename="SYSTEM.ndjson.gz",
            line_no=1,
            raw=json.dumps({"event": "GAP", "t": 1_700_000_000_000, "duration": 5000}),
            is_system=True,
        )
        result = v.parse(line)
        assert isinstance(result, SystemEvent)
        assert result.event == SystemEventType.GAP

    def test_accumulates_statistics(self):
        v = RecordValidator()
        # Parse 3 valid + 1 invalid
        for i in range(3):
            v.parse(self._make_raw_tick_line({"seq": i + 1, "t": 1_700_000_000_000 + i * 100}))
        v.parse(self._make_raw_tick_line({"imbalance": 99.0}))

        stats = v.all_stats()
        assert len(stats) == 1
        assert stats[0].accepted == 3
        assert stats[0].total_rejected == 1

    def test_validation_stats_rejection_rate(self):
        stats = ValidationStats(symbol="TEST")
        stats.total_seen = 100
        stats.accepted = 80
        assert abs(stats.rejection_rate - 0.20) < 0.001
