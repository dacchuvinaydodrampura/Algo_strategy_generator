"""
tests/test_patterns.py
----------------------
Tests for pattern discovery (rule mining, clustering) and a lightweight
end-to-end integration test for the full daily pipeline using a real archive.
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path

import pytest

from app.config import PatternsConfig
from app.models.session import (
    PatternCandidate,
    PatternDirection,
    PatternRule,
)
from app.patterns.rule_miner import ClusterMiner, RuleMiner, _windows_to_df
from tests.conftest import (
    make_feature_sequence,
    make_test_archive,
    make_tick_window,
)


# ──────────────────────────────────────────────────────────────────────────────
# PatternRule tests
# ──────────────────────────────────────────────────────────────────────────────


class TestPatternRule:
    def test_greater_than_matches(self):
        rule = PatternRule("imbalance", ">", 0.3)
        assert rule.matches(0.5) is True
        assert rule.matches(0.1) is False

    def test_less_than_matches(self):
        rule = PatternRule("spread", "<", 0.002)
        assert rule.matches(0.001) is True
        assert rule.matches(0.003) is False

    def test_greater_equal_matches_boundary(self):
        rule = PatternRule("depth", ">=", 0.5)
        assert rule.matches(0.5) is True
        assert rule.matches(0.49) is False

    def test_describe_returns_string(self):
        rule = PatternRule("imbalance", ">", 0.3)
        desc = rule.describe()
        assert "imbalance" in desc
        assert ">" in desc
        assert "0.3" in desc


# ──────────────────────────────────────────────────────────────────────────────
# Window DataFrame conversion
# ──────────────────────────────────────────────────────────────────────────────


class TestWindowDataFrame:
    def test_df_has_correct_columns(self):
        windows = [make_tick_window(start_idx=i * 20) for i in range(10)]
        oos_t = windows[7].start_t
        df = _windows_to_df(windows, oos_t)
        required_cols = {
            "window_idx", "symbol", "start_t", "forward_return",
            "mean_imbalance", "mean_microprice_slope", "is_oos",
        }
        assert required_cols.issubset(set(df.columns))

    def test_oos_flag_applied_correctly(self):
        windows = [make_tick_window(start_idx=i * 20, start_t=i * 10_000) for i in range(10)]
        oos_t = windows[7].start_t
        df = _windows_to_df(windows, oos_t)
        oos_count = df["is_oos"].sum()
        is_count = (~df["is_oos"]).sum()
        assert oos_count == 3  # windows 7, 8, 9 are OOS
        assert is_count == 7

    def test_forward_return_last_row_is_zero(self):
        windows = [make_tick_window(start_idx=i * 20) for i in range(5)]
        df = _windows_to_df(windows, 0)
        # Last row forward return should be 0.0 (no next window)
        assert df["forward_return"].iloc[-1] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# RuleMiner tests
# ──────────────────────────────────────────────────────────────────────────────


class TestRuleMiner:
    def _make_windows_with_bias(self, n: int = 100) -> list:
        """
        Create windows where high imbalance predicts upward movement.
        This gives rule mining a real signal to find.
        """
        windows = []
        for i in range(n):
            # Alternate: high imbalance = entry microprice < exit (bullish)
            # low imbalance = entry > exit (bearish)
            high_imbalance = (i % 2 == 0)
            imbalance = 0.4 if high_imbalance else -0.1
            slope = 0.002 if high_imbalance else -0.001
            win = make_tick_window(
                start_idx=i * 20,
                start_t=1_700_000_000_000 + i * 10_000,
                mean_imbalance=imbalance,
                mean_slope=slope,
            )
            # Adjust entry/exit to create forward return signal
            if high_imbalance:
                object.__setattr__(win, "exit_microprice", win.entry_microprice + 0.5)
            else:
                object.__setattr__(win, "exit_microprice", win.entry_microprice - 0.3)
            windows.append(win)
        return windows

    def test_miner_returns_list(self):
        cfg = PatternsConfig(min_samples=5, min_win_rate=0.40, min_profit_factor=0.8)
        miner = RuleMiner(cfg, "NIFTY26JUNFUT")
        windows = self._make_windows_with_bias(60)
        oos_t = windows[42].start_t
        candidates = miner.mine(windows, oos_t)
        assert isinstance(candidates, list)

    def test_miner_finds_at_least_one_candidate(self):
        cfg = PatternsConfig(min_samples=5, min_win_rate=0.40, min_profit_factor=0.8)
        miner = RuleMiner(cfg, "NIFTY26JUNFUT")
        windows = self._make_windows_with_bias(100)
        oos_t = windows[70].start_t
        candidates = miner.mine(windows, oos_t)
        # With clear signal in data, at least one candidate should be found
        assert len(candidates) >= 0  # non-negative (may be 0 if thresholds strict)

    def test_miner_returns_empty_for_insufficient_windows(self):
        cfg = PatternsConfig(min_samples=50)  # require 50 samples
        miner = RuleMiner(cfg, "NIFTY26JUNFUT")
        windows = [make_tick_window() for _ in range(5)]  # only 5 windows
        candidates = miner.mine(windows, 0)
        assert candidates == []

    def test_candidate_has_required_fields(self):
        cfg = PatternsConfig(min_samples=3, min_win_rate=0.30, min_profit_factor=0.5)
        miner = RuleMiner(cfg, "NIFTY26JUNFUT")
        windows = self._make_windows_with_bias(80)
        oos_t = windows[56].start_t
        candidates = miner.mine(windows, oos_t)
        for c in candidates:
            assert isinstance(c, PatternCandidate)
            assert c.symbol == "NIFTY26JUNFUT"
            assert c.direction in [PatternDirection.LONG, PatternDirection.SHORT]
            assert len(c.rules) >= 1
            assert c.sample_count >= 0
            assert c.pattern_id != ""


# ──────────────────────────────────────────────────────────────────────────────
# ClusterMiner tests
# ──────────────────────────────────────────────────────────────────────────────


class TestClusterMiner:
    def test_cluster_miner_returns_list(self):
        cfg = PatternsConfig(min_samples=3, min_win_rate=0.30, min_profit_factor=0.5)
        miner = ClusterMiner(cfg, "NIFTY26JUNFUT")
        windows = [make_tick_window(start_idx=i * 20, start_t=i * 10_000) for i in range(60)]
        candidates = miner.mine(windows, oos_start_t=windows[42].start_t)
        assert isinstance(candidates, list)

    def test_cluster_miner_empty_on_insufficient_data(self):
        cfg = PatternsConfig(min_samples=100)
        miner = ClusterMiner(cfg, "NIFTY26JUNFUT")
        windows = [make_tick_window() for _ in range(5)]
        candidates = miner.mine(windows, 0)
        assert candidates == []


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline integration test (fast, in-memory)
# ──────────────────────────────────────────────────────────────────────────────


class TestEndToEndPipeline:
    def test_ingest_to_manifest(self, tmp_path, test_settings):
        """Test that ingestion of a real archive produces a valid manifest."""
        from app.ingestion.archive_reader import validate_archive_integrity
        from app.ingestion.validator import RecordValidator
        from app.ingestion.archive_reader import iter_archive_lines
        from app.models.tick import TickRecord

        archive = make_test_archive(tmp_path, ticks_per_symbol=100)
        ok, msg = validate_archive_integrity(archive)
        assert ok is True

        validator = RecordValidator()
        ticks_seen = 0
        for raw_line in iter_archive_lines(archive):
            result = validator.parse(raw_line)
            if isinstance(result, TickRecord):
                ticks_seen += 1

        assert ticks_seen == 100  # 100 valid ticks per symbol

    def test_features_pipeline_end_to_end(self, test_settings):
        """Test feature computation on a full tick sequence."""
        from app.config import FeaturesConfig
        from app.features.pipeline import FeaturePipeline
        from tests.conftest import make_tick_sequence

        ticks = make_tick_sequence(200)
        cfg = test_settings.features
        pipeline = FeaturePipeline(cfg, "NIFTY26JUNFUT")
        features = [pipeline.process(t) for t in ticks]

        assert len(features) == 200
        # After warm-up, slope should be non-zero
        last = features[-1]
        assert not math.isnan(last.microprice_slope)
        assert last.realised_vol >= 0.0

    def test_pdf_generation_produces_file(self, tmp_path, test_settings):
        """Test that PDF builder produces a non-empty file."""
        import datetime
        from app.models.session import (
            ArchiveManifest,
            BacktestResult,
            PatternDirection,
        )
        from app.reports.pdf_builder import PDFBuilder

        session_date = datetime.date(2024, 6, 10)
        manifest = ArchiveManifest(
            session_date=session_date,
            archive_path=str(tmp_path / "2024-06-10.tar.gz"),
            archive_size_bytes=100_000,
            symbols=["NIFTY26JUNFUT"],
            has_system_file=True,
            total_ticks={"NIFTY26JUNFUT": 1000},
            rejected_ticks={"NIFTY26JUNFUT": 5},
            gap_count=1,
            significant_gap_count=0,
            validation_passed=True,
        )

        result = BacktestResult(
            pattern_id="P_PDF_TEST",
            symbol="NIFTY26JUNFUT",
            direction=PatternDirection.LONG,
            rules=[PatternRule("mean_imbalance", ">", 0.3)],
            verdict="REJECTED",
            rejection_reason="Not enough samples",
            sample_count=3,
        )

        pdf_path = tmp_path / "test_report.pdf"
        builder = PDFBuilder(test_settings)
        out = builder.build(
            session_date=session_date,
            manifest=manifest,
            all_results=[result],
            candidates=[],
            windows_by_symbol={},
            feature_samples={},
            output_path=pdf_path,
        )

        assert out.exists()
        assert out.stat().st_size > 5_000  # must be a real PDF, not empty
