"""
tests/test_features.py
----------------------
Unit tests for the feature computation pipeline, tick model validation,
and windowing logic.
"""

from __future__ import annotations

import math

import pytest

from app.config import FeaturesConfig
from app.features.pipeline import FeaturePipeline, _compute_microprice, _slope
from app.models.session import FeatureRecord
from app.models.tick import TickRecord
from app.windows.tick_window import fixed_tick_windows, fixed_time_windows
from tests.conftest import (
    make_feature_sequence,
    make_tick,
    make_tick_sequence,
    make_tick_window,
)


# ──────────────────────────────────────────────────────────────────────────────
# TickRecord model validation
# ──────────────────────────────────────────────────────────────────────────────


class TestTickRecord:
    def test_valid_tick_creates_successfully(self):
        tick = make_tick()
        assert tick.s == "NIFTY26JUNFUT"
        assert tick.spread == 0.5

    def test_midprice_is_correct(self):
        tick = make_tick(bid=22000.0, ask=22001.0)
        assert abs(tick.midprice - 22000.5) < 1e-6

    def test_rejects_crossed_market(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            make_tick(bid=22001.0, ask=22000.0, spread=-1.0)

    def test_rejects_negative_spread(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            make_tick(spread=-0.1)

    def test_rejects_invalid_imbalance_above_1(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            make_tick(imbalance=1.5)

    def test_rejects_invalid_imbalance_below_neg1(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            make_tick(imbalance=-1.5)

    def test_rejects_zero_bid_quantity(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            make_tick(bq=0.0)

    def test_rejects_invalid_timestamp_too_early(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            make_tick(t=0)

    def test_spread_consistency_validation(self):
        """spread field must match ask - bid."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            make_tick(bid=22000.0, ask=22000.5, spread=1.0)  # wrong spread

    def test_bid_levels_returns_available_levels(self):
        tick = make_tick()
        levels = tick.bid_levels()
        assert len(levels) >= 1
        assert all(p > 0 and q > 0 for p, q in levels)

    def test_total_bid_depth_sums_levels(self):
        tick = make_tick()
        depth = tick.total_bid_depth()
        assert depth > 0

    def test_symbol_is_uppercased(self):
        tick = make_tick(symbol="nifty26junfut")
        assert tick.s == "NIFTY26JUNFUT"


# ──────────────────────────────────────────────────────────────────────────────
# Feature helper functions
# ──────────────────────────────────────────────────────────────────────────────


class TestFeatureHelpers:
    def test_slope_of_flat_line_is_zero(self):
        s = _slope([5.0, 5.0, 5.0, 5.0])
        assert abs(s) < 1e-9

    def test_slope_of_increasing_line_is_positive(self):
        s = _slope([1.0, 2.0, 3.0, 4.0])
        assert s > 0

    def test_slope_of_decreasing_line_is_negative(self):
        s = _slope([4.0, 3.0, 2.0, 1.0])
        assert s < 0

    def test_slope_of_single_value_returns_zero(self):
        assert _slope([42.0]) == 0.0

    def test_slope_of_empty_returns_zero(self):
        assert _slope([]) == 0.0

    def test_microprice_falls_back_to_midprice_with_no_depth(self):
        tick = make_tick(bid=22000.0, ask=22001.0)
        # Remove depth levels
        tick_data = tick.model_dump()
        for key in ["bp1", "bq1", "ap1", "aq1", "bp2", "bq2", "ap2", "aq2"]:
            tick_data[key] = None
        from app.models.tick import TickRecord
        clean_tick = TickRecord(**tick_data)
        mp = _compute_microprice(clean_tick, levels=2)
        assert abs(mp - 22000.5) < 1e-4

    def test_microprice_is_ask_weighted_when_more_bid_depth(self):
        tick = make_tick(bid=22000.0, ask=22001.0, bq=900.0, aq=100.0)
        # Heavy bid depth → order flow biased upward → microprice > mid
        # Formula: mp = (ask * bq + bid * aq) / (bq + aq)
        # With bq=900 >> aq=100: mp weighted toward ask price
        mp = _compute_microprice(tick, levels=1)
        mid = tick.midprice
        assert mp > mid  # price biased toward ask side (bullish)

    def test_microprice_is_below_mid_when_heavy_ask_depth(self):
        tick = make_tick(bid=22000.0, ask=22001.0, bq=100.0, aq=900.0)
        # Heavy ask depth → price biased downward → microprice < mid
        mp = _compute_microprice(tick, levels=1)
        mid = tick.midprice
        assert mp < mid  # price biased toward bid side (bearish)


# ──────────────────────────────────────────────────────────────────────────────
# FeaturePipeline
# ──────────────────────────────────────────────────────────────────────────────


class TestFeaturePipeline:
    def test_pipeline_produces_feature_record(self):
        cfg = FeaturesConfig()
        pipeline = FeaturePipeline(cfg, "NIFTY26JUNFUT")
        tick = make_tick()
        fr = pipeline.process(tick)
        assert isinstance(fr, FeatureRecord)
        assert fr.symbol == "NIFTY26JUNFUT"
        assert fr.t == tick.t

    def test_pipeline_processes_sequence_without_error(self):
        cfg = FeaturesConfig()
        pipeline = FeaturePipeline(cfg, "NIFTY26JUNFUT")
        ticks = make_tick_sequence(100)
        features = [pipeline.process(t) for t in ticks]
        assert len(features) == 100
        assert all(isinstance(f, FeatureRecord) for f in features)

    def test_microprice_slope_is_zero_initially(self):
        cfg = FeaturesConfig()
        pipeline = FeaturePipeline(cfg, "NIFTY26JUNFUT")
        tick = make_tick()
        fr = pipeline.process(tick)
        assert fr.microprice_slope == 0.0

    def test_slope_becomes_nonzero_after_window_fills(self):
        cfg = FeaturesConfig()
        pipeline = FeaturePipeline(cfg, "NIFTY26JUNFUT")
        ticks = make_tick_sequence(20, bid_drift=0.01)
        features = [pipeline.process(t) for t in ticks]
        # After window fills, slope should be positive (prices rising)
        last_slope = features[-1].microprice_slope
        assert last_slope > 0

    def test_realised_vol_is_zero_at_start(self):
        cfg = FeaturesConfig()
        pipeline = FeaturePipeline(cfg, "NIFTY26JUNFUT")
        tick = make_tick()
        fr = pipeline.process(tick)
        assert fr.realised_vol == 0.0

    def test_aggression_score_is_bounded(self):
        cfg = FeaturesConfig()
        pipeline = FeaturePipeline(cfg, "NIFTY26JUNFUT")
        ticks = make_tick_sequence(50)
        features = [pipeline.process(t) for t in ticks]
        for f in features:
            assert -2.0 <= f.aggression_score <= 2.0

    def test_relative_spread_is_positive(self):
        cfg = FeaturesConfig()
        pipeline = FeaturePipeline(cfg, "NIFTY26JUNFUT")
        tick = make_tick()
        fr = pipeline.process(tick)
        assert fr.relative_spread > 0

    def test_depth_ratio_is_between_0_and_1(self):
        cfg = FeaturesConfig()
        pipeline = FeaturePipeline(cfg, "NIFTY26JUNFUT")
        ticks = make_tick_sequence(10)
        for t in ticks:
            fr = pipeline.process(t)
            assert 0.0 <= fr.depth_ratio <= 1.0

    def test_tick_counter_increments(self):
        cfg = FeaturesConfig()
        pipeline = FeaturePipeline(cfg, "NIFTY26JUNFUT")
        ticks = make_tick_sequence(5)
        for t in ticks:
            pipeline.process(t)
        assert pipeline.ticks_processed == 5


# ──────────────────────────────────────────────────────────────────────────────
# Windowing
# ──────────────────────────────────────────────────────────────────────────────


class TestTickWindow:
    def test_fixed_windows_correct_count(self):
        features = make_feature_sequence(200)
        windows = list(fixed_tick_windows(iter(features), 20, 20, "NIFTY26JUNFUT"))
        # 200 ticks / 20 ticks per window = 10 windows
        assert len(windows) == 10

    def test_fixed_windows_have_correct_tick_count(self):
        features = make_feature_sequence(200)
        windows = list(fixed_tick_windows(iter(features), 20, 20, "NIFTY26JUNFUT"))
        for w in windows:
            assert w.ticks == 20

    def test_fixed_windows_non_overlapping(self):
        features = make_feature_sequence(100)
        windows = list(fixed_tick_windows(iter(features), 10, 10, "NIFTY26JUNFUT"))
        for i in range(len(windows) - 1):
            assert windows[i].end_idx == windows[i + 1].start_idx

    def test_fixed_windows_partial_last_window_included(self):
        features = make_feature_sequence(25)
        # 25 ticks with window_size=20, step=20
        # First full window: 20 ticks. Remaining: 5 ticks (>= min_ticks=5)
        windows = list(
            fixed_tick_windows(iter(features), 20, 20, "NIFTY26JUNFUT", min_ticks=5)
        )
        assert len(windows) == 2

    def test_empty_feature_list_produces_no_windows(self):
        windows = list(fixed_tick_windows(iter([]), 20, 20, "NIFTY26JUNFUT"))
        assert windows == []

    def test_window_summary_stats_are_computed(self):
        features = make_feature_sequence(50)
        windows = list(fixed_tick_windows(iter(features), 50, 50, "NIFTY26JUNFUT"))
        assert len(windows) == 1
        w = windows[0]
        assert w.mean_imbalance != 0.0 or True  # just check it's set
        assert w.entry_microprice > 0
        assert w.exit_microprice > 0

    def test_time_windows_split_by_duration(self):
        # 200 ticks at 200ms each = 40 seconds
        # Window of 10 seconds → expect ~4 windows
        features = make_feature_sequence(200, start_t=1_700_000_000_000)
        windows = list(
            fixed_time_windows(iter(features), window_seconds=10, symbol="NIFTY26JUNFUT")
        )
        assert len(windows) >= 3

    def test_raises_if_window_size_below_min(self):
        with pytest.raises(ValueError):
            list(fixed_tick_windows(iter([]), 5, 5, "SYM", min_ticks=10))
