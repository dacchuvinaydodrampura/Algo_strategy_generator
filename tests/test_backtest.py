"""
tests/test_backtest.py
----------------------
Unit tests for the backtesting engine, cost model,
and analytics metrics computation.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pandas as pd
import pytest

from app.analytics.metrics import (
    _equity_curve,
    _max_drawdown,
    _profit_factor,
    _sharpe_ratio,
    _stability_cv,
    _win_rate,
    populate_metrics,
)
from app.backtest.costs import CostBreakdown, CostModel
from app.backtest.engine import BacktestEngine, _get_direction_from_window
from app.config import BacktestConfig, PatternsConfig
from app.models.session import (
    BacktestResult,
    PatternCandidate,
    PatternDirection,
    PatternRule,
    TradeResult,
)
from tests.conftest import (
    make_feature_sequence,
    make_tick_window,
)


# ──────────────────────────────────────────────────────────────────────────────
# CostModel tests
# ──────────────────────────────────────────────────────────────────────────────


class TestCostModel:
    def _cfg(self) -> BacktestConfig:
        return BacktestConfig(
            tick_size=0.05,
            lot_size=25,
            brokerage_per_lot=20.0,
            slippage_ticks=1,
            latency_ms=50,
        )

    def test_total_cost_is_brokerage_plus_slippage(self):
        model = CostModel(self._cfg())
        breakdown = model.breakdown(25)
        expected = 20.0 + (1 * 0.05 * 25)
        assert abs(breakdown.total - expected) < 1e-6

    def test_brokerage_component_is_fixed(self):
        model = CostModel(self._cfg())
        bd = model.breakdown(25)
        assert bd.brokerage == 20.0

    def test_slippage_scales_with_lot_size(self):
        model = CostModel(self._cfg())
        bd50 = model.breakdown(50)
        bd25 = model.breakdown(25)
        assert bd50.slippage == 2 * bd25.slippage

    def test_assumption_text_is_non_empty_list(self):
        model = CostModel(self._cfg())
        lines = model.assumption_text()
        assert isinstance(lines, list)
        assert len(lines) >= 5
        assert all(isinstance(l, str) for l in lines)

    def test_cost_breakdown_total_equals_sum(self):
        model = CostModel(self._cfg())
        bd = model.breakdown(25)
        assert abs(bd.total - (bd.brokerage + bd.slippage)) < 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Analytics metric functions
# ──────────────────────────────────────────────────────────────────────────────


def _make_trade(net_pnl: float, is_oos: bool = False) -> TradeResult:
    return TradeResult(
        pattern_id="P1",
        symbol="NIFTY26JUNFUT",
        entry_t=1_700_000_000_000,
        exit_t=1_700_000_060_000,
        direction=PatternDirection.LONG,
        entry_price=22000.0,
        exit_price=22001.0 if net_pnl > 0 else 21999.0,
        stop_price=21995.0,
        target_price=22010.0,
        exit_reason="TARGET" if net_pnl > 0 else "STOP",
        gross_pnl=net_pnl + 20.0,
        cost=20.0,
        net_pnl=net_pnl,
        hold_ticks=10,
        hold_seconds=60.0,
        is_oos=is_oos,
    )


class TestAnalyticsMetrics:
    def test_win_rate_all_wins(self):
        trades = [_make_trade(100.0) for _ in range(10)]
        wr = _win_rate(trades, PatternDirection.LONG)
        assert abs(wr - 1.0) < 1e-9

    def test_win_rate_all_losses(self):
        trades = [_make_trade(-50.0) for _ in range(10)]
        wr = _win_rate(trades, PatternDirection.LONG)
        assert abs(wr - 0.0) < 1e-9

    def test_win_rate_mixed(self):
        trades = [_make_trade(100.0) for _ in range(6)] + [_make_trade(-50.0) for _ in range(4)]
        wr = _win_rate(trades, PatternDirection.LONG)
        assert abs(wr - 0.6) < 1e-9

    def test_profit_factor_no_losses(self):
        trades = [_make_trade(100.0) for _ in range(5)]
        pf = _profit_factor(trades, PatternDirection.LONG)
        assert pf > 100  # essentially infinity

    def test_profit_factor_no_wins(self):
        trades = [_make_trade(-50.0) for _ in range(5)]
        pf = _profit_factor(trades, PatternDirection.LONG)
        assert pf < 0.01

    def test_profit_factor_balanced(self):
        trades = (
            [_make_trade(100.0) for _ in range(5)]
            + [_make_trade(-50.0) for _ in range(5)]
        )
        pf = _profit_factor(trades, PatternDirection.LONG)
        assert abs(pf - 2.0) < 0.01

    def test_equity_curve_starts_at_initial_capital(self):
        trades = [_make_trade(100.0) for _ in range(5)]
        curve = _equity_curve(trades, 1_000_000.0)
        assert curve[0] == 1_000_000.0

    def test_equity_curve_length_is_trades_plus_1(self):
        trades = [_make_trade(50.0) for _ in range(10)]
        curve = _equity_curve(trades, 1_000_000.0)
        assert len(curve) == 11

    def test_max_drawdown_no_loss(self):
        curve = [1_000_000, 1_001_000, 1_002_000, 1_003_000]
        dd = _max_drawdown(curve)
        assert abs(dd) < 1e-6

    def test_max_drawdown_with_loss(self):
        # Peak at 1_100_000, then falls to 1_050_000
        curve = [1_000_000, 1_100_000, 1_050_000]
        dd = _max_drawdown(curve)
        expected = 50_000 / 1_100_000
        assert abs(dd - expected) < 1e-6

    def test_sharpe_ratio_constant_pnl_returns_zero(self):
        # std of identical values = 0 → sharpe = 0
        trades = [_make_trade(100.0) for _ in range(10)]
        sr = _sharpe_ratio(trades)
        assert abs(sr) < 1e-9

    def test_stability_cv_small_trades_returns_1(self):
        # Too few trades → return conservative 1.0
        trades = [_make_trade(100.0) for _ in range(3)]
        cv = _stability_cv(trades, PatternDirection.LONG)
        assert cv == 1.0


# ──────────────────────────────────────────────────────────────────────────────
# populate_metrics integration test
# ──────────────────────────────────────────────────────────────────────────────


class TestPopulateMetrics:
    def _make_result_with_trades(
        self, n_win: int, n_loss: int, n_oos: int = 0
    ) -> BacktestResult:
        result = BacktestResult(
            pattern_id="P_TEST",
            symbol="NIFTY26JUNFUT",
            direction=PatternDirection.LONG,
            rules=[PatternRule("mean_imbalance", ">", 0.2)],
        )
        result.trades = (
            [_make_trade(100.0) for _ in range(n_win)]
            + [_make_trade(-50.0) for _ in range(n_loss)]
            + [_make_trade(80.0, is_oos=True) for _ in range(n_oos)]
        )
        return result

    def test_accepted_pattern_with_good_metrics(self):
        patterns_cfg = PatternsConfig(
            min_samples=5,
            min_win_rate=0.50,
            min_profit_factor=1.0,
            stability_cv_threshold=1.0,  # permissive for test
        )
        bt_cfg = BacktestConfig(initial_capital=1_000_000.0)
        result = self._make_result_with_trades(n_win=15, n_loss=5)
        result = populate_metrics(result, patterns_cfg, bt_cfg)
        assert result.win_rate == pytest.approx(0.75, abs=0.01)
        assert result.verdict == "ACCEPTED"

    def test_rejected_pattern_low_win_rate(self):
        patterns_cfg = PatternsConfig(
            min_samples=5,
            min_win_rate=0.60,
            min_profit_factor=1.0,
        )
        bt_cfg = BacktestConfig(initial_capital=1_000_000.0)
        result = self._make_result_with_trades(n_win=4, n_loss=6)
        result = populate_metrics(result, patterns_cfg, bt_cfg)
        assert result.verdict == "REJECTED"
        assert "Win rate" in result.rejection_reason

    def test_rejected_pattern_too_few_trades(self):
        patterns_cfg = PatternsConfig(min_samples=20)
        bt_cfg = BacktestConfig(initial_capital=1_000_000.0)
        result = self._make_result_with_trades(n_win=5, n_loss=3)
        result = populate_metrics(result, patterns_cfg, bt_cfg)
        assert result.verdict == "REJECTED"
        assert "few" in result.rejection_reason.lower()

    def test_oos_metrics_populated_when_oos_trades_exist(self):
        patterns_cfg = PatternsConfig(
            min_samples=5,
            min_win_rate=0.40,
            min_profit_factor=0.8,
            stability_cv_threshold=1.0,
        )
        bt_cfg = BacktestConfig(initial_capital=1_000_000.0)
        result = self._make_result_with_trades(n_win=10, n_loss=5, n_oos=5)
        result = populate_metrics(result, patterns_cfg, bt_cfg)
        assert result.oos_sample_count == 5
        assert not math.isnan(result.oos_win_rate)

    def test_no_trades_produces_rejected_result(self):
        patterns_cfg = PatternsConfig()
        bt_cfg = BacktestConfig()
        result = BacktestResult(
            pattern_id="EMPTY",
            symbol="NIFTY26JUNFUT",
            direction=PatternDirection.LONG,
            rules=[],
        )
        result = populate_metrics(result, patterns_cfg, bt_cfg)
        assert result.verdict == "REJECTED"


# ──────────────────────────────────────────────────────────────────────────────
# BacktestEngine integration test
# ──────────────────────────────────────────────────────────────────────────────


class TestBacktestEngine:
    def test_engine_produces_result_object(self, test_settings):
        bt_cfg = test_settings.backtest
        engine = BacktestEngine(bt_cfg, CostModel(bt_cfg))

        features = make_feature_sequence(
            500, start_t=1_700_000_000_000, slope_trend=0.001
        )
        windows = [make_tick_window(n_ticks=20, start_idx=i * 20) for i in range(20)]
        features_df = pd.DataFrame([dataclasses.asdict(f) for f in features])

        candidate = PatternCandidate(
            pattern_id="P_BACKTEST_01",
            symbol="NIFTY26JUNFUT",
            direction=PatternDirection.LONG,
            rules=[PatternRule("mean_imbalance", ">", 0.1)],
            matched_windows=list(range(10)),
            sample_count=10,
        )

        result = engine.run(
            candidate=candidate,
            windows=windows,
            features_df=features_df,
            oos_start_t=1_700_000_050_000,
        )

        assert isinstance(result, BacktestResult)
        assert result.pattern_id == "P_BACKTEST_01"
        # Some trades may be produced (or zero if features don't align)
        assert isinstance(result.trades, list)

    def test_get_direction_long_from_positive_slope_and_imbalance(self):
        win = make_tick_window(mean_imbalance=0.5, mean_slope=0.002)
        assert _get_direction_from_window(win) == PatternDirection.LONG

    def test_get_direction_short_from_negative_slope(self):
        win = make_tick_window(mean_imbalance=-0.3, mean_slope=-0.002)
        assert _get_direction_from_window(win) == PatternDirection.SHORT

    def test_dynamic_stops_and_targets(self, test_settings):
        bt_cfg = test_settings.backtest
        bt_cfg_dynamic = bt_cfg.model_copy(update={
            "use_dynamic_stops": True,
            "stop_vol_multiplier": 3.0,
            "target_vol_multiplier": 6.0,
            "default_stop_ticks": 5,
            "default_target_ticks": 10
        })
        engine = BacktestEngine(bt_cfg_dynamic, CostModel(bt_cfg_dynamic))

        features = make_feature_sequence(
            500, start_t=1_700_000_000_000, slope_trend=0.001
        )
        for f in features:
            f.realised_vol = 2.5
            f.spread = 1.0

        windows = [make_tick_window(n_ticks=20, start_idx=i * 20) for i in range(20)]
        features_df = pd.DataFrame([dataclasses.asdict(f) for f in features])

        candidate = PatternCandidate(
            pattern_id="P_BACKTEST_DYNAMIC",
            symbol="NIFTY26JUNFUT",
            direction=PatternDirection.LONG,
            rules=[PatternRule("mean_imbalance", ">", 0.1)],
            matched_windows=[0, 1],
            sample_count=2,
        )

        result = engine.run(
            candidate=candidate,
            windows=windows,
            features_df=features_df,
            oos_start_t=1_700_000_050_000,
        )

        assert len(result.trades) > 0
        trade = result.trades[0]
        # stop_distance = max(vol * multiplier, spread + 2 * tick_size, default_stop_ticks * tick_size)
        # vol * multiplier = 2.5 * 3.0 = 7.5
        # spread + 2 * tick_size = 1.0 + 2 * 0.05 = 1.1
        # default_stop_ticks * tick_size = 5 * 0.05 = 0.25
        # Expected stop distance = 7.5
        assert abs((trade.entry_price - trade.stop_price) - 7.5) < 1e-6
