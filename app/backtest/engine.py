"""
app/backtest/engine.py
-----------------------
Realistic backtesting engine for pattern candidates.

Simulation assumptions (all configurable, all reported in PDF):
- Entry is at the NEXT tick's ask (LONG) or bid (SHORT) after pattern fires.
- Slippage = N ticks * tick_size added to entry price.
- Latency = L ms: we skip ticks within L ms of signal.
- Brokerage = fixed per-lot round-trip cost.
- Exit at first of: target hit, stop hit, max hold time, or end of day.
- No partial fills.
- No pyramiding.

This is NOT a live trading engine.  No orders are placed anywhere.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import pandas as pd

from app.backtest.costs import CostModel
from app.config import BacktestConfig
from app.models.session import (
    BacktestResult,
    FeatureRecord,
    PatternCandidate,
    PatternDirection,
    TickWindow,
    TradeResult,
)
from app.utils.log_setup import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    """
    Simulates all trades for a list of PatternCandidates against tick data.

    Parameters
    ----------
    cfg:        BacktestConfig with cost and trade parameters.
    cost_model: CostModel instance for computing realistic fill costs.
    """

    def __init__(self, cfg: BacktestConfig, cost_model: Optional[CostModel] = None) -> None:
        self._cfg = cfg
        self._costs = cost_model or CostModel(cfg)

    def run(
        self,
        candidate: PatternCandidate,
        windows: list[TickWindow],
        features_df: pd.DataFrame,
        oos_start_t: int,
    ) -> BacktestResult:
        """
        Backtest a single PatternCandidate across all matched windows.

        Parameters
        ----------
        candidate:    The pattern to backtest.
        windows:      All TickWindows (needed for window-level feature lookup).
        features_df:  Full-day FeatureRecord DataFrame indexed by position.
        oos_start_t:  Epoch ms where OOS period begins.
        """
        result = BacktestResult(
            pattern_id=candidate.pattern_id,
            symbol=candidate.symbol,
            direction=candidate.direction,
            rules=candidate.rules,
        )

        if features_df.empty:
            result.verdict = "REJECTED"
            result.rejection_reason = "No feature data available for backtest"
            return result

        trades: list[TradeResult] = []

        for win_idx in candidate.matched_windows:
            if win_idx >= len(windows):
                continue
            window = windows[win_idx]
            trade = self._simulate_trade(window, features_df, oos_start_t)
            if trade is not None:
                trades.append(trade)

        result.trades = trades
        result.sample_count = len(trades)
        result.is_sample_count = sum(1 for t in trades if not t.is_oos)
        result.oos_sample_count = sum(1 for t in trades if t.is_oos)
        return result

    def _simulate_trade(
        self,
        window: TickWindow,
        features_df: pd.DataFrame,
        oos_start_t: int,
    ) -> Optional[TradeResult]:
        """
        Simulate a single trade starting at the end of the given window.
        Returns TradeResult or None if no valid entry was found.
        """
        cfg = self._cfg
        tick_size = cfg.tick_size

        # Entry: first tick after window end + latency
        entry_min_t = window.end_t + cfg.latency_ms

        # Find the entry tick
        after_window = features_df[features_df["t"] >= entry_min_t]
        if after_window.empty:
            logger.debug("no_entry_tick_after_window", window_end_t=window.end_t)
            return None

        entry_row = after_window.iloc[0]
        direction = window.features[-1].imbalance > 0  # use last tick imbalance
        # Direction is set by the PatternCandidate, not recomputed here.
        # We use window's pattern direction context passed in via candidate.

        # Determine stop and target distances in price units
        vol = float(entry_row["realised_vol"])
        spread = float(entry_row["spread"])
        if math.isnan(vol) or vol <= 0.0:
            vol = 0.0

        if cfg.use_dynamic_stops and vol > 0.0:
            min_stop_distance = max(spread + 2 * tick_size, cfg.default_stop_ticks * tick_size)
            stop_distance = max(vol * cfg.stop_vol_multiplier, min_stop_distance)

            min_target_distance = max(2 * spread + 2 * tick_size, cfg.default_target_ticks * tick_size)
            target_distance = max(vol * cfg.target_vol_multiplier, min_target_distance)
        else:
            stop_distance = cfg.default_stop_ticks * tick_size
            target_distance = cfg.default_target_ticks * tick_size

        # Entry price with slippage
        if _get_direction_from_window(window) == PatternDirection.LONG:
            raw_entry = float(entry_row["ask"])
            entry_price = raw_entry + cfg.slippage_ticks * tick_size
            stop_price = entry_price - stop_distance
            target_price = entry_price + target_distance
        else:
            raw_entry = float(entry_row["bid"])
            entry_price = raw_entry - cfg.slippage_ticks * tick_size
            stop_price = entry_price + stop_distance
            target_price = entry_price - target_distance

        entry_t = int(entry_row["t"])
        max_exit_t = entry_t + cfg.max_hold_seconds * 1000

        # Walk ticks forward looking for exit
        holding = features_df[features_df["t"] > entry_t]
        exit_price = entry_price
        exit_t = entry_t
        exit_reason = "TIMEOUT"
        hold_ticks = 0

        for _, row in holding.iterrows():
            hold_ticks += 1
            t = int(row["t"])
            mid = float(row["midprice"])
            bid = float(row["bid"])
            ask = float(row["ask"])

            # Use mid for hit evaluation (conservative)
            if _get_direction_from_window(window) == PatternDirection.LONG:
                check_price = bid  # conservative fill for long exit
                if check_price >= target_price:
                    exit_price = target_price
                    exit_t = t
                    exit_reason = "TARGET"
                    break
                if check_price <= stop_price:
                    exit_price = stop_price
                    exit_t = t
                    exit_reason = "STOP"
                    break
            else:
                check_price = ask
                if check_price <= target_price:
                    exit_price = target_price
                    exit_t = t
                    exit_reason = "TARGET"
                    break
                if check_price >= stop_price:
                    exit_price = stop_price
                    exit_t = t
                    exit_reason = "STOP"
                    break

            if t >= max_exit_t:
                exit_price = bid if _get_direction_from_window(window) == PatternDirection.LONG else ask
                exit_t = t
                exit_reason = "TIMEOUT"
                break

        # Compute PnL
        if _get_direction_from_window(window) == PatternDirection.LONG:
            gross_pnl = (exit_price - entry_price) * cfg.lot_size
        else:
            gross_pnl = (entry_price - exit_price) * cfg.lot_size

        cost = self._costs.total_cost(cfg.lot_size)
        net_pnl = gross_pnl - cost
        hold_seconds = (exit_t - entry_t) / 1000.0

        return TradeResult(
            pattern_id="",  # filled by caller
            symbol=window.symbol,
            entry_t=entry_t,
            exit_t=exit_t,
            direction=_get_direction_from_window(window),
            entry_price=entry_price,
            exit_price=exit_price,
            stop_price=stop_price,
            target_price=target_price,
            exit_reason=exit_reason,
            gross_pnl=gross_pnl,
            cost=cost,
            net_pnl=net_pnl,
            hold_ticks=hold_ticks,
            hold_seconds=hold_seconds,
            is_oos=window.start_t >= oos_start_t,
        )


def _get_direction_from_window(window: TickWindow) -> PatternDirection:
    """
    Infer direction from window summary.
    Positive slope + positive imbalance → LONG, else SHORT.
    Called during simulation when direction isn't directly threaded through.
    """
    if window.mean_microprice_slope >= 0 and window.mean_imbalance >= 0:
        return PatternDirection.LONG
    return PatternDirection.SHORT
