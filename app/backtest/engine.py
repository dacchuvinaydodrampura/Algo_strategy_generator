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


def _simulate_sweep_fill(
    row: pd.Series,
    direction: PatternDirection,
    size: float,
    tick_size: float,
    is_exit: bool = False,
) -> float:
    """
    Simulates sweeping the order book levels 1-5 to compute a volume-weighted fill price.
    For entries:
        LONG buys from ASK side (ap1..ap5, aq1..aq5)
        SHORT sells to BID side (bp1..bp5, bq1..bq5)
    For exits:
        LONG sells to BID side (bp1..bp5, bq1..bq5)
        SHORT buys from ASK side (ap1..ap5, aq1..aq5)
    """
    is_buying = (direction == PatternDirection.LONG and not is_exit) or (direction == PatternDirection.SHORT and is_exit)
    prefix = "a" if is_buying else "b"

    total_val = 0.0
    remaining = size

    for i in range(1, 6):
        p_val = getattr(row, f"{prefix}p{i}", None)
        q_val = getattr(row, f"{prefix}q{i}", None)
        if p_val is None or q_val is None or math.isnan(p_val) or math.isnan(q_val) or q_val <= 0:
            break
        take = min(remaining, q_val)
        total_val += p_val * take
        remaining -= take
        if remaining <= 0:
            break

    if remaining > 0:
        # Fallback to touch price with a 5-tick penalty for remaining size
        touch_p = getattr(row, f"{prefix}p1", row["ask"] if is_buying else row["bid"])
        penalty = 5.0 * tick_size
        avg_p = touch_p + penalty if is_buying else touch_p - penalty
        total_val += avg_p * remaining

    return total_val / size


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
        last_exit_idx = -999999

        for win_idx in candidate.matched_windows:
            if win_idx >= len(windows):
                continue
            window = windows[win_idx]

            # Cooldown check
            if window.start_idx < last_exit_idx + self._cfg.cooldown_ticks:
                continue

            trade = self._simulate_trade(window, candidate.direction, features_df, oos_start_t)
            if trade is not None:
                trades.append(trade)
                # Enforce cooldown using actual holding ticks
                last_exit_idx = window.start_idx + trade.hold_ticks

        result.trades = trades
        result.sample_count = len(trades)
        result.is_sample_count = sum(1 for t in trades if not t.is_oos)
        result.oos_sample_count = sum(1 for t in trades if t.is_oos)
        return result

    def _simulate_trade(
        self,
        window: TickWindow,
        direction: PatternDirection,
        features_df: pd.DataFrame,
        oos_start_t: int,
    ) -> Optional[TradeResult]:
        """
        Simulate a single trade starting at the end of the given window in specified direction.
        Returns TradeResult or None if no valid entry was found.
        """
        cfg = self._cfg
        tick_size = cfg.tick_size

        # Session boundaries check
        session_start_t = int(features_df["t"].iloc[0])
        session_end_t = int(features_df["t"].iloc[-1])

        # Entry: first tick after window end + latency
        entry_min_t = window.end_t + cfg.latency_ms

        # Find the entry tick
        after_window = features_df[features_df["t"] >= entry_min_t]
        if after_window.empty:
            logger.debug("no_entry_tick_after_window", window_end_t=window.end_t)
            return None

        entry_row = after_window.iloc[0]
        entry_t_start = int(entry_row["t"])

        # Exclude entries near session boundaries (within 10 minutes)
        # Only apply if session duration is substantial (e.g. > 1 hour)
        session_duration = session_end_t - session_start_t
        if session_duration > 60 * 60 * 1000:
            if entry_t_start < session_start_t + 10 * 60 * 1000 or entry_t_start > session_end_t - 10 * 60 * 1000:
                return None

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

        # Entry execution
        if cfg.entry_order_type == "limit":
            # Place limit order at best touch price (bid for LONG, ask for SHORT)
            limit_price = float(entry_row["bid"]) if direction == PatternDirection.LONG else float(entry_row["ask"])
            queue_pos = (float(entry_row["bq"]) if direction == PatternDirection.LONG else float(entry_row["aq"])) * cfg.queue_position_multiplier

            filled = False
            entry_price = limit_price
            entry_t = entry_t_start

            # Walk future ticks to simulate limit fill
            limit_holding = features_df[features_df["t"] >= entry_t_start]
            for _, row in limit_holding.iterrows():
                t = int(row["t"])
                bid = float(row["bid"])
                ask = float(row["ask"])
                db = float(row["db"] or 0.0)
                da = float(row["da"] or 0.0)

                # Check limit order timeout
                if (t - entry_t_start) > cfg.limit_order_timeout_seconds * 1000:
                    break

                # Queue and price checks
                if direction == PatternDirection.LONG:
                    if ask <= limit_price:
                        filled = True
                        entry_t = t
                        break
                    if bid > limit_price:
                        break  # price moved away, cancel
                    if bid == limit_price:
                        queue_pos -= max(0.0, -db)
                        if queue_pos <= 0.0:
                            filled = True
                            entry_t = t
                            break
                else:  # SHORT
                    if bid >= limit_price:
                        filled = True
                        entry_t = t
                        break
                    if ask < limit_price:
                        break  # price moved away, cancel
                    if ask == limit_price:
                        queue_pos -= max(0.0, -da)
                        if queue_pos <= 0.0:
                            filled = True
                            entry_t = t
                            break

            if not filled:
                return None  # entry unfilled, skip trade

        else:  # market order: sweeps order book levels instantly
            entry_price = _simulate_sweep_fill(entry_row, direction, cfg.lot_size, tick_size, is_exit=False)
            entry_price = entry_price + (cfg.slippage_ticks * tick_size if direction == PatternDirection.LONG else -cfg.slippage_ticks * tick_size)
            entry_t = entry_t_start

        # Stop and target prices from entry price
        if direction == PatternDirection.LONG:
            stop_price = entry_price - stop_distance
            target_price = entry_price + target_distance
        else:
            stop_price = entry_price + stop_distance
            target_price = entry_price - target_distance

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
            bid = float(row["bid"])
            ask = float(row["ask"])

            if direction == PatternDirection.LONG:
                check_price = bid  # conservative exit bid
                if check_price >= target_price:
                    exit_price = _simulate_sweep_fill(row, direction, cfg.lot_size, tick_size, is_exit=True)
                    exit_price = max(target_price, exit_price)
                    exit_t = t
                    exit_reason = "TARGET"
                    break
                if check_price <= stop_price:
                    exit_price = _simulate_sweep_fill(row, direction, cfg.lot_size, tick_size, is_exit=True)
                    exit_price = min(stop_price, exit_price)
                    exit_t = t
                    exit_reason = "STOP"
                    break
            else:
                check_price = ask
                if check_price <= target_price:
                    exit_price = _simulate_sweep_fill(row, direction, cfg.lot_size, tick_size, is_exit=True)
                    exit_price = min(target_price, exit_price)
                    exit_t = t
                    exit_reason = "TARGET"
                    break
                if check_price >= stop_price:
                    exit_price = _simulate_sweep_fill(row, direction, cfg.lot_size, tick_size, is_exit=True)
                    exit_price = max(stop_price, exit_price)
                    exit_t = t
                    exit_reason = "STOP"
                    break

            if t >= max_exit_t:
                exit_price = _simulate_sweep_fill(row, direction, cfg.lot_size, tick_size, is_exit=True)
                exit_t = t
                exit_reason = "TIMEOUT"
                break

        # Compute PnL
        if direction == PatternDirection.LONG:
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
            direction=direction,
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
