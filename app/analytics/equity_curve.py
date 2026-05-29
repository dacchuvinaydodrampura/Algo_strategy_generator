"""
app/analytics/equity_curve.py
------------------------------
Equity curve construction and related time-series analytics.

Kept separate from metrics.py to maintain single-responsibility.
These functions produce data structures used by the PDF chart builder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.models.session import BacktestResult, TradeResult


@dataclass
class EquityCurveData:
    """All data needed to render the equity curve and drawdown charts."""

    trade_numbers: list[int]        # x-axis: 1, 2, 3, ...
    equity_values: list[float]      # cumulative equity
    drawdown_pct: list[float]       # drawdown percentage (positive values)
    peak_values: list[float]        # running peak
    initial_capital: float
    final_capital: float
    max_drawdown_pct: float
    max_drawdown_trade: int         # trade number where max DD occurred


def build_equity_curve(
    result: BacktestResult,
    initial_capital: float,
    is_only: bool = False,
) -> EquityCurveData:
    """
    Build equity curve data from a BacktestResult.

    Parameters
    ----------
    result:          Backtest result containing trade list.
    initial_capital: Starting capital.
    is_only:         If True, only use in-sample trades.
    """
    trades = sorted(result.trades, key=lambda t: t.entry_t)
    if is_only:
        trades = [t for t in trades if not t.is_oos]

    trade_numbers = [0]
    equity_values = [initial_capital]
    peak_values = [initial_capital]
    drawdown_pct = [0.0]

    peak = initial_capital
    max_dd = 0.0
    max_dd_trade = 0

    for i, trade in enumerate(trades, 1):
        equity = equity_values[-1] + trade.net_pnl
        if equity > peak:
            peak = equity
        dd = (peak - equity) / (peak + 1e-9) * 100.0

        trade_numbers.append(i)
        equity_values.append(equity)
        peak_values.append(peak)
        drawdown_pct.append(dd)

        if dd > max_dd:
            max_dd = dd
            max_dd_trade = i

    return EquityCurveData(
        trade_numbers=trade_numbers,
        equity_values=equity_values,
        drawdown_pct=drawdown_pct,
        peak_values=peak_values,
        initial_capital=initial_capital,
        final_capital=equity_values[-1],
        max_drawdown_pct=max_dd,
        max_drawdown_trade=max_dd_trade,
    )


@dataclass
class RegimeBucket:
    """Performance metrics for one time bucket."""

    bucket_idx: int
    label: str
    trade_count: int
    win_count: int
    win_rate: float
    gross_pnl: float
    net_pnl: float
    avg_hold_seconds: float
    is_stable: bool   # win rate within 10pp of overall win rate


def build_regime_breakdown(
    result: BacktestResult,
    n_buckets: int = 4,
) -> list[RegimeBucket]:
    """
    Split trades into N time buckets and compute per-bucket metrics.
    Used for the Regime Breakdown section of the PDF.
    """
    is_trades = sorted(
        [t for t in result.trades if not t.is_oos],
        key=lambda t: t.entry_t,
    )
    if not is_trades or n_buckets < 2:
        return []

    overall_wr = result.win_rate
    bucket_size = max(1, len(is_trades) // n_buckets)
    buckets: list[RegimeBucket] = []

    for i in range(n_buckets):
        start = i * bucket_size
        end = start + bucket_size if i < n_buckets - 1 else len(is_trades)
        bucket_trades = is_trades[start:end]

        if not bucket_trades:
            continue

        wins = sum(1 for t in bucket_trades if t.net_pnl > 0)
        wr = wins / len(bucket_trades)
        avg_hold = float(np.mean([t.hold_seconds for t in bucket_trades]))
        net_pnl = sum(t.net_pnl for t in bucket_trades)
        gross_pnl = sum(t.gross_pnl for t in bucket_trades)
        is_stable = abs(wr - overall_wr) <= 0.10  # within 10pp of overall

        # Time label: rough hour-based from first trade timestamp
        first_t_s = bucket_trades[0].entry_t / 1000
        hour = int((first_t_s % 86400) // 3600)
        minute = int((first_t_s % 3600) // 60)
        label = f"Q{i+1} (~{hour:02d}:{minute:02d})"

        buckets.append(RegimeBucket(
            bucket_idx=i + 1,
            label=label,
            trade_count=len(bucket_trades),
            win_count=wins,
            win_rate=wr,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            avg_hold_seconds=avg_hold,
            is_stable=is_stable,
        ))

    return buckets
