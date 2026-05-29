"""
app/analytics/regime.py
------------------------
Classifies market regimes from tick-level features and evaluates
pattern performance within each regime.

Regime classification (rule-based, not ML):
  - TRENDING_UP:   microprice_slope > high_threshold
  - TRENDING_DOWN: microprice_slope < -high_threshold
  - VOLATILE:      realised_vol > vol_threshold
  - THIN:          liquidity_thin == 1.0
  - NORMAL:        none of the above

Each TickWindow is assigned a dominant regime based on its feature summary.
BacktestResult trades are then split by the regime of their entry window.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from app.models.session import BacktestResult, TickWindow, TradeResult
from app.utils.log_setup import get_logger

logger = get_logger(__name__)


class Regime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    VOLATILE = "VOLATILE"
    THIN = "THIN"
    NORMAL = "NORMAL"


@dataclass
class RegimePerformance:
    """Pattern performance within one market regime."""

    regime: Regime
    trade_count: int
    win_rate: float
    net_pnl: float
    profit_factor: float


def classify_window(
    window: TickWindow,
    slope_threshold: float = 0.002,
    vol_threshold: float = 0.15,
) -> Regime:
    """
    Classify a single window into a market regime.

    Parameters
    ----------
    slope_threshold: microprice_slope magnitude above which we call it trending.
    vol_threshold:   realised_vol above which we call it volatile.
    """
    if window.mean_liquidity_thin if hasattr(window, 'mean_liquidity_thin') else False:
        return Regime.THIN
    if window.mean_realised_vol > vol_threshold:
        return Regime.VOLATILE
    if window.mean_microprice_slope > slope_threshold:
        return Regime.TRENDING_UP
    if window.mean_microprice_slope < -slope_threshold:
        return Regime.TRENDING_DOWN
    return Regime.NORMAL


def analyse_regime_performance(
    result: BacktestResult,
    windows: list[TickWindow],
) -> list[RegimePerformance]:
    """
    Split trades by entry-window regime and compute per-regime metrics.

    Matching is by trade.entry_t to the closest window start_t.
    """
    if not result.trades or not windows:
        return []

    # Build a quick t → window mapping
    window_by_start_t: dict[int, TickWindow] = {w.start_t: w for w in windows}
    sorted_starts = sorted(window_by_start_t.keys())

    def _find_window(entry_t: int) -> Optional[TickWindow]:
        """Find the window that was active at entry_t."""
        for start in reversed(sorted_starts):
            if start <= entry_t:
                return window_by_start_t[start]
        return None

    regime_trades: dict[Regime, list[TradeResult]] = {r: [] for r in Regime}

    for trade in result.trades:
        win = _find_window(trade.entry_t)
        if win is None:
            regime_trades[Regime.NORMAL].append(trade)
        else:
            r = classify_window(win)
            regime_trades[r].append(trade)

    performances: list[RegimePerformance] = []
    for regime, trades in regime_trades.items():
        if not trades:
            continue
        wins = sum(1 for t in trades if t.net_pnl > 0)
        wr = wins / len(trades)
        net_pnl = sum(t.net_pnl for t in trades)
        gross_wins = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        gross_losses = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
        pf = gross_wins / (gross_losses + 1e-9)

        performances.append(RegimePerformance(
            regime=regime,
            trade_count=len(trades),
            win_rate=wr,
            net_pnl=net_pnl,
            profit_factor=pf,
        ))

    logger.info(
        "regime_analysis_complete",
        pattern_id=result.pattern_id,
        regimes_with_trades=len(performances),
    )
    return sorted(performances, key=lambda p: -p.trade_count)
