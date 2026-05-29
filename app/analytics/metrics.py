"""
app/analytics/metrics.py
-------------------------
Computes all performance metrics for a BacktestResult.

Metrics are computed once and stored back on the BacktestResult object.
No metric is hallucinated: if there are no trades, metrics are 0 or NaN,
not fabricated.

Regime breakdown splits the trading day into N buckets and measures
win rate stability across those buckets (used for stability filtering).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from app.config import BacktestConfig, PatternsConfig
from app.models.session import BacktestResult, PatternDirection, TradeResult
from app.utils.log_setup import get_logger
from app.analytics.validation import run_monte_carlo_test, run_sensitivity_test, run_multi_day_test

logger = get_logger(__name__)

_EPS = 1e-9
_N_REGIME_BUCKETS = 4   # split day into 4 time buckets for stability


def populate_metrics(
    result: BacktestResult,
    patterns_cfg: PatternsConfig,
    bt_cfg: BacktestConfig,
) -> BacktestResult:
    """
    Compute and populate all metrics on `result`.
    Modifies result in-place and returns it.
    """
    all_trades = result.trades
    is_trades = [t for t in all_trades if not t.is_oos]
    oos_trades = [t for t in all_trades if t.is_oos]

    result.sample_count = len(all_trades)
    result.is_sample_count = len(is_trades)
    result.oos_sample_count = len(oos_trades)

    if not is_trades:
        result.verdict = "REJECTED"
        result.rejection_reason = "No in-sample trades produced"
        return result

    # ── IS metrics ────────────────────────────────────────────────────────────
    result.win_rate = _win_rate(is_trades, result.direction)
    result.profit_factor = _profit_factor(is_trades, result.direction)
    result.expectancy = _expectancy(is_trades)
    result.avg_win = _avg_win(is_trades, result.direction)
    result.avg_loss = _avg_loss(is_trades, result.direction)
    result.total_gross_pnl = sum(t.gross_pnl for t in all_trades)
    result.total_net_pnl = sum(t.net_pnl for t in all_trades)
    result.total_costs = sum(t.cost for t in all_trades)

    # ── Equity curve + drawdown ───────────────────────────────────────────────
    equity_curve = _equity_curve(is_trades, bt_cfg.initial_capital)
    result.max_drawdown = _max_drawdown(equity_curve)
    result.sharpe_ratio = _sharpe_ratio(is_trades)

    # ── OOS metrics ───────────────────────────────────────────────────────────
    if oos_trades:
        result.oos_win_rate = _win_rate(oos_trades, result.direction)
        result.oos_profit_factor = _profit_factor(oos_trades, result.direction)
    else:
        result.oos_win_rate = float("nan")
        result.oos_profit_factor = float("nan")

    # ── Stability (CV of win rate across time buckets) ────────────────────────
    result.win_rate_cv = _stability_cv(is_trades, result.direction)
    result.is_stable = result.win_rate_cv <= patterns_cfg.stability_cv_threshold

    # ── Advanced Statistical Validation Checks ──────────────────────────────
    # Monte Carlo simulation
    result.mc_pass = run_monte_carlo_test(
        trades=is_trades,
        trials=patterns_cfg.mc_trials,
        min_win_rate=patterns_cfg.min_win_rate,
    )
    # Sensitivity analysis
    result.sensitivity_pass = run_sensitivity_test(
        trades=is_trades,
        tick_size=bt_cfg.tick_size,
        lot_size=bt_cfg.lot_size,
    )
    # Multi-day cross stability (assesses inter-day consistency on all trades)
    result.multi_day_pass = run_multi_day_test(
        trades=all_trades,
        stability_cv_threshold=patterns_cfg.stability_cv_threshold,
        min_sessions=patterns_cfg.min_multi_day_sessions,
    )

    # ── Verdict ───────────────────────────────────────────────────────────────
    result.verdict, result.rejection_reason = _compute_verdict(
        result, patterns_cfg
    )

    logger.info(
        "metrics_computed",
        pattern_id=result.pattern_id,
        is_trades=len(is_trades),
        oos_trades=len(oos_trades),
        win_rate=f"{result.win_rate:.2%}",
        profit_factor=f"{result.profit_factor:.2f}",
        verdict=result.verdict,
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Individual metric functions
# ──────────────────────────────────────────────────────────────────────────────


def _win_rate(trades: list[TradeResult], direction: PatternDirection) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if _is_win(t, direction))
    return wins / len(trades)


def _is_win(trade: TradeResult, direction: PatternDirection) -> bool:
    return trade.net_pnl > 0


def _profit_factor(trades: list[TradeResult], direction: PatternDirection) -> float:
    gross_wins = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_losses = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
    return gross_wins / (gross_losses + _EPS)


def _expectancy(trades: list[TradeResult]) -> float:
    if not trades:
        return 0.0
    return sum(t.net_pnl for t in trades) / len(trades)


def _avg_win(trades: list[TradeResult], direction: PatternDirection) -> float:
    wins = [t.net_pnl for t in trades if t.net_pnl > 0]
    return float(np.mean(wins)) if wins else 0.0


def _avg_loss(trades: list[TradeResult], direction: PatternDirection) -> float:
    losses = [t.net_pnl for t in trades if t.net_pnl < 0]
    return float(np.mean(losses)) if losses else 0.0


def _equity_curve(
    trades: list[TradeResult], initial_capital: float
) -> list[float]:
    curve = [initial_capital]
    running = initial_capital
    for t in sorted(trades, key=lambda x: x.entry_t):
        running += t.net_pnl
        curve.append(running)
    return curve


def _max_drawdown(equity_curve: list[float]) -> float:
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / (peak + _EPS)
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _sharpe_ratio(trades: list[TradeResult]) -> float:
    """Simplified trade-level Sharpe: mean(pnl) / std(pnl)."""
    if len(trades) < 2:
        return 0.0
    pnls = np.array([t.net_pnl for t in trades])
    std = float(np.std(pnls))
    if std < _EPS:
        return 0.0
    return float(np.mean(pnls) / std)


def _stability_cv(
    trades: list[TradeResult], direction: PatternDirection
) -> float:
    """
    Coefficient of variation of win rate across N time buckets.
    High CV → unstable pattern (regime-dependent).
    """
    if len(trades) < _N_REGIME_BUCKETS * 2:
        return 1.0  # conservative: too few trades to assess stability

    sorted_trades = sorted(trades, key=lambda x: x.entry_t)
    bucket_size = len(sorted_trades) // _N_REGIME_BUCKETS
    win_rates = []

    for i in range(_N_REGIME_BUCKETS):
        start = i * bucket_size
        end = start + bucket_size if i < _N_REGIME_BUCKETS - 1 else len(sorted_trades)
        bucket = sorted_trades[start:end]
        if bucket:
            wr = _win_rate(bucket, direction)
            win_rates.append(wr)

    if not win_rates or np.mean(win_rates) < _EPS:
        return 1.0

    return float(np.std(win_rates) / (np.mean(win_rates) + _EPS))


def _compute_verdict(
    result: BacktestResult,
    cfg: PatternsConfig,
) -> tuple[str, str]:
    """
    Determine ACCEPTED / MARGINAL / REJECTED and reason.
    """
    if result.sample_count < cfg.min_samples:
        return "REJECTED", f"Too few trades: {result.sample_count} < {cfg.min_samples}"

    if result.win_rate < cfg.min_win_rate:
        return "REJECTED", f"Win rate {result.win_rate:.2%} < threshold {cfg.min_win_rate:.2%}"

    if result.profit_factor < cfg.min_profit_factor:
        return (
            "REJECTED",
            f"Profit factor {result.profit_factor:.2f} < threshold {cfg.min_profit_factor:.2f}",
        )

    # Rejection based on validation failure
    if not result.mc_pass:
        return "REJECTED", "Failed Monte Carlo robustness test"

    if not result.sensitivity_pass:
        return "REJECTED", "Failed parameter sensitivity analysis"

    if not result.multi_day_pass:
        return "REJECTED", "Failed multi-day consistency check"

    if not result.is_stable:
        return (
            "MARGINAL",
            f"Unstable pattern: win-rate CV={result.win_rate_cv:.2f} > {cfg.stability_cv_threshold:.2f}",
        )

    # OOS degradation check
    if not math.isnan(result.oos_win_rate):
        degradation = result.win_rate - result.oos_win_rate
        if degradation > 0.15:  # >15 percentage point degradation
            return (
                "MARGINAL",
                f"OOS win rate {result.oos_win_rate:.2%} is {degradation:.1%} below IS {result.win_rate:.2%}",
            )

    return "ACCEPTED", ""


def compute_regime_breakdown(
    result: BacktestResult,
) -> list[dict[str, object]]:
    """
    Return per-time-bucket performance breakdown for the PDF report.
    """
    if not result.trades:
        return []

    sorted_trades = sorted(result.trades, key=lambda x: x.entry_t)
    bucket_size = max(1, len(sorted_trades) // _N_REGIME_BUCKETS)
    breakdown = []

    for i in range(_N_REGIME_BUCKETS):
        start = i * bucket_size
        end = start + bucket_size if i < _N_REGIME_BUCKETS - 1 else len(sorted_trades)
        bucket = sorted_trades[start:end]
        if not bucket:
            continue

        wins = sum(1 for t in bucket if t.net_pnl > 0)
        breakdown.append({
            "bucket": i + 1,
            "trade_count": len(bucket),
            "win_rate": wins / len(bucket),
            "net_pnl": sum(t.net_pnl for t in bucket),
            "avg_hold_s": float(np.mean([t.hold_seconds for t in bucket])),
        })

    return breakdown
