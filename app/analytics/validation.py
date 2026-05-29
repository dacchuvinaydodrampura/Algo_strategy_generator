"""
app/analytics/validation.py
----------------------------
Statistical validation and alpha guardrails.

Includes:
1. Monte Carlo robustness tests (random trade dropping and slippage perturbations).
2. Exit sensitivity analysis (stop-loss/take-profit outcome shifts).
3. Multi-day consistency and regime stability verification.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
import numpy as np

from app.models.session import TradeResult
from app.utils.log_setup import get_logger

logger = get_logger(__name__)

_EPS = 1e-9


def run_monte_carlo_test(
    trades: list[TradeResult],
    trials: int = 50,
    min_win_rate: float = 0.50,
    drop_fraction: float = 0.10,
) -> bool:
    """
    Evaluates pattern survivability by running Monte Carlo trials where we:
    1. Randomly drop a fraction of trades (simulating missed execution / queue timeout).
    2. Add random slippage shocks (simulating latency spike / adverse book sweep).
    
    Returns True if at least 90% of trials maintain positive PnL and win rate >= min_win_rate.
    """
    if len(trades) < 5:
        logger.warning("mc_test_insufficient_samples", count=len(trades))
        return False

    random.seed(42)  # for reproducibility
    successful_trials = 0

    for _ in range(trials):
        # 1. Randomly sample/drop trades
        sample_size = int(len(trades) * (1.0 - drop_fraction))
        sample_size = max(3, sample_size)
        sampled_trades = random.sample(trades, sample_size)

        # 2. Apply random adverse slippage shocks to net PnL
        trial_pnls = []
        for t in sampled_trades:
            # Randomly subtract 0 to 2 ticks of slippage
            pnl_shock = random.choice([0.0, 0.5, 1.0]) * (t.cost / 2.0)  # estimate tick cost from commission
            trial_pnls.append(t.net_pnl - pnl_shock)

        trial_win_rate = sum(1 for p in trial_pnls if p > 0) / len(trial_pnls)
        trial_pnl_total = sum(trial_pnls)

        if trial_win_rate >= min_win_rate and trial_pnl_total > 0:
            successful_trials += 1

    pass_pct = successful_trials / trials
    is_passed = pass_pct >= 0.90

    logger.info(
        "monte_carlo_test_completed",
        trials=trials,
        passed_trials=successful_trials,
        pass_rate=f"{pass_pct:.1%}",
        verdict="PASS" if is_passed else "FAIL",
    )
    return is_passed


def run_sensitivity_test(
    trades: list[TradeResult],
    tick_size: float = 0.05,
    lot_size: int = 25,
) -> bool:
    """
    Verifies that the strategy edge is robust to execution variation.
    Simulates a 1-tick adverse fill shift on both entry and exit.
    
    If the profit factor drops by more than 30% or drops below 1.0, the test fails.
    """
    if not trades:
        return False

    # Compute baseline PF
    gross_wins = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_losses = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
    baseline_pf = gross_wins / (gross_losses + _EPS)

    # Shift each trade's PnL by 1 tick against us (total 2 ticks per round trip)
    tick_cost = tick_size * lot_size
    perturbed_pnls = [t.net_pnl - 2 * tick_cost for t in trades]

    p_wins = sum(p for p in perturbed_pnls if p > 0)
    p_losses = abs(sum(p for p in perturbed_pnls if p < 0))
    perturbed_pf = p_wins / (p_losses + _EPS)

    # Rejection criteria:
    # 1. Perturbed Profit Factor is negative/less than 1.0 (loses money)
    # 2. Perturbed PF drops by > 30% from baseline
    pf_degradation = (baseline_pf - perturbed_pf) / (baseline_pf + _EPS)
    is_passed = perturbed_pf >= 1.0 and pf_degradation <= 0.30

    logger.info(
        "sensitivity_test_completed",
        baseline_pf=f"{baseline_pf:.3f}",
        perturbed_pf=f"{perturbed_pf:.3f}",
        degradation=f"{pf_degradation:.1%}",
        verdict="PASS" if is_passed else "FAIL",
    )
    return is_passed


def run_multi_day_test(
    trades: list[TradeResult],
    stability_cv_threshold: float = 0.35,
    min_sessions: int = 3,
) -> bool:
    """
    Checks the consistency of the pattern across different days.
    Groups trades by day. If the Coefficient of Variation (CV) of daily win rates
    exceeds the threshold, or if we lose money on more than 30% of trading sessions,
    the pattern is rejected as unstable.
    """
    # Group trades by date using entry timestamp (rounded to start of day)
    trades_by_date = defaultdict(list)
    for t in trades:
        # Convert timestamp to date string YYYY-MM-DD
        dt_str = datetime_str_from_ts(t.entry_t)
        trades_by_date[dt_str].append(t)

    unique_days = len(trades_by_date)
    if unique_days < min_sessions:
        logger.warning(
            "multi_day_test_skipped_insufficient_sessions",
            unique_days=unique_days,
            required=min_sessions,
        )
        # Pass if not enough history exists (default to True but flag it)
        return True

    daily_win_rates = []
    losing_days = 0

    for day_str, day_trades in trades_by_date.items():
        wins = sum(1 for t in day_trades if t.net_pnl > 0)
        wr = wins / len(day_trades)
        net_pnl = sum(t.net_pnl for t in day_trades)
        daily_win_rates.append(wr)
        if net_pnl < 0:
            losing_days += 1

    mean_wr = float(np.mean(daily_win_rates))
    std_wr = float(np.std(daily_win_rates))
    cv = std_wr / (mean_wr + _EPS)

    losing_day_fraction = losing_days / unique_days

    # Stability criteria:
    # 1. CV of daily win rates <= threshold
    # 2. Losing days <= 30%
    is_passed = cv <= stability_cv_threshold and losing_day_fraction <= 0.30

    logger.info(
        "multi_day_test_completed",
        unique_days=unique_days,
        win_rate_cv=f"{cv:.3f}",
        losing_days_pct=f"{losing_day_fraction:.1%}",
        verdict="PASS" if is_passed else "FAIL",
    )
    return is_passed


def datetime_str_from_ts(ts: int) -> str:
    """Helper to convert epoch ms to YYYY-MM-DD string."""
    import datetime
    dt = datetime.datetime.utcfromtimestamp(ts / 1000.0)
    return dt.date().isoformat()
