"""
app/patterns/ranker.py
-----------------------
Ranks, deduplicates, and filters PatternCandidate lists before backtesting.

Problems this solves:
- Rule mining can produce hundreds of near-identical candidates.
- Backtesting all of them wastes time and inflates multiple-testing risk.
- We rank by a composite quality score and deduplicate by rule overlap.

Ranking criteria (all IS-only, before backtest):
1. Sample count           — more samples = more reliable
2. Win-rate estimate      — derived from window forward returns
3. Profit-factor estimate — derived from window forward returns

Deduplication:
- Two candidates are considered duplicates if they share >= 80% of
  matched windows. The higher-scoring one is kept.

The returned list is capped at MAX_CANDIDATES_TO_BACKTEST to prevent
the backtest stage from running thousands of patterns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.config import PatternsConfig
from app.models.session import PatternCandidate, PatternDirection, TickWindow
from app.utils.log_setup import get_logger

logger = get_logger(__name__)

_MAX_CANDIDATES_TO_BACKTEST = 20
_DUPLICATE_OVERLAP_THRESHOLD = 0.80


@dataclass
class _ScoredCandidate:
    candidate: PatternCandidate
    score: float


def _jaccard_overlap(set_a: set[int], set_b: set[int]) -> float:
    """Jaccard similarity between two sets of window indices."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _estimate_quality(
    candidate: PatternCandidate,
    windows: list[TickWindow],
    oos_start_t: int,
) -> tuple[float, float]:
    """
    Estimate win rate and profit factor from matched IS windows
    using microprice forward returns.

    Returns (win_rate, profit_factor) estimated from window data only.
    This is a fast pre-screen — the real metrics come from backtesting.
    """
    matched_is = [
        idx for idx in candidate.matched_windows
        if idx < len(windows) and windows[idx].start_t < oos_start_t
    ]

    if len(matched_is) < 3:
        return 0.0, 0.0

    wins, gross_wins, gross_losses = 0, 0.0, 0.0

    for win_idx in matched_is:
        if win_idx + 1 >= len(windows):
            continue
        this_win = windows[win_idx]
        next_win = windows[win_idx + 1]
        fwd_return = next_win.entry_microprice - this_win.exit_microprice

        if candidate.direction == PatternDirection.SHORT:
            fwd_return = -fwd_return

        if fwd_return > 0:
            wins += 1
            gross_wins += fwd_return
        else:
            gross_losses += abs(fwd_return)

    n = len(matched_is)
    wr = wins / n if n > 0 else 0.0
    pf = gross_wins / (gross_losses + 1e-9)
    return wr, pf


def rank_and_deduplicate(
    candidates: list[PatternCandidate],
    windows: list[TickWindow],
    oos_start_t: int,
    cfg: PatternsConfig,
) -> list[PatternCandidate]:
    """
    Score, deduplicate, and return the top-N candidates for backtesting.

    Parameters
    ----------
    candidates:   Raw candidates from rule_miner + cluster_miner.
    windows:      All TickWindows (for forward-return estimation).
    oos_start_t:  OOS boundary timestamp.
    cfg:          Patterns config for quality thresholds.

    Returns
    -------
    Ranked and deduplicated list, capped at MAX_CANDIDATES_TO_BACKTEST.
    """
    if not candidates:
        return []

    logger.info("ranker_input", total_candidates=len(candidates))

    # ── Score each candidate ──────────────────────────────────────────────────
    scored: list[_ScoredCandidate] = []

    for c in candidates:
        wr, pf = _estimate_quality(c, windows, oos_start_t)

        # Composite score: harmonic-ish blend of sample size, WR, PF
        # Logarithm of sample count normalises size differences
        sample_score = math.log1p(c.sample_count) / math.log1p(cfg.min_samples * 10)
        wr_score = max(0.0, wr - cfg.min_win_rate) / (1.0 - cfg.min_win_rate + 1e-9)
        pf_score = max(0.0, pf - cfg.min_profit_factor) / (3.0 - cfg.min_profit_factor + 1e-9)

        composite = 0.4 * sample_score + 0.4 * wr_score + 0.2 * pf_score
        # Penalty for rule complexity to prevent overfitting
        complexity_penalty = 0.05 * max(0, len(c.rules) - 1)
        composite -= complexity_penalty
        scored.append(_ScoredCandidate(candidate=c, score=composite))

    # ── Sort descending by score ──────────────────────────────────────────────
    scored.sort(key=lambda s: -s.score)

    # ── Deduplicate by matched-window overlap ─────────────────────────────────
    kept: list[_ScoredCandidate] = []
    kept_sets: list[set[int]] = []

    for sc in scored:
        win_set = set(sc.candidate.matched_windows)
        is_duplicate = False

        for ks in kept_sets:
            if _jaccard_overlap(win_set, ks) >= _DUPLICATE_OVERLAP_THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(sc)
            kept_sets.append(win_set)

    # ── Cap at maximum ────────────────────────────────────────────────────────
    final = [sc.candidate for sc in kept[:_MAX_CANDIDATES_TO_BACKTEST]]

    logger.info(
        "ranker_output",
        before_dedup=len(scored),
        after_dedup=len(kept),
        final=len(final),
    )
    return final
