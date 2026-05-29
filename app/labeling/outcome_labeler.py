"""
app/labeling/outcome_labeler.py
--------------------------------
Computes forward outcome labels for each TickWindow.

Labels are assigned based on what happens AFTER the window ends:
- forward_return:   microprice change over next N ticks
- target_hit:       did price hit the target before stop?
- stop_hit:         did price hit stop before target?
- timeout:          neither hit within max_hold_seconds

The labeling logic is deterministic and uses tick data only.
Splits are always time-based (IS = first 70%, OOS = last 30%).
Random splits are never used.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

from app.models.session import FeatureRecord, TickWindow
from app.utils.log_setup import get_logger

logger = get_logger(__name__)


@dataclass
class WindowLabel:
    """Forward outcome label for one TickWindow."""

    window_idx: int
    symbol: str
    start_t: int

    # Forward return (microprice change over next N ticks)
    forward_return_5: float   # 5 ticks ahead
    forward_return_10: float  # 10 ticks ahead
    forward_return_20: float  # 20 ticks ahead

    # Binary outcomes
    target_hit: bool
    stop_hit: bool
    timeout: bool

    # Direction implied by label
    bullish: bool   # forward_return_10 > 0
    bearish: bool   # forward_return_10 < 0

    is_oos: bool


def label_windows(
    windows: list[TickWindow],
    features: list[FeatureRecord],
    oos_start_t: int,
    target_ticks: int = 10,
    stop_ticks: int = 5,
    tick_size: float = 0.05,
    max_forward_ticks: int = 20,
) -> list[WindowLabel]:
    """
    Label all windows with their forward outcomes.

    Uses the feature list to look up post-window tick data.
    Only labelled windows whose exit_idx falls within the feature array.
    """
    if not windows or not features:
        return []

    labels: list[WindowLabel] = []
    n_features = len(features)

    for win_idx, window in enumerate(windows):
        exit_feat_idx = window.end_idx  # features are aligned with tick positions

        if exit_feat_idx >= n_features - max_forward_ticks:
            continue  # not enough forward data to label

        entry_mp = window.exit_microprice

        # Compute forward returns at 5, 10, 20 ticks
        def _fwd_return(n: int) -> float:
            idx = min(exit_feat_idx + n, n_features - 1)
            return features[idx].microprice - entry_mp

        fwd5 = _fwd_return(5)
        fwd10 = _fwd_return(10)
        fwd20 = _fwd_return(20)

        # Target / stop evaluation (long-biased check)
        target_delta = target_ticks * tick_size
        stop_delta = stop_ticks * tick_size
        target_hit = False
        stop_hit = False

        for i in range(1, max_forward_ticks + 1):
            idx = exit_feat_idx + i
            if idx >= n_features:
                break
            mp = features[idx].microprice
            change = mp - entry_mp
            if change >= target_delta:
                target_hit = True
                break
            if change <= -stop_delta:
                stop_hit = True
                break

        labels.append(WindowLabel(
            window_idx=win_idx,
            symbol=window.symbol,
            start_t=window.start_t,
            forward_return_5=fwd5,
            forward_return_10=fwd10,
            forward_return_20=fwd20,
            target_hit=target_hit,
            stop_hit=stop_hit,
            timeout=not target_hit and not stop_hit,
            bullish=fwd10 > 0,
            bearish=fwd10 < 0,
            is_oos=window.start_t >= oos_start_t,
        ))

    logger.info(
        "labeling_complete",
        total_windows=len(windows),
        labelled=len(labels),
        bullish=sum(1 for lb in labels if lb.bullish),
        bearish=sum(1 for lb in labels if lb.bearish),
        target_hit=sum(1 for lb in labels if lb.target_hit),
        stop_hit=sum(1 for lb in labels if lb.stop_hit),
    )
    return labels
