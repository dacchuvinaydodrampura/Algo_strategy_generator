"""
app/windows/tick_window.py
--------------------------
Builds rolling windows of FeatureRecords for pattern discovery.

Supported modes:
- Fixed tick count windows (every N ticks, step by S ticks)
- Fixed time windows (every N seconds)
- Event-driven windows (centred around anomaly ticks)

Output: TickWindow dataclass with summary feature statistics.
Windows are yielded as a generator — never all in RAM at once.
"""

from __future__ import annotations

import dataclasses
import math
from collections import deque
from typing import Generator, Iterator

import numpy as np

from app.models.session import FeatureRecord, TickWindow
from app.utils.log_setup import get_logger

logger = get_logger(__name__)


def _summarise_window(
    features: list[FeatureRecord],
    symbol: str,
    start_idx: int,
) -> TickWindow:
    """
    Build a TickWindow from a list of FeatureRecords.
    Computes all summary statistics in one pass.
    """
    n = len(features)
    if n == 0:
        raise ValueError("Cannot summarise an empty feature list")

    imbalances = [f.imbalance for f in features]
    slopes = [f.microprice_slope for f in features]
    aggressions = [f.aggression_score for f in features]
    rel_spreads = [f.relative_spread for f in features]
    depth_ratios = [f.depth_ratio for f in features]
    vols = [f.realised_vol for f in features]
    imbalances_5 = [f.imbalance_5 for f in features]
    imbalance_vels = [f.imbalance_vel for f in features]
    microprice_accs = [f.microprice_acc for f in features]
    spread_ratios = [f.spread_ratio for f in features]
    liquidity_vacuums = [f.liquidity_vacuum for f in features]
    queue_depletions = [f.queue_depletion for f in features]
    replenishments = [f.replenishment for f in features]
    iceberg_indicators = [f.iceberg_indicator for f in features]
    aggressive_bursts = [f.aggressive_burst for f in features]
    of_persistences = [f.of_persistence for f in features]
    vol_clusterings = [f.vol_clustering for f in features]

    return TickWindow(
        symbol=symbol,
        start_idx=start_idx,
        end_idx=start_idx + n,
        start_t=features[0].t,
        end_t=features[-1].t,
        ticks=n,
        features=features,
        mean_imbalance=float(np.mean(imbalances)),
        mean_microprice_slope=float(np.mean(slopes)),
        mean_aggression=float(np.mean(aggressions)),
        mean_relative_spread=float(np.mean(rel_spreads)),
        mean_depth_ratio=float(np.mean(depth_ratios)),
        mean_realised_vol=float(np.mean(vols)),
        entry_microprice=features[0].microprice,
        exit_microprice=features[-1].microprice,
        mean_imbalance_5=float(np.mean(imbalances_5)),
        mean_imbalance_vel=float(np.mean(imbalance_vels)),
        mean_microprice_acc=float(np.mean(microprice_accs)),
        mean_spread_ratio=float(np.mean(spread_ratios)),
        mean_liquidity_vacuum=float(np.mean(liquidity_vacuums)),
        mean_queue_depletion=float(np.mean(queue_depletions)),
        mean_replenishment=float(np.mean(replenishments)),
        mean_iceberg_indicator=float(np.mean(iceberg_indicators)),
        mean_aggressive_burst=float(np.mean(aggressive_bursts)),
        mean_of_persistence=float(np.mean(of_persistences)),
        mean_vol_clustering=float(np.mean(vol_clusterings)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Fixed-tick sliding window
# ──────────────────────────────────────────────────────────────────────────────


def fixed_tick_windows(
    feature_iter: Iterator[FeatureRecord],
    window_size: int,
    step_size: int,
    symbol: str,
    min_ticks: int = 10,
) -> Generator[TickWindow, None, None]:
    """
    Yield non-overlapping (step=window) or overlapping (step < window) windows.

    Parameters
    ----------
    feature_iter: source of FeatureRecord objects (must be time-ordered)
    window_size:  number of ticks per window
    step_size:    how many ticks to advance between windows
    symbol:       symbol name for labeling
    min_ticks:    minimum ticks required to yield a window (last window may be short)
    """
    if window_size < min_ticks:
        raise ValueError(f"window_size {window_size} < min_ticks {min_ticks}")
    if step_size < 1:
        raise ValueError("step_size must be >= 1")

    buffer: list[FeatureRecord] = []
    global_idx = 0          # absolute tick position in the day
    window_start_idx = 0
    step_counter = 0

    for fr in feature_iter:
        buffer.append(fr)
        global_idx += 1
        step_counter += 1

        if len(buffer) >= window_size and step_counter >= step_size:
            # Slice the window
            window_features = buffer[-window_size:]
            yield _summarise_window(window_features, symbol, window_start_idx)
            window_start_idx += step_size
            step_counter = 0

            # Keep only what we need in memory
            # For non-overlapping windows, clear; for overlapping, keep tail
            overlap = window_size - step_size
            if overlap > 0:
                buffer = buffer[-overlap:]
            else:
                buffer.clear()

    # Final partial window
    if len(buffer) >= min_ticks:
        yield _summarise_window(buffer, symbol, window_start_idx)


# ──────────────────────────────────────────────────────────────────────────────
# Fixed-time sliding window
# ──────────────────────────────────────────────────────────────────────────────


def fixed_time_windows(
    feature_iter: Iterator[FeatureRecord],
    window_seconds: float,
    symbol: str,
    min_ticks: int = 10,
) -> Generator[TickWindow, None, None]:
    """
    Yield windows of fixed time duration.

    Windows are emitted when the current tick's timestamp exceeds
    window_start_t + window_seconds.
    """
    window_ms = int(window_seconds * 1000)
    buffer: list[FeatureRecord] = []
    window_start_t: int | None = None
    start_idx = 0
    global_idx = 0

    for fr in feature_iter:
        if window_start_t is None:
            window_start_t = fr.t

        # Check if we've passed the end of this time window
        if fr.t >= window_start_t + window_ms:
            if len(buffer) >= min_ticks:
                yield _summarise_window(buffer, symbol, start_idx)
            start_idx = global_idx
            buffer = [fr]
            window_start_t = fr.t
        else:
            buffer.append(fr)

        global_idx += 1

    # Emit the remaining partial window
    if len(buffer) >= min_ticks:
        yield _summarise_window(buffer, symbol, start_idx)
