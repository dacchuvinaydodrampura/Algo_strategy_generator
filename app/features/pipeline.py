"""
app/features/pipeline.py
------------------------
Microstructure feature pipeline.

Takes a stream of TickRecord objects and produces FeatureRecord objects.
All computation is stateful (uses rolling windows) and transparent.

Feature definitions:

1.  imbalance           - tick's own imbalance field (validated)
2.  microprice          - depth-weighted midprice using best 2 levels
3.  microprice_slope    - linear regression slope of microprice over last N ticks
4.  relative_spread     - spread / midprice
5.  total_bid_depth     - sum bq1..bq5
6.  total_ask_depth     - sum aq1..aq5
7.  depth_ratio         - total_bid / (total_bid + total_ask)
8.  aggression_score    - EWMA of (db - da) / (|db| + |da| + eps)
9.  realised_vol        - std of microprice changes over last N ticks
10. liquidity_thin      - 1 if (total_bid + total_ask) < session_25th_pct
11. momentum            - sign(slope) * |slope| normalised
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

from app.config import FeaturesConfig
from app.models.session import FeatureRecord
from app.models.tick import TickRecord
from app.utils.log_setup import get_logger

logger = get_logger(__name__)

_EPS = 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Rolling window helpers
# ──────────────────────────────────────────────────────────────────────────────


def _slope(values: list[float]) -> float:
    """
    Ordinary least squares slope of values (y) over position index (x).
    Returns 0.0 if fewer than 2 values.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    xm = x.mean()
    ym = y.mean()
    denom = float(np.sum((x - xm) ** 2))
    if denom < _EPS:
        return 0.0
    return float(np.sum((x - xm) * (y - ym)) / denom)


def _compute_microprice(tick: TickRecord, levels: int = 2) -> float:
    """
    Microprice using depth-weighted average of best `levels` bid/ask levels.

        microprice = sum(ask_p_i * bid_q_i + bid_p_i * ask_q_i, i=1..L)
                     / sum(bid_q_i + ask_q_i, i=1..L)

    Falls back to simple mid if depth data is unavailable.
    """
    total_weight = 0.0
    weighted_price = 0.0

    for i in range(1, levels + 1):
        bp = getattr(tick, f"bp{i}", None)
        bq = getattr(tick, f"bq{i}", None)
        ap = getattr(tick, f"ap{i}", None)
        aq = getattr(tick, f"aq{i}", None)
        if None in (bp, bq, ap, aq):
            break
        weighted_price += float(ap) * float(bq) + float(bp) * float(aq)
        total_weight += float(bq) + float(aq)

    if total_weight < _EPS:
        return tick.midprice  # fallback

    return weighted_price / total_weight


# ──────────────────────────────────────────────────────────────────────────────
# Feature pipeline (stateful, one instance per symbol per session)
# ──────────────────────────────────────────────────────────────────────────────


class FeaturePipeline:
    """
    Stateful feature computation engine for a single symbol.

    Usage:
        pipeline = FeaturePipeline(cfg, symbol="NIFTY26JUNFUT")
        for tick in ticks:
            feature_record = pipeline.process(tick)
            # use feature_record

    State is maintained internally via rolling deques.
    Create a new FeaturePipeline for each (symbol, session) pair.
    """

    def __init__(self, cfg: FeaturesConfig, symbol: str) -> None:
        self._cfg = cfg
        self._symbol = symbol

        # Rolling windows (deque auto-evicts oldest items)
        self._microprice_window: Deque[float] = deque(
            maxlen=cfg.slope_window_ticks
        )
        self._vol_window: Deque[float] = deque(
            maxlen=cfg.volatility_window_ticks
        )
        self._aggression_window: Deque[float] = deque(
            maxlen=cfg.aggression_window_ticks
        )
        self._depth_window: Deque[float] = deque(
            maxlen=cfg.liquidity_window_ticks
        )

        # New deques for advanced features
        self._spread_window: Deque[float] = deque(maxlen=50)
        self._rolling_depths: Deque[float] = deque(maxlen=100)
        self._bq_window: Deque[float] = deque(maxlen=50)
        self._aq_window: Deque[float] = deque(maxlen=50)
        self._vol_short_window: Deque[float] = deque(maxlen=15)
        self._vol_long_window: Deque[float] = deque(maxlen=60)

        # For EWMA aggression
        self._ewma_aggression: float = 0.0
        self._ewma_alpha: float = 2.0 / (cfg.aggression_window_ticks + 1)

        # Running 25th pct for liquidity_thin flag
        self._all_depths: list[float] = []
        self._depth_p25: float = 0.0
        self._ticks_processed: int = 0

        # Previous values for deltas
        self._prev_microprice: Optional[float] = None
        self._prev_imbalance: float = 0.0
        self._prev_microprice_slope: float = 0.0
        self._prev_bid: float = 0.0
        self._prev_ask: float = 0.0
        self._prev_bq: float = 0.0
        self._prev_aq: float = 0.0

        # EWMA states for volume deltas
        self._ewma_aggressive_burst: float = 0.0
        self._ewma_of_persistence: float = 0.0
        self._trade_alpha: float = 0.1

    def process(self, tick: TickRecord) -> FeatureRecord:
        """
        Consume one tick and return a fully populated FeatureRecord.
        Must be called in ascending-timestamp order.
        """
        self._ticks_processed += 1

        # ── Raw values ────────────────────────────────────────────────────────
        microprice = _compute_microprice(tick, self._cfg.microprice_levels)
        total_bid = tick.total_bid_depth()
        total_ask = tick.total_ask_depth()
        total_depth = total_bid + total_ask
        midprice = tick.midprice
        spread = tick.spread
        db = tick.db or 0.0
        da = tick.da or 0.0
        bq = tick.bq
        aq = tick.aq

        # ── Update windows ────────────────────────────────────────────────────
        self._microprice_window.append(microprice)

        mp_change = 0.0
        if self._prev_microprice is not None:
            mp_change = microprice - self._prev_microprice
            self._vol_window.append(mp_change)
            self._vol_short_window.append(mp_change)
            self._vol_long_window.append(mp_change)
        self._prev_microprice = microprice

        # Aggression: (db - da) / (|db| + |da| + eps)
        raw_aggression = (db - da) / (abs(db) + abs(da) + _EPS)
        self._ewma_aggression = (
            self._ewma_alpha * raw_aggression
            + (1 - self._ewma_alpha) * self._ewma_aggression
        )

        self._depth_window.append(total_depth)
        self._all_depths.append(total_depth)
        self._spread_window.append(spread)
        self._rolling_depths.append(total_depth)
        self._bq_window.append(bq)
        self._aq_window.append(aq)

        # Update depth percentile every 100 ticks (expensive if done each tick)
        if self._ticks_processed % 100 == 0 and self._all_depths:
            self._depth_p25 = float(np.percentile(self._all_depths, 25))

        # ── Derived features ──────────────────────────────────────────────────
        microprice_slope = _slope(list(self._microprice_window))
        relative_spread = spread / (midprice + _EPS)
        depth_ratio = total_bid / (total_depth + _EPS)
        realised_vol = float(np.std(self._vol_window)) if len(self._vol_window) > 1 else 0.0
        liquidity_thin = 1.0 if (self._depth_p25 > 0 and total_depth < self._depth_p25) else 0.0
        momentum = math.copysign(microprice_slope ** 2, microprice_slope)

        # Advanced features
        # 1. Multi-level imbalance
        imbalance_5 = (total_bid - total_ask) / (total_bid + total_ask + _EPS)

        # 2. Imbalance velocity
        imbalance_vel = tick.imbalance - self._prev_imbalance
        self._prev_imbalance = tick.imbalance

        # 3. Microprice acceleration
        microprice_acc = microprice_slope - self._prev_microprice_slope
        self._prev_microprice_slope = microprice_slope

        # 4. Spread regime
        rolling_median_spread = float(np.median(self._spread_window)) if self._spread_window else spread
        spread_ratio = spread / (rolling_median_spread + _EPS)

        # 5. Liquidity vacuum
        rolling_mean_depth = float(np.mean(self._rolling_depths)) if self._rolling_depths else total_depth
        liquidity_vacuum = 1.0 if total_depth < 0.5 * rolling_mean_depth else 0.0

        # 6. Queue depletion
        p10_bq = float(np.percentile(self._bq_window, 10)) if len(self._bq_window) >= 10 else 0.0
        p10_aq = float(np.percentile(self._aq_window, 10)) if len(self._aq_window) >= 10 else 0.0
        queue_depletion = 1.0 if (bq <= p10_bq or aq <= p10_aq) else 0.0

        # 7. Replenishment detection
        replenished = 0.0
        if self._ticks_processed > 1:
            if tick.bid == self._prev_bid and bq > self._prev_bq * 1.5:
                replenished = 1.0
            elif tick.ask == self._prev_ask and aq > self._prev_aq * 1.5:
                replenished = 1.0
        replenishment = replenished

        # 8. Iceberg behavior
        iceberg = 0.0
        if self._ticks_processed > 1:
            if db < 0 and tick.bid == self._prev_bid and bq >= self._prev_bq:
                iceberg = 1.0
            elif da < 0 and tick.ask == self._prev_ask and aq >= self._prev_aq:
                iceberg = 1.0
        iceberg_indicator = iceberg

        # 9. Aggressive trade bursts
        self._ewma_aggressive_burst = self._trade_alpha * (abs(db) + abs(da)) + (1 - self._trade_alpha) * self._ewma_aggressive_burst
        aggressive_burst = self._ewma_aggressive_burst

        # 10. Order flow persistence
        self._ewma_of_persistence = self._trade_alpha * (db - da) + (1 - self._trade_alpha) * self._ewma_of_persistence
        of_persistence = self._ewma_of_persistence

        # 11. Volatility clustering
        vol_short = float(np.std(self._vol_short_window)) if len(self._vol_short_window) > 1 else 0.0
        vol_long = float(np.std(self._vol_long_window)) if len(self._vol_long_window) > 1 else 0.0
        vol_clustering = vol_short / (vol_long + _EPS)

        # Update previous tick variables
        self._prev_bid = tick.bid
        self._prev_ask = tick.ask
        self._prev_bq = bq
        self._prev_aq = aq

        return FeatureRecord(
            t=tick.t,
            seq=tick.seq,
            symbol=tick.s,
            bid=tick.bid,
            ask=tick.ask,
            midprice=midprice,
            spread=spread,
            bq=bq,
            aq=aq,
            imbalance=tick.imbalance,
            microprice=microprice,
            microprice_slope=microprice_slope,
            relative_spread=relative_spread,
            total_bid_depth=total_bid,
            total_ask_depth=total_ask,
            depth_ratio=depth_ratio,
            aggression_score=self._ewma_aggression,
            realised_vol=realised_vol,
            liquidity_thin=liquidity_thin,
            momentum=momentum,
            imbalance_5=imbalance_5,
            imbalance_vel=imbalance_vel,
            microprice_acc=microprice_acc,
            spread_ratio=spread_ratio,
            liquidity_vacuum=liquidity_vacuum,
            queue_depletion=queue_depletion,
            replenishment=replenishment,
            iceberg_indicator=iceberg_indicator,
            aggressive_burst=aggressive_burst,
            of_persistence=of_persistence,
            vol_clustering=vol_clustering,
        )

    @property
    def ticks_processed(self) -> int:
        return self._ticks_processed
