"""
app/models/session.py
---------------------
Models for session-level metadata, archive manifests, feature records,
and window/pattern containers.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Archive / session metadata
# ──────────────────────────────────────────────────────────────────────────────


class ArchiveManifest(BaseModel):
    """
    Metadata about one processed .tar.gz archive.
    Created during ingestion; persisted to storage.
    """

    session_date: datetime.date
    archive_path: str
    archive_size_bytes: int
    symbols: list[str] = Field(default_factory=list)
    has_system_file: bool = False
    total_ticks: dict[str, int] = Field(default_factory=dict)   # symbol -> count
    rejected_ticks: dict[str, int] = Field(default_factory=dict)
    gap_count: int = 0
    significant_gap_count: int = 0
    total_gap_seconds: float = 0.0
    validation_passed: bool = True
    validation_errors: list[str] = Field(default_factory=list)
    ingestion_started_at: Optional[datetime.datetime] = None
    ingestion_finished_at: Optional[datetime.datetime] = None

    @property
    def total_tick_count(self) -> int:
        return sum(self.total_ticks.values())

    @property
    def total_rejected_count(self) -> int:
        return sum(self.rejected_ticks.values())

    @property
    def rejection_rate(self) -> float:
        total = self.total_tick_count + self.total_rejected_count
        if total == 0:
            return 0.0
        return self.total_rejected_count / total


# ──────────────────────────────────────────────────────────────────────────────
# Feature record (one row per tick, enriched with microstructure features)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class FeatureRecord:
    """
    Microstructure features computed for a single tick.
    This is a flat dataclass for efficient columnar storage.
    All feature values are floats; NaN means 'not computable at this tick'.
    """

    # Identity
    t: int                          # epoch ms
    seq: int
    symbol: str

    # Raw tick features
    bid: float
    ask: float
    midprice: float
    spread: float
    bq: float                       # qty at best bid
    aq: float                       # qty at best ask

    # Computed features
    imbalance: float                # (bq - aq) / (bq + aq)  from tick, validated
    microprice: float               # weighted mid using best-2-level depth
    microprice_slope: float         # slope of microprice over last N ticks
    relative_spread: float          # spread / midprice
    total_bid_depth: float          # sum of bq1..bq5
    total_ask_depth: float          # sum of aq1..aq5
    depth_ratio: float              # total_bid / (total_bid + total_ask)
    aggression_score: float         # smoothed (db - da) / (|db| + |da| + eps)
    realised_vol: float             # std of microprice changes over window
    liquidity_thin: float           # 1 if total depth < threshold else 0
    momentum: float                 # sign of microprice slope * magnitude


# ──────────────────────────────────────────────────────────────────────────────
# Window
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TickWindow:
    """
    A fixed slice of tick data with pre-computed features.
    Used as the unit of analysis for pattern discovery.
    """

    symbol: str
    start_idx: int              # position in the day's tick array
    end_idx: int                # exclusive
    start_t: int                # epoch ms of first tick
    end_t: int                  # epoch ms of last tick
    ticks: int                  # number of ticks in window
    features: list[FeatureRecord] = field(default_factory=list)

    # Summary stats for the window (computed from features)
    mean_imbalance: float = 0.0
    mean_microprice_slope: float = 0.0
    mean_aggression: float = 0.0
    mean_relative_spread: float = 0.0
    mean_depth_ratio: float = 0.0
    mean_realised_vol: float = 0.0
    entry_microprice: float = 0.0
    exit_microprice: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Pattern definition and backtest result
# ──────────────────────────────────────────────────────────────────────────────


class PatternDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass
class PatternRule:
    """
    A single threshold rule on a feature.
    e.g.  imbalance > 0.40  AND  microprice_slope > 0.0
    """

    feature: str
    operator: str       # ">" | "<" | ">=" | "<=" | "=="
    threshold: float

    def matches(self, value: float) -> bool:
        ops = {
            ">": value > self.threshold,
            "<": value < self.threshold,
            ">=": value >= self.threshold,
            "<=": value <= self.threshold,
            "==": abs(value - self.threshold) < 1e-9,
        }
        return ops.get(self.operator, False)

    def describe(self) -> str:
        return f"{self.feature} {self.operator} {self.threshold:.4f}"


@dataclass
class PatternCandidate:
    """
    A discovered pattern definition before backtesting.
    """

    pattern_id: str
    symbol: str
    direction: PatternDirection
    rules: list[PatternRule] = field(default_factory=list)
    matched_windows: list[int] = field(default_factory=list)  # window indices
    sample_count: int = 0
    discovery_method: str = "rule_mining"   # rule_mining | clustering | motif
    description: str = ""

    def describe(self) -> str:
        rule_str = " AND ".join(r.describe() for r in self.rules)
        return f"[{self.pattern_id}] {self.direction.value}: {rule_str}"


@dataclass
class TradeResult:
    """Outcome of a single simulated trade."""

    pattern_id: str
    symbol: str
    entry_t: int
    exit_t: int
    direction: PatternDirection
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    exit_reason: str        # "TARGET" | "STOP" | "TIMEOUT" | "EOD"
    gross_pnl: float
    cost: float
    net_pnl: float
    hold_ticks: int
    hold_seconds: float
    is_oos: bool = False    # True if this trade is in out-of-sample period


@dataclass
class BacktestResult:
    """
    Full backtest results for one pattern.
    Contains both in-sample and out-of-sample trades.
    """

    pattern_id: str
    symbol: str
    direction: PatternDirection
    rules: list[PatternRule]

    # Trade list
    trades: list[TradeResult] = field(default_factory=list)

    # Aggregate metrics (populated by analytics module)
    sample_count: int = 0
    is_sample_count: int = 0
    oos_sample_count: int = 0

    win_rate: float = 0.0
    oos_win_rate: float = 0.0
    profit_factor: float = 0.0
    oos_profit_factor: float = 0.0

    expectancy: float = 0.0         # per-trade expected net pnl
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0

    total_gross_pnl: float = 0.0
    total_net_pnl: float = 0.0
    total_costs: float = 0.0

    # Stability
    win_rate_cv: float = 0.0        # coefficient of variation across time buckets
    is_stable: bool = False
    verdict: str = "REJECTED"       # "ACCEPTED" | "MARGINAL" | "REJECTED"
    rejection_reason: str = ""
