"""
app/models/tick.py
------------------
Canonical data model for a single tick record as received in .ndjson.gz files.

Fields come directly from the data contract.  We use Optional for all
order-book depth levels because some feeds may omit deeper levels under
thin market conditions.  The core fields (t, seq, s, bid, ask) are required.

Validation logic is deliberately strict — any record that fails must be
rejected and logged, never silently passed downstream.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TickRecord(BaseModel):
    """
    Single tick from an NDJSON symbol file.

    Field names match the archive data contract exactly.
    Do not add, rename, or remove fields without updating the contract.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    t: int = Field(..., description="Epoch milliseconds timestamp")
    seq: int = Field(..., description="Monotonically increasing sequence number")
    s: str = Field(..., description="Symbol identifier e.g. BANKNIFTY26JUNFUT")

    # ── Best bid/ask ──────────────────────────────────────────────────────────
    bid: float = Field(..., description="Best bid price")
    ask: float = Field(..., description="Best ask price")
    bq: float = Field(..., description="Quantity at best bid")
    aq: float = Field(..., description="Quantity at best ask")
    spread: float = Field(..., description="Ask minus bid")
    imbalance: float = Field(..., description="Order book imbalance [-1, 1]")

    # ── Bid book depth (levels 1-5) ───────────────────────────────────────────
    bp1: Optional[float] = None
    bp2: Optional[float] = None
    bp3: Optional[float] = None
    bp4: Optional[float] = None
    bp5: Optional[float] = None

    bq1: Optional[float] = None
    bq2: Optional[float] = None
    bq3: Optional[float] = None
    bq4: Optional[float] = None
    bq5: Optional[float] = None

    # ── Ask book depth (levels 1-5) ───────────────────────────────────────────
    ap1: Optional[float] = None
    ap2: Optional[float] = None
    ap3: Optional[float] = None
    ap4: Optional[float] = None
    ap5: Optional[float] = None

    aq1: Optional[float] = None
    aq2: Optional[float] = None
    aq3: Optional[float] = None
    aq4: Optional[float] = None
    aq5: Optional[float] = None

    # ── Delta fields ──────────────────────────────────────────────────────────
    db: Optional[float] = Field(default=None, description="Delta bid quantity")
    da: Optional[float] = Field(default=None, description="Delta ask quantity")

    # ──────────────────────────────────────────────────────────────────────────
    # Validators
    # ──────────────────────────────────────────────────────────────────────────

    @field_validator("t")
    @classmethod
    def validate_timestamp(cls, v: int) -> int:
        # Sanity: reject timestamps clearly outside trading hours or epoch=0
        # We accept any ms timestamp after 2000-01-01 and before 2100-01-01
        _MIN_MS = 946_684_800_000   # 2000-01-01 00:00:00 UTC
        _MAX_MS = 4_102_444_800_000  # 2100-01-01 00:00:00 UTC
        if not (_MIN_MS <= v <= _MAX_MS):
            raise ValueError(f"Timestamp {v} outside acceptable epoch range")
        return v

    @field_validator("s")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Symbol must be a non-empty string")
        # Strip exchange prefix (e.g. 'NSE:NIFTY26JUNFUT' -> 'NIFTY26JUNFUT')
        parts = v.strip().split(":")
        return parts[-1].upper()

    @field_validator("spread")
    @classmethod
    def validate_spread(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Spread cannot be negative, got {v}")
        return v

    @field_validator("imbalance")
    @classmethod
    def validate_imbalance(cls, v: float) -> float:
        # Imbalance is defined as (bq - aq) / (bq + aq), range [-1, 1]
        if not (-1.0 <= v <= 1.0):
            raise ValueError(f"Imbalance {v} outside [-1, 1]")
        return v

    @model_validator(mode="after")
    def validate_bid_ask_consistency(self) -> "TickRecord":
        if self.bid >= self.ask:
            raise ValueError(
                f"Bid {self.bid} >= ask {self.ask}: crossed/locked market"
            )
        expected_spread = round(self.ask - self.bid, 6)
        if abs(expected_spread - self.spread) > 1e-4:
            raise ValueError(
                f"Spread field {self.spread} inconsistent with ask-bid={expected_spread}"
            )
        if self.bq <= 0:
            raise ValueError(f"Best bid quantity {self.bq} must be positive")
        if self.aq <= 0:
            raise ValueError(f"Best ask quantity {self.aq} must be positive")
        return self

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience helpers
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def midprice(self) -> float:
        """Simple mid = (bid + ask) / 2."""
        return (self.bid + self.ask) / 2.0

    @property
    def t_seconds(self) -> float:
        """Timestamp in seconds (float)."""
        return self.t / 1000.0

    def bid_levels(self) -> list[tuple[float, float]]:
        """Return available bid (price, qty) levels, sorted best to worst."""
        levels: list[tuple[float, float]] = []
        for i in range(1, 6):
            p = getattr(self, f"bp{i}")
            q = getattr(self, f"bq{i}")
            if p is not None and q is not None:
                levels.append((p, q))
        return levels

    def ask_levels(self) -> list[tuple[float, float]]:
        """Return available ask (price, qty) levels, sorted best to worst."""
        levels: list[tuple[float, float]] = []
        for i in range(1, 6):
            p = getattr(self, f"ap{i}")
            q = getattr(self, f"aq{i}")
            if p is not None and q is not None:
                levels.append((p, q))
        return levels

    def total_bid_depth(self) -> float:
        """Sum of all available bid quantities."""
        return sum(q for _, q in self.bid_levels())

    def total_ask_depth(self) -> float:
        """Sum of all available ask quantities."""
        return sum(q for _, q in self.ask_levels())
