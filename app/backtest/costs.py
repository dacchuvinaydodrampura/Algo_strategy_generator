"""
app/backtest/costs.py
---------------------
Transparent cost model for the backtesting engine.

All cost assumptions are configurable and explicitly reported in the PDF.

Cost components:
1. Brokerage: fixed per-lot round-trip fee (entry + exit).
2. Slippage:  tick_size * slippage_ticks per side (applied at entry only;
              exits at target/stop are assumed to fill at the signal price).
3. Spread:    entry is at ask for LONG, bid for SHORT (so spread is half-paid).
              This is accounted for in entry_price, not here separately.
4. Latency:   modelled as tick skip (handled in engine, not in costs).

We do NOT model market impact beyond fixed slippage ticks.
We do NOT model partial fills.
This is a research engine — conservative flat-cost assumption is correct.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import BacktestConfig


@dataclass
class CostBreakdown:
    """Per-trade cost breakdown for transparency in reports."""

    brokerage: float        # fixed per-lot
    slippage: float         # ticks * tick_size * lot_size
    total: float


class CostModel:
    """
    Computes realistic trading costs for a single round-trip trade.

    Parameters
    ----------
    cfg: BacktestConfig containing all cost parameters.
    """

    def __init__(self, cfg: BacktestConfig) -> None:
        self._cfg = cfg

    def breakdown(self, lot_size: int) -> CostBreakdown:
        """Return itemised cost for one round-trip trade of `lot_size` units."""
        brokerage = self._cfg.brokerage_per_lot
        slippage = (
            self._cfg.slippage_ticks
            * self._cfg.tick_size
            * lot_size
        )
        return CostBreakdown(
            brokerage=brokerage,
            slippage=slippage,
            total=brokerage + slippage,
        )

    def total_cost(self, lot_size: int) -> float:
        """Return scalar total cost for one round-trip."""
        return self.breakdown(lot_size).total

    def assumption_text(self) -> list[str]:
        """Return human-readable cost assumption strings for the PDF report."""
        cfg = self._cfg
        latency_str = f"{cfg.latency_ms} ms signal-to-fill delay (entry tick skipped)" if cfg.latency_ms > 0 else "0 ms latency (no entry tick skipped)"
        return [
            f"Brokerage: ₹{cfg.brokerage_per_lot:.2f} per lot (round trip)",
            f"Slippage: {cfg.slippage_ticks} tick(s) × ₹{cfg.tick_size:.2f} × lot size",
            f"Latency: {latency_str}",
            f"Tick size: ₹{cfg.tick_size:.2f}",
            f"Lot size: {cfg.lot_size} units",
            "Exit fills: at exact target/stop price (no exit slippage assumed)",
            "Spread cost: implicit in buy-at-ask / sell-at-bid entry assumption",
            "Market impact: not modelled beyond fixed slippage",
            "Partial fills: not modelled (full fill assumed)",
        ]
