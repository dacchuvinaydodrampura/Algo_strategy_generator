"""
Data Models for Strategy Research Engine.

Defines the core data structures for strategies, trades, and backtest results.
All models are JSON-serializable for storage and Telegram notifications.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime
import uuid
import json


@dataclass
class Strategy:
    """
    Represents a rule-based trading strategy.
    
    All fields are algorithmically defined - no discretionary logic.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    
    # Core rules
    entry_rules: List[str] = field(default_factory=list)
    exit_rules: List[str] = field(default_factory=list)
    stop_loss_logic: str = ""
    take_profit_logic: str = ""
    
    # Parameters
    risk_reward: float = 2.0
    timeframe: str = "5m"
    market: str = "NSE:NIFTY50"
    
    # Indicator parameters (for backtesting)
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    use_vwap: bool = True
    use_rsi: bool = False
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Strategy":
        """Create Strategy from dictionary."""
        return cls(**data)
    
    def get_entry_description(self) -> str:
        """Get human-readable entry rules."""
        return " AND ".join(self.entry_rules) if self.entry_rules else "No entry rules"
    
    def get_exit_description(self) -> str:
        """Get human-readable exit rules."""
        return " OR ".join(self.exit_rules) if self.exit_rules else "No exit rules"


@dataclass
class Trade:
    """Represents a single backtest trade."""
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    direction: str  # "LONG" or "SHORT"
    pnl: float
    pnl_percent: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PeriodResult:
    """Backtest result for a single time period."""
    period_days: int
    total_pnl: float
    total_pnl_percent: float
    win_rate: float
    max_drawdown: float
    max_drawdown_percent: float
    expectancy: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    largest_win_contribution: float  # % of total profit from single trade
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (without heavy equity curve for storage)."""
        data = asdict(self)
        # Trim equity curve for storage
        if len(data["equity_curve"]) > 100:
            step = len(data["equity_curve"]) // 100
            data["equity_curve"] = data["equity_curve"][::step]
        # Remove individual trades for storage
        data["trades"] = []
        return data
    
    def is_profitable(self) -> bool:
        """Check if period was profitable."""
        return self.total_pnl > 0


@dataclass
class BacktestResult:
    """Complete backtest results across all time periods."""
    strategy: Strategy
    period_results: Dict[int, PeriodResult] = field(default_factory=dict)
    
    # Aggregate metrics
    all_periods_profitable: bool = False
    worst_drawdown: float = 0.0
    avg_win_rate: float = 0.0
    avg_expectancy: float = 0.0
    avg_profit_factor: float = 0.0
    total_trades_per_year: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "strategy": self.strategy.to_dict(),
            "period_results": {
                str(k): v.to_dict() for k, v in self.period_results.items()
            },
            "all_periods_profitable": self.all_periods_profitable,
            "worst_drawdown": self.worst_drawdown,
            "avg_win_rate": self.avg_win_rate,
            "avg_expectancy": self.avg_expectancy,
            "avg_profit_factor": self.avg_profit_factor,
            "total_trades_per_year": self.total_trades_per_year,
        }
    
    def calculate_aggregates(self):
        """Calculate aggregate metrics from period results."""
        if not self.period_results:
            return
        
        results = list(self.period_results.values())
        
        # Check if all periods profitable
        self.all_periods_profitable = all(r.is_profitable() for r in results)
        
        # Worst drawdown across all periods
        self.worst_drawdown = max(r.max_drawdown_percent for r in results)
        
        # Average metrics
        self.avg_win_rate = sum(r.win_rate for r in results) / len(results)
        self.avg_expectancy = sum(r.expectancy for r in results) / len(results)
        
        # Profit factor (handle zero division)
        pf_values = [r.profit_factor for r in results if r.profit_factor > 0]
        self.avg_profit_factor = sum(pf_values) / len(pf_values) if pf_values else 0
        
        # Trades per year (extrapolate from 365-day period if available)
        if 365 in self.period_results:
            self.total_trades_per_year = self.period_results[365].total_trades
        elif 180 in self.period_results:
            self.total_trades_per_year = int(self.period_results[180].total_trades * 2)
        elif 60 in self.period_results:
            self.total_trades_per_year = int(self.period_results[60].total_trades * 6)


@dataclass
class StoredStrategy:
    """Strategy with backtest results for repository storage."""
    strategy: Strategy
    backtest_result: BacktestResult
    stored_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    hash: str = ""
    
    def __post_init__(self):
        """Generate hash for deduplication."""
        if not self.hash:
            # Hash based on strategy rules
            rules_str = "|".join(self.strategy.entry_rules + self.strategy.exit_rules)
            self.hash = str(hash(rules_str))[:12]
    
    def to_dict(self) -> Dict:
        return {
            "hash": self.hash,
            "stored_at": self.stored_at,
            "strategy": self.strategy.to_dict(),
            "backtest_result": self.backtest_result.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "StoredStrategy":
        strategy = Strategy.from_dict(data["strategy"])
        # Reconstruct backtest result (simplified)
        backtest = BacktestResult(strategy=strategy)
        backtest_data = data.get("backtest_result", {})
        backtest.all_periods_profitable = backtest_data.get("all_periods_profitable", False)
        backtest.worst_drawdown = backtest_data.get("worst_drawdown", 0)
        backtest.avg_win_rate = backtest_data.get("avg_win_rate", 0)
        backtest.avg_expectancy = backtest_data.get("avg_expectancy", 0)
        backtest.avg_profit_factor = backtest_data.get("avg_profit_factor", 0)
        backtest.total_trades_per_year = backtest_data.get("total_trades_per_year", 0)
        
        return cls(
            strategy=strategy,
            backtest_result=backtest,
            stored_at=data.get("stored_at", ""),
            hash=data.get("hash", ""),
        )
