"""
Consistency Filter Module.

Filters strategies based on strict consistency rules.
A strategy PASSES only if ALL conditions are met:
- All 4 periods are profitable
- Max drawdown < 25%
- Expectancy > 0
- No single trade contributes > 30% of total profit
"""

from typing import Tuple, List
from models import BacktestResult
from config import Config


class ConsistencyFilter:
    """
    Filters strategies for consistent performance across all time periods.
    
    This is a strict filter - any failure results in permanent discard.
    """
    
    def __init__(self):
        """Initialize filter with config thresholds."""
        self.max_drawdown_limit = Config.MAX_DRAWDOWN_LIMIT * 100  # Convert to %
        self.max_single_trade_contribution = Config.MAX_SINGLE_TRADE_CONTRIBUTION * 100
        self.min_expectancy = Config.MIN_EXPECTANCY
    
    def check(self, result: BacktestResult) -> Tuple[bool, List[str]]:
        """
        Check if backtest result passes all consistency rules.
        
        Args:
            result: BacktestResult to check
        
        Returns:
            Tuple of (passed, list of failure reasons)
        """
        failures = []
        
        # Check 1: All periods must be profitable
        if not self._check_all_profitable(result, failures):
            pass  # Failure already added
        
        # Check 2: Max drawdown < 25% across all periods
        if not self._check_drawdown(result, failures):
            pass
        
        # Check 3: Expectancy > 0 across all periods
        if not self._check_expectancy(result, failures):
            pass
        
        # Check 4: No single trade > 30% of total profit
        if not self._check_single_trade_contribution(result, failures):
            pass
        
        passed = len(failures) == 0
        return passed, failures
    
    def _check_all_profitable(self, result: BacktestResult, 
                               failures: List[str]) -> bool:
        """Check that all periods are profitable."""
        unprofitable_periods = []
        
        for period, period_result in result.period_results.items():
            if period_result.total_pnl <= 0:
                unprofitable_periods.append(f"{period}D")
        
        if unprofitable_periods:
            failures.append(
                f"Unprofitable periods: {', '.join(unprofitable_periods)}"
            )
            return False
        
        return True
    
    def _check_drawdown(self, result: BacktestResult, 
                         failures: List[str]) -> bool:
        """Check that max drawdown is below threshold across all periods."""
        exceeded_periods = []
        
        for period, period_result in result.period_results.items():
            if period_result.max_drawdown_percent > self.max_drawdown_limit:
                exceeded_periods.append(
                    f"{period}D ({period_result.max_drawdown_percent:.1f}%)"
                )
        
        if exceeded_periods:
            failures.append(
                f"Drawdown > {self.max_drawdown_limit}%: {', '.join(exceeded_periods)}"
            )
            return False
        
        return True
    
    def _check_expectancy(self, result: BacktestResult, 
                           failures: List[str]) -> bool:
        """Check that expectancy is positive across all periods."""
        negative_periods = []
        
        for period, period_result in result.period_results.items():
            if period_result.expectancy <= self.min_expectancy:
                negative_periods.append(
                    f"{period}D ({period_result.expectancy:.2f})"
                )
        
        if negative_periods:
            failures.append(
                f"Negative expectancy: {', '.join(negative_periods)}"
            )
            return False
        
        return True
    
    def _check_single_trade_contribution(self, result: BacktestResult,
                                          failures: List[str]) -> bool:
        """Check that no single trade contributes > 30% of total profit."""
        exceeded_periods = []
        
        for period, period_result in result.period_results.items():
            if period_result.largest_win_contribution > self.max_single_trade_contribution:
                exceeded_periods.append(
                    f"{period}D ({period_result.largest_win_contribution:.1f}%)"
                )
        
        if exceeded_periods:
            failures.append(
                f"Single trade > {self.max_single_trade_contribution}%: {', '.join(exceeded_periods)}"
            )
            return False
        
        return True


# Module-level instance
_filter = ConsistencyFilter()


def check_consistency(result: BacktestResult) -> Tuple[bool, List[str]]:
    """
    Convenience function to check consistency.
    
    Args:
        result: BacktestResult to check
    
    Returns:
        Tuple of (passed, failure_reasons)
    """
    return _filter.check(result)


def filter_consistent(results: List[BacktestResult]) -> List[BacktestResult]:
    """
    Filter multiple results and return only consistent ones.
    
    Args:
        results: List of BacktestResults to filter
    
    Returns:
        List of results that passed all consistency checks
    """
    consistent = []
    
    for result in results:
        passed, failures = _filter.check(result)
        if passed:
            consistent.append(result)
        else:
            print(f"❌ {result.strategy.name} failed: {failures[0]}")
    
    return consistent
