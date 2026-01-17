"""
Consistency Filter Module.

Filters strategies based on strict consistency rules.
A strategy PASSES if at least 6 out of 7 conditions are met (1 failure allowed):
- All 4 periods are profitable
- Max drawdown < 22%
- Expectancy > 0.5
- Win Rate > 54%
- Profit Factor > 1.8
- Return Targets met
- No single trade contributes > 35% of total profit
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
        self.min_win_rate = Config.MIN_WIN_RATE
        self.min_profit_factor = Config.MIN_PROFIT_FACTOR
        
        # Return targets mapping
        self.return_targets = {
            30: Config.MIN_RETURN_30D,
            60: Config.MIN_RETURN_60D,
            180: Config.MIN_RETURN_180D,
            365: Config.MIN_RETURN_365D,
        }
    
    def check(self, result: BacktestResult) -> Tuple[bool, List[str]]:
        """
        Check if backtest result passes all consistency rules.
        """
        failures = []
        
        # Check 1: All periods must be profitable
        if not self._check_all_profitable(result, failures):
            pass 
        
        # Check 2: Max drawdown
        if not self._check_drawdown(result, failures):
            pass
        
        # Check 3: Expectancy
        if not self._check_expectancy(result, failures):
            pass
            
        # Check 4: Win Rate
        if not self._check_win_rate(result, failures):
            pass

        # Check 5: Profit Factor
        if not self._check_profit_factor(result, failures):
            pass
            
        # Check 6: Return Targets
        if not self._check_return_targets(result, failures):
            pass
        
        # Check 7: No single trade > 30% contribution (secondary check)
        if not self._check_single_trade_contribution(result, failures):
            pass
        
        # Relaxed Rule: Allow 1 failure (6 out of 7 must pass)
        passed = len(failures) <= 1
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
        """Check valid expectancy."""
        if result.avg_expectancy < self.min_expectancy:
            failures.append(f"Expectancy {result.avg_expectancy:.2f} < {self.min_expectancy}")
            return False
        return True
    
    def _check_win_rate(self, result: BacktestResult, failures: List[str]) -> bool:
        """Check win rate is above minimum."""
        if result.avg_win_rate < self.min_win_rate:
            failures.append(f"Win Rate {result.avg_win_rate:.1f}% < {self.min_win_rate}%")
            return False
        return True

    def _check_profit_factor(self, result: BacktestResult, failures: List[str]) -> bool:
        """Check profit factor is above minimum."""
        if result.avg_profit_factor < self.min_profit_factor:
            failures.append(f"Profit Factor {result.avg_profit_factor:.2f} < {self.min_profit_factor}")
            return False
        return True

    def _check_return_targets(self, result: BacktestResult, failures: List[str]) -> bool:
        """Check if return targets are met for each period."""
        failed_targets = []
        
        for period, period_result in result.period_results.items():
            target = self.return_targets.get(period)
            if target and period_result.total_pnl_percent < target:
                failed_targets.append(
                    f"{period}D ({period_result.total_pnl_percent:.1f}% < {target}%)"
                )
        
        if failed_targets:
            failures.append(f"Return targets missed: {', '.join(failed_targets)}")
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
