"""
Backtest Engine Module.

Runs backtests on OHLCV data across multiple time periods.
Computes all required metrics:
- Total PnL
- Win rate
- Max drawdown
- Expectancy
- Profit factor
- Equity curve

No look-ahead bias. Uses only past data for signals.
"""

import numpy as np
from typing import List, Dict, Optional
from models import Strategy, Trade, PeriodResult, BacktestResult
from data_provider import DataProvider, OHLCVData
from config import Config
import indicators


class BacktestEngine:
    """
    Backtests strategies on historical OHLCV data.
    
    Runs same strategy logic across all time periods:
    - 30 days
    - 60 days
    - 180 days
    - 365 days
    """
    
    def __init__(self):
        """Initialize backtest engine."""
        self.data_provider = DataProvider()
    
    def run(self, strategy: Strategy, 
            periods: Optional[List[int]] = None) -> BacktestResult:
        """
        Run backtest across all time periods.
        
        Args:
            strategy: Strategy to test
            periods: List of periods in days (default from config)
        
        Returns:
            BacktestResult with results for all periods
        """
        if periods is None:
            periods = Config.BACKTEST_PERIODS
        
        result = BacktestResult(strategy=strategy)
        
        for period in periods:
            period_result = self._run_period(strategy, period)
            result.period_results[period] = period_result
        
        # Calculate aggregate metrics
        result.calculate_aggregates()
        
        return result
    
    def _run_period(self, strategy: Strategy, days: int) -> PeriodResult:
        """
        Run backtest for a single time period.
        
        Args:
            strategy: Strategy to test
            days: Number of days to backtest
        
        Returns:
            PeriodResult with all metrics
        """
        # Get OHLCV data
        data = self.data_provider.get_data(
            market=strategy.market,
            timeframe=strategy.timeframe,
            days=days,
        )
        
        # Calculate indicators
        indicators_data = self._calculate_indicators(strategy, data)
        
        # Run simulation
        trades = self._simulate_trades(strategy, data, indicators_data)
        
        # Calculate metrics
        return self._calculate_metrics(trades, days, data)
    
    def _calculate_indicators(self, strategy: Strategy, 
                               data: OHLCVData) -> Dict[str, np.ndarray]:
        """Calculate all required indicators for the strategy."""
        result = {}
        
        # Always calculate EMAs
        result["ema_fast"] = indicators.ema(data.close, strategy.ema_fast)
        result["ema_slow"] = indicators.ema(data.close, strategy.ema_slow)
        
        # VWAP if needed
        if strategy.use_vwap:
            result["vwap"] = indicators.vwap(
                data.high, data.low, data.close, data.volume
            )
            result["vwap_upper"], result["vwap_lower"] = indicators.vwap_bands(
                result["vwap"], data.close, Config.VWAP_BAND_MULTIPLIER
            )
        
        # RSI if needed
        if strategy.use_rsi:
            result["rsi"] = indicators.rsi(data.close, strategy.rsi_period)
        
        # ATR for stop loss
        result["atr"] = indicators.atr(data.high, data.low, data.close, 14)
        
        return result
    
    def _simulate_trades(self, strategy: Strategy, data: OHLCVData,
                          ind: Dict[str, np.ndarray]) -> List[Trade]:
        """
        Simulate trades based on strategy rules.
        
        This is the core backtesting logic.
        """
        trades = []
        position = None  # None, "LONG", "SHORT"
        entry_price = 0.0
        entry_time = ""
        stop_loss = 0.0
        take_profit = 0.0
        
        # Skip first 50 bars for indicator warmup
        start_idx = 50
        
        for i in range(start_idx, len(data.close)):
            current_price = data.close[i]
            current_time = str(data.timestamps[i]) if data.timestamps is not None else str(i)
            
            # Check for exit first if in position
            if position is not None:
                exit_triggered, exit_price = self._check_exit(
                    position, current_price, data.high[i], data.low[i],
                    stop_loss, take_profit
                )
                
                if exit_triggered:
                    # Calculate P&L
                    if position == "LONG":
                        pnl = exit_price - entry_price
                    else:
                        pnl = entry_price - exit_price
                    
                    pnl_percent = (pnl / entry_price) * 100
                    
                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=current_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction=position,
                        pnl=pnl,
                        pnl_percent=pnl_percent,
                    ))
                    
                    position = None
                    continue
            
            # Check for entry if not in position
            if position is None:
                entry_signal = self._check_entry(strategy, data, ind, i)
                
                if entry_signal is not None:
                    position = entry_signal
                    entry_price = current_price
                    entry_time = current_time
                    
                    # Calculate stop loss and take profit
                    atr = ind["atr"][i] if not np.isnan(ind["atr"][i]) else current_price * 0.01
                    
                    stop_distance = self._calculate_stop_distance(
                        strategy.stop_loss_logic, atr, current_price
                    )
                    
                    if position == "LONG":
                        stop_loss = entry_price - stop_distance
                        take_profit = entry_price + (stop_distance * strategy.risk_reward)
                    else:
                        stop_loss = entry_price + stop_distance
                        take_profit = entry_price - (stop_distance * strategy.risk_reward)
        
        # Close any open position at end
        if position is not None:
            exit_price = data.close[-1]
            if position == "LONG":
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price
            
            pnl_percent = (pnl / entry_price) * 100
            
            trades.append(Trade(
                entry_time=entry_time,
                exit_time=str(data.timestamps[-1]) if data.timestamps is not None else str(len(data.close)),
                entry_price=entry_price,
                exit_price=exit_price,
                direction=position,
                pnl=pnl,
                pnl_percent=pnl_percent,
            ))
        
        return trades
    
    def _check_entry(self, strategy: Strategy, data: OHLCVData,
                      ind: Dict[str, np.ndarray], idx: int) -> Optional[str]:
        """
        Check if entry conditions are met.
        
        Returns "LONG", "SHORT", or None.
        """
        # Parse entry rules and check conditions
        rules = strategy.entry_rules
        
        # EMA conditions
        ema_fast = ind["ema_fast"][idx]
        ema_slow = ind["ema_slow"][idx]
        price = data.close[idx]
        
        if np.isnan(ema_fast) or np.isnan(ema_slow):
            return None
        
        # Check for various entry conditions
        for rule in rules:
            rule_lower = rule.lower()
            
            # EMA crossover conditions
            if "ema_fast_above_slow" in rule_lower:
                if ema_fast <= ema_slow:
                    return None
            elif "ema_fast_below_slow" in rule_lower:
                if ema_fast >= ema_slow:
                    return None
            
            # Price vs EMA
            if "price_above_both_emas" in rule_lower:
                if price <= ema_fast or price <= ema_slow:
                    return None
            elif "price_below_both_emas" in rule_lower:
                if price >= ema_fast or price >= ema_slow:
                    return None
            
            # VWAP conditions
            if strategy.use_vwap and "vwap" in ind:
                vwap_val = ind["vwap"][idx]
                if not np.isnan(vwap_val):
                    if "price_above_vwap" in rule_lower:
                        if price <= vwap_val:
                            return None
                    elif "price_below_vwap" in rule_lower:
                        if price >= vwap_val:
                            return None
            
            # RSI conditions
            if strategy.use_rsi and "rsi" in ind:
                rsi_val = ind["rsi"][idx]
                if not np.isnan(rsi_val):
                    if "rsi_below_70" in rule_lower:
                        if rsi_val >= 70:
                            return None
                    elif "rsi_above_30" in rule_lower:
                        if rsi_val <= 30:
                            return None
            
            # Breakout conditions
            if "breakout_above" in rule_lower:
                lookback = 20
                if idx >= lookback:
                    recent_high = np.max(data.high[idx-lookback:idx])
                    if price <= recent_high:
                        return None
            elif "breakout_below" in rule_lower:
                lookback = 20
                if idx >= lookback:
                    recent_low = np.min(data.low[idx-lookback:idx])
                    if price >= recent_low:
                        return None
            
            # Engulfing patterns
            if "bullish_engulfing" in rule_lower:
                if not indicators.is_bullish_engulfing(
                    data.open, data.close, data.high, data.low, idx
                ):
                    return None
            elif "bearish_engulfing" in rule_lower:
                if not indicators.is_bearish_engulfing(
                    data.open, data.close, data.high, data.low, idx
                ):
                    return None
            
            # Inside bar
            if "inside_bar" in rule_lower:
                if not indicators.is_inside_bar(data.high, data.low, idx):
                    return None
        
        # If all conditions passed, determine direction
        # Based on EMA direction or price action
        if ema_fast > ema_slow:
            return "LONG"
        elif ema_fast < ema_slow:
            return "SHORT"
        
        return None
    
    def _check_exit(self, position: str, current_price: float,
                    high: float, low: float,
                    stop_loss: float, take_profit: float) -> tuple:
        """
        Check if exit conditions are met.
        
        Returns tuple of (exit_triggered, exit_price).
        """
        if position == "LONG":
            # Stop loss hit
            if low <= stop_loss:
                return True, stop_loss
            # Take profit hit
            if high >= take_profit:
                return True, take_profit
        else:  # SHORT
            # Stop loss hit
            if high >= stop_loss:
                return True, stop_loss
            # Take profit hit
            if low <= take_profit:
                return True, take_profit
        
        return False, 0.0
    
    def _calculate_stop_distance(self, sl_logic: str, atr: float, 
                                  price: float) -> float:
        """Calculate stop loss distance based on logic."""
        sl_lower = sl_logic.lower()
        
        if "1.5x_atr" in sl_lower:
            return atr * 1.5
        elif "2x_atr" in sl_lower:
            return atr * 2.0
        elif "0.5pct" in sl_lower:
            return price * 0.005
        elif "1pct" in sl_lower:
            return price * 0.01
        else:
            # Default to 1.5x ATR
            return atr * 1.5
    
    def _calculate_metrics(self, trades: List[Trade], days: int,
                            data: OHLCVData) -> PeriodResult:
        """Calculate all performance metrics from trades."""
        
        if not trades:
            return PeriodResult(
                period_days=days,
                total_pnl=0.0,
                total_pnl_percent=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                max_drawdown_percent=0.0,
                expectancy=0.0,
                profit_factor=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_win=0.0,
                avg_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                largest_win_contribution=0.0,
                equity_curve=[0.0],
                trades=[],
            )
        
        # Separate winners and losers
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        
        # Basic counts
        total_trades = len(trades)
        winning_trades = len(winners)
        losing_trades = len(losers)
        
        # Win rate
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
        
        # P&L calculations
        total_pnl = sum(t.pnl for t in trades)
        total_pnl_percent = sum(t.pnl_percent for t in trades)
        
        gross_profit = sum(t.pnl for t in winners) if winners else 0.0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0.0
        
        # Averages
        avg_win = np.mean([t.pnl for t in winners]) if winners else 0.0
        avg_loss = abs(np.mean([t.pnl for t in losers])) if losers else 0.0
        
        # Largest trades
        largest_win = max([t.pnl for t in winners]) if winners else 0.0
        largest_loss = abs(min([t.pnl for t in losers])) if losers else 0.0
        
        # Largest win contribution (% of total profit)
        if gross_profit > 0:
            largest_win_contribution = (largest_win / gross_profit) * 100
        else:
            largest_win_contribution = 0.0
        
        # Profit factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        if profit_factor == float('inf'):
            profit_factor = 10.0  # Cap at 10 for display
        
        # Expectancy
        win_prob = winning_trades / total_trades if total_trades > 0 else 0
        loss_prob = 1 - win_prob
        expectancy = (win_prob * avg_win) - (loss_prob * avg_loss)
        
        # Equity curve and max drawdown
        equity_curve = [0.0]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade.pnl)
        
        # Max drawdown calculation
        peak = 0.0
        max_dd = 0.0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        
        # Max drawdown as percentage of starting capital (assume 100000)
        starting_capital = 100000.0
        max_drawdown_percent = (max_dd / starting_capital) * 100 if starting_capital > 0 else 0.0
        
        return PeriodResult(
            period_days=days,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            win_rate=win_rate,
            max_drawdown=max_dd,
            max_drawdown_percent=max_drawdown_percent,
            expectancy=expectancy,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            largest_win_contribution=largest_win_contribution,
            equity_curve=equity_curve,
            trades=trades,
        )


# Module-level instance
_engine = BacktestEngine()


def run_backtest(strategy: Strategy, 
                 periods: Optional[List[int]] = None) -> BacktestResult:
    """
    Convenience function to run backtest.
    
    Args:
        strategy: Strategy to test
        periods: Time periods to test (default from config)
    
    Returns:
        BacktestResult with all metrics
    """
    return _engine.run(strategy, periods)
