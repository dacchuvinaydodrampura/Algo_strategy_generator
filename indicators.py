"""
Non-Repainting Technical Indicators.

All indicators use ONLY past data - no look-ahead bias.
Designed for backtesting integrity.
"""

import numpy as np
from typing import Tuple, Optional


def ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average.
    
    Uses standard EMA formula with proper initialization.
    First `period` values use SMA, then EMA afterwards.
    
    Args:
        prices: Array of prices (typically close prices)
        period: EMA period
    
    Returns:
        Array of EMA values (same length as input)
    """
    if len(prices) < period:
        return np.full_like(prices, np.nan)
    
    result = np.full_like(prices, np.nan, dtype=float)
    multiplier = 2.0 / (period + 1)
    
    # Initialize with SMA
    result[period - 1] = np.mean(prices[:period])
    
    # Calculate EMA
    for i in range(period, len(prices)):
        result[i] = (prices[i] - result[i - 1]) * multiplier + result[i - 1]
    
    return result


def sma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average.
    
    Args:
        prices: Array of prices
        period: SMA period
    
    Returns:
        Array of SMA values
    """
    if len(prices) < period:
        return np.full_like(prices, np.nan)
    
    result = np.full_like(prices, np.nan, dtype=float)
    
    for i in range(period - 1, len(prices)):
        result[i] = np.mean(prices[i - period + 1:i + 1])
    
    return result


def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (Wilder's RSI).
    
    Uses Wilder's smoothing method (not SMA-based).
    No repainting - uses only past data.
    
    Args:
        prices: Array of close prices
        period: RSI period (default 14)
    
    Returns:
        Array of RSI values (0-100)
    """
    if len(prices) < period + 1:
        return np.full_like(prices, np.nan)
    
    result = np.full_like(prices, np.nan, dtype=float)
    
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Initial average gain/loss using SMA
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # First RSI value
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))
    
    # Calculate RSI using Wilder's smoothing
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    
    return result


def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
         volume: np.ndarray, reset_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate Volume Weighted Average Price.
    
    Resets at each trading session (if reset_indices provided).
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume
        reset_indices: Indices where VWAP resets (session starts)
    
    Returns:
        Array of VWAP values
    """
    typical_price = (high + low + close) / 3.0
    
    if reset_indices is None:
        # No reset - cumulative VWAP
        cum_tp_vol = np.cumsum(typical_price * volume)
        cum_vol = np.cumsum(volume)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)
        return result
    
    # VWAP with session resets
    result = np.zeros_like(close, dtype=float)
    
    for i in range(len(reset_indices)):
        start_idx = reset_indices[i]
        end_idx = reset_indices[i + 1] if i + 1 < len(reset_indices) else len(close)
        
        session_tp = typical_price[start_idx:end_idx]
        session_vol = volume[start_idx:end_idx]
        
        cum_tp_vol = np.cumsum(session_tp * session_vol)
        cum_vol = np.cumsum(session_vol)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            result[start_idx:end_idx] = np.where(
                cum_vol > 0, cum_tp_vol / cum_vol, np.nan
            )
    
    return result


def vwap_bands(vwap_values: np.ndarray, close: np.ndarray, 
               multiplier: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate VWAP bands based on standard deviation.
    
    Args:
        vwap_values: Pre-calculated VWAP values
        close: Close prices
        multiplier: Band multiplier (default 2.0)
    
    Returns:
        Tuple of (upper_band, lower_band)
    """
    # Rolling standard deviation of close - vwap
    deviation = close - vwap_values
    
    # Use expanding std for session-based calculation
    std_values = np.zeros_like(close, dtype=float)
    for i in range(1, len(close)):
        std_values[i] = np.nanstd(deviation[:i + 1])
    
    upper_band = vwap_values + (multiplier * std_values)
    lower_band = vwap_values - (multiplier * std_values)
    
    return upper_band, lower_band


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
        period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default 14)
    
    Returns:
        Array of ATR values
    """
    if len(high) < period + 1:
        return np.full_like(high, np.nan)
    
    result = np.full_like(high, np.nan, dtype=float)
    
    # True Range calculation
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    
    # First ATR is SMA of TR
    result[period - 1] = np.mean(tr[:period])
    
    # Wilder's smoothing for ATR
    for i in range(period, len(high)):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    
    return result


def is_bullish_engulfing(open_prices: np.ndarray, close_prices: np.ndarray,
                          high_prices: np.ndarray, low_prices: np.ndarray,
                          index: int) -> bool:
    """
    Check for bullish engulfing pattern at given index.
    
    Requirements:
    - Previous candle is bearish (close < open)
    - Current candle is bullish (close > open)
    - Current body engulfs previous body
    """
    if index < 1:
        return False
    
    prev_open = open_prices[index - 1]
    prev_close = close_prices[index - 1]
    curr_open = open_prices[index]
    curr_close = close_prices[index]
    
    # Previous bearish, current bullish
    if prev_close >= prev_open or curr_close <= curr_open:
        return False
    
    # Current body engulfs previous body
    if curr_open < prev_close and curr_close > prev_open:
        return True
    
    return False


def is_bearish_engulfing(open_prices: np.ndarray, close_prices: np.ndarray,
                          high_prices: np.ndarray, low_prices: np.ndarray,
                          index: int) -> bool:
    """
    Check for bearish engulfing pattern at given index.
    """
    if index < 1:
        return False
    
    prev_open = open_prices[index - 1]
    prev_close = close_prices[index - 1]
    curr_open = open_prices[index]
    curr_close = close_prices[index]
    
    # Previous bullish, current bearish
    if prev_close <= prev_open or curr_close >= curr_open:
        return False
    
    # Current body engulfs previous body
    if curr_open > prev_close and curr_close < prev_open:
        return True
    
    return False


def is_inside_bar(high_prices: np.ndarray, low_prices: np.ndarray, 
                  index: int) -> bool:
    """
    Check for inside bar pattern at given index.
    
    Inside bar: Current high/low is within previous high/low range.
    """
    if index < 1:
        return False
    
    return (high_prices[index] < high_prices[index - 1] and 
            low_prices[index] > low_prices[index - 1])


def detect_breakout(high_prices: np.ndarray, low_prices: np.ndarray,
                    close_prices: np.ndarray, index: int, 
                    lookback: int = 20) -> Tuple[bool, bool]:
    """
    Detect price breakout from recent range.
    
    Args:
        high_prices: High prices
        low_prices: Low prices
        close_prices: Close prices
        index: Current bar index
        lookback: Bars to look back for range
    
    Returns:
        Tuple of (bullish_breakout, bearish_breakout)
    """
    if index < lookback:
        return False, False
    
    # Recent range (excluding current bar)
    recent_high = np.max(high_prices[index - lookback:index])
    recent_low = np.min(low_prices[index - lookback:index])
    
    bullish_breakout = close_prices[index] > recent_high
    bearish_breakout = close_prices[index] < recent_low
    
    return bullish_breakout, bearish_breakout


def is_morning_star(open_prices: np.ndarray, close_prices: np.ndarray,
                     high_prices: np.ndarray, low_prices: np.ndarray,
                     index: int) -> bool:
    """
    Check for Morning Star pattern at given index.
    
    Pattern (3 candles):
    1. Long bearish candle
    2. Small body candle (star)
    3. Bullish candle closing above midpoint of first candle
    """
    if index < 2:
        return False
    
    # Candle 1 (Bearish)
    o1, c1 = open_prices[index-2], close_prices[index-2]
    body1 = abs(c1 - o1)
    is_bearish1 = c1 < o1
    
    # Candle 2 (Star/Doji)
    o2, c2 = open_prices[index-1], close_prices[index-1]
    body2 = abs(c2 - o2)
    
    # Candle 3 (Bullish)
    o3, c3 = open_prices[index], close_prices[index]
    is_bullish3 = c3 > o3
    
    # Logic
    # 1. First candle is bearish
    if not is_bearish1:
        return False
        
    # 2. Second candle has small body (relative to first)
    if body2 > (body1 * 0.6):  # allowable limit for "small"
        return False
        
    # 3. Third candle is bullish and closes above midpoint of first
    midpoint1 = (o1 + c1) / 2
    if not (is_bullish3 and c3 > midpoint1):
        return False
        
    return True


# ==============================================================================
# INSTITUTIONAL / SMART MONEY CONCEPTS (SMC)
# ==============================================================================

def detect_bullish_imbalance(open_prices: np.ndarray, close_prices: np.ndarray,
                             high_prices: np.ndarray, low_prices: np.ndarray,
                             index: int) -> bool:
    """
    Detect Bullish Fair Value Gap (FVG).
    
    Condition:
    - Candle 1 High < Candle 3 Low
    - Leaving a gap between wick 1 and wick 3.
    """
    if index < 2:
        return False
        
    # Candle 1 (Index-2), Candle 2 (Index-1), Candle 3 (Index-0)
    high1 = high_prices[index-2]
    low3 = low_prices[index]
    
    # Gap exists if Low of current candle is noticeably higher than High of 2 candles ago
    # We add a small threshold to avoid tiny gaps being counted (0.02% price)
    price_threshold = close_prices[index] * 0.0002
    
    gap_size = low3 - high1
    return gap_size > price_threshold

def detect_bearish_imbalance(open_prices: np.ndarray, close_prices: np.ndarray,
                             high_prices: np.ndarray, low_prices: np.ndarray,
                             index: int) -> bool:
    """
    Detect Bearish Fair Value Gap (FVG).
    
    Condition:
    - Candle 1 Low > Candle 3 High
    """
    if index < 2:
        return False
        
    low1 = low_prices[index-2]
    high3 = high_prices[index]
    
    price_threshold = close_prices[index] * 0.0002
    
    gap_size = low1 - high3
    return gap_size > price_threshold

def detect_liquidity_sweep_low(high_prices: np.ndarray, low_prices: np.ndarray,
                               close_prices: np.ndarray, index: int,
                               lookback: int = 20) -> bool:
    """
    Detect Bullish Liquidity Sweep (Stop Hunt).
    
    Conditions:
    1. Current Low breaks below the Lowest Low of last 'lookback' candles.
    2. Current Close is ABOVE that previous Lowest Low (rejection/reclaim).
    """
    if index < lookback:
        return False
        
    # Find Swing Low in lookback period (excluding current candle)
    prev_lows = low_prices[index-lookback : index]
    swing_low = np.min(prev_lows)
    
    curr_low = low_prices[index]
    curr_close = close_prices[index]
    
    # 1. Sweep: Price dipped below swing low
    swept_liquidity = curr_low < swing_low
    
    # 2. Reclaim: Price closed back above the swing low
    reclaimed_level = curr_close > swing_low
    
    return swept_liquidity and reclaimed_level

def detect_liquidity_sweep_high(high_prices: np.ndarray, low_prices: np.ndarray,
                                close_prices: np.ndarray, index: int,
                                lookback: int = 20) -> bool:
    """
    Detect Bearish Liquidity Sweep (Stop Hunt).
    
    Conditions:
    1. Current High breaks above Highest High.
    2. Current Close is BELOW that previous Highest High.
    """
    if index < lookback:
        return False
        
    prev_highs = high_prices[index-lookback : index]
    swing_high = np.max(prev_highs)
    
    curr_high = high_prices[index]
    curr_close = close_prices[index]
    
    swept_liquidity = curr_high > swing_high
    reclaimed_level = curr_close < swing_high
    
    return swept_liquidity and reclaimed_level

def detect_volatility_squeeze(high_prices: np.ndarray, low_prices: np.ndarray,
                              close_prices: np.ndarray, index: int, 
                              period: int = 20) -> bool:
    """
    Detect Volatility Squeeze (Bollinger Band Squeeze).
    
    Logic: Band Width is at lowest point in 20 bars.
    """
    if index < period:
        return False

    # Calculate BB Width for last 'period' bars
    # This is expensive to do properly for every bar here, so we implement a simplified check.
    # Check if current Range (High-Low) is less than 50% of Average Range
    
    current_range = high_prices[index] - low_prices[index]
    
    # Average range of last 20 bars
    past_ranges = high_prices[index-period:index] - low_prices[index-period:index]
    avg_range = np.mean(past_ranges)
    
    return current_range < (avg_range * 0.5)

def detect_bullish_order_block(open_prices: np.ndarray, close_prices: np.ndarray,
                               index: int) -> bool:
    """
    Detect Bullish Order Block formation.
    
    Simplified Logic:
    1. Candle (Index-1) was Bearish (Red).
    2. Candle (Index) is Bullish (Green).
    3. Candle (Index) Body is > 2x Previous Candle Body (Strong impulse).
    4. Current Close broke above Previous High (Market Structure Shift) - implied by body size here.
    """
    if index < 1:
        return False
        
    # Prev Candle (Bearish)
    o1, c1 = open_prices[index-1], close_prices[index-1]
    is_bearish = c1 < o1
    body1 = abs(o1 - c1)
    
    # Curr Candle (Bullish Impulse)
    o2, c2 = open_prices[index], close_prices[index]
    is_bullish = c2 > o2
    body2 = abs(c2 - o2)
    
    if is_bearish and is_bullish and body2 > (body1 * 2.0):
        return True
        
    return False

def detect_bearish_order_block(open_prices: np.ndarray, close_prices: np.ndarray,
                               index: int) -> bool:
    """
    Detect Bearish Order Block formation.
    """
    if index < 1:
        return False
        
    # Prev Candle (Bullish)
    o1, c1 = open_prices[index-1], close_prices[index-1]
    is_bullish = c1 > o1
    body1 = abs(c1 - o1)
    
    # Curr Candle (Bearish Impulse)
    o2, c2 = open_prices[index], close_prices[index]
    is_bearish = c2 < o2
    body2 = abs(o2 - c2)
    
    if is_bullish and is_bearish and body2 > (body1 * 2.0):
        return True
        
    return False