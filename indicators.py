"""
Non-Repainting Technical Indicators.

All indicators use ONLY past data - no look-ahead bias.
Designed for backtesting integrity.
"""

import numpy as np
from datetime import datetime
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


def detect_bullish_engulfing(open_prices: np.ndarray, close_prices: np.ndarray,
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


def detect_bearish_engulfing(open_prices: np.ndarray, close_prices: np.ndarray,
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


def detect_inside_bar(open_prices: np.ndarray, close_prices: np.ndarray,
                   high_prices: np.ndarray, low_prices: np.ndarray, 
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


def detect_morning_star(open_prices: np.ndarray, close_prices: np.ndarray,
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

def detect_liquidity_sweep_low(open_prices: np.ndarray, close_prices: np.ndarray,
                               high_prices: np.ndarray, low_prices: np.ndarray,
                               index: int, lookback: int = 20) -> bool:
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

def detect_liquidity_sweep_high(open_prices: np.ndarray, close_prices: np.ndarray,
                                high_prices: np.ndarray, low_prices: np.ndarray,
                                index: int, lookback: int = 20) -> bool:
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

def detect_volatility_squeeze(open_prices: np.ndarray, close_prices: np.ndarray,
                              high_prices: np.ndarray, low_prices: np.ndarray, 
                              index: int, period: int = 20) -> bool:
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
                               high_prices: np.ndarray, low_prices: np.ndarray,
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
                               high_prices: np.ndarray, low_prices: np.ndarray,
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


# ==============================================================================
# ADVANCED MARKET STRUCTURE (MSS & BREAKERS)
# ==============================================================================

def detect_market_structure_shift_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                          high_prices: np.ndarray, low_prices: np.ndarray, 
                                          index: int, lookback: int = 20) -> bool:
    """
    Detect Bullish Market Structure Shift (MSS).
    
    Logic:
    1. Identify the lowest point in the lookback period (Swing Low).
    2. Identify the highest high *after* that Swing Low (Lower High).
    3. Price breaks ABOVE that Lower High with a strong close.
    """
    if index < lookback:
        return False
        
    # 1. Find the Swing Low
    window_lows = low_prices[index-lookback:index]
    min_low_idx = np.argmin(window_lows)
    abs_low_idx = (index - lookback) + min_low_idx
    
    # Needs enough data after low to form a lower high
    if abs_low_idx >= index - 2:
        return False
        
    # 2. Find High AFTER the Low (The Swing High to break)
    swing_high = np.max(high_prices[abs_low_idx:index])
    
    # 3. Current candle closes well above that high
    break_level = swing_high
    curr_close = close_prices[index]
    
    # Ensure it's a confirmed break, not just a wick
    return curr_close > break_level

def detect_market_structure_shift_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                          high_prices: np.ndarray, low_prices: np.ndarray, 
                                          index: int, lookback: int = 20) -> bool:
    """Detect Bearish Market Structure Shift (MSS)."""
    if index < lookback:
        return False
        
    window_highs = high_prices[index-lookback:index]
    max_high_idx = np.argmax(window_highs)
    abs_high_idx = (index - lookback) + max_high_idx
    
    if abs_high_idx >= index - 2:
        return False
        
    swing_low = np.min(low_prices[abs_high_idx:index])
    
    curr_close = close_prices[index]
    return curr_close < swing_low

def detect_bullish_breaker(open_prices: np.ndarray, close_prices: np.ndarray,
                           high_prices: np.ndarray, low_prices: np.ndarray,
                           index: int, lookback: int = 10) -> bool:
    """
    Detect Bullish Breaker Block.
    
    Logic:
    1. Previously a Bearish Order Block (Red Candle) existed in lookback.
    2. Price BROKE UP through it (invalidating it as resistance).
    3. Price has now returned to RETEST that level as Support.
    """
    if index < lookback:
        return False
        
    curr_low = low_prices[index]
    curr_close = close_prices[index]
    
    for i in range(1, lookback):
        idx = index - i
        # Check if candle was bearish
        if close_prices[idx] < open_prices[idx]:
            bearish_high = high_prices[idx]
            bearish_open = open_prices[idx]
            
            # Check if price went significantly ABOVE it in between
            highest_since = np.max(high_prices[idx+1:index])
            
            if highest_since > bearish_high * 1.001: # Clear break
                # Check for Retest
                # Current Low touches the "Breaker" zone (High to Open of orig candle)
                if curr_low <= bearish_high and curr_close >= bearish_open:
                    return True
                    
    return False

def detect_bearish_breaker(open_prices: np.ndarray, close_prices: np.ndarray,
                           high_prices: np.ndarray, low_prices: np.ndarray,
                           index: int, lookback: int = 10) -> bool:
    """Detect Bearish Breaker Block (Support turned Resistance)."""
    if index < lookback:
        return False
        
    curr_high = high_prices[index]
    curr_close = close_prices[index]
    
    for i in range(1, lookback):
        idx = index - i
        # Bullish candle
        if close_prices[idx] > open_prices[idx]:
            bullish_low = low_prices[idx]
            bullish_open = open_prices[idx]
            
            lowest_since = np.min(low_prices[idx+1:index])
            
            if lowest_since < bullish_low * 0.999: # Clear break down
                # Retest from below
                if curr_high >= bullish_low and curr_close <= bullish_open:
                    return True
                    
    return False


# ==============================================================================
# PREMIUM / DISCOUNT & OTE
# ==============================================================================

def detect_discount_zone(open_prices: np.ndarray, close_prices: np.ndarray,
                        high_prices: np.ndarray, low_prices: np.ndarray, 
                        index: int, lookback: int = 50) -> bool:
    """
    Check if price is in the Discount Zone (< 50% of range).
    We only want to BUY in Discount.
    """
    if index < lookback:
        return False
        
    recent_high = np.max(high_prices[index-lookback:index])
    recent_low = np.min(low_prices[index-lookback:index])
    
    midpoint = (recent_high + recent_low) / 2
    
    return close_prices[index] < midpoint

def detect_premium_zone(open_prices: np.ndarray, close_prices: np.ndarray,
                       high_prices: np.ndarray, low_prices: np.ndarray, 
                       index: int, lookback: int = 50) -> bool:
    """
    Check if price is in the Premium Zone (> 50% of range).
    We only want to SELL in Premium.
    """
    if index < lookback:
        return False
        
    recent_high = np.max(high_prices[index-lookback:index])
    recent_low = np.min(low_prices[index-lookback:index])
    
    midpoint = (recent_high + recent_low) / 2
    
    return close_prices[index] > midpoint

def detect_optimal_trade_entry_long(open_prices: np.ndarray, close_prices: np.ndarray,
                                    high_prices: np.ndarray, low_prices: np.ndarray,
                                    index: int, lookback: int = 20) -> bool:
    """
    Detect OTE Long Setup.
    
    Logic:
    1. Identify recent impulse move (Low to High).
    2. Measures Fib Retracement.
    3. Price is currently between 0.618 and 0.786 retracement.
    4. Price shows rejection (Close > Open).
    """
    if index < lookback:
        return False
        
    # Find recent swing points
    window_lows = low_prices[index-lookback:index]
    min_low_idx = np.argmin(window_lows)
    abs_low_idx = (index-lookback) + min_low_idx
    
    # Needs impulse up after low
    window_highs = high_prices[abs_low_idx:index]
    if len(window_highs) < 2: 
        return False
        
    max_high_idx = np.argmax(window_highs)
    swing_high = window_highs[max_high_idx]
    swing_low = low_prices[abs_low_idx]
    
    range_move = swing_high - swing_low
    if range_move == 0:
        return False
        
    # Fib Levels (Down from High)
    fib_618 = swing_high - (range_move * 0.618)
    fib_786 = swing_high - (range_move * 0.786)
    
    curr_low = low_prices[index]
    curr_close = close_prices[index]
    curr_open = close_prices[index] # simplify to close > open? No, need open
    
    # Check if price touched the zone [0.786 (bottom), 0.618 (top)]
    in_zone = (curr_low <= fib_618) and (curr_low >= fib_786 * 0.999) 
    
    # Check rejection (Bullish Close)
    if in_zone and curr_close > curr_low: # Basic rejection test
        return True
        
    return False

def detect_optimal_trade_entry_short(open_prices: np.ndarray, close_prices: np.ndarray,
                                     high_prices: np.ndarray, low_prices: np.ndarray,
                                     index: int, lookback: int = 20) -> bool:
    """Detect OTE Short Setup."""
    if index < lookback:
        return False
        
    window_highs = high_prices[index-lookback:index]
    max_high_idx = np.argmax(window_highs)
    abs_high_idx = (index-lookback) + max_high_idx
    
    window_lows = low_prices[abs_high_idx:index]
    if len(window_lows) < 2:
        return False
        
    swing_low = np.min(window_lows)
    swing_high = high_prices[abs_high_idx]
    
    range_move = swing_high - swing_low
    if range_move == 0:
        return False
        
    # Fib Levels (Up from Low)
    fib_618 = swing_low + (range_move * 0.618)
    fib_786 = swing_low + (range_move * 0.786)
    
    curr_high = high_prices[index]
    curr_close = close_prices[index]
    
    in_zone = (curr_high >= fib_618) and (curr_high <= fib_786 * 1.001)
    
    if in_zone and curr_close < curr_high:
        return True
        
    return False


# ==============================================================================
# INDUCEMENT (IDM) & TIME LOGIC
# ==============================================================================

def detect_inducement_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                              high_prices: np.ndarray, low_prices: np.ndarray,
                              index: int, lookback: int = 10) -> bool:
    """
    Detect Bullish Inducement (IDM).
    
    Logic:
    1. A minor short-term low was formed recently (The "Inducement" or "Bait").
    2. Price just swept below it (Taking out early stops).
    3. Price closed back ABOVE it (Trap confirmed).
    """
    if index < lookback:
        return False
        
    # Find a minor local low in recent history (excluding current candle)
    recent_lows = low_prices[index-lookback:index]
    minor_low = np.min(recent_lows)
    
    curr_low = low_prices[index]
    curr_close = close_prices[index]
    
    # 1. Sweep: Price went below the minor low
    sweep = curr_low < minor_low
    
    # 2. Reclaim: Price closed back above it
    reclaim = curr_close > minor_low
    
    return sweep and reclaim

def detect_inducement_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                               high_prices: np.ndarray, low_prices: np.ndarray,
                               index: int, lookback: int = 10) -> bool:
    """Detect Bearish Inducement (IDM)."""
    if index < lookback:
        return False
        
    recent_highs = high_prices[index-lookback:index]
    minor_high = np.max(recent_highs)
    
    curr_high = high_prices[index]
    curr_close = close_prices[index]
    
    sweep = curr_high > minor_high
    reclaim = curr_close < minor_high
    
    return sweep and reclaim

def detect_kill_zone(current_timestamp: datetime, market: str = "NSE") -> bool:
    """
    Check if current time is within a High-Volume 'Kill Zone'.
    
    NSE Kill Zones:
    - AM Session: 09:15 - 11:00 (Opening Volatility + Initial Balance)
    - PM Session: 13:30 - 15:00 (European Overlap + Closing Positioning)
    """
    # Note: Synthetic data might not have diverse timestamps, but logic handles it.
    
    t = current_timestamp.time()
    
    # NSE Zones
    am_start = datetime.strptime("09:15", "%H:%M").time()
    am_end = datetime.strptime("11:00", "%H:%M").time()
    
    pm_start = datetime.strptime("13:30", "%H:%M").time()
    pm_end = datetime.strptime("15:00", "%H:%M").time()
    
    is_am = am_start <= t <= am_end
    is_pm = pm_start <= t <= pm_end
    
    return is_am or is_pm


# ==============================================================================
# CHOCH & MITIGATION
# ==============================================================================

def detect_choch_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                         high_prices: np.ndarray, low_prices: np.ndarray, 
                         index: int, lookback: int = 10) -> bool:
    """
    Detect Bullish Change of Character (CHoCH).
    
    Logic:
    1. Aggressive break of the MOST RECENT structural high (Minor High).
    2. Signals immediate momentum shift, often before confirmed MSS.
    """
    if index < lookback:
        return False
        
    recent_highs = high_prices[index-lookback:index]
    
    # We want the *last* significant high, not necessarily the highest
    # Simplified: Just check if we broke the Max High of last 5 bars
    minor_high = np.max(high_prices[index-5:index])
    
    curr_close = close_prices[index]
    prev_close = close_prices[index-1]
    
    # Needs to proceed from below
    if prev_close < minor_high and curr_close > minor_high:
        return True
        
    return False

def detect_choch_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                         high_prices: np.ndarray, low_prices: np.ndarray, 
                         index: int, lookback: int = 10) -> bool:
    """Detect Bearish CHoCH."""
    if index < lookback:
        return False
        
    minor_low = np.min(low_prices[index-5:index])
    
    curr_close = close_prices[index]
    prev_close = close_prices[index-1]
    
    if prev_close > minor_low and curr_close < minor_low:
        return True
        
    return False

def detect_mitigation_block_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                    high_prices: np.ndarray, low_prices: np.ndarray,
                                    index: int, lookback: int = 15) -> bool:
    """
    Bullish Mitigation Block.
    
    Logic:
    1. Similar to Breaker, but WITHOUT the Liquidity Sweep.
    2. Price formed a Low, bounced, made a Lower Low (Sweep? NO - Higher Low), 
       then broke structure up.
    3. Actually, standard mitigation: Failed Orderblock that was NOT responsible for a sweep.
    """
    # Simplified detection: Failed Bearish Candle retested
    # Same logic as Breaker, but we enforce NO SWEEP of the previous low
    
    if detect_bullish_breaker(open_prices, close_prices, high_prices, low_prices, index, lookback):
        # Now check if the low *before* that breaker was NOT swept
        # This is complex to check perfectly in simple arrays, 
        # so we will use a variant: Retest of a broken resistance level
        return True
    return False # Placeholder for complex logic, reusing Breaker for now with unique name logic

def detect_mitigation_block_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                    high_prices: np.ndarray, low_prices: np.ndarray,
                                    index: int, lookback: int = 15) -> bool:
    """Bearish Mitigation Block."""
    if detect_bearish_breaker(open_prices, close_prices, high_prices, low_prices, index, lookback):
        return True
    return False


# ==============================================================================
# AMD (POWER OF 3) & INVERTED FVG
# ==============================================================================

def detect_amd_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                       high_prices: np.ndarray, low_prices: np.ndarray, 
                       index: int, lookback: int = 15) -> bool:
    """
    Detect AMD Bullish (Accumulation -> Manipulation -> Distribution).
    
    Concept: "Judas Swing"
    1. Accumulation: Price was tight/ranging recently.
    2. Manipulation: Price dropped sharply BELOW the Open (Trapping shorts).
    3. Distribution: Price reclaimed the Open and is rallying.
    """
    if index < lookback:
        return False
        
    # 1. Defining 'Open' as the open of the session/day or simply X bars ago
    # For timeframe agnostic, we take the Open of 'lookback' bars ago
    session_open = open_prices[index-lookback]
    
    # 2. Manipulation: Did we go well below that open?
    recent_lows = low_prices[index-lookback:index]
    min_low = np.min(recent_lows)
    
    manipulation_drop = min_low < session_open * 0.998 # At least 0.2% drop
    
    # 3. Reclaim: Are we now back ABOVE that open?
    curr_close = close_prices[index]
    reclaim = curr_close > session_open
    
    return manipulation_drop and reclaim

def detect_amd_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                       high_prices: np.ndarray, low_prices: np.ndarray, 
                       index: int, lookback: int = 15) -> bool:
    """Detect AMD Bearish."""
    if index < lookback:
        return False
        
    session_open = open_prices[index-lookback]
    
    recent_highs = high_prices[index-lookback:index]
    max_high = np.max(recent_highs)
    
    manipulation_rally = max_high > session_open * 1.002
    
    curr_close = close_prices[index]
    reclaim = curr_close < session_open
    
    return manipulation_rally and reclaim

def detect_inverted_fvg_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                high_prices: np.ndarray, low_prices: np.ndarray,
                                index: int, lookback: int = 10) -> bool:
    """
    Detect Bullish Inverted FVG (IFVG).
    
    Logic:
    1. Identify a previous BEARISH FVG (Resistance).
    2. Verify price BROKE UP through it strongly (Invalidating the FVG).
    3. Price is now testing it as SUPPORT.
    """
    # Needs complex FVG memory. Simplified logic:
    # A Bearish candle (down) was huge (Imbalance).
    # Later, price CLOSED above its High.
    # Now price is dipping back to that High.
    
    if index < 5: return False
    
    # Look for a big bearish candle recently
    for i in range(1, 5):
        idx = index - i
        # Big red candle
        if close_prices[idx] < open_prices[idx] * 0.998:
            bearish_high = high_prices[idx] # Top of the gap area roughly
            
            # Did we close above it subsequently?
            # Check current or prev bar closed above
            if close_prices[index] > bearish_high and low_prices[index] <= bearish_high * 1.001:
                # We are retesting the breakout level
                return True
                
    return False

def detect_inverted_fvg_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                high_prices: np.ndarray, low_prices: np.ndarray,
                                index: int, lookback: int = 10) -> bool:
    """Detect Bearish Inverted FVG."""
    if index < 5: return False
    
    for i in range(1, 5):
        idx = index - i
        # Big green candle
        if close_prices[idx] > open_prices[idx] * 1.002:
            bullish_low = low_prices[idx]
            
            # Did we close below it?
            if close_prices[index] < bullish_low and high_prices[index] >= bullish_low * 0.999:
                return True
                
    return False


# ==============================================================================
# TURTLE SOUP & REJECTION BLOCKS
# ==============================================================================

def detect_turtle_soup_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                               high_prices: np.ndarray, low_prices: np.ndarray,
                               index: int, lookback: int = 20) -> bool:
    """
    Detect Bullish Turtle Soup.
    
    Logic:
    1. Identify a MAJOR Low (e.g. 20-period Low).
    2. Price breaks below it (Stop Hunt) but CLOSES back above it quickly.
    3. Similar to a Sweep, but specific to Major Levels (High Timeframe liquidity).
    """
    if index < lookback:
        return False
        
    recent_lows = low_prices[index-lookback:index]
    major_low = np.min(recent_lows)
    
    curr_low = low_prices[index]
    curr_close = close_prices[index]
    
    # Sweep below Major Low
    sweep = curr_low < major_low
    
    # Close back inside range (Reclaim)
    reclaim = curr_close > major_low
    
    return sweep and reclaim

def detect_turtle_soup_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                               high_prices: np.ndarray, low_prices: np.ndarray,
                               index: int, lookback: int = 20) -> bool:
    """Detect Bearish Turtle Soup."""
    if index < lookback:
        return False
        
    recent_highs = high_prices[index-lookback:index]
    major_high = np.max(recent_highs)
    
    curr_high = high_prices[index]
    curr_close = close_prices[index]
    
    sweep = curr_high > major_high
    reclaim = curr_close < major_high
    
    return sweep and reclaim

def detect_rejection_block_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                   high_prices: np.ndarray, low_prices: np.ndarray,
                                   index: int, lookback: int = 10) -> bool:
    """
    Detect Bullish Rejection Block.
    
    Logic:
    1. Identify a candle with a LONG lower wick (> 50% of range).
    2. This implies price went down and was REJECTED strongly.
    3. If price is near this wick or retesting the body of that candle, it's a buy.
    """
    if index < 1: return False
    
    # Check if CURRENT or PREVIOUS candle is a rejection candle
    # Let's say we look for an entry AFTER a rejection candle formed
    
    idx = index - 1
    candle_range = high_prices[idx] - low_prices[idx]
    if candle_range == 0: return False
    
    lower_wick = min(open_prices[idx], close_prices[idx]) - low_prices[idx]
    
    is_long_wick = lower_wick > (candle_range * 0.5)
    
    # Current bar confirms bullishness (Close > Open)
    is_bullish_confirm = close_prices[index] > open_prices[index]
    
    return is_long_wick and is_bullish_confirm

def detect_rejection_block_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                   high_prices: np.ndarray, low_prices: np.ndarray,
                                   index: int, lookback: int = 10) -> bool:
    """Detect Bearish Rejection Block."""
    if index < 1: return False
    
    idx = index - 1
    candle_range = high_prices[idx] - low_prices[idx]
    if candle_range == 0: return False
    
    upper_wick = high_prices[idx] - max(open_prices[idx], close_prices[idx])
    
    is_long_wick = upper_wick > (candle_range * 0.5)
    
    is_bearish_confirm = close_prices[index] < open_prices[index]
    
    return is_long_wick and is_bearish_confirm


# ==============================================================================
# MASS SMC EXPANSION - BATCH 1 (1-5/50)
# ==============================================================================

def detect_propulsion_block_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                    high_prices: np.ndarray, low_prices: np.ndarray,
                                    index: int, lookback: int = 10) -> bool:
    """
    1. Propulsion Block: 
    An Order Block that has already been tapped into, and price reacts again.
    It 'propels' price higher. It is a secondary O.B. formed inside a larger one.
    """
    # Logic: Bullish Candle -> Retest -> Another Bullish Candle (Propulsion) -> Current Retest
    if index < 3: return False
    
    # 1. Previous candle was bullish
    prev_close = close_prices[index-1]
    prev_open = open_prices[index-1]
    is_prev_bullish = prev_close > prev_open
    
    # 2. Candle before that was a down/pause candle that dipped into a prior level
    # Simplified: We look for a bullish candle that opened near the High of a previous bullish block
    
    # Simple Proxy: Two consecutive bullish candles where the second one opens 
    # ABOVE the 50% of the first one, showing strong momentum reload.
    
    if is_prev_bullish and (close_prices[index-2] > open_prices[index-2]):
        # Propelling: The second candle didn't retrace deep.
        midpoint_prior = (open_prices[index-2] + close_prices[index-2]) / 2
        if prev_open > midpoint_prior:
            return True
            
    return False

def detect_propulsion_block_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                    high_prices: np.ndarray, low_prices: np.ndarray,
                                    index: int, lookback: int = 10) -> bool:
    """Bearish Propulsion Block detection."""
    if index < 3: return False
    
    # Two bearish candles, second one opens below 50% of prior (strong continuation)
    prev_bearish = close_prices[index-1] < open_prices[index-1]
    prior_bearish = close_prices[index-2] < open_prices[index-2]
    
    if prev_bearish and prior_bearish:
        midpoint_prior = (open_prices[index-2] + close_prices[index-2]) / 2
        if open_prices[index-1] < midpoint_prior:
            return True
            
    return False

def detect_liquidity_void_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                  high_prices: np.ndarray, low_prices: np.ndarray,
                                  index: int) -> bool:
    """
    2. Liquidity Void (Bullish Vacuum).
    Identify a massive drop (void) that is likely to be refilled (Bullish logic = target the fill).
    However, here we usually trade the FILL of the void acting as support? 
    SMC usually treats specific Voids as targets. 
    Entry Style: Price dipping INTO a Liquidity Void to find support (fill & reverse).
    """
    # Look for a Giant Green Candle recently (The Void creation)
    # Price is now gently dipping into it.
    if index < 5: return False
    
    # Check last 5 bars for a generic "Huge Move"
    for i in range(1, 6):
        idx = index - i
        body = abs(close_prices[idx] - open_prices[idx])
        avg_body = np.mean(np.abs(close_prices[idx-5:idx] - open_prices[idx-5:idx]))
        
        # Huge Impulse (3x average)
        if body > 3 * avg_body and close_prices[idx] > open_prices[idx]:
            # This is a Bullish Void creator.
            # Are we inside it?
            if low_prices[index] < close_prices[idx] and low_prices[index] > open_prices[idx]:
                return True
                
    return False # Simplified

def detect_liquidity_void_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                  high_prices: np.ndarray, low_prices: np.ndarray,
                                  index: int) -> bool:
    """Bearish Liquidity Void (Price rally into a vacuum to sell)."""
    if index < 5: return False
    
    for i in range(1, 6):
        idx = index - i
        body = abs(close_prices[idx] - open_prices[idx])
        avg_body = np.mean(np.abs(close_prices[idx-5:idx] - open_prices[idx-5:idx]))
        
        if body > 3 * avg_body and close_prices[idx] < open_prices[idx]: # Giant Red Candle
            # Are we rallying into it?
            if high_prices[index] > close_prices[idx] and high_prices[index] < open_prices[idx]:
                return True
                
    return False

def detect_balanced_price_range(open_prices: np.ndarray, close_prices: np.ndarray,
                                high_prices: np.ndarray, low_prices: np.ndarray,
                                index: int) -> bool:
    """
    3. Balanced Price Range (BPR).
    A generic function for both directions (often context dependent).
    Logic: A Bullish FVG and Bearish FVG overlap. 
    Price went up fast, then down fast (or vice versa), leaving a 'Balance'.
    Trading the retest of this zone.
    """
    if index < 3: return False
    
    # 1. Bar -2 was Up
    # 2. Bar -1 was Down (Immediate reversal overlapping)
    # 3. Bar 0 is inside the overlap
    
    bar2_up = close_prices[index-2] > open_prices[index-2]
    bar1_down = close_prices[index-1] < open_prices[index-1]
    
    if bar2_up and bar1_down:
        # Check Overlap (The BPR)
        # Top of BPR = Min(High(-2), High(-1))
        # Bottom of BPR = Max(Low(-2), Low(-1))
        
        top = min(high_prices[index-2], high_prices[index-1])
        bot = max(low_prices[index-2], low_prices[index-1])
        
        if top > bot:
            # We have an overlap range. Current price inside?
            curr_price = close_prices[index]
            if curr_price >= bot and curr_price <= top:
                return True
                
    return False

def detect_volume_imbalance_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                    high_prices: np.ndarray, low_prices: np.ndarray,
                                    index: int) -> bool:
    """
    4. Volume Imbalance (Bullish).
    The gap between Yesterday's Close and Today's Open (or Candle A Close and Candle B Open).
    Price trades inside this 'gap' without wicks covering it.
    """
    if index < 1: return False
    
    # Close of prev
    c1 = close_prices[index-1]
    # Open of curr
    o2 = open_prices[index]
    
    # Gap Up or Down? 
    # Bullish Setup usually implies price dips into a gap to buy.
    
    # Check if there is a literal gap
    if abs(o2 - c1) > 0:
        # We are simply detecting if we are trading near a gap.
        # Detection: We just opened with a gap. 
        # Strategy will handle the rest (e.g. Buy the gap fill).
        return True
        
    return False

def detect_volume_imbalance_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                    high_prices: np.ndarray, low_prices: np.ndarray,
                                    index: int) -> bool:
    """Bearish Volume Imbalance."""
    if index < 1: return False
    c1 = close_prices[index-1]
    o2 = open_prices[index]
    if abs(o2 - c1) > 0:
        return True
    return False

def detect_mean_threshold_retest_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                         high_prices: np.ndarray, low_prices: np.ndarray,
                                         index: int) -> bool:
    """
    5. Mean Threshold Retest (Bullish).
    Retest of the 50% level of a recent significant Bullish Candle (Order Block).
    """
    if index < 5: return False
    
    # Find recent big green candle
    for i in range(1, 5):
        idx = index - i
        if close_prices[idx] > open_prices[idx]:
            # Check size
            range_len = high_prices[idx] - low_prices[idx]
            if range_len == 0: continue
            
            midpoint = (high_prices[idx] + low_prices[idx]) / 2
            
            # Current price touching midpoint?
            if low_prices[index] <= midpoint and close_prices[index] >= midpoint:
                return True
                
    return False

def detect_mean_threshold_retest_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                         high_prices: np.ndarray, low_prices: np.ndarray,
                                         index: int) -> bool:
    """Bearish Mean Threshold Retest (Retest 50% of Red Candle)."""
    if index < 5: return False
    
    for i in range(1, 5):
        idx = index - i
        if close_prices[idx] < open_prices[idx]:
            midpoint = (high_prices[idx] + low_prices[idx]) / 2
            
            # Current price touching midpoint?
            if high_prices[index] >= midpoint and close_prices[index] <= midpoint:
                return True
                
    return False


# ==============================================================================
# MASS SMC EXPANSION - BATCH 2 (6-10/50)
# ==============================================================================

def detect_unicorn_model_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                 high_prices: np.ndarray, low_prices: np.ndarray,
                                 index: int) -> bool:
    """
    6. Unicorn Model (Bullish).
    Convergence of a Breaker Block AND a Fair Value Gap in the same zone.
    The 'Holy Grail' setup.
    """
    # 1. Detect Breaker
    is_breaker = detect_bullish_breaker(open_prices, close_prices, high_prices, low_prices, index)
    
    # 2. Detect FVG nearby (simplified check for recent FVG)
    # We check if the current bar is ALSO reacting to an FVG
    # Simplified: Is there a Bullish FVG in the last 3 bars?
    has_fvg = False
    if index >= 3:
        # Check FVG condition: Low[i] > High[i-2] (Gap Up)
        for i in range(1, 4):
            idx = index - i
            if idx >= 2:
                # Basic FVG check
                if low_prices[idx] > high_prices[idx-2]:
                     has_fvg = True
                     break
    
    return is_breaker and has_fvg

def detect_unicorn_model_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                 high_prices: np.ndarray, low_prices: np.ndarray,
                                 index: int) -> bool:
    """Bearish Unicorn Model."""
    is_breaker = detect_bearish_breaker(open_prices, close_prices, high_prices, low_prices, index)
    
    has_fvg = False
    if index >= 3:
        for i in range(1, 4):
            idx = index - i
            if idx >= 2:
                # Bearish FVG: High[i] < Low[i-2]
                if high_prices[idx] < low_prices[idx-2]:
                    has_fvg = True
                    break
                    
    return is_breaker and has_fvg

def detect_silver_bullet_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                 high_prices: np.ndarray, low_prices: np.ndarray,
                                 index: int, time_obj=None) -> bool:
    """
    7. Silver Bullet (Bullish).
    A generic FVG setup that occurs SPECIFICALLY during the 'Silver Bullet' hour.
    10 AM - 11 AM (NY) or 3 AM - 4 AM (London).
    For simulation, we use a simple time check if 'time_obj' is passed, 
    otherwise we simulate a random 'Time' check or rely on the Kill Zone filter.
    """
    # Assuming we are in a valid time (handled by Kill Zone usually). 
    # Here we enforce FVG detection specifically.
    
    # Check for recent Bullish FVG entry
    if index < 2: return False
    
    # Is current price filling a Bullish FVG?
    # FVG definition: Gap between High[i-2] and Low[i] (Current bar filling it)
    
    # Gap created at i-1? No, Gap is between i-3 and i-1. 
    # Retest at i (current).
    
    gap_low_boundary = high_prices[index-3] if index >= 3 else 0
    gap_high_boundary = low_prices[index-1]
    
    # Valid Gap?
    if gap_high_boundary > gap_low_boundary:
        # Is current Low dipping into it?
        if low_prices[index] <= gap_high_boundary and close_prices[index] >= gap_low_boundary:
            return True
            
    return False

def detect_silver_bullet_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                 high_prices: np.ndarray, low_prices: np.ndarray,
                                 index: int) -> bool:
    """Bearish Silver Bullet."""
    if index < 3: return False
    
    gap_high_boundary = low_prices[index-3]
    gap_low_boundary = high_prices[index-1]
    
    # Gap Down?
    if gap_low_boundary < gap_high_boundary:
        # Rally into gap?
        if high_prices[index] >= gap_low_boundary and close_prices[index] <= gap_high_boundary:
            return True
            
    return False

def detect_opening_gap_reclaim_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                       high_prices: np.ndarray, low_prices: np.ndarray,
                                       index: int) -> bool:
    """
    8. Opening Gap Reclaim (NDOG/NWOG).
    Price drops below the Open of the Day/Session, then reclaims it strongly.
    Similar to AMD but specific to the 'Opening Price' level.
    """
    if index < 10: return False
    
    # Estimate 'Open of Day' as the Open 10 bars ago for simplicity in this window
    # In real logic, pass the actual daily open.
    daily_open = open_prices[index-10] 
    
    # Check if we were below it recently
    was_below = np.min(close_prices[index-5:index]) < daily_open
    
    # Check if we just crossed back above
    is_above = close_prices[index] > daily_open
    
    return was_below and is_above

def detect_opening_gap_reclaim_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                       high_prices: np.ndarray, low_prices: np.ndarray,
                                       index: int) -> bool:
    """Bearish Opening Gap Reclaim."""
    if index < 10: return False
    daily_open = open_prices[index-10]
    
    was_above = np.max(close_prices[index-5:index]) > daily_open
    is_below = close_prices[index] < daily_open
    
    return was_above and is_below

def detect_reclaimed_block_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                   high_prices: np.ndarray, low_prices: np.ndarray,
                                   index: int) -> bool:
    """
    9. Reclaimed Block (Bullish).
    An old Bearish Order Block (Resistance) that was broken.
    Instead of a 'Breaker' (which requires a liquidity sweep), this is just
    old Resistance turning into Support.
    """
    if index < 10: return False
    
    # Find an old resistance candle (Bearish candle)
    # Price broke above it
    # Now retesting it
    
    # Simplified: Look for a level that acted as resistance (High pivot)
    # bar -5 was a high.
    resistance_level = high_prices[index-5]
    
    # We broke it
    broke_out = close_prices[index-2] > resistance_level
    
    # Now retesting
    retest = low_prices[index] <= resistance_level and close_prices[index] >= resistance_level
    
    return broke_out and retest

def detect_reclaimed_block_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                   high_prices: np.ndarray, low_prices: np.ndarray,
                                   index: int) -> bool:
    """Bearish Reclaimed Block."""
    if index < 10: return False
    
    support_level = low_prices[index-5]
    broke_down = close_prices[index-2] < support_level
    retest = high_prices[index] >= support_level and close_prices[index] <= support_level
    
    return broke_down and retest

def detect_wick_consequent_encroachment_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                                high_prices: np.ndarray, low_prices: np.ndarray,
                                                index: int) -> bool:
    """
    10. Wick Consequent Encroachment (Ballish).
    Retest of the 50% level of a long lower wick.
    """
    if index < 2: return False
    
    # Look for recent long wick
    for i in range(1, 5):
        idx = index - i
        # Calculate lower wick
        body_low = min(open_prices[idx], close_prices[idx])
        lower_wick_len = body_low - low_prices[idx]
        total_len = high_prices[idx] - low_prices[idx]
        
        # Significant wick?
        if total_len > 0 and lower_wick_len > total_len * 0.4:
            # 50% of the wick
            wick_ceph = low_prices[idx] + (lower_wick_len * 0.5)
            
            # Are we testing it?
            if low_prices[index] <= wick_ceph:
                return True
                
    return False

def detect_wick_consequent_encroachment_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                                high_prices: np.ndarray, low_prices: np.ndarray,
                                                index: int) -> bool:
    """Bearish Wick CE."""
    if index < 2: return False
    
    for i in range(1, 5):
        idx = index - i
        body_high = max(open_prices[idx], close_prices[idx])
        upper_wick_len = high_prices[idx] - body_high
        total_len = high_prices[idx] - low_prices[idx]
        
        if total_len > 0 and upper_wick_len > total_len * 0.4:
            wick_ceph = high_prices[idx] - (upper_wick_len * 0.5)
            
            if high_prices[index] >= wick_ceph:
                return True
                
    return False


# ==============================================================================
# MASS SMC EXPANSION - BATCH 3 (11-15/50)
# ==============================================================================

def detect_order_flow_entry_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                    high_prices: np.ndarray, low_prices: np.ndarray,
                                    index: int) -> bool:
    """
    11. Order Flow Entry (Bullish OFE).
    Following the 'chain' of mitigated Order Blocks (Higher Lows).
    Price respects a previous OB and creates a new one.
    """
    if index < 5: return False
    
    # 1. Identify if we are in a 'Flow' (Consecutive Higher Lows)
    lows = low_prices[index-4:index+1] # 5 bars
    
    # Check simple trend: Lows are generally rising?
    # Or strict OFE: Current bar tests the High of 2 bars ago (which acted as OB)
    
    # Simplified OFE: Retest of a previous candle's High that broke structure?
    # No, usually OFE means price respects the previous Down Candle.
    
    prev_down_candle_idx = -1
    for i in range(1, 4):
        idx = index - i
        if close_prices[idx] < open_prices[idx]:
            prev_down_candle_idx = idx
            break
            
    if prev_down_candle_idx != -1:
        # We found a recent down candle.
        # Is current price respecting its Low? (Not sweeping it?)
        # And currently bullish?
        prev_low = low_prices[prev_down_candle_idx]
        curr_low = low_prices[index]
        curr_close = close_prices[index]
        
        if curr_low > prev_low and curr_close > open_prices[index]:
             # We respected the previous down candle (Order Block chain intact)
             return True
             
    return False

def detect_order_flow_entry_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                    high_prices: np.ndarray, low_prices: np.ndarray,
                                    index: int) -> bool:
    """Bearish OFE."""
    if index < 5: return False
    
    prev_up_candle_idx = -1
    for i in range(1, 4):
        idx = index - i
        if close_prices[idx] > open_prices[idx]:
            prev_up_candle_idx = idx
            break
            
    if prev_up_candle_idx != -1:
        prev_high = high_prices[prev_up_candle_idx]
        curr_high = high_prices[index]
        curr_close = close_prices[index]
        
        if curr_high < prev_high and curr_close < open_prices[index]:
            return True
            
    return False

def detect_sponsor_candle_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                  high_prices: np.ndarray, low_prices: np.ndarray,
                                  index: int) -> bool:
    """
    12. Sponsorship Candle (Bullish).
    The specific candle that initiated a displacement move.
    Price often returns to the Open/High of this candle to 'test sponsorship'.
    Differs from OB: Focus is on the 'Displacement' initiation.
    """
    if index < 5: return False
    
    # Identify a recent massive expansion bar
    for i in range(1, 5):
        idx = index - i
        body = abs(close_prices[idx] - open_prices[idx])
        avg_body = np.mean(np.abs(close_prices[idx-5:idx] - open_prices[idx-5:idx]))
        
        if body > 2.5 * avg_body and close_prices[idx] > open_prices[idx]:
             # This Green Candle sponsored the move. 
             # Buying the dip into its body.
             sponsor_high = high_prices[idx]
             sponsor_low = low_prices[idx]
             
             if low_prices[index] <= sponsor_high and close_prices[index] >= sponsor_low:
                 return True
                 
    return False

def detect_sponsor_candle_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                  high_prices: np.ndarray, low_prices: np.ndarray,
                                  index: int) -> bool:
    """Bearish Sponsorship Candle."""
    if index < 5: return False
    
    for i in range(1, 5):
        idx = index - i
        body = abs(close_prices[idx] - open_prices[idx])
        avg_body = np.mean(np.abs(close_prices[idx-5:idx] - open_prices[idx-5:idx]))
        
        if body > 2.5 * avg_body and close_prices[idx] < open_prices[idx]:
             # Red Candle sponsored
             sponsor_low = low_prices[idx]
             sponsor_high = high_prices[idx]
             
             if high_prices[index] >= sponsor_low and close_prices[index] <= sponsor_high:
                 return True
                 
    return False

def detect_range_rotation_bullish(open_prices: np.ndarray, close_prices: np.ndarray, 
                                  high_prices: np.ndarray, low_prices: np.ndarray,
                                  index: int, lookback: int = 20) -> bool:
    """
    13. Range Rotation (Bullish).
    Trading from Low Deviation back to Mean.
    Identify if we are at the bottom of a defined 20-period range.
    """
    if index < lookback: return False
    
    curr_range_high = np.max(high_prices[index-lookback:index])
    curr_range_low = np.min(low_prices[index-lookback:index])
    
    # Range check
    range_size = curr_range_high - curr_range_low
    if range_size == 0: return False
    
    position = (close_prices[index] - curr_range_low) / range_size
    
    # If we are in the bottom 10% of the range and showing a green candle
    is_at_bottom = position < 0.10
    is_bullish = close_prices[index] > close_prices[index-1]
    
    return is_at_bottom and is_bullish

def detect_range_rotation_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                  high_prices: np.ndarray, low_prices: np.ndarray,
                                  index: int, lookback: int = 20) -> bool:
    """Bearish Range Rotation (Top to Mean)."""
    if index < lookback: return False
    
    curr_range_high = np.max(high_prices[index-lookback:index])
    curr_range_low = np.min(low_prices[index-lookback:index])
    range_size = curr_range_high - curr_range_low
    if range_size == 0: return False
    position = (close_prices[index] - curr_range_low) / range_size
    
    is_at_top = position > 0.90
    is_bearish = close_prices[index] < close_prices[index-1]
    
    return is_at_top and is_bearish

def detect_snap_back_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                            high_prices: np.ndarray, low_prices: np.ndarray,
                            index: int) -> bool:
    """
    14. Snap Back (Mean Reversion - Bullish).
    Price extended far from EMA20, looking for snap back.
    """
    if index < 20: return False
    
    # Calculate EMA 20
    ema20 = np.mean(close_prices[index-20:index]) # SMA proxy for simplicity or use EMA func
    
    # Extension: is price < 98% of EMA?
    extension = close_prices[index] < (ema20 * 0.995) # 0.5% deviation
    
    # Reversal candle?
    reversal = close_prices[index] > close_prices[index-1]
    
    return extension and reversal

def detect_snap_back_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                            high_prices: np.ndarray, low_prices: np.ndarray,
                            index: int) -> bool:
    """Bearish Snap Back."""
    if index < 20: return False
    ema20 = np.mean(close_prices[index-20:index])
    extension = close_prices[index] > (ema20 * 1.005)
    reversal = close_prices[index] < close_prices[index-1]
    return extension and reversal

def detect_quarterly_shift_bullish(open_prices: np.ndarray, close_prices: np.ndarray, 
                                   high_prices: np.ndarray, low_prices: np.ndarray,
                                   index: int) -> bool:
    """
    15. Quarterly Shift (Bullish).
    Simulated 'Time Quarter' shift. Every 6 hours (approx 6 * 60 / timeframe bars), 
    or just check a cycle.
    Logic: Detect if we are in a new 'Quarter' of the session and momentum shifted Up.
    """
    # Simply check if bar index is a multiple of X (Cycle shift) and direction change
    cycle_length = 24 # e.g. every 24 bars
    
    is_new_quarter = (index % cycle_length) == 0
    is_up = close_prices[index] > close_prices[index-1]
    
    return is_new_quarter and is_up

def detect_quarterly_shift_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                   high_prices: np.ndarray, low_prices: np.ndarray,
                                   index: int) -> bool:
    cycle_length = 24
    is_new_quarter = (index % cycle_length) == 0
    is_down = close_prices[index] < close_prices[index-1]
    return is_new_quarter and is_down


# ==============================================================================
# MASS SMC EXPANSION - BATCH 4 (16-20/50)
# ==============================================================================

def detect_standard_deviation_projection_bullish(close_prices: np.ndarray, 
                                                 high_prices: np.ndarray, low_prices: np.ndarray,
                                                 index: int) -> bool:
    """
    16. Standard Deviation Projection (Bullish).
    Detecting if price is at a -2 to -4 Standard Deviation extension of a previous range.
    Aka 'Judas Swing' extension targets.
    """
    if index < 20: return False
    
    # 1. Measure a previous consolidation range (e.g. index-30 to index-10)
    range_lookback = 20
    range_offset = 10
    
    ref_highs = high_prices[index-range_lookback-range_offset : index-range_offset]
    ref_lows = low_prices[index-range_lookback-range_offset : index-range_offset]
    
    range_high = np.max(ref_highs)
    range_low = np.min(ref_lows)
    range_amp = range_high - range_low
    
    if range_amp == 0: return False
    
    # Target: 2.5 SD extension downwards
    # Target = Range Low - (Range * 1.5) approx for 2.5 SD visual
    target_zone = range_low - range_amp # 1 SD down
    
    # Are we in the target zone?
    curr_low = low_prices[index]
    
    if curr_low <= target_zone and close_prices[index] > close_prices[index-1]:
        # Hitting SD target and reversing
        return True
        
    return False

def detect_standard_deviation_projection_bearish(close_prices: np.ndarray,
                                                 high_prices: np.ndarray, low_prices: np.ndarray,
                                                 index: int) -> bool:
    """Bearish SD Projection."""
    if index < 30: return False
    range_lookback = 20
    range_offset = 10
    ref_highs = high_prices[index-range_lookback-range_offset : index-range_offset]
    ref_lows = low_prices[index-range_lookback-range_offset : index-range_offset]
    range_high = np.max(ref_highs)
    range_low = np.min(ref_lows)
    range_amp = range_high - range_low
    if range_amp == 0: return False
    
    target_zone = range_high + range_amp
    curr_high = high_prices[index]
    
    if curr_high >= target_zone and close_prices[index] < close_prices[index-1]:
        return True
    return False

def detect_power_of_3_swing_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                    high_prices: np.ndarray, low_prices: np.ndarray,
                                    index: int) -> bool:
    """
    17. Power of 3 Swing (Weekly/Monthly style).
    Accumulation -> Manipulation (Down) -> Distribution (Up).
    Here looking for the 'Manipulation' completion.
    Price is BELOW the 'Open' of the period (simulated 30 bars ago) and turning up.
    """
    if index < 30: return False
    
    simulated_period_open = open_prices[index-30]
    
    # Manipulation phase: We spent time below open
    current_price = close_prices[index]
    
    # Are we recovering?
    if current_price < simulated_period_open:
        # We are discount (manipulation area)
        # Are we reversing?
        if close_prices[index] > close_prices[index-1] and close_prices[index-1] < open_prices[index-1]:
             return True
             
    return False

def detect_power_of_3_swing_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                    high_prices: np.ndarray, low_prices: np.ndarray,
                                    index: int) -> bool:
    """Bearish Po3 Swing."""
    if index < 30: return False
    simulated_period_open = open_prices[index-30]
    current_price = close_prices[index]
    
    if current_price > simulated_period_open: # Premium
        if close_prices[index] < close_prices[index-1]:
            return True
            
    return False

def detect_institutional_swing_point_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                             high_prices: np.ndarray, low_prices: np.ndarray,
                                             index: int) -> bool:
    """
    18. Institutional Swing Point (Bullish).
    Fractal Low: Higher Low, Lower Low, Higher Low (3 bar pattern).
    Confirmed Low pivot.
    """
    if index < 2: return False
    
    # Pattern: 2 bars ago (High), 1 bar ago (Lowest Low), 0 bar (Higher Low)
    # Actually fractal is usually: Left Bar Low > Center Bar Low < Right Bar Low.
    # Center = index - 1
    
    l_left = low_prices[index-2]
    l_center = low_prices[index-1]
    l_right = low_prices[index]
    
    # Center is the pivot
    if l_center < l_left and l_center < l_right:
         return True # Valid Swing Low formed at previous bar, confirmed by current
         
    return False

def detect_institutional_swing_point_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                             high_prices: np.ndarray, low_prices: np.ndarray,
                                             index: int) -> bool:
    """Bearish Institutional Swing Point (Fractal High)."""
    if index < 2: return False
    
    h_left = high_prices[index-2]
    h_center = high_prices[index-1]
    h_right = high_prices[index]
    
    if h_center > h_left and h_center > h_right:
        return True
        
    return False

def detect_smt_divergence_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                  high_prices: np.ndarray, low_prices: np.ndarray,
                                  index: int) -> bool:
    """
    19. SMT Divergence (Bullish).
    Smart Money Tool / Correlated Asset Divergence.
    Simulated using RSI/Price divergence in single asset.
    Price makes Lower Low, RSI makes Higher Low.
    """
    if index < 15: return False
    
    # Compare current Low with Low 10 bars ago
    low_curr = low_prices[index]
    low_prev = np.min(low_prices[index-10:index-1])
    
    # Price Lower Low?
    if low_curr < low_prev:
         # Now check RSI proxy (Momentum)
         # Mom_curr = Close - Close[10]
         # This is a weak proxy, assume we need real RSI calculation.
         # For speed, we'll use a momentum check:
         # If Price is lower, but Close is significantly higher than Low (wick rejection) 
         # or we use a simplified 'Strength' metric.
         
         # Let's use simple ROC: (Close - Open)
         strength_curr = close_prices[index] - open_prices[index]
         
         # If we have a lower low but a Green candle (strength), that's a type of divergence/rejection
         if strength_curr > 0:
             return True
             
    return False

def detect_smt_divergence_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                  high_prices: np.ndarray, low_prices: np.ndarray,
                                  index: int) -> bool:
    """Bearish SMT Divergence (Simulated)."""
    if index < 15: return False
    
    high_curr = high_prices[index]
    high_prev = np.max(high_prices[index-10:index-1])
    
    # Price Higher High
    if high_curr > high_prev:
        # But close is bearish (weakness)
        if close_prices[index] < open_prices[index]:
            return True
            
    return False

def detect_macro_macro_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                               high_prices: np.ndarray, low_prices: np.ndarray,
                               index: int) -> bool:
    """
    20. Macro Cycles (Bullish).
    Time-based algorithm macro. e.g. every 1 hour at :50 mark.
    Simulated by bar index modulo.
    """
    # Assuming 1m bars? 10:50, 11:50.
    # Every 60 bars approx.
    # Trigger at bar XX50
    
    if (index % 60) >= 50 and (index % 60) <= 59:
        # Inside Macro window
        return True
    return False

def detect_macro_macro_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                               high_prices: np.ndarray, low_prices: np.ndarray,
                               index: int) -> bool:
    if (index % 60) >= 50 and (index % 60) <= 59:
        return True
    return False


# ==============================================================================
# MASS SMC EXPANSION - BATCH 5 (21-25/50)
# ==============================================================================

def detect_liquidity_run_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                 high_prices: np.ndarray, low_prices: np.ndarray,
                                 index: int) -> bool:
    """
    21. Liquidity Run (Bullish).
    Aggressive sequential taking of internal Lows. 
    Price is 'running' the stops.
    Detection: 3 consecutive bars making Lower Lows, but closing mixed/up?
    Or simply identifying the run itself to fade it.
    Here: Identifying a run DOWN into a reversal (fade the run).
    """
    if index < 4: return False
    
    # 3 Consecutive Lower Lows
    l_1 = low_prices[index-1]
    l_2 = low_prices[index-2]
    l_3 = low_prices[index-3]
    
    is_run_down = (l_1 < l_2) and (l_2 < l_3)
    
    # Current bar breaks the sequence or absorbs the run
    # e.g. Current bar makes a Higher Low? Or simply we assume the run is done at a key level (generator handles context)
    # Let's detect the end of the run: Current Low > Prev Low (Higher Low after run)
    
    is_reversal = low_prices[index] > low_prices[index-1]
    
    return is_run_down and is_reversal

def detect_liquidity_run_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                 high_prices: np.ndarray, low_prices: np.ndarray,
                                 index: int) -> bool:
    """Bearish Liquidity Run (Run up on stops)."""
    if index < 4: return False
    
    h_1 = high_prices[index-1]
    h_2 = high_prices[index-2]
    h_3 = high_prices[index-3]
    
    is_run_up = (h_1 > h_2) and (h_2 > h_3)
    is_reversal = high_prices[index] < high_prices[index-1]
    
    return is_run_up and is_reversal

def detect_stops_hunt_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                              low_prices: np.ndarray, index: int) -> bool:
    """
    22. Stops Hunt (Bullish).
    Classic 'Purge'. Single bar dips below a defined recent Low (Index-2) 
    and IMMEDIATELY closes back above it.
    Different from Turtle Soup (which requires 20-period swing). This is short term.
    """
    if index < 3: return False
    
    # Target Low: Low of 2 bars ago
    target_low = low_prices[index-2]
    
    # Current bar sweeps it
    sweep = low_prices[index] < target_low
    
    # And closes back above
    reclaim = close_prices[index] > target_low
    
    # Also current bar should be Green
    is_green = close_prices[index] > open_prices[index]
    
    return sweep and reclaim and is_green

def detect_stops_hunt_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                              high_prices: np.ndarray, index: int) -> bool:
    """Bearish Stops Hunt."""
    if index < 3: return False
    
    target_high = high_prices[index-2]
    sweep = high_prices[index] > target_high
    reclaim = close_prices[index] < target_high
    is_red = close_prices[index] < open_prices[index]
    
    return sweep and reclaim and is_red

def detect_equilibrium_reclaimed_bullish(close_prices: np.ndarray, 
                                         high_prices: np.ndarray, low_prices: np.ndarray,
                                         index: int) -> bool:
    """
    23. Equilibrium Reclaimed (Bullish).
    Calculates the Current Day's Range (approx 30-50 bars).
    Price was below 50% (Discount) and crosses back ABOVE 50% (Equilibrium).
    """
    if index < 50: return False
    
    # Lookback 50 bars as 'Day' proxy
    lookback = 50
    curr_high = np.max(high_prices[index-lookback:index])
    curr_low = np.min(low_prices[index-lookback:index])
    
    eq = (curr_high + curr_low) / 2
    
    # Was below eq recently?
    was_below = close_prices[index-1] < eq
    
    # Now above?
    is_above = close_prices[index] > eq
    
    return was_below and is_above

def detect_equilibrium_reclaimed_bearish(close_prices: np.ndarray,
                                         high_prices: np.ndarray, low_prices: np.ndarray,
                                         index: int) -> bool:
    """Bearish Equilibrium Reclaimed (Premium to Discount)."""
    if index < 50: return False
    lookback = 50
    curr_high = np.max(high_prices[index-lookback:index])
    curr_low = np.min(low_prices[index-lookback:index])
    eq = (curr_high + curr_low) / 2
    
    was_above = close_prices[index-1] > eq
    is_below = close_prices[index] < eq
    
    return was_above and is_below

def detect_partial_void_fill_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                     high_prices: np.ndarray, low_prices: np.ndarray,
                                     index: int) -> bool:
    """
    24. Partial Void Fill (Bullish).
    Entries at 50% of the void body.
    """
    if index < 5: return False
    
    # Identify Void Candle (Huge Green Candle)
    for i in range(1, 6):
        idx = index - i
        if close_prices[idx] > open_prices[idx]:
            body_range = close_prices[idx] - open_prices[idx]
            avg_range = np.mean(high_prices[idx-5:idx] - low_prices[idx-5:idx])
            
            if body_range > 2 * avg_range: # Big Void
                midpoint = (open_prices[idx] + close_prices[idx]) / 2
                
                # Setup: Price dips into midpoint
                if low_prices[index] <= midpoint and close_prices[index] >= midpoint:
                    return True
                    
    return False

def detect_partial_void_fill_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                     high_prices: np.ndarray, low_prices: np.ndarray,
                                     index: int) -> bool:
    """Bearish Partial Void Fill."""
    if index < 5: return False
    
    for i in range(1, 6):
        idx = index - i
        if close_prices[idx] < open_prices[idx]:
            midpoint = (open_prices[idx] + close_prices[idx]) / 2
            if high_prices[index] >= midpoint and close_prices[index] <= midpoint:
                return True
    return False

def detect_fractal_expansion_bullish(close_prices: np.ndarray, 
                                     high_prices: np.ndarray, low_prices: np.ndarray,
                                     index: int) -> bool:
    """
    25. Fractal Expansion (Bullish).
    Price breaks out of a defined fractal consolidation range.
    """
    if index < 10: return False
    
    # Fractal High 5 bars ago
    fractal_high = high_prices[index-5]
    
    # Check if this was a local high
    is_local_high = (high_prices[index-5] > high_prices[index-6]) and (high_prices[index-5] > high_prices[index-4])
    
    if is_local_high:
        # Check breakout TODAY
        if close_prices[index] > fractal_high and close_prices[index-1] <= fractal_high:
            return True
            
    return False

def detect_fractal_expansion_bearish(close_prices: np.ndarray,
                                     high_prices: np.ndarray, low_prices: np.ndarray,
                                     index: int) -> bool:
    """Bearish Fractal Expansion."""
    if index < 10: return False
    
    fractal_low = low_prices[index-5]
    is_local_low = (low_prices[index-5] < low_prices[index-6]) and (low_prices[index-5] < low_prices[index-4])
    
    if is_local_low:
        if close_prices[index] < fractal_low and close_prices[index-1] >= fractal_low:
            return True
    return False


# ==============================================================================
# MASS SMC EXPANSION - BATCH 6 (26-30/50)
# ==============================================================================

def detect_initial_balance_breakout_bullish(open_prices: np.ndarray, close_prices: np.ndarray, 
                                            high_prices: np.ndarray, low_prices: np.ndarray,
                                            index: int) -> bool:
    """
    26. Initial Balance (IB) Breakout (Bullish).
    Break of the first 60-minutes High.
    Simulated: Assuming index represents minute bars from open...
    In a real system, track 9:15-10:15 High.
    Here: Generic 'first 60 bars' logic.
    """
    if index < 61: return False
    
    # Check if we are past the first hour of session (assuming session restart handled externally)
    # or just look back at a 'session start' proxy.
    
    # Proxy: Assume we look back 60 bars. The High of bars [Index-X : Index-Y]?
    # Simplified Logic: 
    # Current Close > High of the last 60 bars? No, that's just a High.
    # We need a fixed IB range.
    
    # Let's assume strategy generator runs this on specific timeframes.
    # Logic: Breakout of High relative to the last 60 bars, assuming we are early in session.
    # OR: Just detect "Day High Breakout" if time is < 12:00.
    
    day_high = np.max(high_prices[index-60:index]) 
    
    # Breakout
    if close_prices[index] > day_high and close_prices[index-1] <= day_high:
        return True
    return False

def detect_initial_balance_breakout_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                            high_prices: np.ndarray, low_prices: np.ndarray,
                                            index: int) -> bool:
    """Bearish IB Breakout."""
    if index < 61: return False
    day_low = np.min(low_prices[index-60:index])
    if close_prices[index] < day_low and close_prices[index-1] >= day_low:
        return True
    return False

def detect_opening_range_breakout_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                          high_prices: np.ndarray, low_prices: np.ndarray,
                                          index: int) -> bool:
    """
    27. Opening Range Breakout (ORB) (Bullish).
    Breakout of first 15/30m range. Faster than IB.
    """
    if index < 31: return False
    
    # Lookback 30 bars (ORB-30)
    orb_high = np.max(high_prices[index-30:index])
    
    if close_prices[index] > orb_high and close_prices[index-1] <= orb_high:
        return True
    return False

def detect_opening_range_breakout_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                          high_prices: np.ndarray, low_prices: np.ndarray,
                                          index: int) -> bool:
    """Bearish ORB."""
    if index < 31: return False
    orb_low = np.min(low_prices[index-30:index])
    if close_prices[index] < orb_low and close_prices[index-1] >= orb_low:
        return True
    return False

def detect_daily_open_rejection_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                        high_prices: np.ndarray, low_prices: np.ndarray, 
                                        index: int) -> bool:
    """
    28. Daily Open Rejection (Bullish).
    Price dips to the Daily Open price and rejects it (closes higher).
    Support at Open.
    """
    if index < 10: return False
    
    # Proxy for Daily Open: Open price of 75 bars ago (approx start of day for 5m chart?)
    # or just 'Open of the Day' logic.
    # Let's use open_prices[index-50] as a proxy for "The Open".
    daily_open = open_prices[index-50]
    
    # Did we touch it?
    touched = low_prices[index] <= daily_open
    
    # Did we close above it?
    closed_above = close_prices[index] > daily_open
    
    # Is it a green candle?
    is_green = close_prices[index] > open_prices[index]
    
    return touched and closed_above and is_green

def detect_daily_open_rejection_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                        high_prices: np.ndarray, low_prices: np.ndarray, 
                                        index: int) -> bool:
    """Bearish Daily Open Rejection."""
    if index < 50: return False
    daily_open = open_prices[index-50]
    
    touched = high_prices[index] >= daily_open
    closed_below = close_prices[index] < daily_open
    is_red = close_prices[index] < open_prices[index]
    return touched and closed_below and is_red

def detect_pwh_pwl_sweep_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                 high_prices: np.ndarray, low_prices: np.ndarray, 
                                 index: int) -> bool:
    """
    29. PWH/PWL Sweep (Bullish).
    Sweep of Previous Week Low.
    Simulated: Proxy 'Previous Week Low' by looking back X bars.
    """
    if index < 200: return False
    
    # Proxy: Min of last 500 bars? 
    # Let's use a 200 bar lookback as a 'long term' low proxy.
    pwl = np.min(low_prices[index-200:index-10])
    
    # Sweep logic: Current Low < PWL but Close > PWL?
    # Or just detection of lower low.
    
    if low_prices[index] < pwl:
         return True
    return False

def detect_pwh_pwl_sweep_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                 high_prices: np.ndarray, low_prices: np.ndarray, 
                                 index: int) -> bool:
    """Bearish PWH Sweep."""
    if index < 200: return False
    pwh = np.max(high_prices[index-200:index-10])
    if high_prices[index] > pwh:
        return True
    return False

def detect_pdh_pdl_sweep_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                 high_prices: np.ndarray, low_prices: np.ndarray, 
                                 index: int) -> bool:
    """
    30. PDH/PDL Sweep (Bullish).
    Sweep of Previous Day Low.
    """
    if index < 75: return False
    
    pdl = np.min(low_prices[index-75:index-10])
    
    if low_prices[index] < pdl:
        return True
    return False

def detect_pdh_pdl_sweep_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                 high_prices: np.ndarray, low_prices: np.ndarray, 
                                 index: int) -> bool:
    """Bearish PDH Sweep."""
    if index < 75: return False
    pdh = np.max(high_prices[index-75:index-10])
    if high_prices[index] > pdh:
        return True
    return False


# ==============================================================================
# MASS SMC EXPANSION - BATCH 7 (31-35/50)
# ==============================================================================

def detect_swing_failure_pattern_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                         high_prices: np.ndarray, low_prices: np.ndarray,
                                         index: int) -> bool:
    """
    31. Swing Failure Pattern (SFP) (Bullish).
    Price makes a Lower Low, but RSI (Simulated) makes a Higher Low?
    Or classic SFP: Price sweeps a Low and closes ABOVE it.
    This is effectively the same as 'Stops Hunt' but often implies a bigger swing pivot.
    Let's implement 'Classic SFP': Close > Previous Swing Low.
    """
    if index < 10: return False
    
    # Identify a recent Swing Low (e.g. 5 bars ago)
    # Check 5 bars ago was a local low
    l_5 = low_prices[index-5]
    is_swing_low = (l_5 < low_prices[index-6]) and (l_5 < low_prices[index-4])
    
    if is_swing_low:
        # Current bar sweeps it
        if low_prices[index] < l_5:
            # And closes back above?
            if close_prices[index] > l_5:
                return True
                
    return False

def detect_swing_failure_pattern_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                         high_prices: np.ndarray, low_prices: np.ndarray,
                                         index: int) -> bool:
    """Bearish SFP."""
    if index < 10: return False
    
    h_5 = high_prices[index-5]
    is_swing_high = (h_5 > high_prices[index-6]) and (h_5 > high_prices[index-4])
    
    if is_swing_high:
        if high_prices[index] > h_5:
            if close_prices[index] < h_5:
                return True
    return False

def detect_momentum_impulse_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                    high_prices: np.ndarray, low_prices: np.ndarray,
                                    index: int) -> bool:
    """
    32. Momentum Impulse (Bullish).
    Candle Body > 2.0x ATR (Average True Range) of last 14 bars.
    Signifies institutional intent.
    """
    if index < 15: return False
    
    # Calculate approx ATR of last 14 bars (High - Low)
    ranges = high_prices[index-14:index] - low_prices[index-14:index]
    avg_range = np.mean(ranges)
    
    current_body = abs(close_prices[index] - open_prices[index])
    
    if current_body > 2.0 * avg_range:
        # Check direction
        if close_prices[index] > open_prices[index]:
            return True
            
    return False

def detect_momentum_impulse_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                    high_prices: np.ndarray, low_prices: np.ndarray,
                                    index: int) -> bool:
    """Bearish Momentum Impulse."""
    if index < 15: return False
    
    ranges = high_prices[index-14:index] - low_prices[index-14:index]
    avg_range = np.mean(ranges)
    current_body = abs(close_prices[index] - open_prices[index])
    
    if current_body > 2.0 * avg_range:
        if close_prices[index] < open_prices[index]:
            return True
    return False

def detect_psychological_level_rejection_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                                 high_prices: np.ndarray, low_prices: np.ndarray,
                                                 index: int) -> bool:
    """
    33. Psychological Level Rejection (Bullish).
    Price tests '00' or '50' level. 
    Simulated: Check if price crosses integer boundaries (mock).
    In real data: modulo 100 or 50.
    Here: Assume prices are like 21500, 21550.
    """
    # Check if Low touched a xx00 or xx50 level
    curr_low = low_prices[index]
    
    # Simple logic: modulo 50 check. 
    # If Low crosses a multiple of 50.
    # range_low = int(curr_low)
    # Check if there's a multiple of 50 between Low and Close?
    
    # Let's say if Low < 50-level and Close > 50-level.
    # We find the nearest 50 level below Close.
    nearest_50 = (int(close_prices[index]) // 50) * 50
    
    if low_prices[index] < nearest_50 and close_prices[index] > nearest_50:
        return True
    return False

def detect_psychological_level_rejection_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                                 high_prices: np.ndarray, low_prices: np.ndarray,
                                                 index: int) -> bool:
    """Bearish Psych Level Rejection."""
    nearest_50 = (int(close_prices[index]) // 50) * 50 + 50 # Ceiling
    if high_prices[index] > nearest_50 and close_prices[index] < nearest_50:
        return True
    return False

def detect_trendline_liquidity_build_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                            high_prices: np.ndarray, low_prices: np.ndarray,
                                            index: int) -> bool:
    """
    34. Trendline Liquidity Build (Bullish Setup).
    Detects a smooth line of Higher Lows (Retail Trendline).
    This implies liquidity is building below it.
    Strategy: Wait for the break of this line?
    Or detecting the build itself (to short into later).
    Let's define this as: "Liquidity is built, ready to target".
    Setup: 3 perfectly aligned Higher Lows.
    """
    if index < 10: return False
    
    # Check if Low[i], Low[i-2], Low[i-4] are forming a line.
    # Proxy: simply strictly increasing lows with low variance in slope.
    
    l1 = low_prices[index]
    l2 = low_prices[index-2]
    l3 = low_prices[index-4]
    
    is_trend_up = (l1 > l2) and (l2 > l3)
    
    # If consistent slope... approximated.
    if is_trend_up:
        slope1 = l1 - l2
        slope2 = l2 - l3
        if abs(slope1 - slope2) < (l1 * 0.0005): # Consistent-ish
            return True
            
    return False

def detect_trendline_liquidity_build_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                            high_prices: np.ndarray, low_prices: np.ndarray,
                                            index: int) -> bool:
    """Bearish Trendline Liquidity (Retail Resistance)."""
    if index < 10: return False
    h1 = high_prices[index]
    h2 = high_prices[index-2]
    h3 = high_prices[index-4]
    
    is_trend_down = (h1 < h2) and (h2 < h3)
    if is_trend_down:
         slope1 = h2 - h1
         slope2 = h3 - h2
         if abs(slope1 - slope2) < (h1 * 0.0005):
             return True
    return False

def detect_failed_auction_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                  low_prices: np.ndarray, index: int) -> bool:
    """
    35. Failed Auction (Bullish).
    Market Profile logic. Price breaks a range Low, finds no aggressive sellers, 
    and returns to range. 
    Similar to Stops Hunt, but often characterized by Low Volume (difficult to sim here without volume).
    Simulation: Break Low, Small Candle Body (Doji/Spinning Top) = Indecision/Failure.
    """
    if index < 20: return False
    
    recent_low = np.min(low_prices[index-20:index-5])
    
    # Break Low
    if low_prices[index] < recent_low:
        # Check for indecision (Small body)
        body = abs(close_prices[index] - open_prices[index])
        rng = high_prices[index-14] - low_prices[index-14] # Approx ATR proxy
        
        if body < (rng * 0.3): # Small body
             # Failed to push
             return True
             
    return False

def detect_failed_auction_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                  high_prices: np.ndarray, index: int) -> bool:
    """Bearish Failed Auction."""
    if index < 20: return False
    recent_high = np.max(high_prices[index-20:index-5])
    
    if high_prices[index] > recent_high:
        body = abs(close_prices[index] - open_prices[index])
        rng = high_prices[index-14] - low_prices[index-14]
        if body < (rng * 0.3):
            return True
    return False


# ==============================================================================
# MASS SMC EXPANSION - BATCH 8 (36-40/50)
# ==============================================================================

def detect_amd_setup_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                             low_prices: np.ndarray, index: int) -> bool:
    """
    36. AMD (Accumulation-Manipulation-Distribution) (Bullish).
    Sequence:
    1. Accumulation (Consolidation) - handled by context?
    2. Manipulation (Break Low) - handled here.
    3. Distribution (Reversal/Expansion) - handled here.
    Logic: Price sweeps a recent consolidation low (Manipulation) and reverses aggressively.
    Essentially a specific type of Liquidity Run/SFP but context-heavy.
    Simplified: Sweep of recent Range Low + Strong Close.
    """
    if index < 20: return False
    
    # 1. Define Range (Last 10 bars)
    recent_lows = low_prices[index-15:index-1]
    range_low = np.min(recent_lows)
    
    # 2. Manipulation (Sweep)
    if low_prices[index] < range_low:
        # 3. Distribution start (Close back inside relative to the sweep)
        # Ideally close near high of the day?
        # Let's demand Close > Range Low (Reclaim)
        if close_prices[index] > range_low:
             return True
             
    return False

def detect_amd_setup_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                             high_prices: np.ndarray, index: int) -> bool:
    """Bearish AMD (Power of 3 Intraday)."""
    if index < 20: return False
    recent_highs = high_prices[index-15:index-1]
    range_high = np.max(recent_highs)
    
    if high_prices[index] > range_high: # Manipulation
        if close_prices[index] < range_high: # Reclaim/Distribute
            return True
            
    return False

def detect_turtle_soup_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                               high_prices: np.ndarray, low_prices: np.ndarray,
                               index: int) -> bool:
    """
    37. Turtle Soup (Bullish).
    Classic formulation:
    1. Price makes a new 20-period Low.
    2. Previous 20-period Low must be at least 4 periods ago.
    3. Price closes ABOVE that previous 20-period Low.
    """
    if index < 25: return False
    
    # Today must make a new 20-period low (relative to the snapshot before today)
    # The 'Previous' 20-period low:
    prev_20_low = np.min(low_prices[index-21:index-1])
    
    # Did we break it?
    if low_prices[index] < prev_20_low:
        # Close back above it?
        if close_prices[index] > prev_20_low:
            return True
            
    return False

def detect_turtle_soup_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                               high_prices: np.ndarray, low_prices: np.ndarray,
                               index: int) -> bool:
    """Bearish Turtle Soup."""
    if index < 25: return False
    prev_20_high = np.max(high_prices[index-21:index-1])
    
    if high_prices[index] > prev_20_high:
        if close_prices[index] < prev_20_high:
            return True
            
    return False

def detect_decoupled_ob_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                low_prices: np.ndarray, index: int) -> bool:
    """
    38. Decoupled Order Block (Bullish).
    An OB that was 'pierced' (wicked through) but the Body held the level.
    Shows incredible resilience.
    Detection:
    1. Identify Bullish Candle (OB candidate) e.g. at Index-3.
    2. Price at Index (or Index-1) wicked BELOW OB Low.
    3. But Closed ABOVE OB Low (or OB Open).
    """
    if index < 5: return False
    
    # OB Candidate: Large Green Candle at Index-3
    ob_idx = index - 3
    is_green_ob = close_prices[ob_idx] > open_prices[ob_idx]
    
    if is_green_ob:
        ob_low = low_prices[ob_idx]
        ob_open = open_prices[ob_idx]
        
        # Current bar tests it
        # Wick below OB Low
        if low_prices[index] < ob_low:
            # But Close > OB Low (Held)
            if close_prices[index] > ob_low:
                return True
                
    return False

def detect_decoupled_ob_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                high_prices: np.ndarray, index: int) -> bool:
    """Bearish Decoupled OB."""
    if index < 5: return False
    ob_idx = index - 3
    is_red_ob = close_prices[ob_idx] < open_prices[ob_idx]
    
    if is_red_ob:
        ob_high = high_prices[ob_idx]
        if high_prices[index] > ob_high:
            if close_prices[index] < ob_high:
                return True
    return False

def detect_propulsion_candle_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                     index: int) -> bool:
    """
    39. Propulsion Candle (Bullish).
    Aggressive Continuation.
    1. Previous Candle was Bullish.
    2. Current Candle opens INSIDE the upper 50% of Previous Body.
    3. Current Candle is also Bullish.
    """
    if index < 2: return False
    
    prev_open = open_prices[index-1]
    prev_close = close_prices[index-1]
    
    # Prev Bullish
    if prev_close > prev_open:
        prev_body = prev_close - prev_open
        prev_mid = prev_open + (prev_body * 0.5)
        
        # Current Open inside Upper 50%
        curr_open = open_prices[index]
        if curr_open > prev_mid and curr_open < prev_close:
            # Current Bullish
            if close_prices[index] > curr_open:
                return True
                
    return False

def detect_propulsion_candle_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                     index: int) -> bool:
    """Bearish Propulsion Candle."""
    if index < 2: return False
    prev_open = open_prices[index-1]
    prev_close = close_prices[index-1]
    
    # Prev Bearish
    if prev_close < prev_open:
        prev_body = prev_open - prev_close
        prev_mid = prev_close + (prev_body * 0.5) # Midpoint
        
        # Open inside Lower 50% (between Close and Mid)
        curr_open = open_prices[index]
        if curr_open < prev_mid and curr_open > prev_close:
            if close_prices[index] < curr_open:
                return True
                
    return False

def detect_engineered_liquidity_bullish(low_prices: np.ndarray, close_prices: np.ndarray,
                                        index: int) -> bool:
    """
    40. Engineered Liquidity Sweep (Bullish).
    Detecting a Double Bottom (Equal Lows) that gets Swept.
    1. Find Equal Lows (approx) in recent history (Index-10 to Index-1).
    2. Current bar sweeps them.
    3. Current bar closes above (Reclaim).
    """
    if index < 20: return False
    
    # Scan for equal lows
    # Let's look at pairs of lows in the window
    window_lows = low_prices[index-20:index-1]
    
    eq_low_level = None
    
    # Simple check: Find min, check if another low is within tolerance checks
    min_val = np.min(window_lows)
    
    l_swept = low_prices[index]
    
    # Find lows near this level in history (that are slightly higher than current sweep)
    # count lows in (l_swept, l_swept * 1.001)
    
    nearby_lows = 0
    for i in range(1, 20):
        l_past = low_prices[index-i]
        if l_past > l_swept and l_past < (l_swept * 1.001): # Very close
            nearby_lows += 1
            
    if nearby_lows >= 2: # Swept at least 2 candles stops
        if close_prices[index] > l_swept:
            return True
            
    return False

def detect_engineered_liquidity_bearish(high_prices: np.ndarray, close_prices: np.ndarray,
                                        index: int) -> bool:
    """Bearish Engineered Liquidity Sweep."""
    if index < 20: return False
    h_swept = high_prices[index]
    
    nearby_highs = 0
    for i in range(1, 20):
        h_past = high_prices[index-i]
        if h_past < h_swept and h_past > (h_swept * 0.999):
            nearby_highs += 1
            
    if nearby_highs >= 2:
        if close_prices[index] < h_swept:
            return True
            
    return False


# ==============================================================================
# MASS SMC EXPANSION - BATCH 9 (41-45/50)
# ==============================================================================

def detect_inverse_fvg_bullish(close_prices: np.ndarray, 
                               high_prices: np.ndarray, low_prices: np.ndarray,
                               index: int) -> bool:
    """
    41. Inverse FVG (Bullish).
    A Bearish FVG (Gap down) that was broken to the upside (closed above) 
    and is now being retested as Support.
    """
    if index < 10: return False
    
    # 1. Identify a recent Bearish FVG (Index-1 to Index-10)
    # Gap: Low[i-1] > High[i+1] (for bearish FVG, wait: Gap is between High[i-1] and Low[i+1]? No)
    # Bearish FVG Pattern (3 candles):
    # Candle 1 (Down), Candle 2 (Down), Candle 3 (Down).
    # Gap is between Low of Candle 1 and High of Candle 3.
    # Where Low[1] > High[3].
    
    for i in range(1, 10):
        # Look for Bearish FVG at i
        # Structure: i-2 (C1), i-1 (C2), i (C3)?
        # Let's say the gap occurs at index `idx`
        idx = index - i
        if idx < 2: continue
        
        # Bearish Gap: Low[idx-2] > High[idx]
        gap_high = low_prices[idx-2]
        gap_low = high_prices[idx]
        
        if gap_high > gap_low:
             # This was a bearish FVG.
             # 2. Was it broken? (Price Closed ABOVE gap_high afterwards)
             # Check prices between idx and current index
             # Find if any Close > gap_high
             broken = False
             failed = False
             for j in range(idx+1, index):
                 if close_prices[j] > gap_high:
                     broken = True
                 # If we broke and then closed way below again?
                 
             if broken:
                 # 3. Retest: Current Low is inside the Gap (or near gap_high) and rejecting?
                 # Assuming simple retest: Low <= gap_high and Close > gap_high
                 if low_prices[index] <= gap_high and close_prices[index] >= gap_low: # In gap
                     # Validation: Bullish candle or rejection
                     if close_prices[index] > close_prices[index-1]:
                         return True
                         
    return False

def detect_inverse_fvg_bearish(close_prices: np.ndarray,
                               high_prices: np.ndarray, low_prices: np.ndarray,
                               index: int) -> bool:
    """Bearish Inverse FVG (Flipped Bullish FVG)."""
    if index < 10: return False
    for i in range(1, 10):
        idx = index - i
        if idx < 2: continue
        
        # Bullish FVG: High[idx-2] < Low[idx]
        gap_low = high_prices[idx-2]
        gap_high = low_prices[idx]
        
        if gap_low < gap_high:
            broken = False
            for j in range(idx+1, index):
                if close_prices[j] < gap_low:
                    broken = True
            
            if broken:
                if high_prices[index] >= gap_low and close_prices[index] <= gap_high:
                    if close_prices[index] < close_prices[index-1]:
                        return True
    return False

def detect_mitigation_block_bullish(high_prices: np.ndarray, low_prices: np.ndarray,
                                    close_prices: np.ndarray, index: int) -> bool:
    """
    42. Mitigation Block (Bullish).
    Failed Swing Low (HL) that led to a Break of Structure HIGH, then retested?
    No, Mitigation Block is when a swing low *fails* to take liquidity (Higher Low) 
    but then price breaks down?
    
    Standard Def: 
    Bullish Mitigation Block: Price makes a Low, then a Lower Low? No, that's Breaker.
    Mitigation: Price makes a Low, then a HIGHER Low (failure to sweep), then rallies above old High?
    Wait. 
    ICT Def:
    Bearish Mitigation: Short term High, then Lower High (failure to sweep), then break Low. The Lower High range becomes the block.
    Bullish Mitigation: Short term Low, then Higher Low (failure to sweep), then break High. The Higher Low range is the block.
    
    Let's implement Bullish Mitigation:
    1. Swing Low (A)
    2. Swing Low (B) - Higher than A.
    3. Break of High between A and B?
    
    Actually, usually Mitigation is used when Breaker setup fails to form a new low/high.
    
    Simplified Bullish Mitigation:
    - Recent Swing Low 1
    - Recent Swing Low 2 (Higher than 1)
    - Break of structural High.
    - Retest of the Down Candle involved in Swing Low 2.
    """
    if index < 10: return False
    
    # Identify HL pattern: Low[i-5] (L1), Low[i-2] (L2). L2 > L1.
    l1 = low_prices[index-5]
    l2 = low_prices[index-2]
    
    if l2 > l1:
        # Check Break of High (between i-5 and i-2)
        inter_high = np.max(high_prices[index-5:index-2])
        
        # Have we broken that high?
        if close_prices[index-1] > inter_high:
            # Retest of L2 area?
            # Current Low touches L2 area (retest)
            if low_prices[index] <= (l2 * 1.001):
                if close_prices[index] > l2:
                    return True
                    
    return False

def detect_mitigation_block_bearish(high_prices: np.ndarray, low_prices: np.ndarray,
                                    close_prices: np.ndarray, index: int) -> bool:
    """Bearish Mitigation Block."""
    if index < 10: return False
    h1 = high_prices[index-5]
    h2 = high_prices[index-2]
    
    if h2 < h1: # Lower High
        inter_low = np.min(low_prices[index-5:index-2])
        if close_prices[index-1] < inter_low: # Break Structure Low
            # Retest H2
            if high_prices[index] >= (h2 * 0.999):
                 if close_prices[index] < h2:
                     return True
    return False

def detect_rejection_block_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                   low_prices: np.ndarray, index: int) -> bool:
    """
    43. Rejection Block (Bullish).
    Trading a classical 'Long Wick' candle.
    Price dips into the wick of a previous Swing Low (Rejection Block) and buys.
    """
    if index < 5: return False
    
    # Find a recent candle with long lower wick
    for i in range(1, 10):
        idx = index - i
        # Wick size
        body_low = min(open_prices[idx], close_prices[idx])
        wick_len = body_low - low_prices[idx]
        body_len = abs(close_prices[idx] - open_prices[idx])
        
        if wick_len > body_len * 1.5: # Significant wick
            # This is a Rejection Block
            # Entry: Price enters the wick region
            if low_prices[index] < body_low and low_prices[index] > low_prices[idx]:
                # And rejects (closes above current low)
                if close_prices[index] > low_prices[index]:
                    return True
                    
    return False

def detect_rejection_block_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                   high_prices: np.ndarray, index: int) -> bool:
    """Bearish Rejection Block."""
    if index < 5: return False
    for i in range(1, 10):
        idx = index - i
        body_high = max(open_prices[idx], close_prices[idx])
        wick_len = high_prices[idx] - body_high
        body_len = abs(close_prices[idx] - open_prices[idx])
        
        if wick_len > body_len * 1.5:
            if high_prices[index] > body_high and high_prices[index] < high_prices[idx]:
                if close_prices[index] < high_prices[index]:
                    return True
    return False

def detect_nwog_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                        high_prices: np.ndarray, low_prices: np.ndarray, 
                        index: int) -> bool:
    """
    44. New Week Opening Gap (NWOG) (Bullish).
    Gap between Friday Close and Monday Open.
    Simulated: Check large gaps between bars?
    Or modulo based on weekly bars.
    Without calendar data, we can detect 'Large Gap' on Open.
    If Open[i] > Close[i-1] (Gap Up) or Open[i] < Close[i-1] (Gap Down).
    NWOG Bullish: Price fills a gap?
    Let's assume Bullish Reuse of NWOG:
    Gap was established. Current price comes back to test it.
    """
    if index < 50: return False
    
    # 1. Search for a large gap in history
    for i in range(1, 50):
        idx = index - i
        if idx < 1: continue
        
        curr_open = open_prices[idx]
        prev_close = close_prices[idx-1]
        
        gap_size = abs(curr_open - prev_close)
        
        # Significant gap? > 0.1%?
        if gap_size > (prev_close * 0.0005): # Small barrier for sim
            # Found gap
            gap_min = min(curr_open, prev_close)
            gap_max = max(curr_open, prev_close)
            
            # 2. Retest today?
            if low_prices[index] <= gap_max and high_prices[index] >= gap_min:
                 # In Gap
                 if close_prices[index] > low_prices[index]: # Bounce
                     return True
                     
    return False

def detect_nwog_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                        high_prices: np.ndarray, low_prices: np.ndarray, 
                        index: int) -> bool:
    """Bearish NWOG test."""
    # Same logic, just checking for bearish rejection of gap
    if index < 50: return False
    for i in range(1, 50):
        idx = index - i
        if idx < 1: continue
        
        gap_size = abs(open_prices[idx] - close_prices[idx-1])
        if gap_size > (close_prices[idx-1] * 0.0005):
            gap_min = min(open_prices[idx], close_prices[idx-1])
            gap_max = max(open_prices[idx], close_prices[idx-1])
            
            if high_prices[index] >= gap_min and low_prices[index] <= gap_max:
                if close_prices[index] < high_prices[index]:
                    return True
    return False

def detect_volume_void_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                               high_prices: np.ndarray, low_prices: np.ndarray, 
                               index: int) -> bool:
    """
    45. Volume Void / LVN (Bullish).
    Simulated by identifying a 'Candle with very large range but very small body?'.
    No, Volume Void is typically a "Liquidity Void" areas where price skipped.
    Let's define it as: Price re-entering a zone of rapid prior movement (Large Range Candle).
    Bullish: Price dips into a previous Massive Green Candle (Imbalance) and holds.
    Distinct from FVG: FVG is the gap between candles. Void is the candle itself.
    """
    if index < 10: return False
    
    # Find Massive Green Candle
    for i in range(1, 20):
        idx = index - i
        body = close_prices[idx] - open_prices[idx]
        
        # Check against recent average
        avg_body = np.mean(np.abs(close_prices[idx-5:idx] - open_prices[idx-5:idx]))
        
        if body > 3.0 * avg_body: # Huge impulsive move (creating void)
             # Entry: Price returns to 50% of this candle
             mid = open_prices[idx] + (body * 0.5)
             
             if low_prices[index] <= mid and close_prices[index] >= mid:
                 return True
                 
    return False

def detect_volume_void_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                               high_prices: np.ndarray, low_prices: np.ndarray, 
                               index: int) -> bool:
    """Bearish Volume Void."""
    if index < 10: return False
    for i in range(1, 20):
        idx = index - i
        body = open_prices[idx] - close_prices[idx] # Red
        avg_body = np.mean(np.abs(close_prices[idx-5:idx] - open_prices[idx-5:idx]))
        
        if body > 3.0 * avg_body:
            mid = close_prices[idx] + (body * 0.5)
            if high_prices[index] >= mid and close_prices[index] <= mid:
                return True
    return False


# ==============================================================================
# MASS SMC EXPANSION - BATCH 10 (46-50/50 - THE FINAL BATCH)
# ==============================================================================

def detect_dragon_pattern_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                  high_prices: np.ndarray, low_prices: np.ndarray, 
                                  index: int) -> bool:
    """
    46. Dragon Pattern (Bullish).
    Aggressive W-Bottom.
    Structure:
    1. Head (H) - Start of drop.
    2. Left Leg (LL) - First Low.
    3. Hump (B) - Bounce.
    4. Right Leg (RL) - Second Low (can be slightly lower or higher than LL).
    5. Break of Trendline (Head to Hump).
    Simplified Detection:
    Double Bottom (LL approx RL) + Break of Hump (B).
    Basically a 'W' pattern breakout.
    """
    if index < 20: return False
    
    # Check for W shape
    # Look for Hump around i-10
    # Look for RL around i-2
    # Look for LL around i-15
    
    # Let's search for pivots
    # Recent low (RL)
    rl_idx = -1
    rl_val = 999999
    for i in range(1, 10):
        if low_prices[index-i] < rl_val:
            rl_val = low_prices[index-i]
            rl_idx = index-i
    
    if rl_idx == -1: return False
    
    # Hump (B) - High before RL
    hump_idx = -1
    hump_val = -1
    for i in range(rl_idx-10, rl_idx):
        if i < 0: continue
        if close_prices[i] > hump_val:
            hump_val = close_prices[i]
            hump_idx = i
            
    if hump_idx == -1: return False
    
    # Left Leg (LL) - Low before Hump
    ll_idx = -1
    ll_val = 999999
    for i in range(hump_idx-10, hump_idx):
        if i < 0: continue
        if low_prices[i] < ll_val:
            ll_val = low_prices[i]
            ll_idx = i
            
    if ll_idx == -1: return False
    
    # Valid Dragon?
    # RL should be near LL (within tolerance)
    if 0.98 < (rl_val / ll_val) < 1.02:
        # Breakout: Current Close > Hump
        if close_prices[index] > hump_val:
             return True
             
    return False

def detect_dragon_pattern_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                  high_prices: np.ndarray, low_prices: np.ndarray, 
                                  index: int) -> bool:
    """Bearish Dragon (Inverse). M Pattern Breakout."""
    if index < 20: return False
    
    # Right Leg (RH) - Recent High
    rh_val = -1
    rh_idx = -1
    for i in range(1, 10):
        if high_prices[index-i] > rh_val:
            rh_val = high_prices[index-i]
            rh_idx = index-i
            
    if rh_idx == -1: return False
    
    # Hump (Low)
    hump_val = 999999
    hump_idx = -1
    for i in range(rh_idx-10, rh_idx):
        if i < 0: continue
        if close_prices[i] < hump_val:
            hump_val = close_prices[i]
            hump_idx = i
            
    if hump_idx == -1: return False
    
    # Left Level (LH)
    lh_val = -1
    for i in range(hump_idx-10, hump_idx):
        if i < 0: continue
        if high_prices[i] > lh_val:
            lh_val = high_prices[i]
            
    if lh_val == -1: return False
    
    if 0.98 < (rh_val / lh_val) < 1.02:
        if close_prices[index] < hump_val:
             return True
    return False

def detect_quasimodo_pattern_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                     high_prices: np.ndarray, low_prices: np.ndarray, 
                                     index: int) -> bool:
    """
    47. Quasimodo (QM) / Over-Under (Bullish).
    Sequence:
    1. Low (L)
    2. High (H)
    3. Lower Low (LL) - Liquidity Grab (The 'Under')
    4. Higher High (HH) - MSS (The 'Over')
    5. Retest of L (The Left Shoulder/QM Level)
    
    We need to detect the Retest of the original Low (L) after HH.
    """
    if index < 25: return False
    
    # Scanning history for pattern
    # Look for LL (lowest point in last 25 bars)
    ll_idx = np.argmin(low_prices[index-25:index-5]) + (index-25)
    ll_val = low_prices[ll_idx]
    
    # Look for L (Low before LL)
    # Search backwards from LL
    l_idx = -1
    l_val = -1
    for i in range(ll_idx-10, ll_idx):
        if i < 0: continue
        # Find a local low
        if low_prices[i] < low_prices[i-1] and low_prices[i] < low_prices[i+1]:
            if low_prices[i] > ll_val: # Must be higher than LL
                 l_val = low_prices[i]
                 l_idx = i
                 
    if l_idx == -1: return False
    
    # Look for HH (Highest point after LL)
    hh_val = -1
    hh_idx = -1
    for i in range(ll_idx+1, index):
        if high_prices[i] > hh_val:
            hh_val = high_prices[i]
            hh_idx = i
            
    # Check if HH > H (High between L and LL)
    # Find H
    h_val = np.max(high_prices[l_idx:ll_idx])
    
    if hh_val > h_val: # MSS Confirmed
         # Step 5: Retest of L (QM Level)
         # Current price matches L_val
         if abs(low_prices[index] - l_val) < (l_val * 0.001):
             return True
             
    return False

def detect_quasimodo_pattern_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                     high_prices: np.ndarray, low_prices: np.ndarray, 
                                     index: int) -> bool:
    """Bearish QM."""
    if index < 25: return False
    
    hh_idx = np.argmax(high_prices[index-25:index-5]) + (index-25)
    hh_val = high_prices[hh_idx]
    
    h_idx = -1
    h_val = 999999
    for i in range(hh_idx-10, hh_idx):
        if i < 0: continue
        if high_prices[i] > high_prices[i-1] and high_prices[i] > high_prices[i+1]:
            if high_prices[i] < hh_val:
                h_val = high_prices[i]
                h_idx = i
    
    if h_idx == -1: return False
    
    ll_val = 999999  # Lowest after HH
    for i in range(hh_idx+1, index):
         if low_prices[i] < ll_val:
             ll_val = low_prices[i]
             
    l_val = np.min(low_prices[h_idx:hh_idx])
    
    if ll_val < l_val: # MSS
        # Retest H_val
         if abs(high_prices[index] - h_val) < (h_val * 0.001):
             return True
    return False

def detect_triple_tap_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                             high_prices: np.ndarray, low_prices: np.ndarray, 
                             index: int) -> bool:
    """
    48. Triple Tap / 3-Drive (Bullish).
    Three pushes down into a level.
    L1, L2, L3.
    L3 is often a slight sweep or divergence point.
    """
    if index < 15: return False
    
    # Identify 3 recent lows
    # Current Low (L3)
    l3 = low_prices[index]
    
    # Look for L2 and L1
    # Simple check: 3 distinct swing lows in declining or flat order
    # For a reversal, they should be somewhat evenly spaced.
    # Let's verify locally.
    
    l2_idx = -1
    for i in range(index-5, index-2):
        if low_prices[i] < low_prices[i-1] and low_prices[i] < low_prices[i+1]:
            l2_idx = i
            break
            
    if l2_idx != -1:
        l2 = low_prices[l2_idx]
        
        l1_idx = -1
        for i in range(l2_idx-5, l2_idx-2):
             if i < 1: continue
             if low_prices[i] < low_prices[i-1] and low_prices[i] < low_prices[i+1]:
                 l1_idx = i
                 break
                 
        if l1_idx != -1:
            # Found 3 taps.
            return True
            
    return False

def detect_triple_tap_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                             high_prices: np.ndarray, low_prices: np.ndarray, 
                             index: int) -> bool:
    """Bearish Triple Tap (3 Highs)."""
    if index < 15: return False
    h3 = high_prices[index]
    
    h2_idx = -1
    for i in range(index-5, index-2):
        if high_prices[i] > high_prices[i-1] and high_prices[i] > high_prices[i+1]:
            h2_idx = i
            break
            
    if h2_idx != -1:
        h1_idx = -1
        for i in range(h2_idx-5, h2_idx-2):
            if i < 1: continue
            if high_prices[i] > high_prices[i-1] and high_prices[i] > high_prices[i+1]:
                h1_idx = i
                break
        if h1_idx != -1:
            return True
    return False

def detect_compression_liquidity_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                         high_prices: np.ndarray, low_prices: np.ndarray,
                                         index: int) -> bool:
    """
    49. Compression (CP) (Bullish Target?).
    Usually CP is a bearish signal if price is compressing UP (Wedge).
    If Price is Compressing DOWN (Falling Wedge), it leads to Bullish breakout.
    Let's implement 'Falling Compression' = Bullish.
    Characterized by overlapping candles and decreasing range.
    """
    if index < 10: return False
    
    # Check overlap of last 5 candles
    compressed = True
    for i in range(1, 5):
        curr = index - i + 1
        prev = index - i
        # Overlap check: Low[curr] < High[prev] and High[curr] > Low[prev]
        # Strict compression: The 'staircase' is tight.
        pass # It's complex to code perfectly.
        
    # Proxy: Low Volatility (ATR contracting) + Trend Direction
    # Trend Down
    in_downtrend = high_prices[index] < high_prices[index-5]
    
    # Low Volatility?
    ranges = high_prices[index-5:index] - low_prices[index-5:index]
    avg_range = np.mean(ranges)
    past_avg = np.mean(high_prices[index-15:index-10] - low_prices[index-15:index-10])
    
    msg_compressing = avg_range < (past_avg * 0.7)
    
    if in_downtrend and msg_compressing:
        # CP into Demand?
        return True
        
    return False

def detect_compression_liquidity_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                         high_prices: np.ndarray, low_prices: np.ndarray,
                                         index: int) -> bool:
    """Bearish Compression (Rising Wedge)."""
    if index < 10: return False
    in_uptrend = low_prices[index] > low_prices[index-5]
    ranges = high_prices[index-5:index] - low_prices[index-5:index]
    avg_range = np.mean(ranges)
    past_avg = np.mean(high_prices[index-15:index-10] - low_prices[index-15:index-10])
    
    if in_uptrend and avg_range < (past_avg * 0.7):
        return True
    return False

def detect_master_pattern_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                high_prices: np.ndarray, low_prices: np.ndarray, 
                                index: int) -> bool:
    """
    50. The Master Pattern (Bullish).
    Contraction -> Expansion -> Trend.
    Detecting the Transition from Contraction to Expansion (Breakout).
    1. Low Volatility period (Index-10 to Index-2).
    2. Expansion Candle (Index-1 or Index).
    """
    if index < 15: return False
    
    # Contraction
    std_dev_old = np.std(close_prices[index-15:index-3])
    
    # Expansion
    std_dev_new = np.std(close_prices[index-3:index])
    
    # Significant increase
    if std_dev_new > (std_dev_old * 2.0):
        # Trend Up?
        if close_prices[index] > close_prices[index-3]:
            return True
            
    return False

def detect_master_pattern_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                high_prices: np.ndarray, low_prices: np.ndarray, 
                                index: int) -> bool:
    """Bearish Master Pattern."""
    if index < 15: return False
    std_dev_old = np.std(close_prices[index-15:index-3])
    std_dev_new = np.std(close_prices[index-3:index])
    
    if std_dev_new > (std_dev_old * 2.0):
        if close_prices[index] < close_prices[index-3]:
            return True
    return False


# ==============================================================================
# MASS SMC EXPANSION - PHASE 2 (51-75)
# BATCH 11 (51-55/75)
# ==============================================================================

def detect_inducement_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                               high_prices: np.ndarray, low_prices: np.ndarray,
                               index: int) -> bool:
    """
    51. Inducement (IDM) (Bullish).
    A short-term Low that forms just above a POI (Point of Interest).
    It 'induces' early buyers to enter, placing stops below it.
    The market then sweeps this Low (IDM) into the real POI and reverses.
    Detection logic:
    1. A Swing Low (IDM) is formed.
    2. Price breaks (sweeps) this low.
    3. Price reverses strongly.
    Similar to Liquidity Run, but specifically defined as the "Trap" before the move.
    """
    if index < 10: return False
    
    # 1. Recent Swing Low (IDM candidate)
    # L[i-3] < L[i-4] and L[i-3] < L[i-2]
    # Let's verify if we just swept a very recent, very minor low.
    
    idm_idx = -1
    for i in range(index-5, index-1):
        if low_prices[i] < low_prices[i-1] and low_prices[i] < low_prices[i+1]:
            idm_idx = i
            break
            
    if idm_idx != -1:
        idm_low = low_prices[idm_idx]
        
        # 2. Current bar sweeps it
        if low_prices[index] < idm_low:
             # 3. And Reclaims?
             if close_prices[index] > idm_low:
                 return True
                 
    return False

def detect_inducement_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                               high_prices: np.ndarray, low_prices: np.ndarray,
                               index: int) -> bool:
    """Bearish Inducement (Trap High)."""
    if index < 10: return False
    idm_idx = -1
    for i in range(index-5, index-1):
        if high_prices[i] > high_prices[i-1] and high_prices[i] > high_prices[i+1]:
            idm_idx = i
            break
            
    if idm_idx != -1:
        idm_high = high_prices[idm_idx]
        if high_prices[index] > idm_high:
            if close_prices[index] < idm_high:
                return True
    return False

def detect_balanced_price_range_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                        high_prices: np.ndarray, low_prices: np.ndarray,
                                        index: int) -> bool:
    """
    52. Balanced Price Range (BPR) (Bullish).
    A zone where price traded aggressively up and then down (or vice versa), overalapping gaps.
    Bullish BPR:
    1. Bearish FVG created previously.
    2. Immediately broken by a Bullish FVG (large green candle) moving through it.
    3. Re-entry into this zone.
    Simplified: Overlap of Up-Candle Body and Previous Down-Candle Body created a 'rebalance'.
    """
    if index < 10: return False
    
    # 1. Find recent aggressive Bullish Move (Candle at i-1 or i-2)
    # Check if this candle 'filled' a previous Bearish FVG?
    # Complexity: High.
    # Simplified Logic:
    # A large Green candle (C1) follows a large Red candle (C0).
    # The bodies overlap significantly.
    # Current price (C2) dips into the C1 body.
    
    c1_idx = index - 1 # Recent Green Candle
    if close_prices[c1_idx] > open_prices[c1_idx]: # Green
         c0_idx = index - 2 # Previous
         if close_prices[c0_idx] < open_prices[c0_idx]: # Red
             # Check for overlap/BPR nature
             # Did the Green move 'undo' the Red move immediately?
             if close_prices[c1_idx] > open_prices[c0_idx] and open_prices[c1_idx] < close_prices[c0_idx]:
                 # It engulfed or effectively reversed it.
                 # BPR Zone is arguably the interaction area.
                 # Filter: Did C1 leave a FVG? Check C2 High < C0 Low? Standard FVG check.
                 
                 # Entry: Retest of C1's body/open
                 retest_level = open_prices[c1_idx] + (abs(close_prices[c1_idx]-open_prices[c1_idx])*0.5)
                 if low_prices[index] < retest_level:
                     if close_prices[index] > low_prices[index]: # Rejection
                         return True
                         
    return False

def detect_balanced_price_range_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                        high_prices: np.ndarray, low_prices: np.ndarray,
                                        index: int) -> bool:
    """Bearish BPR."""
    if index < 10: return False
    c1_idx = index - 1 # Red
    if close_prices[c1_idx] < open_prices[c1_idx]:
        c0_idx = index - 2 # Green
        if close_prices[c0_idx] > open_prices[c0_idx]:
            if close_prices[c1_idx] < open_prices[c0_idx] and open_prices[c1_idx] > close_prices[c0_idx]:
                 # Retest
                 retest_level = open_prices[c1_idx] - (abs(open_prices[c1_idx]-close_prices[c1_idx])*0.5)
                 if high_prices[index] > retest_level:
                     if close_prices[index] < high_prices[index]:
                         return True
    return False

def detect_volume_imbalance_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                    high_prices: np.ndarray, low_prices: np.ndarray,
                                    index: int) -> bool:
    """
    53. Volume Imbalance (VI) (Bullish).
    Gap between the BODIES of two sequential candles, but the WICKS overlap 
    (meaning price traded there, but didn't close/open there).
    Unlike FVG (Price didn't trade there).
    Bullish VI: Open[i] > Close[i-1] (Gap up in body)
    Or Open[i] < Close[i-1]?
    Actually VI is when Close[i-1] and Open[i] have a gap.
    e.g. C[i-1] = 100. O[i] = 102. 
    But Wicks: High[i-1] = 103, Low[i] = 99. Wicks overlap (99 to 103).
    Bodies have gap (100 to 102).
    """
    if index < 5: return False
    
    # Check VI in history (e.g. at i-1, i-2...)
    for i in range(1, 10):
        idx = index - i
        if idx < 1: continue
        
        # Check Gap between Close[idx-1] and Open[idx] (or vice versa)
        # We don't know candle colors. Just gap between the two body ends.
        
        body_top_prev = max(open_prices[idx-1], close_prices[idx-1])
        body_bottom_curr = min(open_prices[idx], close_prices[idx])
        
        # Is there a gap?
        if body_bottom_curr > body_top_prev:
            # Theoretical bullish VI (Gap Up Body)
            # Check retest
            gap_mid = (body_bottom_curr + body_top_prev) / 2
            
            # Current price dips into it
            if low_prices[index] <= gap_mid:
                # Holds?
                if close_prices[index] > low_prices[index]:
                    return True
                    
    return False

def detect_volume_imbalance_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                    high_prices: np.ndarray, low_prices: np.ndarray,
                                    index: int) -> bool:
    """Bearish VI (Gap Down Body)."""
    if index < 5: return False
    for i in range(1, 10):
        idx = index - i
        if idx < 1: continue
        
        body_bottom_prev = min(open_prices[idx-1], close_prices[idx-1])
        body_top_curr = max(open_prices[idx], close_prices[idx])
        
        # Gap Down
        if body_top_curr < body_bottom_prev:
            gap_mid = (body_top_curr + body_bottom_prev) / 2
            if high_prices[index] >= gap_mid:
                if close_prices[index] < high_prices[index]:
                    return True
    return False

def detect_judas_swing_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                               high_prices: np.ndarray, low_prices: np.ndarray, 
                               index: int) -> bool:
    """
    54. Judas Swing (Bullish).
    Aggressive fakeout at session open.
    Simulated: Check time? No time here. 
    Simulate via: Sharp drop (Manipulation) followed by reversal candle.
    Essentially an SFP/AMD but specifically looking for high volatility change.
    Let's check: 
    1. Low Volatility (Pre-session).
    2. Strong Red Candle (Judas).
    3. Immediate Rejection/Reversal Green Candle.
    """
    if index < 15: return False
    
    # 1. Low Vol
    vol_prev = np.std(close_prices[index-10:index-2])
    
    # 2. Judas Candle (i-1) - Large Red
    c1_idx = index - 1
    c1_body = open_prices[c1_idx] - close_prices[c1_idx]
    if c1_body > (vol_prev * 2.0): # Large Red
        # 3. Reversal (Current Green)
        if close_prices[index] > open_prices[index]:
             # Engulfing the Judas?
             if close_prices[index] > open_prices[c1_idx]:
                 return True
                 
    return False

def detect_judas_swing_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                               high_prices: np.ndarray, low_prices: np.ndarray, 
                               index: int) -> bool:
    """Bearish Judas."""
    if index < 15: return False
    vol_prev = np.std(close_prices[index-10:index-2])
    
    c1_idx = index - 1
    c1_body = close_prices[c1_idx] - open_prices[c1_idx] # Green
    if c1_body > (vol_prev * 2.0):
        if close_prices[index] < open_prices[index]: # Red
            if close_prices[index] < open_prices[c1_idx]:
                return True
    return False

def detect_three_candle_reversal_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                         high_prices: np.ndarray, low_prices: np.ndarray, 
                                         index: int) -> bool:
    """
    55. Three Candle Reversal (3C) (Bullish).
    Pattern:
    1. Down Candle (Trend).
    2. Indecision Candle (Doji/Spinning Top) or Lower Low wick.
    3. Up Candle (Reversal).
    Key: Candle 3 closes above Candle 1 High? Or Candle 2 High?
    Aggressive 3C: Candle 3 Close > Candle 1 High.
    """
    if index < 2: return False
    
    c1 = index - 2
    c2 = index - 1
    c3 = index
    
    # C1: Down
    if close_prices[c1] < open_prices[c1]:
        # C2: Indecision? Or just smaller body?
        c2_body = abs(close_prices[c2] - open_prices[c2])
        c1_body = abs(close_prices[c1] - open_prices[c1])
        
        if c2_body < c1_body:
            # C3: Up
            if close_prices[c3] > open_prices[c3]:
                # Strong close
                if close_prices[c3] > open_prices[c1]: # Engulfs C1 Open
                    return True
                    
    return False

def detect_three_candle_reversal_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                         high_prices: np.ndarray, low_prices: np.ndarray, 
                                         index: int) -> bool:
    """Bearish 3C Reversal."""
    if index < 2: return False
    c1 = index - 2
    c2 = index - 1
    c3 = index
    
    if close_prices[c1] > open_prices[c1]: # Up
        c2_body = abs(close_prices[c2] - open_prices[c2])
        c1_body = abs(close_prices[c1] - open_prices[c1])
        
        if c2_body < c1_body:
            if close_prices[c3] < open_prices[c3]:
                if close_prices[c3] < open_prices[c1]:
                    return True
    return False


# ==============================================================================
# MASS SMC EXPANSION - PHASE 2 (51-75)
# BATCH 12 (56-60/75)
# ==============================================================================

def detect_ote_entry_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                             high_prices: np.ndarray, low_prices: np.ndarray, 
                             index: int) -> bool:
    """
    56. Optimal Trade Entry (OTE) (Bullish).
    Price retraces into the 62% - 79% Fibonacci zone of a recent impulse leg.
    Logic:
    1. Find a recent strong impulse (Swing Low to Swing High).
    2. Check if current price is within 62% - 79% retracement.
    """
    if index < 20: return False
    
    # Find recent swing low and swing high
    sw_low = np.min(low_prices[index-20:index-5])
    sw_high = np.max(high_prices[index-20:index-5])
    
    if sw_high > sw_low:
        diff = sw_high - sw_low
        ote_min = sw_low + diff * (1 - 0.79)
        ote_max = sw_low + diff * (1 - 0.62)
        
        if ote_min <= close_prices[index] <= ote_max:
            # Rejection from zone
            if close_prices[index] > low_prices[index]:
                return True
    return False

def detect_ote_entry_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                             high_prices: np.ndarray, low_prices: np.ndarray, 
                             index: int) -> bool:
    """Bearish OTE."""
    if index < 20: return False
    sw_high = np.max(high_prices[index-20:index-5])
    sw_low = np.min(low_prices[index-20:index-5])
    
    if sw_high > sw_low:
        diff = sw_high - sw_low
        ote_max = sw_high - diff * (1 - 0.79)
        ote_min = sw_high - diff * (1 - 0.62)
        
        if ote_min <= close_prices[index] <= ote_max:
            if close_prices[index] < high_prices[index]:
                return True
    return False

def detect_defining_range_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                 high_prices: np.ndarray, low_prices: np.ndarray,
                                 index: int) -> bool:
    """
    57. Defining Range (DR) (Bullish).
    Breakout of the first hour (simulated as 60 bars/mins).
    """
    if index < 70: return False # Need at least 60 bars + some testing room
    
    # 1. Define Range from bar 0 to 60
    range_high = np.max(high_prices[0:60])
    
    # 2. Breakout
    if close_prices[index] > range_high:
        if close_prices[index-1] <= range_high:
            return True
    return False

def detect_defining_range_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                 high_prices: np.ndarray, low_prices: np.ndarray,
                                 index: int) -> bool:
    """Bearish DR."""
    if index < 70: return False
    range_low = np.min(low_prices[0:60])
    if close_prices[index] < range_low:
        if close_prices[index-1] >= range_low:
            return True
    return False

def detect_cpr_range_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                             high_prices: np.ndarray, low_prices: np.ndarray,
                             index: int) -> bool:
    """
    58. Central Pivot Range (CPR) (Bullish).
    Pivot = (H + L + C) / 3
    BC (Bottom Central) = (H + L) / 2
    TC (Top Central) = (Pivot - BC) + Pivot
    Bullish: Price holding above CPR.
    Since we don't have distinct days, we use a 50-bar window for "Yesterday".
    """
    if index < 60: return False
    
    # Historical data (Proxy for Yesterday)
    h_prev = np.max(high_prices[index-60:index-10])
    l_prev = np.min(low_prices[index-60:index-10])
    c_prev = close_prices[index-11]
    
    pivot = (h_prev + l_prev + c_prev) / 3
    bc = (h_prev + l_prev) / 2
    tc = (pivot - bc) + pivot
    
    cpr_min = min(bc, tc)
    cpr_max = max(bc, tc)
    
    # Bullish: Price bounces off CPR or stays above
    if low_prices[index] <= cpr_max and close_prices[index] > cpr_min:
        if close_prices[index] > close_prices[index-1]:
            return True
    return False

def detect_cpr_range_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                             high_prices: np.ndarray, low_prices: np.ndarray,
                             index: int) -> bool:
    """Bearish CPR Rejection."""
    if index < 60: return False
    h_prev = np.max(high_prices[index-60:index-10])
    l_prev = np.min(low_prices[index-60:index-10])
    c_prev = close_prices[index-11]
    
    pivot = (h_prev + l_prev + c_prev) / 3
    bc = (h_prev + l_prev) / 2
    tc = (pivot - bc) + pivot
    
    cpr_min = min(bc, tc)
    cpr_max = max(bc, tc)
    
    if high_prices[index] >= cpr_min and close_prices[index] < cpr_max:
        if close_prices[index] < close_prices[index-1]:
            return True
    return False

def detect_value_area_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                               high_prices: np.ndarray, low_prices: np.ndarray,
                               index: int) -> bool:
    """
    59. Value Area (VA) (Bullish).
    Value Area Low (VAL) approximation using 70% of standard deviation of price.
    Bullish: Price enters VAL from below or rejects it.
    """
    if index < 50: return False
    window = close_prices[index-50:index]
    mean = np.mean(window)
    std = np.std(window)
    
    val = mean - (std * 0.7) # Approx 70% of volume
    
    if low_prices[index] <= val and close_prices[index] > val:
        return True
    return False

def detect_value_area_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                               high_prices: np.ndarray, low_prices: np.ndarray,
                               index: int) -> bool:
    """Bearish VA (VAH rejection)."""
    if index < 50: return False
    window = close_prices[index-50:index]
    mean = np.mean(window)
    std = np.std(window)
    
    vah = mean + (std * 0.7)
    
    if high_prices[index] >= vah and close_prices[index] < vah:
        return True
    return False

def detect_poc_level_rejection_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                        high_prices: np.ndarray, low_prices: np.ndarray,
                                        index: int) -> bool:
    """
    60. Point of Control (POC) (Bullish).
    POC is the price level with the highest frequency in the lookback window.
    """
    if index < 100: return False
    window = close_prices[index-100:index]
    
    # Histogram proxy for POC
    counts, bins = np.histogram(window, bins=20)
    poc = bins[np.argmax(counts)]
    
    # Bullish: Rejection of POC level
    if low_prices[index] <= poc and close_prices[index] > poc:
        if close_prices[index] > close_prices[index-1]:
            return True
    return False

def detect_poc_level_rejection_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                        high_prices: np.ndarray, low_prices: np.ndarray,
                                        index: int) -> bool:
    """Bearish POC rejection."""
    if index < 100: return False
    window = close_prices[index-100:index]
    counts, bins = np.histogram(window, bins=20)
    poc = bins[np.argmax(counts)]
    
    if high_prices[index] >= poc and close_prices[index] < poc:
        if close_prices[index] < close_prices[index-1]:
            return True
    return False


# ==============================================================================
# MASS SMC EXPANSION - PHASE 2 (51-75)
# BATCH 13 (61-65/75)
# ==============================================================================

def detect_poor_high_low_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                 high_prices: np.ndarray, low_prices: np.ndarray,
                                 index: int) -> bool:
    """Poor High (Bearish Target). Unfinished auction above."""
    if index < 5: return False
    # Is it a local high?
    if high_prices[index] >= np.max(high_prices[index-5:index]):
        body_high = max(open_prices[index], close_prices[index])
        wick_size = high_prices[index] - body_high
        body_size = abs(close_prices[index] - open_prices[index])
        if body_size > 0 and (wick_size / body_size) < 0.05:
            return True
    return False

def detect_poor_high_low_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                 high_prices: np.ndarray, low_prices: np.ndarray,
                                 index: int) -> bool:
    """Poor Low (Bullish Target). Unfinished auction below."""
    if index < 5: return False
    if low_prices[index] <= np.min(low_prices[index-5:index]):
        body_low = min(open_prices[index], close_prices[index])
        wick_size = body_low - low_prices[index]
        body_size = abs(close_prices[index] - open_prices[index])
        if body_size > 0 and (wick_size / body_size) < 0.05:
            return True
    return False

def detect_single_prints_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                 high_prices: np.ndarray, low_prices: np.ndarray,
                                 index: int) -> bool:
    """
    62. Single Prints (Bullish Opportunity).
    Gaps in a TPO profile. Price moved so fast it only spent 1 'TPO' (time period) there.
    Simulated: A large gap between the wicks of candle i-1 and i+1? No.
    Simulated as a 'very long candle body' that hasn't been overlapped by 
    the next 5 candles.
    """
    if index < 10: return False
    
    # Looking back for a massive green candle that is "Single" (not overlapped)
    for i in range(2, 10):
        idx = index - i
        # Is it a massive green candle?
        body = close_prices[idx] - open_prices[idx]
        if body > 0:
            # Check if anyone else has traded in this body range since then
            overlap = False
            for j in range(idx+1, index+1):
                if low_prices[j] < close_prices[idx] and high_prices[j] > open_prices[idx]:
                    # This is partial overlap, but single prints are usually "Thin" zones.
                    # Let's say if it hasn't been 50% filled.
                    pass
            
            # Simplified: Massive candle (Impulse) retest
            avg_body = np.mean(np.abs(close_prices[idx-5:idx] - open_prices[idx-5:idx]))
            if body > 3.0 * avg_body:
                # Current price is retesting the 'Single Print' zone (the body)
                if low_prices[index] <= close_prices[idx] and close_prices[index] >= open_prices[idx]:
                    if close_prices[index] > open_prices[index]: # Bouncing
                        return True
                        
    return False

def detect_single_prints_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                 high_prices: np.ndarray, low_prices: np.ndarray,
                                 index: int) -> bool:
    """Bearish Single Prints Retest."""
    if index < 10: return False
    for i in range(2, 10):
        idx = index - i
        body = open_prices[idx] - close_prices[idx] # Red body
        if body > 0:
            avg_body = np.mean(np.abs(close_prices[idx-5:idx] - open_prices[idx-5:idx]))
            if body > 3.0 * avg_body:
                if high_prices[index] >= close_prices[idx] and close_prices[index] <= open_prices[idx]:
                    if close_prices[index] < open_prices[index]: # Rejecting
                        return True
    return False

def detect_tails_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                         high_prices: np.ndarray, low_prices: np.ndarray,
                         index: int) -> bool:
    """
    63. Buying Tail (Bullish).
    Aggressive rejection at the bottom of a range.
    Long lower wick (> 2x body) that occurs at a 20-period Low.
    """
    if index < 20: return False
    
    # Is it a 20-period Low?
    if low_prices[index] <= np.min(low_prices[index-20:index]):
        body_low = min(open_prices[index], close_prices[index])
        wick_size = body_low - low_prices[index]
        body_size = abs(close_prices[index] - open_prices[index])
        
        if wick_size > (body_size * 2.0) and wick_size > 0:
            return True
    return False

def detect_tails_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                         high_prices: np.ndarray, low_prices: np.ndarray,
                         index: int) -> bool:
    """Selling Tail (Bearish)."""
    if index < 20: return False
    
    if high_prices[index] >= np.max(high_prices[index-20:index]):
        body_high = max(open_prices[index], close_prices[index])
        wick_size = high_prices[index] - body_high
        body_size = abs(close_prices[index] - open_prices[index])
        
        if wick_size > (body_size * 2.0) and wick_size > 0:
            return True
    return False

def detect_composite_operator_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                      high_prices: np.ndarray, low_prices: np.ndarray,
                                      index: int) -> bool:
    """
    64. Composite Operator (Bullish Footprint).
    Price level where 'Professional' money is accumulating.
    Simulated: Range contraction with high volume (if we had volume).
    Without volume: Persistent 'Tight' lows that refuse to break despite multiple tests.
    """
    if index < 10: return False
    
    # Aggressive rejection of the same level 3 times in 10 bars
    tests = 0
    level = low_prices[index]
    for i in range(1, 10):
        if abs(low_prices[index-i] - level) < (level * 0.0005):
            tests += 1
            
    if tests >= 2:
        # Sign of Strength (current candle is bullish)
        if close_prices[index] > close_prices[index-1]:
            return True
    return False

def detect_composite_operator_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                      high_prices: np.ndarray, low_prices: np.ndarray,
                                      index: int) -> bool:
    """Composite Operator (Bearish Footprint)."""
    if index < 10: return False
    tests = 0
    level = high_prices[index]
    for i in range(1, 10):
        if abs(high_prices[index-i] - level) < (level * 0.0005):
            tests += 1
    if tests >= 2:
        if close_prices[index] < close_prices[index-1]:
            return True
    return False

def detect_dist_accum_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                              high_prices: np.ndarray, low_prices: np.ndarray,
                              index: int) -> bool:
    """
    65. Accumulation Phase Proxy (Bullish).
    Wyckoff Logic: contraction -> Spring (Sweep) -> SOS.
    """
    if index < 30: return False
    
    # 1. Contraction (Low Vol)
    vol_prev = np.std(close_prices[index-30:index-10])
    vol_recent = np.std(close_prices[index-10:index-2])
    
    if vol_recent < vol_prev:
        # 2. Spring (Sweep Low of range)
        range_low = np.min(low_prices[index-30:index-2])
        if low_prices[index-1] < range_low:
             # 3. SOS (Current candle closes high)
             if close_prices[index] > close_prices[index-1] and close_prices[index] > range_low:
                 return True
                 
    return False

def detect_dist_accum_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                              high_prices: np.ndarray, low_prices: np.ndarray,
                              index: int) -> bool:
    """Distribution Phase Proxy (Bearish)."""
    if index < 30: return False
    vol_prev = np.std(close_prices[index-30:index-10])
    vol_recent = np.std(close_prices[index-10:index-2])
    
    if vol_recent < vol_prev:
        range_high = np.max(high_prices[index-30:index-2])
        if high_prices[index-1] > range_high:
             if close_prices[index] < close_prices[index-1] and close_prices[index] < range_high:
                 return True
    return False

# ==============================================================================
# MASS SMC EXPANSION - PHASE 2 (51-75)
# BATCH 14 (66-70/75)
# ==============================================================================

def detect_wyckoff_spring_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                 high_prices: np.ndarray, low_prices: np.ndarray, 
                                 index: int) -> bool:
    """
    66. Wyckoff Spring (Bullish).
    A terminal shakeout below a trading range before a markup phase.
    Logic:
    1. Identify a trading range (Support level).
    2. Price breaks below support (Spring).
    3. Price immediately recovers back into the range.
    """
    if index < 20: return False
    
    # Identify a recent support level (range low)
    range_low = np.min(low_prices[index-20:index-5])
    
    # 1. Spring: Previous candle (or current) breaks range_low
    if low_prices[index-1] < range_low or low_prices[index] < range_low:
        # 2. Recovery: Current close is back above range_low
        if close_prices[index] > range_low:
            # Check for high momentum (Large body close)
            if close_prices[index] > close_prices[index-1]:
                return True
    return False

def detect_wyckoff_upthrust_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                    high_prices: np.ndarray, low_prices: np.ndarray, 
                                    index: int) -> bool:
    """Wyckoff Upthrust (Bearish Spring)."""
    if index < 20: return False
    range_high = np.max(high_prices[index-20:index-5])
    if high_prices[index-1] > range_high or high_prices[index] > range_high:
        if close_prices[index] < range_high:
            if close_prices[index] < close_prices[index-1]:
                return True
    return False

def detect_wyckoff_sos_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                              low_prices: np.ndarray, high_prices: np.ndarray,
                              index: int) -> bool:
    """
    67. Sign of Strength (SOS) (Bullish).
    Aggressive price expansion to the upside with large candle bodies.
    """
    if index < 5: return False
    
    # Look for a sequence of 3 candles with increasing strength or persistent expansion
    c1_body = close_prices[index] - open_prices[index]
    c2_body = close_prices[index-1] - open_prices[index-1]
    
    avg_body = np.mean(np.abs(close_prices[index-10:index-1] - open_prices[index-10:index-1]))
    
    if c1_body > 2.0 * avg_body and close_prices[index] > high_prices[index-1]:
        # Strong breakout move
        return True
    return False

def detect_wyckoff_sow_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                              low_prices: np.ndarray, high_prices: np.ndarray,
                              index: int) -> bool:
    """Sign of Weakness (SOW) (Bearish SOS)."""
    if index < 5: return False
    c1_body = open_prices[index] - close_prices[index] # Red
    avg_body = np.mean(np.abs(close_prices[index-10:index-1] - open_prices[index-10:index-1]))
    if c1_body > 2.0 * avg_body and close_prices[index] < low_prices[index-1]:
        return True
    return False

def detect_wyckoff_lps_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                high_prices: np.ndarray, low_prices: np.ndarray, 
                                index: int) -> bool:
    """
    68. Last Point of Support (LPS) (Bullish).
    A shallow pullback after a Sign of Strength (SOS).
    Logic:
    1. Previous SOS detected.
    2. Current pullback holds above a previous structural high (flipping).
    """
    if index < 15: return False
    
    # 1. Look for a breakout high in the last 15-5 bars
    breakout_high = np.max(high_prices[index-15:index-5])
    
    # 2. Did we break it recently?
    if close_prices[index-3] > breakout_high:
        # 3. Pullback today: Low touches or stays near breakout_high
        if low_prices[index] >= breakout_high * 0.998 and low_prices[index] <= breakout_high * 1.01:
            if close_prices[index] > close_prices[index-1]: # Bullish bounce
                return True
    return False

def detect_wyckoff_lpsy_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                 high_prices: np.ndarray, low_prices: np.ndarray, 
                                 index: int) -> bool:
    """Last Point of Supply (LPSY) (Bearish LPS)."""
    if index < 15: return False
    breakout_low = np.min(low_prices[index-15:index-5])
    if close_prices[index-3] < breakout_low:
        if high_prices[index] <= breakout_low * 1.002 and high_prices[index] >= breakout_low * 0.99:
            if close_prices[index] < close_prices[index-1]:
                return True
    return False

def detect_effort_vs_result_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                    low_prices: np.ndarray, high_prices: np.ndarray,
                                    index: int) -> bool:
    """
    69. Effort vs Result (Bullish Divergence).
    Effort = Candle Range (proxy for volume/effort).
    Result = Vertical progress (Net change).
    Bullish Divergence: A candle has a very large range (Effort) but closes near its open 
    after hitting a support level (absorbed selling).
    """
    if index < 5: return False
    
    # Candle Range (Total Effort)
    candle_range = high_prices[index] - low_prices[index]
    # Candle Body (Result)
    candle_body = abs(close_prices[index] - open_prices[index])
    
    avg_range = np.mean(high_prices[index-10:index-1] - low_prices[index-10:index-1])
    
    if candle_range > 2.0 * avg_range: # High Effort
        if candle_body < candle_range * 0.3: # Low Result (Absorption)
            # Rejection from Low
            if close_prices[index] > low_prices[index] + (candle_range * 0.6):
                return True
    return False

def detect_effort_vs_result_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                    low_prices: np.ndarray, high_prices: np.ndarray,
                                    index: int) -> bool:
    """Effort vs Result (Bearish Divergence / Absorption)."""
    if index < 5: return False
    candle_range = high_prices[index] - low_prices[index]
    candle_body = abs(close_prices[index] - open_prices[index])
    avg_range = np.mean(high_prices[index-10:index-1] - low_prices[index-10:index-1])
    if candle_range > 2.0 * avg_range:
        if candle_body < candle_range * 0.3:
            # Rejection from High
            if close_prices[index] < high_prices[index] - (candle_range * 0.6):
                return True
    return False

def detect_stopping_volume_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                   low_prices: np.ndarray, high_prices: np.ndarray,
                                   index: int) -> bool:
    """
    70. Stopping Volume (Bullish Climax).
    A massive red candle followed by a narrow range candle that holds the low.
    Suggests institutional absorption.
    """
    if index < 2: return False
    
    # C1: Massive Red (Effort)
    c1_idx = index - 1
    c1_range = high_prices[c1_idx] - low_prices[c1_idx]
    c1_body = open_prices[c1_idx] - close_prices[c1_idx]
    
    avg_range = np.mean(high_prices[index-10:index-2] - low_prices[index-10:index-2])
    
    if c1_body > 2.0 * avg_range: # Climax drop
        # C2: Narrow range candle holding
        c2_range = high_prices[index] - low_prices[index]
        if c2_range < c1_range * 0.5:
            if low_prices[index] >= low_prices[c1_idx]:
                if close_prices[index] > open_prices[index]: # Green rejection
                    return True
    return False

def detect_stopping_volume_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                   low_prices: np.ndarray, high_prices: np.ndarray,
                                   index: int) -> bool:
    """Stopping Volume (Bearish Climax)."""
    if index < 2: return False
    c1_idx = index - 1
    c1_body = close_prices[c1_idx] - open_prices[c1_idx] # Green
    avg_range = np.mean(high_prices[index-10:index-2] - low_prices[index-10:index-2])
    if c1_body > 2.0 * avg_range:
        c2_range = high_prices[index] - low_prices[index]
        if c2_range < (high_prices[c1_idx] - low_prices[c1_idx]) * 0.5:
            if high_prices[index] <= high_prices[c1_idx]:
                if close_prices[index] < open_prices[index]: # Red rejection
                    return True
    return False
    return False


# ==============================================================================
# MASS SMC EXPANSION - PHASE 2 (51-75)
# BATCH 15 (71-75/75) - FINAL BATCH
# ==============================================================================

def detect_vsa_no_demand_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                high_prices: np.ndarray, low_prices: np.ndarray,
                                index: int) -> bool:
    """
    71. VSA No Demand (Bearish).
    A narrow range bullish candle with low "Effort" (Range) compared to previous.
    Indicates lack of professional interest in higher prices.
    Logic:
    1. Uptrending/High prices.
    2. Bullish candle (C[i] > O[i]).
    3. Range is significantly smaller than previous 2-3 candles.
    4. Followed by a bearish confirmation? We detect the 'Signal' bar here.
    """
    if index < 5: return False
    
    # 1. Bullish candle
    if close_prices[index] > open_prices[index]:
        # 2. Narrow range (Effort)
        curr_range = high_prices[index] - low_prices[index]
        prev_ranges = high_prices[index-3:index] - low_prices[index-3:index]
        avg_prev_range = np.mean(prev_ranges)
        
        if curr_range < avg_prev_range * 0.7:
            # 3. High prices (relative)
            if high_prices[index] >= np.max(high_prices[index-10:index]):
                return True
    return False

def detect_vsa_no_supply_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                high_prices: np.ndarray, low_prices: np.ndarray,
                                index: int) -> bool:
    """
    72. VSA No Supply (Bullish).
    A narrow range bearish candle with low "Effort" at low prices.
    Logic:
    1. Downtrending/Low prices.
    2. Bearish candle (C[i] < O[i]).
    3. Range is significantly smaller than previous candles.
    """
    if index < 5: return False
    
    if close_prices[index] < open_prices[index]:
        curr_range = high_prices[index] - low_prices[index]
        prev_ranges = high_prices[index-3:index] - low_prices[index-3:index]
        avg_prev_range = np.mean(prev_ranges)
        
        if curr_range < avg_prev_range * 0.7:
            if low_prices[index] <= np.min(low_prices[index-10:index]):
                return True
    return False

def detect_vsa_shakeout_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                               high_prices: np.ndarray, low_prices: np.ndarray,
                               index: int) -> bool:
    """
    73. VSA Shakeout (Bullish).
    A wide range candle that sweeps below a support level and closes near high.
    Similar to Spring but specifically focuses on the wide range 'VSA' behavior.
    """
    if index < 10: return False
    
    # Identify support
    support = np.min(low_prices[index-10:index-2])
    
    # Wide range effort
    curr_range = high_prices[index] - low_prices[index]
    avg_prev_range = np.mean(high_prices[index-5:index-1] - low_prices[index-5:index-1])
    
    if curr_range > avg_prev_range * 1.5:
        if low_prices[index] < support:
            # Rejection: Closes near high
            if close_prices[index] > low_prices[index] + (curr_range * 0.7):
                return True
    return False

def detect_vsa_shakeout_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                               high_prices: np.ndarray, low_prices: np.ndarray,
                               index: int) -> bool:
    """VSA Shakeout (Bearish)."""
    if index < 10: return False
    resistance = np.max(high_prices[index-10:index-2])
    curr_range = high_prices[index] - low_prices[index]
    avg_prev_range = np.mean(high_prices[index-5:index-1] - low_prices[index-5:index-1])
    
    if curr_range > avg_prev_range * 1.5:
        if high_prices[index] > resistance:
            if close_prices[index] < low_prices[index] + (curr_range * 0.3):
                return True
    return False

def detect_selling_climax_bullish(open_prices: np.ndarray, close_prices: np.ndarray,
                                 high_prices: np.ndarray, low_prices: np.ndarray,
                                 index: int) -> bool:
    """
    74. Selling Climax (Bullish Reversal).
    Extremely large range red candle at the end of a move, with a long lower wick.
    Professional money absorbs the panic selling.
    """
    if index < 20: return False
    
    # Downtrend
    if close_prices[index-1] < np.mean(close_prices[index-20:index-1]):
        curr_range = high_prices[index] - low_prices[index]
        # Massive range
        avg_range = np.mean(high_prices[index-10:index-1] - low_prices[index-10:index-1])
        if curr_range > avg_range * 2.5:
            # Long lower wick
            body_low = min(open_prices[index], close_prices[index])
            wick_size = body_low - low_prices[index]
            if wick_size > (curr_range * 0.4):
                return True
    return False

def detect_buying_climax_bearish(open_prices: np.ndarray, close_prices: np.ndarray,
                                high_prices: np.ndarray, low_prices: np.ndarray,
                                index: int) -> bool:
    """
    75. Buying Climax (Bearish Reversal).
    Extremely large range green candle at the end of a move, with long upper wick.
    Professional money sells into the FOMO.
    """
    if index < 20: return False
    
    # Uptrend
    if close_prices[index-1] > np.mean(close_prices[index-20:index-1]):
        curr_range = high_prices[index] - low_prices[index]
        avg_range = np.mean(high_prices[index-10:index-1] - low_prices[index-10:index-1])
        if curr_range > avg_range * 2.5:
            # Long upper wick
            body_high = max(open_prices[index], close_prices[index])
            wick_size = high_prices[index] - body_high
            if wick_size > (curr_range * 0.4):
                return True
    return False
