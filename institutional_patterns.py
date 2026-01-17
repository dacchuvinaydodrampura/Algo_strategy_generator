

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
