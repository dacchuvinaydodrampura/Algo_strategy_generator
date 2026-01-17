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
 
  
 d e f   d e t e c t _ b u l l i s h _ i m b a l a n c e ( o p e n _ p r i c e s :   n p . n d a r r a y ,   c l o s e _ p r i c e s :   n p . n d a r r a y ,  
                                                           h i g h _ p r i c e s :   n p . n d a r r a y ,   l o w _ p r i c e s :   n p . n d a r r a y ,  
                                                           i n d e x :   i n t )   - >   b o o l :  
         " " "  
         D e t e c t   B u l l i s h   F a i r   V a l u e   G a p   ( F V G ) .  
          
         C o n d i t i o n :  
         -   C a n d l e   1   H i g h   <   C a n d l e   3   L o w  
         -   L e a v i n g   a   g a p   b e t w e e n   w i c k   1   a n d   w i c k   3 .  
         " " "  
         i f   i n d e x   <   2 :  
                 r e t u r n   F a l s e  
                  
         #   C a n d l e   1   ( I n d e x - 2 ) ,   C a n d l e   2   ( I n d e x - 1 ) ,   C a n d l e   3   ( I n d e x - 0 )  
         h i g h 1   =   h i g h _ p r i c e s [ i n d e x - 2 ]  
         l o w 3   =   l o w _ p r i c e s [ i n d e x ]  
          
         #   G a p   e x i s t s   i f   L o w   o f   c u r r e n t   c a n d l e   i s   n o t i c e a b l y   h i g h e r   t h a n   H i g h   o f   2   c a n d l e s   a g o  
         #   W e   a d d   a   s m a l l   t h r e s h o l d   t o   a v o i d   t i n y   g a p s   b e i n g   c o u n t e d   ( 0 . 0 2 %   p r i c e )  
         p r i c e _ t h r e s h o l d   =   c l o s e _ p r i c e s [ i n d e x ]   *   0 . 0 0 0 2  
          
         g a p _ s i z e   =   l o w 3   -   h i g h 1  
         r e t u r n   g a p _ s i z e   >   p r i c e _ t h r e s h o l d  
  
 d e f   d e t e c t _ b e a r i s h _ i m b a l a n c e ( o p e n _ p r i c e s :   n p . n d a r r a y ,   c l o s e _ p r i c e s :   n p . n d a r r a y ,  
                                                           h i g h _ p r i c e s :   n p . n d a r r a y ,   l o w _ p r i c e s :   n p . n d a r r a y ,  
                                                           i n d e x :   i n t )   - >   b o o l :  
         " " "  
         D e t e c t   B e a r i s h   F a i r   V a l u e   G a p   ( F V G ) .  
          
         C o n d i t i o n :  
         -   C a n d l e   1   L o w   >   C a n d l e   3   H i g h  
         " " "  
         i f   i n d e x   <   2 :  
                 r e t u r n   F a l s e  
                  
         l o w 1   =   l o w _ p r i c e s [ i n d e x - 2 ]  
         h i g h 3   =   h i g h _ p r i c e s [ i n d e x ]  
          
         p r i c e _ t h r e s h o l d   =   c l o s e _ p r i c e s [ i n d e x ]   *   0 . 0 0 0 2  
          
         g a p _ s i z e   =   l o w 1   -   h i g h 3  
         r e t u r n   g a p _ s i z e   >   p r i c e _ t h r e s h o l d  
  
 d e f   d e t e c t _ l i q u i d i t y _ s w e e p _ l o w ( h i g h _ p r i c e s :   n p . n d a r r a y ,   l o w _ p r i c e s :   n p . n d a r r a y ,  
                                                               c l o s e _ p r i c e s :   n p . n d a r r a y ,   i n d e x :   i n t ,  
                                                               l o o k b a c k :   i n t   =   2 0 )   - >   b o o l :  
         " " "  
         D e t e c t   B u l l i s h   L i q u i d i t y   S w e e p   ( S t o p   H u n t ) .  
          
         C o n d i t i o n s :  
         1 .   C u r r e n t   L o w   b r e a k s   b e l o w   t h e   L o w e s t   L o w   o f   l a s t   ' l o o k b a c k '   c a n d l e s .  
         2 .   C u r r e n t   C l o s e   i s   A B O V E   t h a t   p r e v i o u s   L o w e s t   L o w   ( r e j e c t i o n / r e c l a i m ) .  
         " " "  
         i f   i n d e x   <   l o o k b a c k :  
                 r e t u r n   F a l s e  
                  
         #   F i n d   S w i n g   L o w   i n   l o o k b a c k   p e r i o d   ( e x c l u d i n g   c u r r e n t   c a n d l e )  
         p r e v _ l o w s   =   l o w _ p r i c e s [ i n d e x - l o o k b a c k   :   i n d e x ]  
         s w i n g _ l o w   =   n p . m i n ( p r e v _ l o w s )  
          
         c u r r _ l o w   =   l o w _ p r i c e s [ i n d e x ]  
         c u r r _ c l o s e   =   c l o s e _ p r i c e s [ i n d e x ]  
          
         #   1 .   S w e e p :   P r i c e   d i p p e d   b e l o w   s w i n g   l o w  
         s w e p t _ l i q u i d i t y   =   c u r r _ l o w   <   s w i n g _ l o w  
          
         #   2 .   R e c l a i m :   P r i c e   c l o s e d   b a c k   a b o v e   t h e   s w i n g   l o w  
         r e c l a i m e d _ l e v e l   =   c u r r _ c l o s e   >   s w i n g _ l o w  
          
         r e t u r n   s w e p t _ l i q u i d i t y   a n d   r e c l a i m e d _ l e v e l  
  
 d e f   d e t e c t _ l i q u i d i t y _ s w e e p _ h i g h ( h i g h _ p r i c e s :   n p . n d a r r a y ,   l o w _ p r i c e s :   n p . n d a r r a y ,  
                                                                 c l o s e _ p r i c e s :   n p . n d a r r a y ,   i n d e x :   i n t ,  
                                                                 l o o k b a c k :   i n t   =   2 0 )   - >   b o o l :  
         " " "  
         D e t e c t   B e a r i s h   L i q u i d i t y   S w e e p   ( S t o p   H u n t ) .  
          
         C o n d i t i o n s :  
         1 .   C u r r e n t   H i g h   b r e a k s   a b o v e   H i g h e s t   H i g h .  
         2 .   C u r r e n t   C l o s e   i s   B E L O W   t h a t   p r e v i o u s   H i g h e s t   H i g h .  
         " " "  
         i f   i n d e x   <   l o o k b a c k :  
                 r e t u r n   F a l s e  
                  
         p r e v _ h i g h s   =   h i g h _ p r i c e s [ i n d e x - l o o k b a c k   :   i n d e x ]  
         s w i n g _ h i g h   =   n p . m a x ( p r e v _ h i g h s )  
          
         c u r r _ h i g h   =   h i g h _ p r i c e s [ i n d e x ]  
         c u r r _ c l o s e   =   c l o s e _ p r i c e s [ i n d e x ]  
          
         s w e p t _ l i q u i d i t y   =   c u r r _ h i g h   >   s w i n g _ h i g h  
         r e c l a i m e d _ l e v e l   =   c u r r _ c l o s e   <   s w i n g _ h i g h  
          
         r e t u r n   s w e p t _ l i q u i d i t y   a n d   r e c l a i m e d _ l e v e l  
  
 d e f   d e t e c t _ v o l a t i l i t y _ s q u e e z e ( h i g h _ p r i c e s :   n p . n d a r r a y ,   l o w _ p r i c e s :   n p . n d a r r a y ,  
                                                             c l o s e _ p r i c e s :   n p . n d a r r a y ,   i n d e x :   i n t ,    
                                                             p e r i o d :   i n t   =   2 0 )   - >   b o o l :  
         " " "  
         D e t e c t   V o l a t i l i t y   S q u e e z e   ( B o l l i n g e r   B a n d   S q u e e z e ) .  
          
         L o g i c :   B a n d   W i d t h   i s   a t   l o w e s t   p o i n t   i n   2 0   b a r s .  
         " " "  
         i f   i n d e x   <   p e r i o d :  
                 r e t u r n   F a l s e  
  
         #   C a l c u l a t e   B B   W i d t h   f o r   l a s t   ' p e r i o d '   b a r s  
         #   T h i s   i s   e x p e n s i v e   t o   d o   p r o p e r l y   f o r   e v e r y   b a r   h e r e ,   s o   w e   i m p l e m e n t   a   s i m p l i f i e d   c h e c k .  
         #   C h e c k   i f   c u r r e n t   R a n g e   ( H i g h - L o w )   i s   l e s s   t h a n   5 0 %   o f   A v e r a g e   R a n g e  
          
         c u r r e n t _ r a n g e   =   h i g h _ p r i c e s [ i n d e x ]   -   l o w _ p r i c e s [ i n d e x ]  
          
         #   A v e r a g e   r a n g e   o f   l a s t   2 0   b a r s  
         p a s t _ r a n g e s   =   h i g h _ p r i c e s [ i n d e x - p e r i o d : i n d e x ]   -   l o w _ p r i c e s [ i n d e x - p e r i o d : i n d e x ]  
         a v g _ r a n g e   =   n p . m e a n ( p a s t _ r a n g e s )  
          
         r e t u r n   c u r r e n t _ r a n g e   <   ( a v g _ r a n g e   *   0 . 5 )  
  
 d e f   d e t e c t _ b u l l i s h _ o r d e r _ b l o c k ( o p e n _ p r i c e s :   n p . n d a r r a y ,   c l o s e _ p r i c e s :   n p . n d a r r a y ,  
                                                               i n d e x :   i n t )   - >   b o o l :  
         " " "  
         D e t e c t   B u l l i s h   O r d e r   B l o c k   f o r m a t i o n .  
          
         S i m p l i f i e d   L o g i c :  
         1 .   C a n d l e   ( I n d e x - 1 )   w a s   B e a r i s h   ( R e d ) .  
         2 .   C a n d l e   ( I n d e x )   i s   B u l l i s h   ( G r e e n ) .  
         3 .   C a n d l e   ( I n d e x )   B o d y   i s   >   2 x   P r e v i o u s   C a n d l e   B o d y   ( S t r o n g   i m p u l s e ) .  
         4 .   C u r r e n t   C l o s e   b r o k e   a b o v e   P r e v i o u s   H i g h   ( M a r k e t   S t r u c t u r e   S h i f t )   -   i m p l i e d   b y   b o d y   s i z e   h e r e .  
         " " "  
         i f   i n d e x   <   1 :  
                 r e t u r n   F a l s e  
                  
         #   P r e v   C a n d l e   ( B e a r i s h )  
         o 1 ,   c 1   =   o p e n _ p r i c e s [ i n d e x - 1 ] ,   c l o s e _ p r i c e s [ i n d e x - 1 ]  
         i s _ b e a r i s h   =   c 1   <   o 1  
         b o d y 1   =   a b s ( o 1   -   c 1 )  
          
         #   C u r r   C a n d l e   ( B u l l i s h   I m p u l s e )  
         o 2 ,   c 2   =   o p e n _ p r i c e s [ i n d e x ] ,   c l o s e _ p r i c e s [ i n d e x ]  
         i s _ b u l l i s h   =   c 2   >   o 2  
         b o d y 2   =   a b s ( c 2   -   o 2 )  
          
         i f   i s _ b e a r i s h   a n d   i s _ b u l l i s h   a n d   b o d y 2   >   ( b o d y 1   *   2 . 0 ) :  
                 r e t u r n   T r u e  
                  
         r e t u r n   F a l s e  
  
 d e f   d e t e c t _ b e a r i s h _ o r d e r _ b l o c k ( o p e n _ p r i c e s :   n p . n d a r r a y ,   c l o s e _ p r i c e s :   n p . n d a r r a y ,  
                                                               i n d e x :   i n t )   - >   b o o l :  
         " " "  
         D e t e c t   B e a r i s h   O r d e r   B l o c k   f o r m a t i o n .  
         " " "  
         i f   i n d e x   <   1 :  
                 r e t u r n   F a l s e  
                  
         #   P r e v   C a n d l e   ( B u l l i s h )  
         o 1 ,   c 1   =   o p e n _ p r i c e s [ i n d e x - 1 ] ,   c l o s e _ p r i c e s [ i n d e x - 1 ]  
         i s _ b u l l i s h   =   c 1   >   o 1  
         b o d y 1   =   a b s ( c 1   -   o 1 )  
          
         #   C u r r   C a n d l e   ( B e a r i s h   I m p u l s e )  
         o 2 ,   c 2   =   o p e n _ p r i c e s [ i n d e x ] ,   c l o s e _ p r i c e s [ i n d e x ]  
         i s _ b e a r i s h   =   c 2   <   o 2  
         b o d y 2   =   a b s ( o 2   -   c 2 )  
          
         i f   i s _ b u l l i s h   a n d   i s _ b e a r i s h   a n d   b o d y 2   >   ( b o d y 1   *   2 . 0 ) :  
                 r e t u r n   T r u e  
                  
         r e t u r n   F a l s e  
 