"""
OHLCV Data Provider.

Generates realistic synthetic market data for backtesting.
Can be replaced with real data sources (Fyers, Yahoo Finance, etc.)

IMPORTANT: This generates simulated data for testing.
For production, replace with actual market data API.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib


@dataclass
class OHLCVData:
    """OHLCV data container."""
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    
    def __len__(self) -> int:
        return len(self.close)


class DataProvider:
    """
    Provides OHLCV data for backtesting.
    
    Currently uses synthetic data generation.
    Replace with Fyers API for real data.
    """
    
    # Cache for data (refresh once per hour for efficiency)
    _cache: Dict[str, OHLCVData] = {}
    _cache_time: Optional[datetime] = None
    _cache_duration: timedelta = timedelta(hours=1)
    
    @classmethod
    def get_data(cls, market: str, timeframe: str, days: int, 
                 seed: Optional[int] = None) -> OHLCVData:
        """
        Get OHLCV data for specified market and timeframe.
        
        Args:
            market: Market symbol (e.g., "NSE:NIFTY50")
            timeframe: Candle timeframe ("1m", "5m", "15m")
            days: Number of trading days
            seed: Random seed for reproducibility
        
        Returns:
            OHLCVData object
        """
        cache_key = f"{market}_{timeframe}_{days}"
        
        # Check cache
        if (cls._cache_time and 
            datetime.now() - cls._cache_time < cls._cache_duration and
            cache_key in cls._cache):
            return cls._cache[cache_key]
        
        # Generate new data
        data = cls._generate_synthetic_data(market, timeframe, days, seed)
        
        # Update cache
        cls._cache[cache_key] = data
        cls._cache_time = datetime.now()
        
        return data
    
    @classmethod
    def _generate_synthetic_data(cls, market: str, timeframe: str, 
                                  days: int, seed: Optional[int] = None) -> OHLCVData:
        """
        Generate realistic synthetic OHLCV data.
        
        Uses geometric Brownian motion with mean reversion
        to simulate realistic price movement.
        """
        # Deterministic seed based on market if not provided
        if seed is None:
            seed = int(hashlib.md5(f"{market}_{datetime.now().date()}".encode()).hexdigest()[:8], 16)
        
        np.random.seed(seed)
        
        # Candles per day
        candles_per_day = {
            "1m": 375,
            "5m": 75,
            "15m": 25,
        }.get(timeframe, 75)
        
        total_candles = days * candles_per_day
        
        # Base price based on market
        base_prices = {
            "NSE:NIFTY50": 22000.0,
            "NSE:BANKNIFTY": 48000.0,
            "NSE:NIFTYIT": 35000.0,
        }
        base_price = base_prices.get(market, 22000.0)
        
        # Volatility based on market
        volatilities = {
            "NSE:NIFTY50": 0.0012,
            "NSE:BANKNIFTY": 0.0018,
            "NSE:NIFTYIT": 0.0015,
        }
        volatility = volatilities.get(market, 0.0012)
        
        # Generate timestamps
        start_date = datetime.now() - timedelta(days=days)
        timestamps = np.array([
            start_date + timedelta(minutes=i * (375 // candles_per_day))
            for i in range(total_candles)
        ])
        
        # Generate price series using GBM with mean reversion
        returns = np.random.normal(0, volatility, total_candles)
        
        # Add trend and mean reversion
        trend = np.random.choice([-1, 1]) * 0.00001  # Slight trend
        mean_reversion_strength = 0.02
        
        prices = np.zeros(total_candles)
        prices[0] = base_price
        
        for i in range(1, total_candles):
            # Mean reversion component
            mean_rev = mean_reversion_strength * (base_price - prices[i-1]) / base_price
            
            # Price change
            change = prices[i-1] * (returns[i] + trend + mean_rev)
            prices[i] = prices[i-1] + change
        
        # Generate OHLC from price series
        open_prices = np.zeros(total_candles)
        high_prices = np.zeros(total_candles)
        low_prices = np.zeros(total_candles)
        close_prices = np.zeros(total_candles)
        volumes = np.zeros(total_candles)
        
        for i in range(total_candles):
            base = prices[i]
            spread = base * volatility * 0.5
            
            open_prices[i] = base + np.random.uniform(-spread, spread)
            close_prices[i] = base + np.random.uniform(-spread, spread)
            
            # High is max of open/close plus some
            high_prices[i] = max(open_prices[i], close_prices[i]) + abs(np.random.normal(0, spread))
            # Low is min of open/close minus some
            low_prices[i] = min(open_prices[i], close_prices[i]) - abs(np.random.normal(0, spread))
            
            # Volume with some randomness
            base_volume = 10000
            volumes[i] = base_volume * (1 + abs(np.random.normal(0, 0.5)))
        
        return OHLCVData(
            timestamps=timestamps,
            open=open_prices,
            high=high_prices,
            low=low_prices,
            close=close_prices,
            volume=volumes,
        )
    
    @classmethod
    def clear_cache(cls):
        """Clear cached data."""
        cls._cache.clear()
        cls._cache_time = None
