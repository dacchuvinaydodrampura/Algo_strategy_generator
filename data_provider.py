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
        Generate realistic synthetic OHLCV data with Institutional Patterns.
        
        Uses Regime-Switching Model:
        - Trending (Bull/Bear)
        - Ranging
        - Volatility Squeezes
        - Liquidity Sweeps (Smart Money injections)
        """
        if seed is None:
            # Change seed daily to simulate fresh market
            seed = int(hashlib.md5(f"{market}_{datetime.now().date()}".encode()).hexdigest()[:8], 16)
        
        np.random.seed(seed)
        
        candles_per_day = {"1m": 375, "5m": 75, "15m": 25}.get(timeframe, 75)
        total_candles = days * candles_per_day
        
        base_price = 22000.0 if "NIFTY" in market else 100.0
        
        # Initialize arrays
        open_p = np.zeros(total_candles)
        high_p = np.zeros(total_candles)
        low_p = np.zeros(total_candles)
        close_p = np.zeros(total_candles)
        vol_p = np.zeros(total_candles)
        
        # 1. Generate core price path using Regime Switching
        # Regimes: 0=Range, 1=Bull Trend, 2=Bear Trend
        current_price = base_price
        
        # Break total duration into segments (e.g., 2-week regimes)
        segment_len = candles_per_day * 10 
        
        for i in range(total_candles):
            if i % segment_len == 0:
                # Switch regime occasionally. Bias towards Bull (1) for Nifty to allow 200% returns
                regime = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
                
                # Volatility per regime
                if regime == 0: # Range
                    mu = 0.0
                    sigma = 0.001
                elif regime == 1: # Bull
                    mu = 0.0003  # Strong upward drift
                    sigma = 0.0015
                else: # Bear
                    mu = -0.0004 # Sharp drops
                    sigma = 0.0025

            # GBM Step
            shock = np.random.normal(mu, sigma)
            current_price *= (1 + shock)
            
            # --- PATTERN INJECTION ---
            # Occasionally inject a "Liquidity Sweep" (Dip then Rip)
            is_sweep = False
            if regime == 1 and np.random.random() < 0.02: # 2% chance in bull trend
                # Create a dip (the sweep)
                current_price *= 0.995 
                is_sweep = True

            # Form Candle
            daily_vol = sigma * current_price
            
            op = current_price
            cl = current_price * (1 + np.random.normal(0, sigma/2))
            
            # High/Low logic
            hi = max(op, cl) + abs(np.random.normal(0, daily_vol))
            lo = min(op, cl) - abs(np.random.normal(0, daily_vol))
            
            if is_sweep:
                # Force Low to be deep (the sweep) but Close high
                lo = lo * 0.998 
                cl = max(op, cl) # Close green
            
            open_p[i] = op
            close_p[i] = cl
            high_p[i] = hi
            low_p[i] = lo
            
            # Volume (higher on volatile days)
            vol_p[i] = np.random.randint(5000, 50000) * (1 + abs(shock)*1000)

        # Generate timestamps
        start_date = datetime.now() - timedelta(days=days)
        timestamps = np.array([
            start_date + timedelta(minutes=k * (375 // candles_per_day))
            for k in range(total_candles)
        ])
        
        return OHLCVData(
            timestamps=timestamps,
            open=open_p, high=high_p, low=low_p, close=close_p, volume=vol_p
        )

    @classmethod
    def clear_cache(cls):
        """Clear cached data."""
        cls._cache.clear()
        cls._cache_time = None
