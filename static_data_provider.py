"""
Static Data Provider.

Loads OHLCV data from local CSV files in 'data/' directory.
Used for Render deployment where API access is restricted.
"""

import os
import pandas as pd
import numpy as np
from data_provider import OHLCVData

class StaticDataProvider:
    """
    Loads data from static CSV files.
    """
    
    @staticmethod
    def get_data(market: str, timeframe: str, days: int) -> OHLCVData:
        """
        Load data from CSV and filter by requested days.
        """
        # File naming convention: NSE_NIFTY50-INDEX_5m.csv
        safe_market = market.replace(':', '_')
        filename = f"data/{safe_market}_{timeframe}.csv"
        
        if not os.path.exists(filename):
            print(f"⚠️ Data file not found: {filename}")
            # Fallback to generator
            from data_provider import DataProvider as SyntheticProvider
            return SyntheticProvider.get_data(market, timeframe, days)
            
        try:
            # Load CSV
            df = pd.read_csv(filename)
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Filter for last N days
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
            mask = df['datetime'] > cutoff_date
            filtered_df = df.loc[mask]
            
            if filtered_df.empty:
                print(f"⚠️ No data found for last {days} days in {filename}")
                return OHLCVData([], [], [], [], [], [])
            
            return OHLCVData(
                timestamps=filtered_df['datetime'].values,
                open=filtered_df['open'].values,
                high=filtered_df['high'].values,
                low=filtered_df['low'].values,
                close=filtered_df['close'].values,
                volume=filtered_df['volume'].values
            )
            
        except Exception as e:
            print(f"❌ Error loading CSV {filename}: {e}")
            from data_provider import DataProvider as SyntheticProvider
            return SyntheticProvider.get_data(market, timeframe, days)

# Global accessor
def get_static_data(market, timeframe, days):
    return StaticDataProvider.get_data(market, timeframe, days)
