"""
Fyers Data Provider.

Fetches real OHLCV data from Fyers API v3.
Replaces synthetic data generator for production use.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from fyers_apiv3 import fyersModel
from data_provider import OHLCVData
from config import Config

class FyersDataProvider:
    """
    Fetches historical data from Fyers API.
    """
    
    def __init__(self):
        self.api = None
        self.connected = False
        self._connect()
        
    def _connect(self):
        """Connect to Fyers using stored token or env vars."""
        try:
            app_id = os.getenv('FYERS_APP_ID')
            token = None
            
            # Try finding token in various locations
            possible_paths = [
                '.fyers_token', 
                '../algo_trading_bot/.fyers_token',
                '../../algo_trading_bot/.fyers_token'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        token = f.read().strip()
                    print(f"✅ Found Fyers token at {path}")
                    break
            
            if not token:
                token = os.getenv('FYERS_ACCESS_TOKEN')
            
            if app_id and token:
                self.api = fyersModel.FyersModel(
                    client_id=app_id,
                    token=token,
                    log_path=""
                )
                
                # Verify connection
                try:
                    profile = self.api.get_profile()
                    if profile.get('s') == 'ok':
                        self.connected = True
                        print("✅ Connected to Fyers API")
                    else:
                        print(f"⚠️ Fyers Token Invalid: {profile}")
                except Exception as e:
                    print(f"⚠️ Fyers Connection Failed: {e}")
            else:
                print("⚠️ Missing Fyers Credentials (APP_ID or Token)")
                
        except Exception as e:
            print(f"❌ Fyers Init Error: {e}")

    def get_data(self, market: str, timeframe: str, days: int) -> OHLCVData:
        """
        Get OHLCV data for market.
        
        Args:
            market: Symbol (should be valid Fyers symbol, e.g., "NSE:NIFTY50-INDEX")
            timeframe: "1m", "5m", "15m"
            days: History length in days
        """
        if not self.connected:
            print("⚠️ Using synthetic data (Fyers not connected)")
            from data_provider import DataProvider as SyntheticProvider
            return SyntheticProvider.get_data(market, timeframe, days)
        
        try:
            # Map timeframe
            tf_map = {'1m': '1', '5m': '5', '15m': '15', '30m': '30', '1h': '60'}
            resolution = tf_map.get(timeframe, '5')
            
            # Calculate dates
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            # Fyers History API limit handling
            # Limit is usually based on candle count or date range.
            # We'll fetch in chunks if needed or just request the range.
            # Fyers Python SDK handles date ranges usually, but for safe side:
            
            data = {
                "symbol": market,
                "resolution": resolution,
                "date_format": "1",
                "range_from": from_date.strftime("%Y-%m-%d"),
                "range_to": to_date.strftime("%Y-%m-%d"),
                "cont_flag": "1"
            }
            
            response = self.api.history(data)
            
            if response.get('s') != 'ok':
                print(f"⚠️ Fyers History Error: {response}")
                # Fallback
                from data_provider import DataProvider as SyntheticProvider
                return SyntheticProvider.get_data(market, timeframe, days)
            
            candles = response.get('candles', [])
            if not candles:
                print("⚠️ No data returned from Fyers")
                from data_provider import DataProvider as SyntheticProvider
                return SyntheticProvider.get_data(market, timeframe, days)
                
            # Parse candles: [timestamp, open, high, low, close, volume]
            df = pd.DataFrame(candles, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            
            return OHLCVData(
                timestamps=pd.to_datetime(df['ts'], unit='s').values,
                open=df['o'].values,
                high=df['h'].values,
                low=df['l'].values,
                close=df['c'].values,
                volume=df['v'].values
            )
            
        except Exception as e:
            print(f"❌ Data Fetch Error: {e}")
            from data_provider import DataProvider as SyntheticProvider
            return SyntheticProvider.get_data(market, timeframe, days)

# Global instance
_fyers_provider = FyersDataProvider()

def get_fyers_data(market, timeframe, days):
    return _fyers_provider.get_data(market, timeframe, days)
