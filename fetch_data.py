"""
Static Data Fetcher.

Run this script LOCALLY to fetch historical data from Fyers.
Saves data to 'data/' csv files which are then pushed to GitHub.
This avoids API rate limits and authentication issues on Render.

Run this every 15 days to update your dataset.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
from config import Config

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

def connect_fyers():
    """Connect using local token file or from algo_trading_bot."""
    token_path = ".fyers_token"
    if not os.path.exists(token_path):
        token_path = "../algo_trading_bot/.fyers_token"
    
    if not os.path.exists(token_path):
        print("❌ .fyers_token not found!")
        return None
        
    with open(token_path, "r") as f:
        access_token = f.read().strip()
        
    app_id = os.getenv("FYERS_APP_ID")
    if not app_id:
        print("❌ FYERS_APP_ID not found in .env")
        return None
        
    try:
        fyers = fyersModel.FyersModel(client_id=app_id, token=access_token, log_path="")
        profile = fyers.get_profile()
        
        if profile.get("s") == "ok":
            print("✅ Connected to Fyers API")
            return fyers
        else:
            print("⚠️ Access token invalid, attempting refresh...")
            
            # Try refresh
            refresh_path = "../algo_trading_bot/.fyers_refresh_token"
            if not os.path.exists(refresh_path):
                refresh_path = ".fyers_refresh_token"
                
            if os.path.exists(refresh_path):
                with open(refresh_path, "r") as f:
                    refresh_token = f.read().strip()
                
                secret_id = os.getenv("FYERS_SECRET_ID")
                if not secret_id:
                    print("❌ FYERS_SECRET_ID needed for refresh")
                    return None

                import requests
                import hashlib
                
                # AppIdHash = SHA256(app_id + ":" + app_secret)
                app_id_hash = hashlib.sha256(f"{app_id}:{secret_id}".encode()).hexdigest()
                
                url = "https://api-t1.fyers.in/api/v3/validate-refresh-token"
                payload = {
                    "grant_type": "refresh_token",
                    "appIdHash": app_id_hash,
                    "refresh_token": refresh_token,
                    "pin": os.getenv("FYERS_PIN", "")
                }
                
                res = requests.post(url, json=payload).json()
                if res.get("s") == "ok":
                    new_access = res.get("access_token")
                    print("✅ Token Refreshed Successfully!")
                    
                    # Update fyers object
                    fyers = fyersModel.FyersModel(client_id=app_id, token=new_access, log_path="")
                    return fyers
                else:
                    print(f"❌ Refresh failed: {res}")
            else:
                 print("❌ No refresh token found")
                 
            return None
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None

def fetch_and_save(fyers, symbol, timeframe, total_days):
    """Fetch history in chunks and save to CSV."""
    print(f"📥 Fetching {symbol} ({timeframe}) for last {total_days} days...")
    
    tf_map = {'1m': '1', '5m': '5', '15m': '15'}
    resolution = tf_map.get(timeframe, '5')
    
    # Chunk size based on timeframe
    # Fyers limit is approx 100 days for 5m/15m, less for 1m
    chunk_size = 90 if timeframe != '1m' else 30
    
    all_candles = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=total_days)
    
    current_to = end_date
    
    while current_to > start_date:
        current_from = current_to - timedelta(days=chunk_size)
        if current_from < start_date:
            current_from = start_date
            
        print(f"   ... fetching {current_from.date()} to {current_to.date()}")
        
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": current_from.strftime("%Y-%m-%d"),
            "range_to": current_to.strftime("%Y-%m-%d"),
            "cont_flag": "1"
        }
        
        try:
            response = fyers.history(data)
            if response.get("s") == "ok":
                candles = response.get("candles", [])
                if candles:
                    all_candles.extend(candles)
            else:
                 print(f"      ⚠️ Chunk failed: {response}")
                 
        except Exception as e:
            print(f"      ❌ Chunk error: {e}")
            
        # Move back for next chunk
        current_to = current_from - timedelta(days=1)
    
    if not all_candles:
        print(f"   ❌ No data fetched for {symbol}")
        return

    # Process and Save
    try:
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(all_candles, columns=cols)
        
        # Deduplicate based on timestamp
        df = df.drop_duplicates(subset=['timestamp'])
        
        # Sort by time
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values('datetime')
        
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        # Save to CSV
        filename = f"data/{symbol.replace(':', '_')}_{timeframe}.csv"
        df.to_csv(filename, index=False)
        print(f"   ✅ Saved {len(df)} rows to {filename}")
        
    except Exception as e:
        print(f"   ❌ Save Error: {e}")

def main():
    fyers = connect_fyers()
    if not fyers:
        return

    # Markets to fetch
    markets = Config.MARKETS
    timeframes = ["1m", "5m", "15m"]   # Added 1m
    max_days = 365                     # Always fetch max history (365 days)
    
    print(f"\n🚀 Starting Data Update Cycle for {len(markets)} markets...")
    
    for market in markets:
        for tf in timeframes:
            fetch_and_save(fyers, market, tf, max_days)
            
    print("\n✅ Update Complete! Now do:")
    print("git add data/")
    print("git commit -m 'Update market data'")
    print("git push")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
