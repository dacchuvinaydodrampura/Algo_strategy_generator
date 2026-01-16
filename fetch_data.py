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
    """Connect using existing token from algo_trading_bot or local file."""
    token_path = "../algo_trading_bot/.fyers_token"
    if not os.path.exists(token_path):
        token_path = ".fyers_token"
    
    if not os.path.exists(token_path):
        print("❌ .fyers_token not found! Please run your bot login first.")
        return None
        
    with open(token_path, "r") as f:
        access_token = f.read().strip()
        
    app_id = os.getenv("FYERS_APP_ID")
    if not app_id:
        print("❌ FYERS_APP_ID not found in .env")
        return None
        
    try:
        fyers = fyersModel.FyersModel(client_id=app_id, token=access_token, log_path="")
        if fyers.get_profile().get("s") == "ok":
            print("✅ Connected to Fyers API")
            return fyers
        else:
            print("❌ Token seems invalid")
            return None
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None

def fetch_and_save(fyers, symbol, timeframe, days):
    """Fetch history and save to CSV."""
    print(f"📥 Fetching {symbol} ({timeframe}) for last {days} days...")
    
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)
    
    tf_map = {'1m': '1', '5m': '5', '15m': '15'}
    resolution = tf_map.get(timeframe, '5')
    
    data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": from_date.strftime("%Y-%m-%d"),
        "range_to": to_date.strftime("%Y-%m-%d"),
        "cont_flag": "1"
    }
    
    try:
        response = fyers.history(data)
        if response.get("s") != "ok":
            print(f"   ⚠️ API Error: {response}")
            return
            
        candles = response.get("candles", [])
        if not candles:
            print("   ⚠️ No candles returned")
            return
            
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(candles, columns=cols)
        
        # Convert timestamp
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        # Save to CSV
        filename = f"data/{symbol.replace(':', '_')}_{timeframe}.csv"
        df.to_csv(filename, index=False)
        print(f"   ✅ Saved {len(df)} rows to {filename}")
        
    except Exception as e:
        print(f"   ❌ Fetch Error: {e}")

def main():
    fyers = connect_fyers()
    if not fyers:
        return

    # Markets to fetch
    markets = Config.MARKETS
    timeframes = ["5m", "15m"]         # Reduced timeframes to save space/time
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
