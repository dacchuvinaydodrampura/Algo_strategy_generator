from fyers_apiv3 import fyersModel
import os
import datetime

# Read token
with open(".fyers_token", "r") as f:
    token = f.read().strip()
    
app_id = os.getenv("FYERS_APP_ID")
fyers = fyersModel.FyersModel(client_id=app_id, token=token, log_path="")

# Test 1: Quote (Verify Symbol)
print("Testing Quote for NSE:NIFTY50-INDEX...")
q = fyers.quotes({"symbols": "NSE:NIFTY50-INDEX"})
print(f"Quote: {q}")

# Test 2: History (Small Range)
print("\nTesting History (5 days)...")
to_date = datetime.datetime.now()
from_date = to_date - datetime.timedelta(days=5)

data = {
    "symbol": "NSE:NIFTY50-INDEX",
    "resolution": "5",
    "date_format": "1",
    "range_from": from_date.strftime("%Y-%m-%d"),
    "range_to": to_date.strftime("%Y-%m-%d"),
    "cont_flag": "1"
}

h = fyers.history(data)
print(f"History: {h.get('s')}")
if h.get('s') != 'ok':
    print(h)
else:
    print(f"Candles: {len(h.get('candles', []))}")
