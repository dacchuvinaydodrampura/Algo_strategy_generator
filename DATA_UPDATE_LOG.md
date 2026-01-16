# Strategy Research Engine - Data Update Guide

This file tracks the last data update.

Last Update: NEVER
Next Due: Immediately

## How to Update Data

1. **Local**: Run the fetch script
   ```bash
   python fetch_data.py
   ```
   (Make sure you are logged into your trading bot / have .fyers_token)

2. **Commit**: Push the new CSV files
   ```bash
   git add data/
   git commit -m "Update market data (Next due: <DATE+15>)"
   git push
   ```

3. **Deploy**: Render will auto-deploy the new data.
