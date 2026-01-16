"""
Strategy Research Engine - Main Entry Point.

Cron-safe entry point for Render deployment.
Runs a single cycle:
1. Generate N strategies
2. Validate logic
3. Backtest all time windows
4. Apply consistency filter
5. Store passing strategies
6. Notify via Telegram

No infinite loop - designed for Render cron jobs.
"""

import sys
import time
from datetime import datetime

# Import all modules
from config import Config
from strategy_generator import generate_strategies
from strategy_validator import validate_strategies
from backtest_engine import run_backtest
from consistency_filter import check_consistency
from strategy_repository import store_strategy, get_stored_count
from telegram_notifier import notify_strategy
from flask import Flask
import threading

# Flask App for Render Web Service (Free Tier Hack)
app = Flask(__name__)

@app.route('/')
def home():
    """Keep-alive endpoint."""
    return f"Strategy Engine Running | Stored: {get_stored_count()}", 200

def run_cycle():
    """Run a single strategy research cycle."""
    start_time = time.time()
    
    print("=" * 60)
    print(f"🚀 Strategy Research Engine - Cycle Started")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # ... rest of the logic ...
    
    # Validate configuration
    Config.validate()
    
    # Stats for this cycle
    generated = 0
    validated = 0
    backtested = 0
    passed = 0
    notified = 0
    
    try:
        # Step 1: Generate strategies
        print(f"\n📝 Generating {Config.STRATEGIES_PER_CYCLE} strategies...")
        strategies = generate_strategies(Config.STRATEGIES_PER_CYCLE)
        generated = len(strategies)
        
        # Step 2: Validate strategies
        valid_strategies = validate_strategies(strategies)
        validated = len(valid_strategies)
        
        if not valid_strategies:
            print("   No valid strategies to backtest")
            return
        
        # Step 3 & 4: Backtest and filter each strategy
        for strategy in valid_strategies:
            # Backtest across all periods
            result = run_backtest(strategy)
            backtested += 1
            
            # Apply consistency filter
            is_consistent, failures = check_consistency(result)
            
            if is_consistent:
                passed += 1
                print(f"   ✅ PASSED all filters: {strategy.name}")
                
                # Store strategy
                store_strategy(strategy, result)
                
                # Send Telegram notification
                if notify_strategy(strategy, result):
                    notified += 1
        
        # Summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"📈 Cycle Complete - {elapsed:.1f}s")
        print(f"   Generated:  {generated}")
        print(f"   Validated:  {validated}")
        print(f"   Backtested: {backtested}")
        print(f"   Passed:     {passed}")
        print(f"   Notified:   {notified}")
        print(f"   Total Stored: {get_stored_count()}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Cycle failed with error: {e}")
        import traceback
        traceback.print_exc()

def background_loop():
    """Continuous loop for background execution."""
    print("⏳ Starting background strategy loop...")
    while True:
        try:
            run_cycle()
        except Exception as e:
            print(f"❌ Critical Error in Loop: {e}")
        
        # Sleep for 60 seconds (configurable)
        print("💤 Sleeping for 60s...")
        time.sleep(60)

# Start background thread when app starts
threading.Thread(target=background_loop, daemon=True).start()

if __name__ == "__main__":
    # Local development
    app.run(host='0.0.0.0', port=8080)
