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

import logging
import sys
import threading
import time
from flask import Flask

from config import Config
from strategy_generator import generate_strategies
from strategy_validator import validate_strategies
from backtest_engine import run_backtest
from consistency_filter import check_consistency
from strategy_repository import store_strategy
from telegram_notifier import notify_strategy
from intelligence_module import train_brain
from git_sync import sync_to_github
from ai_module import get_ai_insight

# Configure logging to stdout (Crucial for Render Logs)
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for verbose output
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def health_check():
    return "Strategy Research Engine Running", 200

# ... (flask app remains same) ...

def run_cycle():
    """Run a single strategy research cycle."""
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info(f"🚀 Strategy Research Engine - Cycle Started")

    try:
        # Validate configuration
        Config.validate()
        
        # 🧠 Intelligence: Learn from past results before generating new ones
        logger.info("🧠 Brain Sync: Training DNA from market experience...")
        train_brain()
        
        # Stats for this cycle
        generated = 0
        validated = 0
        backtested = 0
        passed = 0
        notified = 0

        # 🌐 AI Internet Brain: Consult Gemini for Market Context
        logger.info("🌐 AI Brain: Consulting Google Gemini for Internet Market Context...")
        ai_insight = get_ai_insight()
        
        # Step 1: Generate strategies
        logger.info(f"📝 Generating {Config.STRATEGIES_PER_CYCLE} strategies using AI Bias...")
        strategies = generate_strategies(Config.STRATEGIES_PER_CYCLE, ai_insight)
        generated = len(strategies)
        
        for i, s in enumerate(strategies):
            logger.debug(f"[Gen] Strategy {i+1}: {s.name} ({s.market} {s.timeframe})")
            logger.debug(f"      Rules: {s.entry_rules}")
        
        # Step 2: Validate strategies
        valid_strategies = validate_strategies(strategies)
        validated = len(valid_strategies)
        logger.info(f"✅ Validated {validated}/{generated} strategies.")
        
        if not valid_strategies:
            logger.warning("   No valid strategies to backtest")
            return
        
        # Step 3 & 4: Backtest and filter each strategy
        for strategy in valid_strategies:
            logger.info(f"👉 Testing: {strategy.name}")
            
            # Backtest across all periods
            try:
                result = run_backtest(strategy)
                backtested += 1
                
                # Log Summary of Backtest
                logger.debug(f"   📊 Backtest Results for {strategy.name}:")
                for period_days, res in result.period_results.items():
                    logger.debug(f"      {period_days}d: PnL {res.total_pnl:.2f} | WR {res.win_rate:.1f}% | DD {res.max_drawdown_percent:.1f}% | Trades {res.total_trades}")

                # Apply consistency filter
                is_consistent, failures = check_consistency(result)
                
                if is_consistent:
                    passed += 1
                    if len(failures) > 0:
                        logger.info(f"   ⚠️ PASSED (With {len(failures)} Failure): {failures[0]}")
                    else:
                        logger.info(f"   ✅ PASSED ALL FILTERS (Perfect)")
                    
                    # Store strategy
                    store_strategy(strategy, result)
                    
                    # Send Telegram notification
                    if notify_strategy(strategy, result):
                        notified += 1
                else:
                    logger.info(f"   ❌ Filter Failed: {failures[:1]}") # Log first failure reason
                    logger.debug(f"      All Failures: {failures}")

            except Exception as bt_error:
                logger.error(f"   ⚠️ Backtest crash for {strategy.name}: {bt_error}")

            # Explicit memory cleanup for 512MB constraint
            import gc
            gc.collect()

        # 🔄 Final Step: Auto-Sync Permanent Memory back to GitHub
        if passed > 0 or validated > 0:
            logger.info("📡 GitSync: Backing up Intelligence Memory to GitHub...")
            sync_to_github()

        # Summary
        elapsed = time.time() - start_time
        logger.info("-" * 60)
        logger.info(f"📈 Cycle Complete - {elapsed:.1f}s")
        logger.info(f"   Gen: {generated} | Val: {validated} | B/T: {backtested} | Pass: {passed}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Cycle failed with error: {e}")
        import traceback
        traceback.print_exc()

def background_loop():
    """Continuous loop for background execution."""
    # Small startup delay to ensure logs are ready
    time.sleep(2)
    logger.info("⏳ Starting background strategy loop...")
    
    while True:
        try:
            run_cycle()
        except Exception as e:
            logger.error(f"❌ Critical Error in Loop: {e}")
        
        # Sleep for 60 seconds (configurable)
        logger.info("💤 Sleeping for 60s...")
        time.sleep(60)

# Start background thread when app starts
threading.Thread(target=background_loop, daemon=True).start()

if __name__ == "__main__":
    # Local development
    app.run(host='0.0.0.0', port=8080)
