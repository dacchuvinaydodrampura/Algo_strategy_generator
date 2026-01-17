
"""
Test Telegram Notification Script.
Run this to verify that your bot can properly send alerts to your Telegram.

It simulates a "Super Strategy" that matches your new strict criteria:
- Nifty 50
- 62% Win Rate
- 2.5 Profit Factor
- 2:1 Reward:Risk
"""
from models import Strategy, BacktestResult, PeriodResult
from telegram_notifier import notify_strategy, notify_status, TelegramNotifier
from config import Config

def main():
    print("🚀 Testing Telegram Connection...")
    
    # 1. Check Credentials
    if not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_CHAT_ID:
        print("❌ Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not found in .env")
        print("Please check your .env file.")
        return

    # 2. Send a simple status message first
    print(f"Attempting to send status to Chat ID: {Config.TELEGRAM_CHAT_ID}...")
    success = notify_status("<b>System Test:</b> Telegram connection verified! ✅\nPreparing to send example strategy alert...")
    
    if not success:
        print("❌ Failed to send status message. Check bot token/permissions.")
        return
        
    # 3. Create a Mock Strategy that meets strict rules
    strategy = Strategy(
        name="Nifty Golden Trend V1",
        market="NSE:NIFTY50-INDEX",
        timeframe="15m",
        entry_rules=["morning_star_pattern", "price_within_0.5pct_of_vwap", "rsi_above_30"],
        exit_rules=["stop_loss_hit", "take_profit_hit"],
        stop_loss_logic="entry_candle_stop",
        risk_reward=2.0,  # Min 2.0
    )

    # 4. Create Mock Results (High performance)
    result = BacktestResult(strategy=strategy)
    result.total_trades_per_year = 450
    result.avg_win_rate = 62.5       # > 60%
    result.worst_drawdown = 12.5     # < 20%
    result.avg_profit_factor = 2.45  # > 2.0
    result.avg_expectancy = 0.85     # > 0.5
    
    # Add dummy period results so the message formats correctly
    def mock_per(pnl): return PeriodResult(30,0,pnl,0,0,0,0,0,0,0,0,0,0,0,0,0)
    result.period_results = {
        30: mock_per(35.5),   # > 33%
        60: mock_per(72.0),   # > 66%
        180: mock_per(160.0), # > 150%
        365: mock_per(215.0)  # > 200%
    }

    # 5. Send Strategy Alert
    print("Sending Strategy Alert...")
    success = notify_strategy(strategy, result)
    
    if success:
        print("✅ Strategy Validation Alert Sent Successfully!")
        print("Check your Telegram!")
    else:
        print("❌ Failed to send strategy alert.")

if __name__ == "__main__":
    main()
