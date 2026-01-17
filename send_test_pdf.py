
from models import Strategy, BacktestResult, PeriodResult, Trade
from telegram_notifier import notify_strategy
from config import Config
import random

def main():
    print("🚀 Sending Enhanced Test PDF to Telegram...")
    
    # Check Credentials
    if not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_CHAT_ID:
        print("❌ Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not found in .env")
        return

    # Create Mock Strategy with rich rules for the glossary
    strategy = Strategy(
        name="INSTITUTIONAL_LIQUIDITY_V3",
        market="NSE:NIFTY50",
        timeframe="15m",
        entry_rules=[
            "liquidity_sweep_low", 
            "bullish_order_block_retest", 
            "market_structure_shift_bullish",
            "detect_fair_value_gap_bullish",
            "detect_kill_zone_london_open"
        ],
        exit_rules=["stop_loss_hit", "take_profit_hit", "rsi_overbought"],
        stop_loss_logic="1.5x_atr",
        risk_reward=3.5,
    )

    # Mock Trades for Chart
    mock_trades = [
        Trade("2025-01-01 10:00", "2025-01-01 12:00", 100.0, 105.0, "LONG", 5.0, 5.0),
        Trade("2025-01-02 11:00", "2025-01-01 11:30", 102.0, 101.0, "SHORT", 1.0, 1.0),
        Trade("2025-01-05 09:30", "2025-01-05 14:00", 98.0, 108.0, "LONG", 10.0, 10.2),
    ]

    # Create Rich Multi-Period Results
    periods = {}
    for days in [30, 60, 180, 365]:
        multiplier = days / 30
        periods[days] = PeriodResult(
            period_days=days,
            total_pnl=2500.0 * multiplier,
            total_pnl_percent=15.0 * multiplier,
            win_rate=0.68,
            max_drawdown=1200.0,
            max_drawdown_percent=8.5,
            expectancy=0.65,
            profit_factor=2.2,
            total_trades=int(20 * multiplier),
            winning_trades=int(14 * multiplier),
            losing_trades=int(6 * multiplier),
            avg_win=300.0,
            avg_loss=150.0,
            largest_win=800.0,
            largest_loss=300.0,
            largest_win_contribution=5.0,
            trades=mock_trades if days == 30 else []
        )

    result = BacktestResult(
        strategy=strategy,
        period_results=periods,
        all_periods_profitable=True,
        worst_drawdown=8.5,
        avg_win_rate=0.68,
        avg_expectancy=0.65,
        avg_profit_factor=2.2,
        total_trades_per_year=240
    )

    # Send Notification
    print("Generating PDF and sending...")
    success = notify_strategy(strategy, result)
    
    if success:
        print("✅ Enhanced PDF sent to Telegram successfully!")
    else:
        print("❌ Failed to send Telegram alert.")

if __name__ == "__main__":
    main()
