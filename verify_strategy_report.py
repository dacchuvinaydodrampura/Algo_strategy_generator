
"""
Verification Report Generator.
Simulates the 'NEW STRATEGY PASSED' output for the user's specific strategy
and appends the underlying algorithmic rules used for backtesting.
"""
from models import Strategy, BacktestResult
from telegram_notifier import TelegramNotifier
import datetime

def main():
    # 1. Reconstruct the User's Strategy (ID: 8cda20fc) based on their provided stats
    strategy = Strategy(
        id="8cda20fc",
        name="Morning Star VWAP (User Var)",
        market="NSE:NIFTY50-INDEX",
        timeframe="15m",
        entry_rules=["morning_star_pattern", "price_within_0.5pct_of_vwap", "rsi_above_30"],
        exit_rules=["stop_loss_hit", "take_profit_hit"],
        stop_loss_logic="entry_candle_stop",
        risk_reward=1.7
    )

    # 2. Mock the Backtest Result to match user's values exactly
    # These are hardcoded because the user wants "this data" to be sent/verified
    result = BacktestResult(strategy=strategy)
    result.total_trades_per_year = 521
    result.avg_win_rate = 43.6
    result.worst_drawdown = 1.0  # Assumes user meant 1.0%
    result.avg_profit_factor = 1.16
    result.avg_expectancy = 5.29
    
    # Mock Period Results (stored in simple dict structure for notifier)
    # The notifier uses result.period_results[period].total_pnl_percent
    # We need to create dummy PeriodResult objects
    from models import PeriodResult
    
    def mock_period(pnl_pct):
        return PeriodResult(
            period_days=30, # dummy
            total_pnl=0,
            total_pnl_percent=pnl_pct,
            win_rate=0, max_drawdown=0, max_drawdown_percent=0,
            expectancy=0, profit_factor=0, total_trades=0,
            winning_trades=0, losing_trades=0, avg_win=0, avg_loss=0,
            largest_win=0, largest_loss=0, largest_win_contribution=0
        )

    result.period_results = {
        30: mock_period(0.8),
        60: mock_period(2.4),
        180: mock_period(4.3),
        365: mock_period(9.1)
    }

    # 3. Use TelegramNotifier's format logic
    notifier = TelegramNotifier(bot_token="dummy", chat_id="dummy") # Disabled state
    # We manually invoke the formatting method which now includes detailed rules
    full_output = notifier._format_message(strategy, result)
    
    with open("report_output.txt", "w", encoding="utf-8") as f:
        f.write("-" * 40 + "\n")
        f.write(full_output + "\n")
        f.write("-" * 40 + "\n")
    
    print("Report written to report_output.txt")

if __name__ == "__main__":
    main()
