
from pdf_generator import generate_strategy_pdf
from models import Strategy, BacktestResult, PeriodResult, Trade
import os
from datetime import datetime

def main():
    print("🚀 Verifying PDF Generation with Multi-Period Tables & Charts...")
    
    # Mock data
    s = Strategy(
        name="INSTITUTIONAL_TEST_V2",
        market="NSE:NIFTY50",
        timeframe="15m",
        entry_rules=["liquidity_sweep", "order_block", "is_in_discount"],
        exit_rules=["stop_loss_hit", "take_profit_hit"],
        stop_loss_logic="1.5x_atr",
        risk_reward=3.0
    )
    
    # Create Mock Trades for Chart
    # Need to match dates in your static CSV if you want actual plotting, 
    # but for PDF structure test, we just check if it runs without crashing.
    # The ChartGenerator inside pdf_generator calls get_static_data.
    # If the CSV is missing, get_static_data returns empty/random, chart might fail gracefully.
    # We will assume some data exists or handled gracefully.
    
    mock_trades = [
        Trade("2025-01-01 10:00", "2025-01-01 12:00", 100.0, 105.0, "LONG", 5.0, 5.0),
        Trade("2025-01-02 11:00", "2025-01-01 11:30", 102.0, 101.0, "SHORT", 1.0, 1.0),
    ]

    # Create Results for multiple periods
    periods = {}
    for days in [30, 60, 180, 365]:
        periods[days] = PeriodResult(
            period_days=days,
            total_pnl=1000.0 * (days/30),
            total_pnl_percent=10.0 * (days/30),
            win_rate=0.55,
            max_drawdown=500.0,
            max_drawdown_percent=5.0,
            expectancy=0.5,
            profit_factor=1.5 + (days/1000),
            total_trades=int(10 * (days/30)),
            winning_trades=5,
            losing_trades=5,
            avg_win=200.0,
            avg_loss=100.0,
            largest_win=500.0,
            largest_loss=200.0,
            largest_win_contribution=2.0,
            trades=mock_trades if days == 30 else []
        )
    
    r = BacktestResult(
        strategy=s,
        period_results=periods,
        all_periods_profitable=True,
        worst_drawdown=5.0,
        avg_win_rate=0.55,
        avg_expectancy=0.5,
        avg_profit_factor=1.8,
        total_trades_per_year=120
    )
    
    filename = "test_report_enhanced.pdf"
    
    # Ensure a dummy CSV exists for chart generator to not crash completely if it tries to load
    # (Though ChartGenerator handles empty data, let's just run it.)
    
    success = generate_strategy_pdf(s, r, filename)
    
    if success and os.path.exists(filename):
        print("✅ Enhanced PDF Generated Successfully!")
        print(f"File created: {filename}")
        # Clean up
        # os.remove(filename) # Keep it to inspect if manual run
    else:
        print("❌ PDF Generation Failed!")

if __name__ == "__main__":
    main()
