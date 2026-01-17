
from pdf_generator import generate_strategy_pdf
from models import Strategy, BacktestResult, PeriodResult
import os

def main():
    print("🚀 Verifying PDF Generation...")
    
    # Mock data
    s = Strategy(
        name="INSTITUTIONAL_TEST",
        market="NSE:NIFTY50",
        timeframe="15m",
        entry_rules=[
            "liquidity_sweep_low", 
            "bullish_order_block", 
            "volatility_squeeze",
            "market_structure_shift_bullish",
            "bullish_breaker_retest"
        ],
        exit_rules=["rsi_above_70"],
        stop_loss_logic="1.5x_atr",
        risk_reward=3.0
    )
    
    # Fully populated PeriodResult
    p_res = PeriodResult(
        period_days=365,
        total_pnl=50000.0,
        total_pnl_percent=250.0,
        win_rate=0.65,
        max_drawdown=5000.0,
        max_drawdown_percent=10.0,
        expectancy=0.5,
        profit_factor=2.5,
        total_trades=100,
        winning_trades=65,
        losing_trades=35,
        avg_win=1000.0,
        avg_loss=400.0,
        largest_win=2000.0,
        largest_loss=1000.0,
        largest_win_contribution=4.0
    )
    
    r = BacktestResult(
        strategy=s,
        total_trades_per_year=100,
        avg_expectancy=0.5,
        avg_profit_factor=2.5,
        avg_win_rate=0.65,
        period_results={
            365: p_res
        }
    )
    
    filename = "test_report.pdf"
    success = generate_strategy_pdf(s, r, filename)
    
    if success and os.path.exists(filename):
        print("✅ PDF Generated Successfully!")
        print(f"File created: {filename}")
        # Clean up
        os.remove(filename)
    else:
        print("❌ PDF Generation Failed!")

if __name__ == "__main__":
    main()
