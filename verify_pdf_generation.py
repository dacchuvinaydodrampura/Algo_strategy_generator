
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
            "bullish_breaker_retest",
            "optimal_trade_entry_long",
            "is_in_discount_zone",
            "detect_inducement_bullish",
            "is_in_kill_zone",
            "detect_choch_bullish",
            "detect_mitigation_block_bullish",
            "detect_amd_bullish",
            "detect_inverted_fvg_bullish",
            "detect_turtle_soup_bullish",
            "detect_rejection_block_bullish",
            "detect_propulsion_block_bullish",
            "detect_balanced_price_range",
            "detect_mean_threshold_retest_bullish",
            "detect_unicorn_model_bullish",
            "detect_silver_bullet_bullish",
            "detect_reclaimed_block_bullish",
            "detect_order_flow_entry_bullish",
            "detect_sponsor_candle_bullish",
            "detect_quarterly_shift_bullish",
            "detect_standard_deviation_projection_bullish",
            "detect_power_of_3_swing_bullish",
            "detect_smt_divergence_bullish",
            "detect_stops_hunt_bullish",
            "detect_fractal_expansion_bullish",
            "detect_initial_balance_breakout_bullish",
            "detect_pwh_pwl_sweep_bullish",
            "detect_swing_failure_pattern_bullish",
            "detect_psychological_level_rejection_bullish",
            "detect_trendline_liquidity_build_bullish",
            "detect_amd_setup_bullish",
            "detect_propulsion_candle_bullish",
            "detect_inverse_fvg_bullish",
            "detect_rejection_block_bullish",
            "detect_volume_void_bullish",
            "detect_dragon_pattern_bullish",
            "detect_quasimodo_pattern_bullish",
            "detect_triple_tap_bullish",
            "detect_master_pattern_bullish",
            "detect_inducement_bullish",
            "detect_balanced_price_range_bullish",
            "detect_volume_imbalance_bullish",
            "detect_judas_swing_bullish",
            "detect_three_candle_reversal_bullish",
            "detect_ote_entry_bullish",
            "detect_cpr_range_bullish",
            "detect_poc_level_rejection_bullish",
            "detect_single_prints_bullish",
            "detect_tails_bullish",
            "detect_dist_accum_bullish",
            "detect_wyckoff_spring_bullish",
            "detect_wyckoff_sos_bullish",
            "detect_effort_vs_result_bullish",
            "detect_vsa_no_supply_bullish",
            "detect_vsa_shakeout_bullish",
            "detect_selling_climax_bullish"
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
