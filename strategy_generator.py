"""
Strategy Generator Module.

Auto-generates rule-based intraday trading strategies using:
- Price Action (70% weight)
- VWAP
- EMA (max 2)
- RSI (optional)

All strategies are algorithmically executable with fixed risk-reward.
"""

import random
from typing import List
from models import Strategy
from config import Config


class StrategyGenerator:
    """
    Generates rule-based trading strategies.
    
    Strategy composition:
    - 70% Price Action based
    - VWAP crossovers and bands
    - EMA crossovers (max 2 EMAs)
    - RSI filters (optional)
    """
    
    # ==========================================================================
    # ENTRY RULE TEMPLATES
    # ==========================================================================
    
    # Price Action entries (70% weight)
    PRICE_ACTION_ENTRIES = [
        ("Bullish Engulfing at Support", "LONG", 
         ["bullish_engulfing", "price_near_support"]),
        ("Bearish Engulfing at Resistance", "SHORT", 
         ["bearish_engulfing", "price_near_resistance"]),
        ("Inside Bar Breakout Long", "LONG", 
         ["inside_bar_breakout_up"]),
        ("Inside Bar Breakout Short", "SHORT", 
         ["inside_bar_breakout_down"]),
        ("Range Breakout Long", "LONG", 
         ["breakout_above_20bar_high"]),
        ("Range Breakout Short", "SHORT", 
         ["breakout_below_20bar_low"]),
        ("Morning Star Reversal", "LONG", 
         ["morning_star_pattern"]),
        ("Evening Star Reversal", "SHORT", 
         ["evening_star_pattern"]),
    ]
    
    # VWAP entries
    VWAP_ENTRIES = [
        ("VWAP Bounce Long", "LONG", 
         ["price_above_vwap", "pullback_to_vwap"]),
        ("VWAP Bounce Short", "SHORT", 
         ["price_below_vwap", "rally_to_vwap"]),
        ("VWAP Band Breakout Long", "LONG", 
         ["close_above_vwap_upper_band"]),
        ("VWAP Band Breakout Short", "SHORT", 
         ["close_below_vwap_lower_band"]),
    ]
    
    # EMA entries
    EMA_ENTRIES = [
        ("EMA Crossover Long", "LONG", 
         ["ema_fast_above_slow", "price_above_both_emas"]),
        ("EMA Crossover Short", "SHORT", 
         ["ema_fast_below_slow", "price_below_both_emas"]),
        ("EMA Pullback Long", "LONG", 
         ["price_above_ema_slow", "pullback_to_ema_fast"]),
        ("EMA Pullback Short", "SHORT", 
         ["price_below_ema_slow", "rally_to_ema_fast"]),
    ]
    
    # Institutional / SMC Entries (Higher Win Rate Logic)
    # Institutional / SMC Entries (Higher Win Rate Logic)
    INSTITUTIONAL_ENTRIES = [
        ("Liquidity Sweep Long", "LONG",
         ["liquidity_sweep_low", "bullish_order_block"]),
         
        ("Liquidity Sweep Short", "SHORT",
         ["liquidity_sweep_high", "bearish_order_block"]),
         
        ("Order Block Re-test Long", "LONG",
         ["price_above_ema_slow", "bullish_order_block"]),
         
        ("Order Block Re-test Short", "SHORT",
         ["price_below_ema_slow", "bearish_order_block"]),
         
        ("FVG Fill Reversal Long", "LONG",
         ["bullish_imbalance", "rsi_oversold_bounce"]), 
         
        ("Squeeze Breakout Long", "LONG",
         ["volatility_squeeze", "breakout_above_20bar_high"]),
         
        ("Squeeze Breakout Short", "SHORT",
         ["volatility_squeeze", "breakout_below_20bar_low"]),

        ("MSS Aggressive Long", "LONG",
         ["market_structure_shift_bullish"]),

        ("MSS Aggressive Short", "SHORT",
         ["market_structure_shift_bearish"]),

        ("Bullish Breaker Re-test", "LONG",
         ["bullish_breaker_retest"]),

        ("Bearish Breaker Re-test", "SHORT",
         ["bearish_breaker_retest"]),
         
        ("OTE Pullback Long", "LONG",
         ["optimal_trade_entry_long", "is_in_discount_zone"]),
         
        ("OTE Pullback Short", "SHORT",
         ["optimal_trade_entry_short", "is_in_premium_zone"]),

        ("Inducement Trap Long", "LONG",
         ["detect_inducement_bullish", "bullish_order_block"]),

        ("Inducement Trap Short", "SHORT",
         ["detect_inducement_bearish", "bearish_order_block"]),

        ("Aggressive CHoCH Long", "LONG",
         ["detect_choch_bullish"]),

        ("Aggressive CHoCH Short", "SHORT",
         ["detect_choch_bearish"]),

        ("Mitigation Block Long", "LONG",
         ["detect_mitigation_block_bullish"]),

        ("Mitigation Block Short", "SHORT",
         ["detect_mitigation_block_bearish"]),

        ("AMD PowerOf3 Long", "LONG",
         ["detect_amd_bullish"]),

        ("AMD PowerOf3 Short", "SHORT",
         ["detect_amd_bearish"]),

        ("Inverted FVG Long (Flip)", "LONG",
         ["detect_inverted_fvg_bullish"]),

        ("Inverted FVG Short (Flip)", "SHORT",
         ["detect_inverted_fvg_bearish"]),

        ("Turtle Soup Reversal Long", "LONG",
         ["detect_turtle_soup_bullish"]),

        ("Turtle Soup Reversal Short", "SHORT",
         ["detect_turtle_soup_bearish"]),

        ("Rejection Wick Long", "LONG",
         ["detect_rejection_block_bullish"]),

        ("Rejection Wick Short", "SHORT",
         ["detect_rejection_block_bearish"]),

        ("Propulsion Block Long", "LONG",
         ["detect_propulsion_block_bullish"]),

        ("Propulsion Block Short", "SHORT",
         ["detect_propulsion_block_bearish"]),

        ("Liquidity Void Fill Long", "LONG",
         ["detect_liquidity_void_bullish"]),

        ("Liquidity Void Fill Short", "SHORT",
         ["detect_liquidity_void_bearish"]),

        ("Balanced Price Range (BPR)", "LONG",
         ["detect_balanced_price_range", "price_above_ema_fast"]), # Context needed

        ("Volume Imbalance Gap", "LONG",
         ["detect_volume_imbalance_bullish"]),

        ("Mean Threshold Retest Long", "LONG",
         ["detect_mean_threshold_retest_bullish"]),

        ("Mean Threshold Retest Short", "SHORT",
         ["detect_mean_threshold_retest_bearish"]),

        ("Unicorn Model Long (Breaker+FVG)", "LONG",
         ["detect_unicorn_model_bullish"]),

        ("Unicorn Model Short (Breaker+FVG)", "SHORT",
         ["detect_unicorn_model_bearish"]),

        ("Silver Bullet Setup Long (Time+FVG)", "LONG",
         ["detect_silver_bullet_bullish"]),
        
        ("Silver Bullet Setup Short (Time+FVG)", "SHORT",
         ["detect_silver_bullet_bearish"]),

        ("Opening Gap Reclaim Long (NDOG)", "LONG",
         ["detect_opening_gap_reclaim_bullish"]),

        ("Opening Gap Reclaim Short (NDOG)", "SHORT",
         ["detect_opening_gap_reclaim_bearish"]),

        ("Reclaimed Block Long (Flip)", "LONG",
         ["detect_reclaimed_block_bullish"]),

        ("Reclaimed Block Short (Flip)", "SHORT",
         ["detect_reclaimed_block_bearish"]),

        ("Wick 50% Entry Long", "LONG",
         ["detect_wick_consequent_encroachment_bullish"]),

        ("Wick 50% Entry Short", "SHORT",
         ["detect_wick_consequent_encroachment_bearish"]),

        ("Order Flow Entry Long (Chain)", "LONG",
         ["detect_order_flow_entry_bullish"]),

        ("Order Flow Entry Short (Chain)", "SHORT",
         ["detect_order_flow_entry_bearish"]),

        ("Sponsorship Candle Long", "LONG",
         ["detect_sponsor_candle_bullish"]),

        ("Sponsorship Candle Short", "SHORT",
         ["detect_sponsor_candle_bearish"]),

        ("Range Mean Reversion Long", "LONG",
         ["detect_range_rotation_bullish"]),

        ("Range Mean Reversion Short", "SHORT",
         ["detect_range_rotation_bearish"]),

        ("Snap Back Reversal Long", "LONG",
         ["detect_snap_back_bullish"]),

        ("Snap Back Reversal Short", "SHORT",
         ["detect_snap_back_bearish"]),

        ("Quarterly Time Shift Long", "LONG",
         ["detect_quarterly_shift_bullish"]),

        ("Quarterly Time Shift Short", "SHORT",
         ["detect_quarterly_shift_bearish"]),

        ("Standard Deviation Reversal Long (Judas)", "LONG",
         ["detect_standard_deviation_projection_bullish"]),

        ("Standard Deviation Reversal Short (Judas)", "SHORT",
         ["detect_standard_deviation_projection_bearish"]),

        ("Power of 3 Swing Long (Manipulation)", "LONG",
         ["detect_power_of_3_swing_bullish"]),

        ("Power of 3 Swing Short (Manipulation)", "SHORT",
         ["detect_power_of_3_swing_bearish"]),

        ("Institutional Swing Low (Fractal)", "LONG",
         ["detect_institutional_swing_point_bullish"]),

        ("Institutional Swing High (Fractal)", "SHORT",
         ["detect_institutional_swing_point_bearish"]),

        ("SMT Divergence Long", "LONG",
         ["detect_smt_divergence_bullish"]),

        ("SMT Divergence Short", "SHORT",
         ["detect_smt_divergence_bearish"]),

        ("Macro Cycle Burst Long", "LONG",
         ["detect_macro_macro_bullish"]),
        
        ("Macro Cycle Burst Short", "SHORT",
         ["detect_macro_macro_bearish"]),

        ("Liquidity Run Reversal Long (Fade)", "LONG",
         ["detect_liquidity_run_bullish"]),

        ("Liquidity Run Reversal Short (Fade)", "SHORT",
         ["detect_liquidity_run_bearish"]),

        ("Stops Hunt Long (Purge)", "LONG",
         ["detect_stops_hunt_bullish"]),

        ("Stops Hunt Short (Purge)", "SHORT",
         ["detect_stops_hunt_bearish"]),

        ("Equilibrium Reclaim Long (Day)", "LONG",
         ["detect_equilibrium_reclaimed_bullish"]),

        ("Equilibrium Reclaim Short (Day)", "SHORT",
         ["detect_equilibrium_reclaimed_bearish"]),

        ("Partial Void Fill Long (50%)", "LONG",
         ["detect_partial_void_fill_bullish"]),

        ("Partial Void Fill Short (50%)", "SHORT",
         ["detect_partial_void_fill_bearish"]),

        ("Fractal Expansion Breakout Long", "LONG",
         ["detect_fractal_expansion_bullish"]),
         
        ("Fractal Expansion Breakout Short", "SHORT",
         ["detect_fractal_expansion_bearish"]),

        ("Initial Balance Breakout Long (60m)", "LONG",
         ["detect_initial_balance_breakout_bullish"]),

        ("Initial Balance Breakout Short (60m)", "SHORT",
         ["detect_initial_balance_breakout_bearish"]),

        ("ORB Breakout Long (30m)", "LONG",
         ["detect_opening_range_breakout_bullish"]),

        ("ORB Breakout Short (30m)", "SHORT",
         ["detect_opening_range_breakout_bearish"]),

        ("Daily Open Rejection Long", "LONG",
         ["detect_daily_open_rejection_bullish"]),

        ("Daily Open Rejection Short", "SHORT",
         ["detect_daily_open_rejection_bearish"]),

        ("Previous Week Low Sweep (PWL)", "LONG",
         ["detect_pwh_pwl_sweep_bullish"]),

        ("Previous Week High Sweep (PWH)", "SHORT",
         ["detect_pwh_pwl_sweep_bearish"]),

        ("Previous Day Low Sweep (PDL)", "LONG",
         ["detect_pdh_pdl_sweep_bullish"]),

        ("Previous Day High Sweep (PDH)", "SHORT",
         ["detect_pdh_pdl_sweep_bearish"]),

        ("Swing Failure Pattern Long (SFP)", "LONG",
         ["detect_swing_failure_pattern_bullish"]),

        ("Swing Failure Pattern Short (SFP)", "SHORT",
         ["detect_swing_failure_pattern_bearish"]),

        ("Momentum Impulse Long (2x ATR)", "LONG",
         ["detect_momentum_impulse_bullish"]),

        ("Momentum Impulse Short (2x ATR)", "SHORT",
         ["detect_momentum_impulse_bearish"]),

        ("Psychological Level Rejection Long", "LONG",
         ["detect_psychological_level_rejection_bullish"]),

        ("Psychological Level Rejection Short", "SHORT",
         ["detect_psychological_level_rejection_bearish"]),

        ("Trendline Liquidity Build Long (Target)", "LONG",
         ["detect_trendline_liquidity_build_bullish"]),

        ("Trendline Liquidity Build Short (Target)", "SHORT",
         ["detect_trendline_liquidity_build_bearish"]),

        ("Failed Auction Long (Market Profile)", "LONG",
         ["detect_failed_auction_bullish"]),

        ("Failed Auction Short (Market Profile)", "SHORT",
         ["detect_failed_auction_bearish"]),

        ("AMD Setup Long (Accum-Manip-Dist)", "LONG",
         ["detect_amd_setup_bullish"]),

        ("AMD Setup Short (Accum-Manip-Dist)", "SHORT",
         ["detect_amd_setup_bearish"]),

        ("Turtle Soup Long (20-Period Sweep)", "LONG",
         ["detect_turtle_soup_bullish"]),

        ("Turtle Soup Short (20-Period Sweep)", "SHORT",
         ["detect_turtle_soup_bearish"]),

        ("Decoupled Order Block Long", "LONG",
         ["detect_decoupled_ob_bullish"]),

        ("Decoupled Order Block Short", "SHORT",
         ["detect_decoupled_ob_bearish"]),

        ("Propulsion Candle Long (Continuation)", "LONG",
         ["detect_propulsion_candle_bullish"]),

        ("Propulsion Candle Short (Continuation)", "SHORT",
         ["detect_propulsion_candle_bearish"]),

        ("Engineered Liquidity Sweep Long (Dbl Bottom)", "LONG",
         ["detect_engineered_liquidity_bullish"]),

        ("Engineered Liquidity Sweep Short (Dbl Top)", "SHORT",
         ["detect_engineered_liquidity_bearish"]),

        ("Inverse FVG Long (Flip)", "LONG",
         ["detect_inverse_fvg_bullish"]),

        ("Inverse FVG Short (Flip)", "SHORT",
         ["detect_inverse_fvg_bearish"]),

        ("Mitigation Block Long (Failed Swing)", "LONG",
         ["detect_mitigation_block_bullish"]),

        ("Mitigation Block Short (Failed Swing)", "SHORT",
         ["detect_mitigation_block_bearish"]),

        ("Rejection Block Long (Wick)", "LONG",
         ["detect_rejection_block_bullish"]),

        ("Rejection Block Short (Wick)", "SHORT",
         ["detect_rejection_block_bearish"]),

        ("NWOG Retest Long (Gap Fill)", "LONG",
         ["detect_nwog_bullish"]),

        ("NWOG Retest Short (Gap Fill)", "SHORT",
         ["detect_nwog_bearish"]),

        ("Volume Void Entry Long (Imbalance)", "LONG",
         ["detect_volume_void_bullish"]),

        ("Volume Void Entry Short (Imbalance)", "SHORT",
         ["detect_volume_void_bearish"]),

        ("Dragon Pattern Long (W-Shape)", "LONG",
         ["detect_dragon_pattern_bullish"]),

        ("Dragon Pattern Short (M-Shape)", "SHORT",
         ["detect_dragon_pattern_bearish"]),

        ("Quasimodo Pattern Long (Over/Under)", "LONG",
         ["detect_quasimodo_pattern_bullish"]),

        ("Quasimodo Pattern Short (Over/Under)", "SHORT",
         ["detect_quasimodo_pattern_bearish"]),

        ("Triple Tap Reversal Long (3-Drive)", "LONG",
         ["detect_triple_tap_bullish"]),

        ("Triple Tap Reversal Short (3-Drive)", "SHORT",
         ["detect_triple_tap_bearish"]),

        ("Compression Liquidity Long (Falling CP)", "LONG",
         ["detect_compression_liquidity_bullish"]),

        ("Compression Liquidity Short (Rising CP)", "SHORT",
         ["detect_compression_liquidity_bearish"]),

        ("The Master Pattern Long (Expansion)", "LONG",
         ["detect_master_pattern_bullish"]),

        ("The Master Pattern Short (Expansion)", "SHORT",
         ["detect_master_pattern_bearish"]),

        ("Inducement Trap Long (IDM Reclaim)", "LONG",
         ["detect_inducement_bullish"]),

        ("Inducement Trap Short (IDM Reclaim)", "SHORT",
         ["detect_inducement_bearish"]),

        ("Balanced Price Range Long (BPR)", "LONG",
         ["detect_balanced_price_range_bullish"]),

        ("Balanced Price Range Short (BPR)", "SHORT",
         ["detect_balanced_price_range_bearish"]),

        ("Volume Imbalance Long (Body Gap)", "LONG",
         ["detect_volume_imbalance_bullish"]),

        ("Volume Imbalance Short (Body Gap)", "SHORT",
         ["detect_volume_imbalance_bearish"]),

        ("Judas Swing Long (Session Mock)", "LONG",
         ["detect_judas_swing_bullish"]),

        ("Judas Swing Short (Session Mock)", "SHORT",
         ["detect_judas_swing_bearish"]),

        ("Three Candle Reversal Long (3C)", "LONG",
         ["detect_three_candle_reversal_bullish"]),

        ("Three Candle Reversal Short (3C)", "SHORT",
         ["detect_three_candle_reversal_bearish"]),

        ("Optimal Trade Entry Long (OTE)", "LONG",
         ["detect_ote_entry_bullish"]),

        ("Optimal Trade Entry Short (OTE)", "SHORT",
         ["detect_ote_entry_bearish"]),

        ("Defining Range Breakout Long (DR)", "LONG",
         ["detect_defining_range_bullish"]),

        ("Defining Range Breakout Short (DR)", "SHORT",
         ["detect_defining_range_bearish"]),

        ("CPR Range Bounce Long", "LONG",
         ["detect_cpr_range_bullish"]),

        ("CPR Range Rejection Short", "SHORT",
         ["detect_cpr_range_bearish"]),

        ("Value Area Low (VAL) Long", "LONG",
         ["detect_value_area_bullish"]),

        ("Value Area High (VAH) Short", "SHORT",
         ["detect_value_area_bearish"]),

        ("Point of Control (POC) Rejection Long", "LONG",
         ["detect_poc_level_rejection_bullish"]),

        ("Point of Control (POC) Rejection Short", "SHORT",
         ["detect_poc_level_rejection_bearish"]),

        ("Poor High Target (Unfinished Auction)", "LONG",
         ["detect_poor_high_low_bullish_v2"]),

        ("Poor Low Target (Unfinished Auction)", "SHORT",
         ["detect_poor_high_low_bearish"]),

        ("Single Prints Zone Retest Long", "LONG",
         ["detect_single_prints_bullish"]),

        ("Single Prints Zone Retest Short", "SHORT",
         ["detect_single_prints_bearish"]),

        ("Buying Tail Rejection (Bullish)", "LONG",
         ["detect_tails_bullish"]),

        ("Selling Tail Rejection (Bearish)", "SHORT",
         ["detect_tails_bearish"]),

        ("Composite Operator Accumulation", "LONG",
         ["detect_composite_operator_bullish"]),

        ("Composite Operator Distribution", "SHORT",
         ["detect_composite_operator_bearish"]),

        ("Accumulation Phase (Spring/SOS)", "LONG",
         ["detect_dist_accum_bullish"]),

        ("Distribution Phase (Upthrust/SOW)", "SHORT",
         ["detect_dist_accum_bearish"]),

        ("Wyckoff Spring (Terminal Shakeout)", "LONG",
         ["detect_wyckoff_spring_bullish"]),

        ("Wyckoff Upthrust (Terminal Shakeout)", "SHORT",
         ["detect_wyckoff_upthrust_bearish"]),

        ("Sign of Strength (Aggressive Markup)", "LONG",
         ["detect_wyckoff_sos_bullish"]),

        ("Sign of Weakness (Aggressive Markdown)", "SHORT",
         ["detect_wyckoff_sow_bearish"]),

        ("Last Point of Support (LPS) Retest", "LONG",
         ["detect_wyckoff_lps_bullish"]),

        ("Last Point of Supply (LPSY) Retest", "SHORT",
         ["detect_wyckoff_lpsy_bearish"]),

        ("Effort vs Result (Absorption Long)", "LONG",
         ["detect_effort_vs_result_bullish"]),

        ("Effort vs Result (Absorption Short)", "SHORT",
         ["detect_effort_vs_result_bearish"]),

        ("Stopping Volume (Bullish Climax)", "LONG",
         ["detect_stopping_volume_bullish"]),

        ("Stopping Volume (Bearish Climax)", "SHORT",
         ["detect_stopping_volume_bearish"]),

        ("VSA No Supply (Bullish Test)", "LONG",
         ["detect_vsa_no_supply_bullish"]),

        ("VSA No Demand (Bearish Test)", "SHORT",
         ["detect_vsa_no_demand_bearish"]),

        ("VSA Shakeout (Bullish Sweep)", "LONG",
         ["detect_vsa_shakeout_bullish"]),

        ("VSA Shakeout (Bearish Sweep)", "SHORT",
         ["detect_vsa_shakeout_bearish"]),

        ("Selling Climax (Institutional Absorption)", "LONG",
         ["detect_selling_climax_bullish"]),

        ("Buying Climax (Institutional Absorption)", "SHORT",
         ["detect_buying_climax_bearish"]),
    ]

    # ==========================================================================
    # FILTER TEMPLATES
    # ==========================================================================
    
    RSI_FILTERS = [
        ("RSI not overbought", ["rsi_below_70"]),
        ("RSI not oversold", ["rsi_above_30"]),
        ("RSI oversold bounce", ["rsi_crossed_above_30"]),
        ("RSI overbought rejection", ["rsi_crossed_below_70"]),
        ("RSI neutral zone", ["rsi_between_40_60"]),
    ]
    
    VWAP_FILTERS = [
        ("Above VWAP", ["price_above_vwap"]),
        ("Below VWAP", ["price_below_vwap"]),
        ("Near VWAP", ["price_within_0.5pct_of_vwap"]),
    ]
    
    # ==========================================================================
    # EXIT RULE TEMPLATES
    # ==========================================================================
    
    EXIT_RULES = [
        "stop_loss_hit",
        "take_profit_hit",
        "ema_crossover_reversal",
        "price_crosses_vwap_opposite",
        "end_of_session",
    ]
    
    # ==========================================================================
    # STOP LOSS TEMPLATES
    # ==========================================================================
    
    STOP_LOSS_TYPES = [
        ("ATR-based (1.5x ATR)", "1.5x_atr_from_entry"),
        ("ATR-based (2x ATR)", "2x_atr_from_entry"),
        ("Fixed percentage (0.5%)", "0.5pct_from_entry"),
        ("Fixed percentage (1%)", "1pct_from_entry"),
        ("Below/Above swing low/high", "swing_point_stop"),
        ("Below/Above entry candle", "entry_candle_stop"),
    ]
    
    def __init__(self):
        """Initialize the generator."""
        self.generated_count = 0
    
    def generate(self, count: int = 1) -> List[Strategy]:
        """
        Generate multiple rule-based strategies.
        
        Args:
            count: Number of strategies to generate
        
        Returns:
            List of Strategy objects
        """
        strategies = []
        for _ in range(count):
            strategy = self._generate_single()
            strategies.append(strategy)
            self.generated_count += 1
        
        return strategies
    
    def _generate_single(self) -> Strategy:
        """Generate a single strategy with random rule combinations."""
        
        # Select market and timeframe
        market = random.choice(Config.MARKETS)
        timeframe = random.choice(Config.TIMEFRAMES)
        
        # Decide on main strategy type
        # 50% Institutional (SMC), 30% Price Action, 20% Technical (EMA/VWAP)
        strategy_type = random.choices(
            ["institutional", "price_action", "technical"],
            weights=[0.5, 0.3, 0.2]
        )[0]
        
        # Select base entry
        if strategy_type == "institutional":
             name, direction, entry_rules = random.choice(self.INSTITUTIONAL_ENTRIES)
             # Enforce Premium/Discount logic (50% chance)
             if random.random() < 0.5:
                 if direction == "LONG" and "is_in_discount_zone" not in entry_rules:
                     entry_rules = list(entry_rules) + ["is_in_discount_zone"]
                 elif direction == "SHORT" and "is_in_premium_zone" not in entry_rules:
                     entry_rules = list(entry_rules) + ["is_in_premium_zone"]
        elif strategy_type == "price_action":
            name, direction, entry_rules = random.choice(self.PRICE_ACTION_ENTRIES)
        else: # technical
            if random.random() < 0.5:
                name, direction, entry_rules = random.choice(self.VWAP_ENTRIES)
            else:
                name, direction, entry_rules = random.choice(self.EMA_ENTRIES)
        
        # Copy rules to avoid mutation
        entry_rules = list(entry_rules)
        
        # Add VWAP filter (50% chance)
        use_vwap = random.random() < 0.5
        if use_vwap and strategy_type != "vwap":
            filter_name, filter_rules = random.choice(self.VWAP_FILTERS)
            entry_rules.extend(filter_rules)
        
        # Add RSI filter (30% chance)
        use_rsi = random.random() < 0.3
        if use_rsi:
            filter_name, filter_rules = random.choice(self.RSI_FILTERS)
            entry_rules.extend(filter_rules)
        
        # Select EMA periods
        ema_fast = random.choice([9, 13])
        ema_slow = random.choice([21, 34])
        
        # Select risk-reward ratio
        risk_reward = round(random.uniform(Config.RR_MIN, Config.RR_MAX), 1)
        
        # Select stop loss logic
        sl_name, sl_logic = random.choice(self.STOP_LOSS_TYPES)
        
        # Take profit based on R:R
        tp_logic = f"{risk_reward}x_stop_distance"
        
        # Select exit rules
        exit_rules = ["stop_loss_hit", "take_profit_hit"]
        if random.random() < 0.5:
            exit_rules.append(random.choice(self.EXIT_RULES[2:]))
            
        # Add Kill Zone (Time) Filter (30% chance for any strategy)
        if random.random() < 0.3:
            if "is_in_kill_zone" not in entry_rules:
                entry_rules.append("is_in_kill_zone")
        
        # Generate unique name
        strategy_name = f"{name}_{timeframe}_{self.generated_count}"
        
        return Strategy(
            name=strategy_name,
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            stop_loss_logic=sl_logic,
            take_profit_logic=tp_logic,
            risk_reward=risk_reward,
            timeframe=timeframe,
            market=market,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            rsi_period=Config.RSI_PERIOD,
            rsi_oversold=Config.RSI_OVERSOLD,
            rsi_overbought=Config.RSI_OVERBOUGHT,
            use_vwap=use_vwap or strategy_type == "vwap",
            use_rsi=use_rsi,
        )


# Module-level instance for convenience
_generator = StrategyGenerator()


def generate_strategies(count: int = None) -> List[Strategy]:
    """
    Convenience function to generate strategies.
    
    Args:
        count: Number of strategies (defaults to config value)
    
    Returns:
        List of Strategy objects
    """
    if count is None:
        count = Config.STRATEGIES_PER_CYCLE
    return _generator.generate(count)
