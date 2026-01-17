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
