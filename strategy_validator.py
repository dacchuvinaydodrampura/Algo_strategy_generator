"""
Strategy Validator Module.

Validates strategy logic for:
- Algorithmic executability
- No look-ahead bias
- No discretionary terms
- Proper rule definitions
"""

from typing import List, Tuple
from models import Strategy


class StrategyValidator:
    """
    Validates strategies for backtesting and live execution.
    
    Checks:
    1. All rules are algorithmically defined
    2. No look-ahead bias indicators
    3. Entry/exit rules are non-conflicting
    4. Stop loss and take profit are defined
    5. Timeframe is valid
    """
    
    # Discretionary terms that indicate non-algorithmic logic
    DISCRETIONARY_TERMS = [
        "maybe", "might", "could", "should consider",
        "looks like", "seems", "feels", "opinion",
        "discretion", "judgment", "manual", "visual",
        "news", "sentiment", "announcement", "event",
    ]
    
    # Look-ahead bias indicators (repainting)
    LOOKAHEAD_INDICATORS = [
        "future", "next_candle", "will_be", "going_to",
        "repaint", "zigzag", "pivot_confirmed",
    ]
    
    # Valid entry rule keywords
    VALID_ENTRY_KEYWORDS = [
        "above", "below", "crossover", "crossed",
        "breakout", "breakdown", "pullback", "rally",
        "engulfing", "inside_bar", "morning_star", "evening_star",
        "ema", "vwap", "rsi", "atr", "price", "close", "open",
        "high", "low", "support", "resistance", "swing",
        "band", "upper", "lower", "between", "within",
        "hit", "near", "pct", "candle", "bar",
        # Institutional Pattern Keywords
        "order_block", "sweep", "imbalance", "squeeze",
    ]
    
    # Valid timeframes
    VALID_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h"]
    
    def __init__(self):
        """Initialize validator."""
        self.validation_errors: List[str] = []
    
    def validate(self, strategy: Strategy) -> Tuple[bool, List[str]]:
        """
        Validate a strategy for executability.
        
        Args:
            strategy: Strategy to validate
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        self.validation_errors = []
        
        # Run all validation checks
        self._check_entry_rules(strategy)
        self._check_exit_rules(strategy)
        self._check_stop_loss(strategy)
        self._check_take_profit(strategy)
        self._check_timeframe(strategy)
        self._check_discretionary_terms(strategy)
        self._check_lookahead_bias(strategy)
        self._check_risk_reward(strategy)
        
        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors
    
    def _check_entry_rules(self, strategy: Strategy):
        """Check entry rules are properly defined."""
        if not strategy.entry_rules:
            self.validation_errors.append("No entry rules defined")
            return
        
        for rule in strategy.entry_rules:
            # Check rule contains valid keywords
            rule_lower = rule.lower()
            has_valid_keyword = any(
                kw in rule_lower for kw in self.VALID_ENTRY_KEYWORDS
            )
            if not has_valid_keyword:
                self.validation_errors.append(
                    f"Entry rule '{rule}' does not contain valid indicator keywords"
                )
    
    def _check_exit_rules(self, strategy: Strategy):
        """Check exit rules are properly defined."""
        if not strategy.exit_rules:
            self.validation_errors.append("No exit rules defined")
            return
        
        # Must have stop_loss_hit as minimum
        has_stop_loss_exit = any(
            "stop" in rule.lower() for rule in strategy.exit_rules
        )
        if not has_stop_loss_exit:
            self.validation_errors.append("Exit rules must include stop loss exit")
    
    def _check_stop_loss(self, strategy: Strategy):
        """Check stop loss logic is defined."""
        if not strategy.stop_loss_logic:
            self.validation_errors.append("Stop loss logic not defined")
            return
        
        # Must be quantifiable
        valid_sl_types = ["atr", "pct", "swing", "candle", "fixed"]
        sl_lower = strategy.stop_loss_logic.lower()
        
        has_valid_sl = any(sltype in sl_lower for sltype in valid_sl_types)
        if not has_valid_sl:
            self.validation_errors.append(
                f"Stop loss '{strategy.stop_loss_logic}' is not quantifiable"
            )
    
    def _check_take_profit(self, strategy: Strategy):
        """Check take profit logic is defined."""
        if not strategy.take_profit_logic:
            self.validation_errors.append("Take profit logic not defined")
    
    def _check_timeframe(self, strategy: Strategy):
        """Check timeframe is valid for intraday."""
        if strategy.timeframe not in self.VALID_TIMEFRAMES:
            self.validation_errors.append(
                f"Timeframe '{strategy.timeframe}' is not valid for intraday"
            )
    
    def _check_discretionary_terms(self, strategy: Strategy):
        """Check for discretionary/subjective terms."""
        all_rules = (
            strategy.entry_rules + 
            strategy.exit_rules + 
            [strategy.stop_loss_logic, strategy.take_profit_logic]
        )
        
        for rule in all_rules:
            rule_lower = rule.lower()
            for term in self.DISCRETIONARY_TERMS:
                if term in rule_lower:
                    self.validation_errors.append(
                        f"Rule contains discretionary term: '{term}'"
                    )
    
    def _check_lookahead_bias(self, strategy: Strategy):
        """Check for look-ahead bias indicators."""
        all_rules = (
            strategy.entry_rules + 
            strategy.exit_rules + 
            [strategy.stop_loss_logic, strategy.take_profit_logic]
        )
        
        for rule in all_rules:
            rule_lower = rule.lower()
            for indicator in self.LOOKAHEAD_INDICATORS:
                if indicator in rule_lower:
                    self.validation_errors.append(
                        f"Rule contains look-ahead bias: '{indicator}'"
                    )
    
    def _check_risk_reward(self, strategy: Strategy):
        """Check risk-reward is within allowed range."""
        if strategy.risk_reward < 2.0:
            self.validation_errors.append(
                f"Risk-reward {strategy.risk_reward} is less than 2.0 (User Rule)"
            )
        if strategy.risk_reward > 5.0:
            self.validation_errors.append(
                f"Risk-reward {strategy.risk_reward} is unrealistically high"
            )


# Module-level instance
_validator = StrategyValidator()


def validate_strategy(strategy: Strategy) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate a strategy.
    
    Args:
        strategy: Strategy to validate
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    return _validator.validate(strategy)


def validate_strategies(strategies: List[Strategy]) -> List[Strategy]:
    """
    Validate multiple strategies and return only valid ones.
    
    Args:
        strategies: List of strategies to validate
    
    Returns:
        List of valid strategies
    """
    valid_strategies = []
    
    for strategy in strategies:
        is_valid, errors = _validator.validate(strategy)
        if is_valid:
            valid_strategies.append(strategy)
        else:
            print(f"⚠️ Strategy {strategy.name} invalid: {errors[0]}")
    
    return valid_strategies
