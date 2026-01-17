"""
Configuration Management for Strategy Research Engine.

All settings loaded from environment variables with sensible defaults.
Optimized for 512MB RAM constraint on Render free tier.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Central configuration class."""
    
    # ==========================================================================
    # TELEGRAM SETTINGS
    # ==========================================================================
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    
    # ==========================================================================
    # ENGINE SETTINGS
    # ==========================================================================
    STRATEGIES_PER_CYCLE: int = int(os.getenv("STRATEGIES_PER_CYCLE", "5"))
    STORAGE_PATH: str = os.getenv("STORAGE_PATH", "strategies.json")
    
    # ==========================================================================
    # BACKTEST PERIODS (days)
    # ==========================================================================
    BACKTEST_PERIODS: list = [30, 60, 180, 365]
    
    # ==========================================================================
    # CONSISTENCY FILTER THRESHOLDS
    # ==========================================================================
    MAX_DRAWDOWN_LIMIT: float = 0.20  # 20%
    MAX_SINGLE_TRADE_CONTRIBUTION: float = 0.35  # Relaxed slightly to allow high return strategies
    MIN_EXPECTANCY: float = 0.5   # Positive expectancy required
    MIN_WIN_RATE: float = 60.0    # 60%
    MIN_PROFIT_FACTOR: float = 2.0
    
    # Minimum Return Targets (Percentage)
    MIN_RETURN_30D: float = 33.0
    MIN_RETURN_60D: float = 66.0
    MIN_RETURN_180D: float = 150.0
    MIN_RETURN_365D: float = 200.0
    
    # ==========================================================================
    # STRATEGY GENERATION PARAMETERS
    # ==========================================================================
    TIMEFRAMES: list = ["1m", "5m", "15m"]
    # Valid Fyers Symbols
    MARKETS: list = ["NSE:NIFTY50-INDEX"]
    
    # Risk-Reward ratios (min, max)
    RR_MIN: float = 2.0
    RR_MAX: float = 4.0
    
    # EMA periods to use
    EMA_PERIODS: list = [9, 13, 21, 34, 50]
    
    # RSI levels
    RSI_OVERSOLD: int = 30
    RSI_OVERBOUGHT: int = 70
    RSI_PERIOD: int = 14
    
    # VWAP bands
    VWAP_BAND_MULTIPLIER: float = 2.0
    
    # ==========================================================================
    # DATA SETTINGS
    # ==========================================================================
    # Candles per day approximations
    CANDLES_PER_DAY_1M: int = 375  # NSE trading hours
    CANDLES_PER_DAY_5M: int = 75
    CANDLES_PER_DAY_15M: int = 25
    
    # ==========================================================================
    # LOGGING
    # ==========================================================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate critical configuration settings."""
        errors = []
        
        if not cls.TELEGRAM_BOT_TOKEN:
            errors.append("TELEGRAM_BOT_TOKEN not set")
        if not cls.TELEGRAM_CHAT_ID:
            errors.append("TELEGRAM_CHAT_ID not set")
        
        if errors:
            print(f"⚠️ Configuration warnings: {', '.join(errors)}")
            return False
        return True
    
    @classmethod
    def get_candles_per_day(cls, timeframe: str) -> int:
        """Get approximate candles per trading day for timeframe."""
        mapping = {
            "1m": cls.CANDLES_PER_DAY_1M,
            "5m": cls.CANDLES_PER_DAY_5M,
            "15m": cls.CANDLES_PER_DAY_15M,
        }
        return mapping.get(timeframe, cls.CANDLES_PER_DAY_5M)
