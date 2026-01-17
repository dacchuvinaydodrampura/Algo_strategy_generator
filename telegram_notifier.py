"""
Telegram Notifier Module.

Sends alerts for passing strategies via Telegram Bot API.
Only sends messages when a strategy passes ALL filters.
"""

import requests
from typing import Optional
from models import Strategy, BacktestResult
from config import Config


class TelegramNotifier:
    """
    Sends Telegram notifications for passing strategies.
    
    Uses Telegram Bot API directly (no external dependencies).
    """
    
    BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"
    
    def __init__(self, bot_token: Optional[str] = None, 
                 chat_id: Optional[str] = None):
        """
        Initialize notifier.
        
        Args:
            bot_token: Telegram bot token (from @BotFather)
            chat_id: Target chat/group ID
        """
        self.bot_token = bot_token or Config.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or Config.TELEGRAM_CHAT_ID
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if not self.enabled:
            print("⚠️ Telegram notifications disabled (missing credentials)")
    
    def send(self, strategy: Strategy, 
             backtest_result: BacktestResult) -> bool:
        """
        Send notification for a passing strategy.
        
        Args:
            strategy: Strategy that passed
            backtest_result: Associated backtest results
        
        Returns:
            True if message sent successfully
        """
        if not self.enabled:
            print("📱 [TELEGRAM DISABLED] Would send notification for:", strategy.name)
            return False
        
        message = self._format_message(strategy, backtest_result)
        return self._send_message(message)
    
    def _format_message(self, strategy: Strategy, 
                         result: BacktestResult) -> str:
        """Format the notification message."""
        
        # Build performance summary for each period
        period_lines = []
        for period in sorted(result.period_results.keys()):
            pr = result.period_results[period]
            sign = "+" if pr.total_pnl_percent > 0 else ""
            period_lines.append(f"• {period}D: {sign}{pr.total_pnl_percent:.1f}%")
        
        period_summary = "\n".join(period_lines)
        
        # Entry/exit rules (truncated for readability)
        entry_short = ", ".join(strategy.entry_rules[:3])
        if len(strategy.entry_rules) > 3:
            entry_short += "..."
        
        exit_short = ", ".join(strategy.exit_rules[:2])
        
        message = f"""🎯 NEW STRATEGY PASSED

📊 Market: {strategy.market}
⏱ Timeframe: {strategy.timeframe}
📈 Trades/Year: {result.total_trades_per_year}

📉 Win Rate: {result.avg_win_rate:.1f}%
📉 Max Drawdown: {result.worst_drawdown:.1f}%
📈 Profit Factor: {result.avg_profit_factor:.2f}
📊 Expectancy: {result.avg_expectancy:.2f}

📊 Performance:
{period_summary}

📋 Entry: {entry_short}
🚪 Exit: {exit_short}
🛑 SL: {strategy.stop_loss_logic}
🎯 R:R: 1:{strategy.risk_reward}

ID: {strategy.id}

{self._get_detailed_rules(strategy)}"""
        
        return message

    def _get_detailed_rules(self, strategy: Strategy) -> str:
        """Generate detailed algorithmic verification rules."""
        lines = ["🔹 ALGO BACKTESTING RULES VERIFICATION 🔹\n"]
        counter = 1
        
        def escape(text):
            return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        # 1. Entry Patterns & Filters
        for rule in strategy.entry_rules:
            rule_lower = rule.lower()
            explanation = ""
            
            if "morning_star_pattern" in rule_lower:
                explanation = """   def is_morning_star(candles):
       # Candle 1 (Ind-2): Bearish (Close < Open)
       # Candle 2 (Ind-1): Small Body (Body < 60% of Candle 1 Body)
       # Candle 3 (Ind-0): Bullish (Close > Open) AND (Close > Midpoint of Candle 1)
       return True if all_conditions_met else False"""
            
            elif "bullish_engulfing" in rule_lower:
                explanation = """   def is_bullish_engulfing(candles):
       # Candle 1 (Ind-1): Bearish
       # Candle 2 (Ind-0): Bullish AND Body Engulfs Candle 1 Body
       return True"""

            elif "price_within_0.5pct_of_vwap" in rule_lower:
                explanation = """   # Entry valid ONLY if price is within 0.5% of VWAP line
   dist = abs(Close - VWAP) / Close
   return True if dist <= 0.005 else False"""
            
            elif "rsi_above_30" in rule_lower:
                explanation = f"""   # RSI({strategy.rsi_period}) filter
   return True if RSI[idx] > 30 else False"""
   
            elif "rsi_below_70" in rule_lower:
                 explanation = f"""   # RSI({strategy.rsi_period}) filter
   return True if RSI[idx] < 70 else False"""

            if explanation:
                name = rule.replace("_", " ").title() + " Logic"
                lines.append(f"{counter}. {name}:")
                lines.append(escape(explanation) + "\n")
                counter += 1

            # --- SMC EXPLANATIONS ---
            explanation = ""
            if "liquidity_sweep_low" in rule_lower:
                explanation = """   # 1. Price BREAKS below 20-candle Low
   # 2. Price CLOSES back above that Low (Reclaim)
   # Indicates Stop Hunt / Bear Trap."""
            elif "bullish_order_block" in rule_lower:
                explanation = """   # Bullish Order Block:
   # 1. Bearish Candle followed by Strong Bullish Impulse
   # 2. Impulse Body > 2x Bearish Body (Institutional Footprint)"""
            elif "volatility_squeeze" in rule_lower:
                explanation = """   # Bollinger Squeeze:
   # Current Range < 50% of Average Range (Energy Building)"""
            elif "bullish_imbalance" in rule_lower:
                explanation = """   # Fair Value Gap (Buying):
   # Low(Current) > High(2 candles ago)
   # Price jumped leaving unfilled orders."""
   
            if explanation:
                name = rule.replace("_", " ").title() + " (Institutional)"
                lines.append(f"{counter}. {name}:")
                lines.append(escape(explanation) + "\n")
                counter += 1

        # 2. Stop Loss Logic
        sl_logic = strategy.stop_loss_logic
        sl_expl = ""
        if "entry_candle_stop" in sl_logic:
            sl_expl = """   # Dynamic SL based on entry candle's High/Low volatility
   stop_dist = max(abs(Entry - Low), abs(High - Entry))
   SL_Price = Entry - stop_dist (for Long)"""
        elif "1.5x_atr" in sl_logic:
             sl_expl = """   # Volatility-based Stop Loss
   stop_dist = ATR(14) * 1.5
   SL_Price = Entry - stop_dist"""
        
        if sl_expl:
            lines.append(f"{counter}. Stop Loss ({sl_logic}):")
            lines.append(escape(sl_expl) + "\n")
            counter += 1

        # 3. Take Profit Logic
        lines.append(f"{counter}. Take Profit (1:{strategy.risk_reward} RR):")
        lines.append(f"   TP_Price = Entry + (stop_dist * {strategy.risk_reward})")
        
        return "\n".join(lines)
    
    def _send_message(self, message: str) -> bool:
        """Send message via Telegram API."""
        url = self.BASE_URL.format(token=self.bot_token)
        
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML",
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ Telegram notification sent")
                return True
            else:
                print(f"❌ Telegram error: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Telegram request failed: {e}")
            return False
    
    def send_status(self, message: str) -> bool:
        """
        Send a status/info message.
        
        Args:
            message: Status message to send
        
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False
        
        return self._send_message(f"ℹ️ {message}")


# Module-level instance
_notifier = TelegramNotifier()


def notify_strategy(strategy: Strategy, 
                    backtest_result: BacktestResult) -> bool:
    """
    Convenience function to send strategy notification.
    
    Args:
        strategy: Passing strategy
        backtest_result: Associated results
    
    Returns:
        True if notification sent
    """
    return _notifier.send(strategy, backtest_result)


def notify_status(message: str) -> bool:
    """Send a status message."""
    return _notifier.send_status(message)
