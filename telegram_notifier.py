"""
Telegram Notifier Module.

Sends alerts for passing strategies via Telegram Bot API.
Only sends messages when a strategy passes ALL filters.
"""

import requests
import os
from typing import Optional
from models import Strategy, BacktestResult
from config import Config
from pdf_generator import generate_strategy_pdf

class TelegramNotifier:
    """
    Sends Telegram notifications for passing strategies.
    
    Uses Telegram Bot API directly (no external dependencies).
    """
    
    BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"
    DOC_URL = "https://api.telegram.org/bot{token}/sendDocument"
    
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
        Send notification for a passing strategy with PDF report.
        """
        if not self.enabled:
            print("📱 [TELEGRAM DISABLED] Would send notification for:", strategy.name)
            return False
        
        # 1. Generate PDF
        pdf_filename = f"/tmp/strategy_{strategy.id}.pdf"
        try:
            # Ensure tmp dir exists
            os.makedirs(os.path.dirname(pdf_filename), exist_ok=True)
            
            pdf_success = generate_strategy_pdf(strategy, backtest_result, pdf_filename)
            
            # 2. Format Caption
            caption = self._format_message(strategy, backtest_result)
            
            # 3. Send
            if pdf_success:
                print(f"📤 Sending PDF report: {pdf_filename}")
                res = self._send_document(pdf_filename, caption=caption[:1024])
                
                # Cleanup
                if os.path.exists(pdf_filename):
                    os.remove(pdf_filename)
                return res
            else:
                print("⚠️ PDF Generation failed, falling back to detailed text")
                full_text = self._format_fallback_text(strategy, backtest_result)
                return self._send_message(full_text)
                
        except Exception as e:
            print(f"❌ Error in send flow: {e}")
            return False

    def _format_message(self, strategy: Strategy, 
                         result: BacktestResult) -> str:
        """Format the notification message (Short Summary for Caption)."""
        
        # Build performance summary for each period
        res_365 = result.period_results.get(365)
        stats = ""
        if res_365:
            stats = f"""
💰 <b>PnL (365d):</b> {res_365.total_pnl:,.2f}
🎯 <b>Win Rate:</b> {res_365.win_rate*100:.1f}%
📉 <b>DD:</b> {res_365.max_drawdown*100:.1f}%
📈 <b>PF:</b> {res_365.profit_factor:.2f}"""

        msg = f"""<b>INSTITUTIONAL STRATEGY PASSED</b> 🦅
        
<b>ID:</b> {strategy.name}
<b>Market:</b> {strategy.market} ({strategy.timeframe})
{stats}

<i>Full detailed report attached as PDF.</i> 📄"""
        return msg

    def _format_fallback_text(self, strategy: Strategy, result: BacktestResult) -> str:
        """Fallback detailed text report if PDF fails."""
        
        # Entry rules
        rules_text = ""
        for rule in strategy.entry_rules:
            rules_text += f"• {rule}\n"
            
        msg = f"""🎯 <b>STRATEGY PASSED (TEXT FALLBACK)</b>
        
<b>ID:</b> {strategy.name}
<b>Market:</b> {strategy.market} ({strategy.timeframe})

<b>Rules:</b>
{rules_text}

<b>Institutional Patterns Detected:</b>
{'✅ Liquidity Sweeps' if any('sweep' in r for r in strategy.entry_rules) else ''}
{'✅ Order Blocks' if any('order_block' in r for r in strategy.entry_rules) else ''}

<i>(PDF Report Generation Failed - This is a text backup)</i>
"""
        return msg

    def _send_document(self, filepath: str, caption: str) -> bool:
        """Send PDF document via Telegram."""
        url = self.DOC_URL.format(token=self.bot_token)
        
        try:
            with open(filepath, 'rb') as f:
                files = {'document': f}
                data = {
                    'chat_id': self.chat_id,
                    'caption': caption,
                    'parse_mode': 'HTML'
                }
                response = requests.post(url, files=files, data=data, timeout=30)
                
                if response.status_code == 200:
                    print(f"✅ Telegram PDF sent successfully")
                    return True
                else:
                    print(f"❌ Telegram PDF error: {response.status_code} - {response.text}")
                    return False
        except Exception as e:
            print(f"❌ Telegram document upload failed: {e}")
            return False
            
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
        """Send a status/info message."""
        if not self.enabled:
            return False
        return self._send_message(f"ℹ️ {message}")


# Module-level instance
_notifier = TelegramNotifier()


def notify_strategy(strategy: Strategy, 
                    backtest_result: BacktestResult) -> bool:
    """Convenience function to send strategy notification."""
    return _notifier.send(strategy, backtest_result)


def notify_status(message: str) -> bool:
    """Send a status message."""
    return _notifier.send_status(message)
