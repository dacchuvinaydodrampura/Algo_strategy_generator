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
{'✅ Market Structure Shift (MSS)' if any('market_structure_shift' in r for r in strategy.entry_rules) else ''}
{'✅ Breaker Blocks' if any('breaker' in r for r in strategy.entry_rules) else ''}
{'✅ Premium/Discount Zone' if any('is_in_' in r for r in strategy.entry_rules) else ''}
{'✅ Optimal Trade Entry (OTE)' if any('optimal_trade_entry' in r for r in strategy.entry_rules) else ''}
{'✅ Inducement Trap (IDM)' if any('inducement' in r for r in strategy.entry_rules) else ''}
{'✅ Kill Zone (Time Filter)' if any('kill_zone' in r for r in strategy.entry_rules) else ''}
{'✅ Change of Character (CHoCH)' if any('choch' in r for r in strategy.entry_rules) else ''}
{'✅ Mitigation Block' if any('mitigation' in r for r in strategy.entry_rules) else ''}
{'✅ Power of 3 (AMD)' if any('amd' in r for r in strategy.entry_rules) else ''}
{'✅ Inverted FVG (Flip)' if any('ifvg' in r for r in strategy.entry_rules) else ''}
{'✅ Turtle Soup (Minor Stop Hunt)' if any('turtle' in r for r in strategy.entry_rules) else ''}
{'✅ Rejection Block (Wick)' if any('rejection' in r for r in strategy.entry_rules) else ''}
{'✅ Propulsion Block' if any('propulsion' in r for r in strategy.entry_rules) else ''}
{'✅ Liquidity Void' if any('void' in r for r in strategy.entry_rules) else ''}
{'✅ Balanced Price Range (BPR)' if any('bpr' in r for r in strategy.entry_rules) else ''}
{'✅ Volume Imbalance' if any('volume_imbalance' in r for r in strategy.entry_rules) else ''}
{'✅ Mean Threshold Retest' if any('mean_threshold' in r for r in strategy.entry_rules) else ''}
{'✅ Unicorn Model' if any('unicorn' in r for r in strategy.entry_rules) else ''}
{'✅ Silver Bullet (Time)' if any('silver_bullet' in r for r in strategy.entry_rules) else ''}
{'✅ Opening Gap Reclaim' if any('gap_reclaim' in r for r in strategy.entry_rules) else ''}
{'✅ Reclaimed Block' if any('reclaimed_block' in r for r in strategy.entry_rules) else ''}
{'✅ Wick 50% (C.E.)' if any('wick_ce' in r for r in strategy.entry_rules) else ''}
{'✅ Order Flow Entry' if any('order_flow' in r for r in strategy.entry_rules) else ''}
{'✅ Sponsorship Candle' if any('sponsor' in r for r in strategy.entry_rules) else ''}
{'✅ Range Rotation' if any('range_rotation' in r for r in strategy.entry_rules) else ''}
{'✅ Snap Back' if any('snap_back' in r for r in strategy.entry_rules) else ''}
{'✅ Quarterly Shift' if any('quarterly_shift' in r for r in strategy.entry_rules) else ''}
{'✅ Std Deviation Projection' if any('std_dev' in r for r in strategy.entry_rules) else ''}
{'✅ Power of 3 Swing' if any('po3_swing' in r for r in strategy.entry_rules) else ''}
{'✅ Institutional Swing Point' if any('inst_swing' in r for r in strategy.entry_rules) else ''}
{'✅ SMT Divergence (Simulated)' if any('smt_div' in r for r in strategy.entry_rules) else ''}
{'✅ Macro Cycle Burst' if any('macro' in r for r in strategy.entry_rules) else ''}
{'✅ Liquidity Run (Cascade)' if any('liq_run' in r for r in strategy.entry_rules) else ''}
{'✅ Stops Hunt (Purge)' if any('stops_hunt' in r for r in strategy.entry_rules) else ''}
{'✅ Equilibrium Reclaim' if any('eq_reclaim' in r for r in strategy.entry_rules) else ''}
{'✅ Partial Void Fill (50%)' if any('partial_void' in r for r in strategy.entry_rules) else ''}
{'✅ Fractal Expansion' if any('fractal_exp' in r for r in strategy.entry_rules) else ''}
{'✅ Initial Balance Break' if any('ib_breakout' in r for r in strategy.entry_rules) else ''}
{'✅ ORB Strategy' if any('orb' in r for r in strategy.entry_rules) else ''}
{'✅ Daily Open Rejection' if any('daily_open_rej' in r for r in strategy.entry_rules) else ''}
{'✅ PWH/PWL Sweep' if any('pwh_pwl' in r for r in strategy.entry_rules) else ''}
{'✅ PDH/PDL Sweep' if any('pdh_pdl' in r for r in strategy.entry_rules) else ''}
{'✅ SFP Pattern' if any('sfp' in r for r in strategy.entry_rules) else ''}
{'✅ Momentum Impulse' if any('impulse' in r for r in strategy.entry_rules) else ''}
{'✅ Psych Level Rejection' if any('psych_level' in r for r in strategy.entry_rules) else ''}
{'✅ Trendline Liquidity' if any('tl_liquidity' in r for r in strategy.entry_rules) else ''}
{'✅ Failed Auction' if any('failed_auction' in r for r in strategy.entry_rules) else ''}
{'✅ AMD Structure' if any('amd' in r for r in strategy.entry_rules) else ''}
{'✅ Turtle Soup' if any('turtle_soup' in r for r in strategy.entry_rules) else ''}
{'✅ Decoupled OB' if any('decoupled_ob' in r for r in strategy.entry_rules) else ''}
{'✅ Propulsion Candle' if any('propulsion_candle' in r for r in strategy.entry_rules) else ''}
{'✅ Engineered Liq Sweep' if any('eng_liquidity' in r for r in strategy.entry_rules) else ''}
{'✅ Inverse FVG (Flip)' if any('ifvg' in r for r in strategy.entry_rules) else ''}
{'✅ Mitigation Block' if any('mitigation_block' in r for r in strategy.entry_rules) else ''}
{'✅ Rejection Block' if any('rejection_block' in r for r in strategy.entry_rules) else ''}
{'✅ NWOG Retest' if any('nwog' in r for r in strategy.entry_rules) else ''}
{'✅ Volume Void' if any('vol_void' in r for r in strategy.entry_rules) else ''}
{'✅ Dragon Pattern' if any('dragon' in r for r in strategy.entry_rules) else ''}
{'✅ Quasimodo (QM)' if any('quasimodo' in r for r in strategy.entry_rules) else ''}
{'✅ Triple Tap (3-Drive)' if any('triple_tap' in r for r in strategy.entry_rules) else ''}
{'✅ Compression (CP)' if any('compression' in r for r in strategy.entry_rules) else ''}
{'✅ Master Pattern' if any('master_pattern' in r for r in strategy.entry_rules) else ''}
{'✅ Inducement (IDM)' if any('inducement' in r for r in strategy.entry_rules) else ''}
{'✅ Balanced Price Range' if any('bpr' in r for r in strategy.entry_rules) else ''}
{'✅ Volume Imbalance' if any('vol_imbalance' in r for r in strategy.entry_rules) else ''}
{'✅ Judas Swing' if any('judas' in r for r in strategy.entry_rules) else ''}
{'✅ 3-Candle Formation' if any('3c_rev' in r for r in strategy.entry_rules) else ''}
{'✅ Optimal Trade Entry' if any('ote' in r for r in strategy.entry_rules) else ''}
{'✅ Defining Range (DR)' if any('dr_range' in r for r in strategy.entry_rules) else ''}
{'✅ CPR Range' if any('cpr' in r for r in strategy.entry_rules) else ''}
{'✅ Value Area (VA)' if any('value_area' in r for r in strategy.entry_rules) else ''}
{'✅ Point of Control' if any('poc' in r for r in strategy.entry_rules) else ''}
{'✅ Poor High/Low' if any('poor_high_low' in r for r in strategy.entry_rules) else ''}
{'✅ Single Prints' if any('single_prints' in r for r in strategy.entry_rules) else ''}
{'✅ Buying/Selling Tail' if any('tails' in r for r in strategy.entry_rules) else ''}
{'✅ Composite Operator' if any('composite_op' in r for r in strategy.entry_rules) else ''}
{'✅ Dist/Accum Phase' if any('dist_accum' in r for r in strategy.entry_rules) else ''}
{'✅ Wyckoff Spring/UT' if any('spring_upthrust' in r for r in strategy.entry_rules) else ''}
{'✅ Sign of Strength' if any('sign_of_strength' in r for r in strategy.entry_rules) else ''}
{'✅ LPS/LPSY Retest' if any('last_point_support' in r for r in strategy.entry_rules) else ''}
{'✅ Effort vs Result' if any('effort_result' in r for r in strategy.entry_rules) else ''}
{'✅ Stopping Volume' if any('stopping_vol' in r for r in strategy.entry_rules) else ''}
{'✅ VSA Patterns (ND/NS)' if any('vsa_patterns' in r for r in strategy.entry_rules) else ''}
{'✅ Climax Reversals' if any('climax_logic' in r for r in strategy.entry_rules) else ''}

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
