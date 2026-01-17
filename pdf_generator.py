
from fpdf import FPDF
from models import Strategy, BacktestResult
import os
from datetime import datetime

class StrategyReportPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'Institutional Strategy Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_strategy_pdf(strategy: Strategy, result: BacktestResult, filename: str):
    """Generates a PDF report for a single strategy."""
    
    pdf = StrategyReportPDF()
    pdf.add_page()
    
    # Title Section
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, f"Strategy: {strategy.name}", 0, 1, 'L')
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, f"Market: {strategy.market} | Timeframe: {strategy.timeframe}", 0, 1, 'L')
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, 'L')
    pdf.line(10, 45, 200, 45)
    pdf.ln(10)
    
    # Performance Snapshot
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, "Performance Snapshot (365 Days)", 0, 1, 'L')
    pdf.set_font('Courier', '', 11)
    
    res_365 = result.period_results.get(365)
    if res_365:
        pdf.cell(0, 8, f"Total PnL:       {res_365.total_pnl:,.2f}", 0, 1)
        pdf.cell(0, 8, f"Win Rate:        {res_365.win_rate*100:.1f}%", 0, 1)
        pdf.cell(0, 8, f"Profit Factor:   {res_365.profit_factor:.2f}", 0, 1)
        pdf.cell(0, 8, f"Max Drawdown:    {res_365.max_drawdown*100:.1f}%", 0, 1)
        pdf.cell(0, 8, f"Total Trades:    {res_365.total_trades}", 0, 1)
    else:
        pdf.cell(0, 8, "No 365-day data available.", 0, 1)
    
    pdf.ln(5)
    
    # Strategy Rules
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, " algorithmic Logic (Rules)", 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    pdf.multi_cell(0, 5, "Entry Conditions:")
    pdf.set_font('Courier', '', 9)
    for rule in strategy.entry_rules:
        pdf.cell(10) # Indent
        pdf.cell(0, 5, f"- {rule}", 0, 1)
        
    pdf.ln(2)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, "Exit Conditions:")
    pdf.set_font('Courier', '', 9)
    for rule in strategy.exit_rules:
        pdf.cell(10) # Indent
        pdf.cell(0, 5, f"- {rule}", 0, 1)
        
    pdf.ln(5)
    
    # detailed Explanations (SMC)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "Institutional Concepts Used", 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    explanations = []
    if any("order_block" in r for r in strategy.entry_rules):
        explanations.append("ORDER BLOCK: Institutional footprint of buying/selling. Detected by strong impulse candles engulfing previous structure.")
    if any("liquidity_sweep" in r for r in strategy.entry_rules):
        explanations.append("LIQUIDITY SWEEP: Stop hunt below a key swing low/high followed by a reclaim. Targeted to trap retail traders.")
    if any("imbalance" in r for r in strategy.entry_rules):
        explanations.append("FAIR VALUE GAP (FVG): Price inefficiency caused by aggressive buying/selling. The algorithm targets these gaps for fills.")
    if any("market_structure_shift" in r for r in strategy.entry_rules):
        explanations.append("MARKET STRUCTURE SHIFT (MSS): Confirmed trend reversal where price breaks a key Swing High/Low responsible for the recent trend.")
    if any("breaker" in r for r in strategy.entry_rules):
        explanations.append("BREAKER BLOCK: A failed Order Block that is reclaimed. Old Support becomes New Resistance (or vice-versa). High probability setup.")
    if any("is_in_discount" in r for r in strategy.entry_rules):
        explanations.append("DISCOUNT ZONE FILTRATION: Algo ONLY buys when price is in the lower 50% of the recent range. 'Buy Cheap'.")
    if any("is_in_premium" in r for r in strategy.entry_rules):
        explanations.append("PREMIUM ZONE FILTRATION: Algo ONLY sells when price is in the upper 50% of the recent range. 'Sell Expensive'.")
    if any("optimal_trade_entry" in r for r in strategy.entry_rules):
        explanations.append("OPTIMAL TRADE ENTRY (OTE): Precision entry at the 61.8-78.6% Fibonacci retracement level of a major impulse leg.")
    if any("inducement" in r for r in strategy.entry_rules):
        explanations.append("INDUCEMENT (IDM): A trap set for early traders. A minor swing point is 'swept' to grab liquidity before the real move.")
    if any("kill_zone" in r for r in strategy.entry_rules):
        explanations.append("KILL ZONE (TIME): Trade execution is restricted to high-volume market hours (e.g. Open/Close) to avoid chop.")
    if any("choch" in r for r in strategy.entry_rules):
        explanations.append("CHANGE OF CHARACTER (CHoCH): Aggressive reversal signal. Detecting the *first* break of minor structure.")
    if any("mitigation" in r for r in strategy.entry_rules):
        explanations.append("MITIGATION BLOCK: Trading off a failed Support/Resistance level that was re-tested without a prior liquidity sweep.")
    if any("amd" in r for r in strategy.entry_rules):
        explanations.append("POWER OF 3 (AMD): Accumulation-Manipulation-Distribution. Catching the 'Judas Swing' where price fakes a move before the real trend.")
    if any("ifvg" in r for r in strategy.entry_rules):
        explanations.append("INVERTED FVG (Flip): A Fair Value Gap that was broken and is now acting as Support/Resistance in the opposite direction.")
    if any("turtle" in r for r in strategy.entry_rules):
        explanations.append("TURTLE SOUP: A 'Major Stop Hunt'. Price breaks a 20-period Low/High but fails to close outside, signaling a major reversal.")
    if any("rejection" in r for r in strategy.entry_rules):
        explanations.append("REJECTION BLOCK (WICK): Entry based on a long wick (>50% of candle). Trading the rejection of a price level.")
    if any("propulsion" in r for r in strategy.entry_rules):
        explanations.append("PROPULSION BLOCK: A secondary Order Block formed inside a larger one. Signs of momentum reloading.")
    if any("void" in r for r in strategy.entry_rules):
        explanations.append("LIQUIDITY VOID: A sharp, aggressive price move that creates a vacuum. Logic targets the fill/reversal of this void.")
    if any("bpr" in r for r in strategy.entry_rules):
        explanations.append("BALANCED PRICE RANGE (BPR): Overlap of Bullish/Bearish FVGs. A perfect state of balance often retested.")
    if any("volume_imbalance" in r for r in strategy.entry_rules):
        explanations.append("VOLUME IMBALANCE: Trading the gap between Candle Close and next Candle Open.")
    if any("mean_threshold" in r for r in strategy.entry_rules):
        explanations.append("MEAN THRESHOLD: Precision retest of the exact 50% level of a significant Order Block.")
    if any("unicorn" in r for r in strategy.entry_rules):
        explanations.append("UNICORN MODEL: High-confidence setup combining a Breaker Block with a Fair Value Gap in the same zone.")
    if any("silver_bullet" in r for r in strategy.entry_rules):
        explanations.append("SILVER BULLET: Time-based setup targeting Fair Value Gaps specifically during the 10-11 AM or 3-4 AM windows.")
    if any("gap_reclaim" in r for r in strategy.entry_rules):
        explanations.append("OPENING GAP RECLAIM: Price drops below the Daily/Weekly opening price and strongly reclaims it.")
    if any("reclaimed_block" in r for r in strategy.entry_rules):
        explanations.append("RECLAIMED BLOCK: An old Resistance/Support level that was broken but is now being respected again.")
    if any("wick_ce" in r for r in strategy.entry_rules):
        explanations.append("WICK C.E.: Consequent Encroachment. Precision entry at the 50% level of a long wick.")
    if any("order_flow" in r for r in strategy.entry_rules):
        explanations.append("ORDER FLOW CHAIN: Following the institutional flow by identifying respect of prior Order Block levels (Higher Lows).")
    if any("sponsor" in r for r in strategy.entry_rules):
        explanations.append("SPONSORSHIP CANDLE: Trading the specific candle that initiated a displacement/expansion move.")
    if any("range_rotation" in r for r in strategy.entry_rules):
        explanations.append("RANGE ROTATION: Mean reversion strategy trading from the deviations (High/Low) back to the center.")
    if any("snap_back" in r for r in strategy.entry_rules):
        explanations.append("SNAP BACK (Rubber Band): Aggressive mean reversion when price is statistically over-extended (>3 Sigma/ATR) from Mean.")
    if any("quarterly_shift" in r for r in strategy.entry_rules):
        explanations.append("QUARTERLY SHIFT: Time-based structure shift concept adapted for intraday cycle rotations.")
    if any("std_dev" in r for r in strategy.entry_rules):
        explanations.append("STD DEV PROJECTION: Trading the edges of statistical extremity. Targets 2.5-4.0 Standard Deviation extensions (Judas Swing).")
    if any("po3_swing" in r for r in strategy.entry_rules):
        explanations.append("POWER OF 3 SWING: Long-term Accumulation-Manipulation-Distribution. Catching the end of the 'Manipulation' phase.")
    if any("inst_swing" in r for r in strategy.entry_rules):
        explanations.append("INSTITUTIONAL SWING POINT: Confirmed Fractal High/Low (3-bar pattern) indicating a structural pivot.")
    if any("smt_div" in r for r in strategy.entry_rules):
        explanations.append("SMT DIVERGENCE: 'Smart Money Tool'. Divergence between Price and Momentum/Correlated Asset (Simulated) indicating manipulation.")
    if any("macro" in r for r in strategy.entry_rules):
        explanations.append("MACRO CYCLES: Algorithmic time windows (e.g. 10:50-11:10) where volatility injection is programmed.")
    if any("liq_run" in r for r in strategy.entry_rules):
        explanations.append("LIQUIDITY RUN: Detecting a sequential cascade of stops being triggered (3+ levels). Fading the run.")
    if any("stops_hunt" in r for r in strategy.entry_rules):
        explanations.append("STOPS HUNT (Classic): Single-bar purge of a near-term pivot followed by immediate reclaim.")
    if any("eq_reclaim" in r for r in strategy.entry_rules):
        explanations.append("EQUILIBRIUM RECLAIM: Reclaiming the 50% level of the current Day's Range (Discount <-> Premium transition).")
    if any("partial_void" in r for r in strategy.entry_rules):
        explanations.append("PARTIAL VOID FILL: Aggressive entry at 50% fill of a Liquidity Void (expecting immediate reaction).")
    if any("fractal_exp" in r for r in strategy.entry_rules):
        explanations.append("FRACTAL EXPANSION: Breakout strategy from a defined fractal consolidation range.")
    if any("ib_breakout" in r for r in strategy.entry_rules):
        explanations.append("INITIAL BALANCE BREAKOUT: Trading the break of the High/Low established in the first 60 minutes.")
    if any("orb" in r for r in strategy.entry_rules):
        explanations.append("OPENING RANGE BREAKOUT (ORB): Momentum entry on the break of the first 15/30 minute range.")
    if any("daily_open_rej" in r for r in strategy.entry_rules):
        explanations.append("DAILY OPEN REJECTION: Price returns to the exact Daily Open level and rejects it (Support/Resistance).")
    if any("pwh_pwl" in r for r in strategy.entry_rules):
        explanations.append("PWH/PWL SWEEP: Taking out the Previous Week's High or Low (Liquidity Grab) and reversing.")
    if any("pdh_pdl" in r for r in strategy.entry_rules):
        explanations.append("PDH/PDL SWEEP: Taking out the Previous Day's High or Low (Classic Stop Hunt).")
    if any("sfp" in r for r in strategy.entry_rules):
        explanations.append("SWING FAILURE PATTERN (SFP): RSI/Price divergence at a key swing high/low.")
    if any("impulse" in r for r in strategy.entry_rules):
        explanations.append("MOMENTUM IMPULSE: Validating entry with a candle body > 2.0x ATR.")
    if any("psych_level" in r for r in strategy.entry_rules):
        explanations.append("PSYCH LEVEL REJECTION: Rejection of key '00' or '50' institutional round numbers.")
    if any("tl_liquidity" in r for r in strategy.entry_rules):
        explanations.append("TRENDLINE LIQUIDITY: Identifying smooth trendlines as retail bait, targeting the liquidity pool below/above.")
    if any("failed_auction" in r for r in strategy.entry_rules):
        explanations.append("FAILED AUCTION: Price breaks a range but fails to hold, returning inside (Market Profile).")
    if any("amd" in r for r in strategy.entry_rules):
        explanations.append("AMD SETUP: Accumulation-Manipulation-Distribution. Classic intraday structure trade.")
    if any("turtle_soup" in r for r in strategy.entry_rules):
        explanations.append("TURTLE SOUP: 20-period swing low/high sweep followed by a reclaim (Street Smarts).")
    if any("decoupled_ob" in r for r in strategy.entry_rules):
        explanations.append("DECOUPLED OB: An Order Block that was wicked through but the body held, showing resilience.")
    if any("propulsion_candle" in r for r in strategy.entry_rules):
        explanations.append("PROPULSION CANDLE: Aggressive continuation candle opening inside the upper 50% of previous body.")
    if any("eng_liquidity" in r for r in strategy.entry_rules):
        explanations.append("ENGINEERED LIQUIDITY SWEEP: Taking out the stops below a clean Double Bottom/Top.")
    if any("ifvg" in r for r in strategy.entry_rules):
        explanations.append("INVERSE FVG (IFVG): A broken Fair Value Gap that is retested as support/resistance.")
    if any("mitigation_block" in r for r in strategy.entry_rules):
        explanations.append("MITIGATION BLOCK: A failed swing structure (Breaker without liquidity grab) that gets retested.")
    if any("rejection_block" in r for r in strategy.entry_rules):
        explanations.append("REJECTION BLOCK: Trading the long wick of a high-impact candle.")
    if any("nwog" in r for r in strategy.entry_rules):
        explanations.append("NWOG RE-TEST: Price interacts with a significant New Week Opening Gap.")
    if any("vol_void" in r for r in strategy.entry_rules):
        explanations.append("VOLUME VOID: Re-entering a zone of low volume (high velocity price movement).")
    if any("dragon" in r for r in strategy.entry_rules):
        explanations.append("DRAGON PATTERN: Aggressive W-Bottom with a Trendline break and reclaim.")
    if any("quasimodo" in r for r in strategy.entry_rules):
        explanations.append("QUASIMODO (QM): Over/Under pattern involved a liquidity grab followed by Market Structure Shift.")
    if any("triple_tap" in r for r in strategy.entry_rules):
        explanations.append("TRIPLE TAP (3-Drive): Three distinct pushes into a level, signalling exhaustion.")
    if any("compression" in r for r in strategy.entry_rules):
        explanations.append("COMPRESSION (CP): Price advancing while consuming all orders, leaving a clean path for reversal.")
    if any("master_pattern" in r for r in strategy.entry_rules):
        explanations.append("MASTER PATTERN: The universal Contraction-Expansion-Trend cycle detection.")
    if any("inducement" in r for r in strategy.entry_rules):
        explanations.append("INDUCEMENT (IDM): A short-term internal trap high/low that lures early traders before the real move.")
    if any("bpr" in r for r in strategy.entry_rules):
        explanations.append("BALANCED PRICE RANGE (BPR): A double rebalance event where a FVG is immediately reversed.")
    if any("vol_imbalance" in r for r in strategy.entry_rules):
        explanations.append("VOLUME IMBALANCE (VI): A gap between candle BODIES where wicks overlap.")
    if any("judas" in r for r in strategy.entry_rules):
        explanations.append("JUDAS SWING: An aggressive session-open fakeout designed to trap breakout traders.")
    if any("3c_rev" in r for r in strategy.entry_rules):
        explanations.append("3-CANDLE FORMATION: Specific Impulse -> Pause -> Reversal sequence.")
    if any("ote" in r for r in strategy.entry_rules):
        explanations.append("OPTIMAL TRADE ENTRY (OTE): Fibonacci retracement into the deep 62%-79% discount/premium zone.")
    if any("dr_range" in r for r in strategy.entry_rules):
        explanations.append("DEFINING RANGE (DR): Trading the breakout of mechanical levels established in the first hour of trade.")
    if any("cpr" in r for r in strategy.entry_rules):
        explanations.append("CENTRAL PIVOT RANGE (CPR): Using the relationship between Pivot, BC, and TC to find key support/resistance.")
    if any("value_area" in r for r in strategy.entry_rules):
        explanations.append("VALUE AREA (VA): Trading institutional value where 70% of the day's volume is concentrated.")
    if any("poc" in r for r in strategy.entry_rules):
        explanations.append("POINT OF CONTROL (POC): Targeted rejection of the price level with the highest volume node.")
    if any("poor_high_low" in r for r in strategy.entry_rules):
        explanations.append("POOR HIGH/LOW: Identifying unfinished auctions where price lacked aggressive rejection.")
    if any("single_prints" in r for r in strategy.entry_rules):
        explanations.append("SINGLE PRINTS: High-velocity zones where price skipped, creating a profile imbalance.")
    if any("tails" in r for r in strategy.entry_rules):
        explanations.append("BUYING/SELLING TAIL: Strong institutional rejection at the extremes of the range.")
    if any("composite_op" in r for r in strategy.entry_rules):
        explanations.append("COMPOSITE OPERATOR: Spotting footprints of professional accumulation or distribution.")
    if any("dist_accum" in r for r in strategy.entry_rules):
        explanations.append("DIST/ACCUM PHASE: Identifying market cycle transitions using Wyckoff logic.")
    if any("spring_upthrust" in r for r in strategy.entry_rules):
        explanations.append("WYCKOFF SPRING/UPTHRUST: A terminal shakeout fakeout of a trading range before the real move.")
    if any("sign_of_strength" in r for r in strategy.entry_rules):
        explanations.append("SIGN OF STRENGTH (SOS): Aggressive price expansion confirming institutional direction.")
    if any("last_point_support" in r for r in strategy.entry_rules):
        explanations.append("LAST POINT OF SUPPORT (LPS): A structural retest pullback during a markup/markdown phase.")
    if any("effort_result" in r for r in strategy.entry_rules):
        explanations.append("EFFORT VS RESULT: Identifying divergence between price range (Effort) and final progress (Result).")
    if any("stopping_vol" in r for r in strategy.entry_rules):
        explanations.append("STOPPING VOLUME: Climax action where aggressive moves are absorbed by institutional orders.")
    if any("vsa_patterns" in r for r in strategy.entry_rules):
        explanations.append("VSA PATTERNS: Volume Spread Analysis signals like No Demand, No Supply, and Shakeouts.")
    if any("climax_logic" in r for r in strategy.entry_rules):
        explanations.append("CLIMAX LOGIC: Identifying extreme exhaustion points (Selling/Buying Climax) where reversals occur.")
    
    if not explanations:
        explanations.append("Standard Price Action / Indicator logic.")
        
    for expl in explanations:
        pdf.multi_cell(0, 6, f"* {expl}")
        pdf.ln(2)

    try:
        pdf.output(filename)
        return True
    except Exception as e:
        print(f"PDF Generation Error: {e}")
        return False
