
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
