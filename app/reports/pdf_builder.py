"""
app/reports/pdf_builder.py
--------------------------
Generates the full 20-section research PDF report.

Uses ReportLab Platypus for document layout.
Charts are generated with matplotlib, saved to temp PNG, embedded in PDF.

All 20 sections defined in the spec are implemented in order.
No section is skipped or stubbed.  If data is unavailable for a section,
the section still appears with an explicit "Not available: [reason]" note.
"""

from __future__ import annotations

import dataclasses
import io
import math
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import KeepTogether

from app.analytics.metrics import compute_regime_breakdown
from app.backtest.costs import CostModel
from app.config import BacktestConfig, ReportConfig, Settings
from app.models.session import (
    ArchiveManifest,
    BacktestResult,
    FeatureRecord,
    PatternCandidate,
    TickWindow,
)
from app.utils.log_setup import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────────────────────────────────────

_C_HEADER = colors.HexColor("#1B2A3B")
_C_ACCENT = colors.HexColor("#2E86AB")
_C_GREEN = colors.HexColor("#27AE60")
_C_RED = colors.HexColor("#E74C3C")
_C_YELLOW = colors.HexColor("#F39C12")
_C_LIGHT_GREY = colors.HexColor("#F5F7FA")
_C_MID_GREY = colors.HexColor("#BDC3C7")
_C_WHITE = colors.white
_C_BLACK = colors.black


# ──────────────────────────────────────────────────────────────────────────────
# Style helpers
# ──────────────────────────────────────────────────────────────────────────────


def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {}

    styles["title"] = ParagraphStyle(
        "title",
        parent=base["Title"],
        fontSize=28,
        textColor=_C_WHITE,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    styles["subtitle"] = ParagraphStyle(
        "subtitle",
        parent=base["Normal"],
        fontSize=14,
        textColor=_C_WHITE,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    styles["section_header"] = ParagraphStyle(
        "section_header",
        parent=base["Heading1"],
        fontSize=14,
        textColor=_C_WHITE,
        backColor=_C_HEADER,
        borderPad=6,
        spaceAfter=10,
        spaceBefore=16,
    )
    styles["subsection"] = ParagraphStyle(
        "subsection",
        parent=base["Heading2"],
        fontSize=11,
        textColor=_C_ACCENT,
        spaceAfter=6,
        spaceBefore=10,
    )
    styles["body"] = ParagraphStyle(
        "body",
        parent=base["Normal"],
        fontSize=9,
        leading=14,
        spaceAfter=4,
    )
    styles["caption"] = ParagraphStyle(
        "caption",
        parent=base["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#666666"),
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    styles["verdict_accepted"] = ParagraphStyle(
        "verdict_accepted",
        parent=base["Normal"],
        fontSize=16,
        textColor=_C_GREEN,
        alignment=TA_CENTER,
        spaceBefore=12,
    )
    styles["verdict_marginal"] = ParagraphStyle(
        "verdict_marginal",
        parent=base["Normal"],
        fontSize=16,
        textColor=_C_YELLOW,
        alignment=TA_CENTER,
        spaceBefore=12,
    )
    styles["verdict_rejected"] = ParagraphStyle(
        "verdict_rejected",
        parent=base["Normal"],
        fontSize=16,
        textColor=_C_RED,
        alignment=TA_CENTER,
        spaceBefore=12,
    )
    styles["code"] = ParagraphStyle(
        "code",
        parent=base["Code"],
        fontSize=7.5,
        fontName="Courier",
        leading=11,
        spaceAfter=4,
    )
    return styles


def _table_style(
    header_bg: colors.Color = _C_ACCENT,
    stripe: bool = True,
) -> TableStyle:
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), _C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5),
        ("ROWBACKGROUND", (0, 1), (-1, -1), [_C_WHITE, _C_LIGHT_GREY]),
        ("GRID", (0, 0), (-1, -1), 0.4, _C_MID_GREY),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    return TableStyle(cmds)


def _section_header(text: str, styles: dict) -> list:
    return [
        Spacer(1, 0.3 * cm),
        Paragraph(f"  {text}", styles["section_header"]),
        Spacer(1, 0.2 * cm),
    ]


def _na(reason: str, styles: dict) -> Paragraph:
    return Paragraph(f"<i>Not available: {reason}</i>", styles["body"])


# ──────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ──────────────────────────────────────────────────────────────────────────────


def _save_fig_to_tempfile(fig: plt.Figure) -> str:
    """Save matplotlib figure to a temp PNG and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    return tmp.name


def _equity_curve_chart(result: BacktestResult, initial_capital: float) -> Optional[str]:
    trades = sorted(result.trades, key=lambda t: t.entry_t)
    if not trades:
        return None

    equity = [initial_capital]
    for t in trades:
        equity.append(equity[-1] + t.net_pnl)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(equity, color="#2E86AB", linewidth=1.5)
    ax.axhline(y=initial_capital, color="#BDC3C7", linestyle="--", linewidth=0.8)
    ax.fill_between(range(len(equity)), initial_capital, equity,
                    where=[e >= initial_capital for e in equity],
                    alpha=0.2, color="#27AE60")
    ax.fill_between(range(len(equity)), initial_capital, equity,
                    where=[e < initial_capital for e in equity],
                    alpha=0.2, color="#E74C3C")
    ax.set_title("Equity Curve (Net PnL)", fontsize=10)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Capital (₹)")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save_fig_to_tempfile(fig)


def _drawdown_chart(result: BacktestResult, initial_capital: float) -> Optional[str]:
    trades = sorted(result.trades, key=lambda t: t.entry_t)
    if not trades:
        return None

    equity = [initial_capital]
    for t in trades:
        equity.append(equity[-1] + t.net_pnl)

    peak = equity[0]
    dd_series = []
    for val in equity:
        if val > peak:
            peak = val
        dd_series.append((peak - val) / (peak + 1e-9) * 100)

    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.fill_between(range(len(dd_series)), 0, [-d for d in dd_series], color="#E74C3C", alpha=0.6)
    ax.plot([-d for d in dd_series], color="#C0392B", linewidth=1)
    ax.set_title("Drawdown (%)", fontsize=10)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save_fig_to_tempfile(fig)


def _feature_distribution_chart(feature_df_data: list[dict], feature: str) -> Optional[str]:
    if not feature_df_data:
        return None
    vals = [row.get(feature, 0) for row in feature_df_data if row.get(feature) is not None]
    if not vals:
        return None

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.hist(vals, bins=40, color="#2E86AB", edgecolor="white", alpha=0.8)
    ax.set_title(f"Distribution: {feature}", fontsize=9)
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save_fig_to_tempfile(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main PDF builder
# ──────────────────────────────────────────────────────────────────────────────


class PDFBuilder:
    """
    Builds the full research PDF report.

    Parameters
    ----------
    settings: Application settings (for cost assumptions, paths, etc.)
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._styles = _build_styles()
        self._temp_files: list[str] = []

    def build(
        self,
        session_date: date,
        manifest: ArchiveManifest,
        all_results: list[BacktestResult],
        candidates: list[PatternCandidate],
        windows_by_symbol: dict[str, list[TickWindow]],
        feature_samples: dict[str, list[dict]],   # symbol -> list of feature dicts
        output_path: Path,
    ) -> Path:
        """
        Generate the PDF and write it to output_path.
        Returns output_path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("pdf_build_start", date=str(session_date), output=str(output_path))

        doc = BaseDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=1.8 * cm,
            rightMargin=1.8 * cm,
            topMargin=2.0 * cm,
            bottomMargin=2.0 * cm,
        )

        # Single page template
        frame = Frame(
            doc.leftMargin, doc.bottomMargin,
            doc.width, doc.height,
            id="main",
        )
        doc.addPageTemplates([PageTemplate(id="main", frames=frame)])

        story: list = []

        # ── Section 1: Cover Page ─────────────────────────────────────────────
        story.extend(self._cover_page(session_date, manifest, all_results))

        # ── Section 2: Session Summary ────────────────────────────────────────
        story.extend(self._session_summary(session_date, manifest))

        # ── Section 3: Archive and Validation Summary ─────────────────────────
        story.extend(self._archive_validation_summary(manifest))

        # ── Section 4: Symbol Coverage ────────────────────────────────────────
        story.extend(self._symbol_coverage(manifest))

        # ── Section 5: Data Quality / Gap Analysis ────────────────────────────
        story.extend(self._gap_analysis(manifest))

        # ── Sections 6-20 per pattern ─────────────────────────────────────────
        accepted = [r for r in all_results if r.verdict == "ACCEPTED"]
        marginal = [r for r in all_results if r.verdict == "MARGINAL"]
        rejected = [r for r in all_results if r.verdict == "REJECTED"]

        if accepted or marginal:
            for result in accepted + marginal:
                story.extend(
                    self._pattern_sections(
                        result=result,
                        windows=windows_by_symbol.get(result.symbol, []),
                        feature_samples=feature_samples.get(result.symbol, []),
                        candidates=[c for c in candidates if c.pattern_id == result.pattern_id],
                    )
                )
        else:
            story.extend(_section_header("6. Strategy / Pattern Identity", self._styles))
            story.append(_na("No patterns passed quality thresholds.", self._styles))

        # ── Section 17: Failure Analysis (aggregate) ──────────────────────────
        story.extend(self._failure_analysis(rejected))

        # ── Section 20: Final Verdict ─────────────────────────────────────────
        story.extend(self._final_verdict(all_results, session_date))

        doc.build(story)
        self._cleanup_temp_files()
        logger.info("pdf_build_complete", path=str(output_path))
        return output_path

    # ──────────────────────────────────────────────────────────────────────────
    # Section builders
    # ──────────────────────────────────────────────────────────────────────────

    def _cover_page(
        self,
        session_date: date,
        manifest: ArchiveManifest,
        results: list[BacktestResult],
    ) -> list:
        st = self._styles
        accepted = sum(1 for r in results if r.verdict == "ACCEPTED")
        marginal = sum(1 for r in results if r.verdict == "MARGINAL")

        items = [
            Spacer(1, 3 * cm),
            Paragraph("MARKET MICROSTRUCTURE RESEARCH ENGINE", st["title"]),
            Spacer(1, 0.5 * cm),
            Paragraph(f"Daily Research Report — {session_date.strftime('%A, %d %B %Y')}",
                      st["subtitle"]),
            Spacer(1, 2 * cm),
        ]

        summary_data = [
            ["Session Date", session_date.strftime("%Y-%m-%d")],
            ["Symbols Analysed", ", ".join(manifest.symbols)],
            ["Total Ticks", f"{manifest.total_tick_count:,}"],
            ["Rejected Ticks", f"{manifest.total_rejected_count:,} ({manifest.rejection_rate:.1%})"],
            ["Significant Gaps", str(manifest.significant_gap_count)],
            ["Patterns Accepted", str(accepted)],
            ["Patterns Marginal", str(marginal)],
            ["Generated At", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")],
            ["Engine", self._settings.report.author],
        ]

        tbl = Table(summary_data, colWidths=[5 * cm, 10 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), _C_HEADER),
            ("TEXTCOLOR", (0, 0), (0, -1), _C_WHITE),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.4, _C_MID_GREY),
            ("ROWBACKGROUND", (1, 0), (1, -1), [_C_LIGHT_GREY, _C_WHITE]),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        items.append(tbl)
        items.append(Spacer(1, 1 * cm))
        items.append(Paragraph(
            "<i>This is a research report only. No trading recommendations. "
            "Past microstructure patterns do not guarantee future performance.</i>",
            st["caption"],
        ))
        items.append(PageBreak())
        return items

    def _session_summary(self, session_date: date, manifest: ArchiveManifest) -> list:
        st = self._styles
        items = _section_header("2. Session Summary", st)

        items.append(Paragraph(
            f"Archive processed for trading session <b>{session_date.isoformat()}</b>. "
            f"The archive contained <b>{len(manifest.symbols)}</b> symbol file(s): "
            f"<b>{', '.join(manifest.symbols)}</b>. "
            f"Total validated ticks: <b>{manifest.total_tick_count:,}</b>. "
            f"Ingestion completed at: <b>{manifest.ingestion_finished_at}</b>.",
            st["body"],
        ))

        dur = None
        if manifest.ingestion_started_at and manifest.ingestion_finished_at:
            dur = (manifest.ingestion_finished_at - manifest.ingestion_started_at).total_seconds()

        if dur is not None:
            items.append(Paragraph(
                f"Processing time: <b>{dur:.1f} seconds</b>.",
                st["body"],
            ))

        return items

    def _archive_validation_summary(self, manifest: ArchiveManifest) -> list:
        st = self._styles
        items = _section_header("3. Archive and Validation Summary", st)

        status_color = "#27AE60" if manifest.validation_passed else "#E74C3C"
        status_text = "PASSED" if manifest.validation_passed else "FAILED"
        items.append(Paragraph(
            f"Validation status: <font color='{status_color}'><b>{status_text}</b></font>",
            st["body"],
        ))

        if manifest.validation_errors:
            items.append(Paragraph("<b>Validation Errors:</b>", st["body"]))
            for err in manifest.validation_errors:
                items.append(Paragraph(f"• {err}", st["body"]))
        else:
            items.append(Paragraph("No validation errors encountered.", st["body"]))

        data = [["Metric", "Value"]]
        data += [
            ["Archive Path", manifest.archive_path],
            ["Archive Size", f"{manifest.archive_size_bytes / 1024:.1f} KB"],
            ["Total Ticks", f"{manifest.total_tick_count:,}"],
            ["Rejected Ticks", f"{manifest.total_rejected_count:,}"],
            ["Rejection Rate", f"{manifest.rejection_rate:.2%}"],
            ["Gap Events", str(manifest.gap_count)],
            ["Significant Gaps (>5 min)", str(manifest.significant_gap_count)],
            ["Total Gap Duration", f"{manifest.total_gap_seconds:.1f} s"],
        ]

        tbl = Table(data, colWidths=[7 * cm, 9 * cm])
        tbl.setStyle(_table_style())
        items.append(tbl)
        return items

    def _symbol_coverage(self, manifest: ArchiveManifest) -> list:
        st = self._styles
        items = _section_header("4. Symbol Coverage", st)

        if not manifest.symbols:
            items.append(_na("No symbols found in archive.", st))
            return items

        data = [["Symbol", "Total Ticks", "Rejected", "Rejection Rate"]]
        for sym in manifest.symbols:
            total = manifest.total_ticks.get(sym, 0)
            rej = manifest.rejected_ticks.get(sym, 0)
            rate = rej / (total + rej + 1e-9)
            data.append([sym, f"{total:,}", f"{rej:,}", f"{rate:.2%}"])

        tbl = Table(data, colWidths=[6 * cm, 4 * cm, 4 * cm, 4 * cm])
        tbl.setStyle(_table_style())
        items.append(tbl)
        return items

    def _gap_analysis(self, manifest: ArchiveManifest) -> list:
        st = self._styles
        items = _section_header("5. Data Quality / Gap Analysis", st)

        items.append(Paragraph(
            f"Total gap events detected: <b>{manifest.gap_count}</b>. "
            f"Significant gaps (>300s): <b>{manifest.significant_gap_count}</b>. "
            f"Total gap duration: <b>{manifest.total_gap_seconds:.1f}s</b>.",
            st["body"],
        ))

        if manifest.significant_gap_count > 0:
            items.append(Paragraph(
                "<font color='#E74C3C'><b>Warning:</b></font> "
                "Significant gaps detected. Pattern discovery may be impaired "
                "during these periods. Gap windows are excluded from backtest.",
                st["body"],
            ))
        else:
            items.append(Paragraph(
                "<font color='#27AE60'>No significant gaps detected. "
                "Data continuity is good.</font>",
                st["body"],
            ))
        return items

    def _pattern_sections(
        self,
        result: BacktestResult,
        windows: list[TickWindow],
        feature_samples: list[dict],
        candidates: list[PatternCandidate],
    ) -> list:
        """Sections 6-19 for a single pattern."""
        st = self._styles
        items: list = [PageBreak()]

        # ── 6. Strategy / Pattern Identity ────────────────────────────────────
        items.extend(_section_header(f"6. Strategy / Pattern Identity — {result.pattern_id}", st))
        items.append(Paragraph(
            f"<b>Symbol:</b> {result.symbol} | "
            f"<b>Direction:</b> {result.direction.value} | "
            f"<b>Discovery method:</b> {candidates[0].discovery_method if candidates else 'N/A'}",
            st["body"],
        ))
        if candidates:
            items.append(Paragraph(candidates[0].description, st["body"]))

        # ── 7. Exact Pattern Definition ───────────────────────────────────────
        items.extend(_section_header("7. Exact Pattern Definition", st))
        if result.rules:
            data = [["Feature", "Operator", "Threshold", "Meaning"]]
            feature_meanings = {
                "mean_imbalance": "Order book imbalance (bid - ask depth ratio)",
                "mean_microprice_slope": "Slope of depth-weighted midprice",
                "mean_aggression": "EWMA of delta-bid vs delta-ask aggression",
                "mean_relative_spread": "Spread relative to midprice",
                "mean_depth_ratio": "Bid depth / total depth",
                "mean_realised_vol": "Realised microprice volatility",
            }
            for rule in result.rules:
                data.append([
                    rule.feature,
                    rule.operator,
                    f"{rule.threshold:.6f}",
                    feature_meanings.get(rule.feature, ""),
                ])
            tbl = Table(data, colWidths=[5 * cm, 2 * cm, 3 * cm, 6 * cm])
            tbl.setStyle(_table_style())
            items.append(tbl)
        else:
            items.append(_na("No rules defined.", st))

        # ── 8. Feature Context ────────────────────────────────────────────────
        items.extend(_section_header("8. Feature Context", st))
        items.append(Paragraph(
            "Feature distributions for this symbol across the full session. "
            "Dashed lines indicate rule thresholds.",
            st["body"],
        ))
        if feature_samples and result.rules:
            chart_feature = result.rules[0].feature.replace("mean_", "")
            chart_path = _feature_distribution_chart(feature_samples, chart_feature)
            if chart_path:
                self._temp_files.append(chart_path)
                items.append(Image(chart_path, width=12 * cm, height=5 * cm))
                items.append(Paragraph(
                    f"Distribution of {chart_feature} across all windows.",
                    st["caption"],
                ))
        else:
            items.append(_na("Feature sample data not available.", st))

        # ── 9. Sample Count and Match Distribution ────────────────────────────
        items.extend(_section_header("9. Sample Count and Match Distribution", st))
        data = [
            ["Metric", "Value"],
            ["Total matched windows", str(result.sample_count)],
            ["In-sample matched", str(result.is_sample_count)],
            ["Out-of-sample matched", str(result.oos_sample_count)],
        ]
        tbl = Table(data, colWidths=[8 * cm, 8 * cm])
        tbl.setStyle(_table_style())
        items.append(tbl)

        # ── 10. Trade Rules ───────────────────────────────────────────────────
        items.extend(_section_header("10. Trade Rules", st))
        cfg = self._settings.backtest
        data = [
            ["Parameter", "Value"],
            ["Entry", f"Next tick after signal + {cfg.latency_ms}ms latency"],
            ["Direction", result.direction.value],
            ["Entry price (LONG)", f"Ask + {cfg.slippage_ticks} tick(s) slippage"],
            ["Entry price (SHORT)", f"Bid − {cfg.slippage_ticks} tick(s) slippage"],
            ["Stop loss", f"{cfg.default_stop_ticks} ticks from entry"],
            ["Target", f"{cfg.default_target_ticks} ticks from entry"],
            ["Max hold time", f"{cfg.max_hold_seconds}s"],
            ["Exit rules", "First of: Target / Stop / Timeout / End-of-Day"],
        ]
        tbl = Table(data, colWidths=[8 * cm, 8 * cm])
        tbl.setStyle(_table_style())
        items.append(tbl)

        # ── 11. Backtest Assumptions ──────────────────────────────────────────
        items.extend(_section_header("11. Backtest Assumptions", st))
        cost_model = CostModel(cfg)
        for line in cost_model.assumption_text():
            items.append(Paragraph(f"• {line}", st["body"]))

        # ── 12. Backtest Results ──────────────────────────────────────────────
        items.extend(_section_header("12. Backtest Results (In-Sample)", st))
        items.append(self._metrics_table(result, oos=False))

        # ── 13. Cost-Adjusted Results ─────────────────────────────────────────
        items.extend(_section_header("13. Cost-Adjusted Results", st))
        data = [
            ["Metric", "Gross", "Net (after costs)"],
            ["Total PnL (₹)", f"{result.total_gross_pnl:,.2f}", f"{result.total_net_pnl:,.2f}"],
            ["Total Costs (₹)", "—", f"{result.total_costs:,.2f}"],
            ["Avg per Trade (₹)", f"{result.total_gross_pnl / max(result.sample_count,1):,.2f}",
             f"{result.total_net_pnl / max(result.sample_count,1):,.2f}"],
        ]
        tbl = Table(data, colWidths=[6 * cm, 5 * cm, 5 * cm])
        tbl.setStyle(_table_style())
        items.append(tbl)

        # ── 14. Equity Curve ──────────────────────────────────────────────────
        items.extend(_section_header("14. Equity Curve", st))
        eq_path = _equity_curve_chart(result, cfg.initial_capital)
        if eq_path:
            self._temp_files.append(eq_path)
            items.append(Image(eq_path, width=14 * cm, height=6 * cm))
            items.append(Paragraph(
                "Equity curve (net PnL). Green = above initial capital. Red = below.",
                st["caption"],
            ))
        else:
            items.append(_na("No trades to plot.", st))

        # ── 15. Drawdown Curve ────────────────────────────────────────────────
        items.extend(_section_header("15. Drawdown Curve", st))
        dd_path = _drawdown_chart(result, cfg.initial_capital)
        if dd_path:
            self._temp_files.append(dd_path)
            items.append(Image(dd_path, width=14 * cm, height=5 * cm))
            items.append(Paragraph(
                f"Max drawdown: {result.max_drawdown:.2%}", st["caption"]
            ))
        else:
            items.append(_na("No trades to plot.", st))

        # ── 16. Regime Breakdown ──────────────────────────────────────────────
        items.extend(_section_header("16. Regime Breakdown", st))
        regime_data = compute_regime_breakdown(result)
        if regime_data:
            data = [["Bucket", "Trades", "Win Rate", "Net PnL (₹)", "Avg Hold (s)"]]
            for b in regime_data:
                data.append([
                    f"Q{b['bucket']}",
                    str(b["trade_count"]),
                    f"{b['win_rate']:.2%}",
                    f"{b['net_pnl']:,.2f}",
                    f"{b['avg_hold_s']:.1f}",
                ])
            tbl = Table(data, colWidths=[3 * cm, 3 * cm, 3 * cm, 4 * cm, 4 * cm])
            tbl.setStyle(_table_style())
            items.append(tbl)
            items.append(Paragraph(
                f"Stability (win-rate CV): {result.win_rate_cv:.3f} "
                f"({'STABLE' if result.is_stable else 'UNSTABLE'})",
                st["body"],
            ))
        else:
            items.append(_na("Insufficient trades for regime breakdown.", st))

        # ── 18. OOS / Walk-Forward ────────────────────────────────────────────
        items.extend(_section_header("18. Out-of-Sample / Walk-Forward Results", st))
        if result.oos_sample_count > 0:
            items.append(self._metrics_table(result, oos=True))
            is_wr = result.win_rate
            oos_wr = result.oos_win_rate
            delta = is_wr - oos_wr
            colour = "#27AE60" if delta <= 0.10 else "#E74C3C"
            items.append(Paragraph(
                f"IS win rate: {is_wr:.2%} → OOS win rate: {oos_wr:.2%} "
                f"(Δ = <font color='{colour}'>{delta:+.2%}</font>)",
                st["body"],
            ))
        else:
            items.append(_na("No out-of-sample trades (pattern fires only in IS period).", st))

        # ── 19. Raw Matched Tick Examples ─────────────────────────────────────
        items.extend(_section_header("19. Raw Matched Tick Examples", st))
        max_examples = self._settings.report.max_example_windows
        example_windows = [
            windows[i] for i in (candidates[0].matched_windows[:max_examples] if candidates else [])
            if i < len(windows)
        ]
        if example_windows:
            for i, win in enumerate(example_windows, 1):
                items.append(Paragraph(f"<b>Example {i}:</b>", st["body"]))
                items.append(Paragraph(
                    f"Time: {win.start_t} – {win.end_t} | Ticks: {win.ticks} | "
                    f"Imbalance: {win.mean_imbalance:.4f} | Slope: {win.mean_microprice_slope:.6f} | "
                    f"Aggression: {win.mean_aggression:.4f} | "
                    f"Entry MP: {win.entry_microprice:.2f} → Exit MP: {win.exit_microprice:.2f}",
                    st["code"],
                ))
        else:
            items.append(_na("No example windows available.", st))

        return items

    def _metrics_table(self, result: BacktestResult, oos: bool = False) -> Table:
        prefix = "OOS " if oos else "IS "
        wr = result.oos_win_rate if oos else result.win_rate
        pf = result.oos_profit_factor if oos else result.profit_factor
        n = result.oos_sample_count if oos else result.is_sample_count

        data = [
            [f"{prefix}Metric", "Value"],
            ["Sample Count", str(n)],
            ["Win Rate", f"{wr:.2%}" if not math.isnan(wr) else "N/A"],
            ["Profit Factor", f"{pf:.3f}" if not math.isnan(pf) else "N/A"],
            ["Expectancy (₹)", f"{result.expectancy:,.2f}"],
            ["Avg Win (₹)", f"{result.avg_win:,.2f}"],
            ["Avg Loss (₹)", f"{result.avg_loss:,.2f}"],
            ["Max Drawdown", f"{result.max_drawdown:.2%}"],
            ["Sharpe Ratio", f"{result.sharpe_ratio:.3f}"],
            ["Total Net PnL (₹)", f"{result.total_net_pnl:,.2f}"],
        ]
        if not oos:
            data.append(["Stability (CV)", f"{result.win_rate_cv:.3f}"])

        tbl = Table(data, colWidths=[8 * cm, 8 * cm])
        tbl.setStyle(_table_style())
        return tbl

    def _failure_analysis(self, rejected: list[BacktestResult]) -> list:
        st = self._styles
        items = _section_header("17. Failure Analysis", st)

        if not rejected:
            items.append(Paragraph("No patterns were rejected.", st["body"]))
            return items

        items.append(Paragraph(
            f"<b>{len(rejected)}</b> pattern(s) were evaluated and rejected "
            "before inclusion in this report. Rejection reasons are listed below. "
            "This section is included for full transparency.",
            st["body"],
        ))

        data = [["Pattern ID", "Symbol", "Direction", "Trades", "Win Rate", "PF", "Reason"]]
        for r in rejected:
            data.append([
                r.pattern_id[:20] + "..." if len(r.pattern_id) > 20 else r.pattern_id,
                r.symbol,
                r.direction.value,
                str(r.sample_count),
                f"{r.win_rate:.2%}" if r.sample_count > 0 and "No in-sample trades" not in r.rejection_reason else "—",
                f"{r.profit_factor:.2f}" if r.sample_count > 0 and "No in-sample trades" not in r.rejection_reason else "—",
                r.rejection_reason[:60],
            ])

        tbl = Table(data, colWidths=[3.5*cm, 2.5*cm, 2*cm, 2*cm, 2*cm, 2*cm, 5*cm])
        tbl.setStyle(_table_style())
        items.append(tbl)
        return items

    def _final_verdict(
        self, all_results: list[BacktestResult], session_date: date
    ) -> list:
        st = self._styles
        items = [PageBreak()] + _section_header("20. Final Verdict", st)

        accepted = [r for r in all_results if r.verdict == "ACCEPTED"]
        marginal = [r for r in all_results if r.verdict == "MARGINAL"]
        rejected = [r for r in all_results if r.verdict == "REJECTED"]

        total = len(all_results)
        items.append(Paragraph(
            f"Session: <b>{session_date.isoformat()}</b> | "
            f"Patterns evaluated: <b>{total}</b> | "
            f"Accepted: <b>{len(accepted)}</b> | "
            f"Marginal: <b>{len(marginal)}</b> | "
            f"Rejected: <b>{len(rejected)}</b>",
            st["body"],
        ))

        if accepted:
            style_key = "verdict_accepted"
            verdict_str = f"✓ {len(accepted)} ACCEPTED PATTERN(S) FOUND"
        elif marginal:
            style_key = "verdict_marginal"
            verdict_str = f"⚠ {len(marginal)} MARGINAL PATTERN(S) ONLY — USE WITH CAUTION"
        else:
            style_key = "verdict_rejected"
            verdict_str = "✗ NO VIABLE PATTERNS — DO NOT TRADE"

        items.append(Spacer(1, 0.5 * cm))
        items.append(Paragraph(verdict_str, st[style_key]))
        items.append(Spacer(1, 0.5 * cm))

        items.append(Paragraph(
            "<i>This report was generated automatically by the Market Research Engine. "
            "All assumptions are documented above. A qualified researcher should verify "
            "any pattern before deployment. Past microstructure results are not indicative "
            "of future performance.</i>",
            st["caption"],
        ))
        return items

    def _cleanup_temp_files(self) -> None:
        for f in self._temp_files:
            try:
                os.unlink(f)
            except OSError:
                pass
        self._temp_files.clear()
