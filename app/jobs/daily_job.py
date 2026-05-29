"""
app/jobs/daily_job.py
---------------------
Daily pipeline orchestrator.

Wires every module together in the correct sequence:
  Telegram listen → Download → Ingest → Validate → Features
  → Windows → Pattern discovery → Labeling → Backtest
  → Analytics → PDF report → Telegram send

Design principles:
- One job processes one archive (one trading day).
- All stages are individually logged with timing.
- Failures at any stage are caught, logged, and reported via Telegram.
- The job is idempotent: re-running with the same archive is safe.
- No global state — everything is passed via function arguments.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import time
import traceback
from pathlib import Path
from typing import Optional

from app.analytics.metrics import compute_regime_breakdown, populate_metrics
from app.backtest.costs import CostModel
from app.backtest.engine import BacktestEngine, _get_direction_from_window
from app.config import Settings, load_settings
from app.features.pipeline import FeaturePipeline
from app.ingestion.archive_reader import (
    ArchiveReadError,
    iter_archive_lines,
    validate_archive_integrity,
)
from app.ingestion.validator import RecordValidator
from app.models.session import (
    ArchiveManifest,
    BacktestResult,
    FeatureRecord,
    PatternCandidate,
    PatternDirection,
    TickWindow,
)
from app.models.system_event import SystemEvent
from app.models.tick import TickRecord
from app.patterns.rule_miner import ClusterMiner, RuleMiner
from app.reports.pdf_builder import PDFBuilder
from app.storage.tick_store import TickStore
from app.telegram_io.listener import TelegramListener
from app.telegram_io.sender import TelegramSender
from app.utils.log_setup import configure_logging, get_logger
from app.windows.tick_window import fixed_tick_windows

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Ingest and validate archive
# ──────────────────────────────────────────────────────────────────────────────


def _stage_ingest(
    archive_path: Path,
    settings: Settings,
    store: TickStore,
    session_date: datetime.date,
) -> ArchiveManifest:
    """
    Stream the archive, validate every record, write ticks to Parquet.
    Returns a populated ArchiveManifest.
    """
    logger.info("stage_ingest_start", archive=str(archive_path))
    t0 = time.monotonic()

    # Quick structural check first
    ok, msg = validate_archive_integrity(archive_path)
    manifest = ArchiveManifest(
        session_date=session_date,
        archive_path=str(archive_path),
        archive_size_bytes=archive_path.stat().st_size,
        ingestion_started_at=datetime.datetime.utcnow(),
    )
    if not ok:
        manifest.validation_passed = False
        manifest.validation_errors.append(f"Structural check failed: {msg}")
        logger.error("archive_structural_check_failed", reason=msg)
        return manifest

    validator = RecordValidator(strict_sequence=False)
    ingestion_cfg = settings.ingestion

    # Per-symbol buffers: we write in chunks to avoid large in-memory lists
    symbol_buffers: dict[str, list[TickRecord]] = {}
    system_events: list[SystemEvent] = []

    def _flush_symbol(symbol: str) -> None:
        buf = symbol_buffers.get(symbol, [])
        if not buf:
            return
        store.write_ticks(session_date, symbol, iter(buf))
        symbol_buffers[symbol] = []

    line_count = 0
    for raw_line in iter_archive_lines(archive_path):
        line_count += 1
        record = validator.parse(raw_line)

        if record is None:
            continue

        if isinstance(record, SystemEvent):
            system_events.append(record)
            continue

        if isinstance(record, TickRecord):
            sym = record.s
            if sym not in symbol_buffers:
                symbol_buffers[sym] = []
            symbol_buffers[sym].append(record)

            # Flush to disk in chunks
            if len(symbol_buffers[sym]) >= ingestion_cfg.chunk_size_ticks:
                _flush_symbol(sym)

    # Final flush
    for sym in list(symbol_buffers.keys()):
        _flush_symbol(sym)

    # Build manifest from validator stats
    for stats in validator.all_stats():
        if stats.total_seen == 0:
            continue
        sym = stats.symbol
        manifest.symbols.append(sym)
        manifest.total_ticks[sym] = stats.accepted
        manifest.rejected_ticks[sym] = stats.total_rejected

        if stats.rejected_model_validation > 0 or stats.rejected_missing_fields > 0:
            manifest.validation_errors.append(
                f"{sym}: {stats.total_rejected} ticks rejected "
                f"(model_err={stats.rejected_model_validation}, "
                f"missing={stats.rejected_missing_fields})"
            )

    # Process system events
    sig_gaps = [e for e in system_events if e.is_significant_gap]
    manifest.has_system_file = validator.system_stats().total_seen > 0
    manifest.gap_count = len([e for e in system_events if e.event.value == "GAP"])
    manifest.significant_gap_count = len(sig_gaps)
    manifest.total_gap_seconds = sum(
        e.duration_seconds or 0.0 for e in system_events if e.duration_seconds
    )

    # Reject symbols below minimum tick threshold
    for sym in list(manifest.symbols):
        if manifest.total_ticks.get(sym, 0) < ingestion_cfg.min_ticks_per_symbol:
            manifest.validation_errors.append(
                f"{sym}: only {manifest.total_ticks[sym]} ticks — below minimum "
                f"{ingestion_cfg.min_ticks_per_symbol}, symbol excluded."
            )
            manifest.symbols.remove(sym)

    manifest.validation_passed = len(manifest.validation_errors) == 0
    manifest.ingestion_finished_at = datetime.datetime.utcnow()

    elapsed = time.monotonic() - t0
    logger.info(
        "stage_ingest_complete",
        symbols=manifest.symbols,
        total_ticks=manifest.total_tick_count,
        rejected=manifest.total_rejected_count,
        gaps=manifest.gap_count,
        elapsed_s=f"{elapsed:.1f}",
    )
    return manifest


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Feature computation
# ──────────────────────────────────────────────────────────────────────────────


def _stage_features(
    session_date: datetime.date,
    symbol: str,
    settings: Settings,
    store: TickStore,
) -> list[FeatureRecord]:
    """Load ticks for a symbol and compute microstructure features."""
    logger.info("stage_features_start", symbol=symbol)
    t0 = time.monotonic()

    ticks_df = store.load_ticks(session_date, symbol)
    if ticks_df.empty:
        logger.warning("no_ticks_for_features", symbol=symbol)
        return []

    pipeline = FeaturePipeline(settings.features, symbol)
    features: list[FeatureRecord] = []

    for row in ticks_df.itertuples(index=False):
        try:
            tick = TickRecord(**row._asdict())
            fr = pipeline.process(tick)
            features.append(fr)
        except Exception as exc:
            logger.debug("feature_tick_skipped", symbol=symbol, error=str(exc))
            continue

    store.write_features(session_date, symbol, features)
    logger.info(
        "stage_features_complete",
        symbol=symbol,
        count=len(features),
        elapsed_s=f"{time.monotonic() - t0:.1f}",
    )
    return features


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: Windowing
# ──────────────────────────────────────────────────────────────────────────────


def _stage_windows(
    features: list[FeatureRecord],
    settings: Settings,
    symbol: str,
) -> list[TickWindow]:
    """Build fixed-tick windows from the feature stream."""
    if not features:
        return []

    windows_cfg = settings.windows
    all_windows: list[TickWindow] = []

    # Use the primary window size (first in config list)
    window_size = windows_cfg.tick_sizes[0] if windows_cfg.tick_sizes else 50
    step_size = window_size  # non-overlapping by default

    logger.info("stage_windows_start", symbol=symbol, window_size=window_size)

    for win in fixed_tick_windows(
        feature_iter=iter(features),
        window_size=window_size,
        step_size=step_size,
        symbol=symbol,
        min_ticks=windows_cfg.min_window_ticks,
    ):
        all_windows.append(win)

    logger.info("stage_windows_complete", symbol=symbol, count=len(all_windows))
    return all_windows


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4: Pattern discovery
# ──────────────────────────────────────────────────────────────────────────────


def _stage_patterns(
    windows: list[TickWindow],
    settings: Settings,
    symbol: str,
) -> list[PatternCandidate]:
    """Run rule mining and clustering to discover pattern candidates."""
    if not windows:
        return []

    patterns_cfg = settings.patterns
    oos_start_t = _compute_oos_start(windows, patterns_cfg.oos_split_fraction)

    logger.info(
        "stage_patterns_start",
        symbol=symbol,
        windows=len(windows),
        oos_fraction=patterns_cfg.oos_split_fraction,
    )

    candidates: list[PatternCandidate] = []

    # Rule mining
    rule_miner = RuleMiner(patterns_cfg, symbol)
    rule_candidates = rule_miner.mine(windows, oos_start_t)
    candidates.extend(rule_candidates)

    # Cluster mining
    cluster_miner = ClusterMiner(patterns_cfg, symbol)
    cluster_candidates = cluster_miner.mine(windows, oos_start_t)
    candidates.extend(cluster_candidates)

    logger.info(
        "stage_patterns_complete",
        symbol=symbol,
        total_candidates=len(candidates),
        rule_mining=len(rule_candidates),
        clustering=len(cluster_candidates),
    )
    return candidates, oos_start_t


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5: Backtest
# ──────────────────────────────────────────────────────────────────────────────


def _stage_backtest(
    candidates: list[PatternCandidate],
    windows: list[TickWindow],
    features: list[FeatureRecord],
    oos_start_t: int,
    settings: Settings,
    store: TickStore,
    session_date: datetime.date,
) -> list[BacktestResult]:
    """Backtest all candidates and compute analytics metrics."""
    import dataclasses
    import pandas as pd

    if not candidates or not features:
        return []

    bt_cfg = settings.backtest
    patterns_cfg = settings.patterns

    # Build features DataFrame once (shared across all pattern backtests)
    features_data = [dataclasses.asdict(f) for f in features]
    features_df = pd.DataFrame(features_data)

    engine = BacktestEngine(bt_cfg, CostModel(bt_cfg))
    results: list[BacktestResult] = []

    logger.info("stage_backtest_start", candidates=len(candidates))

    for candidate in candidates:
        try:
            result = engine.run(candidate, windows, features_df, oos_start_t)
            result = populate_metrics(result, patterns_cfg, bt_cfg)

            # Set pattern_id on each trade (engine doesn't have access to candidate)
            for trade in result.trades:
                trade.pattern_id = candidate.pattern_id

            results.append(result)
            store.save_backtest_result(session_date, result)
        except Exception as exc:
            logger.error(
                "backtest_failed_for_candidate",
                pattern_id=candidate.pattern_id,
                error=str(exc),
            )
            # Append a failed result so it appears in rejection analysis
            failed = BacktestResult(
                pattern_id=candidate.pattern_id,
                symbol=candidate.symbol,
                direction=candidate.direction,
                rules=candidate.rules,
                verdict="REJECTED",
                rejection_reason=f"Backtest error: {exc}",
            )
            results.append(failed)

    accepted = sum(1 for r in results if r.verdict == "ACCEPTED")
    marginal = sum(1 for r in results if r.verdict == "MARGINAL")
    rejected_count = sum(1 for r in results if r.verdict == "REJECTED")

    logger.info(
        "stage_backtest_complete",
        total=len(results),
        accepted=accepted,
        marginal=marginal,
        rejected=rejected_count,
    )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Stage 6: Report
# ──────────────────────────────────────────────────────────────────────────────


def _stage_report(
    session_date: datetime.date,
    manifest: ArchiveManifest,
    all_results: list[BacktestResult],
    all_candidates: list[PatternCandidate],
    windows_by_symbol: dict[str, list[TickWindow]],
    features_by_symbol: dict[str, list[FeatureRecord]],
    settings: Settings,
) -> Path:
    """Generate the PDF report and the JSON summary."""
    logger.info("stage_report_start", date=str(session_date))

    pdf_dir = settings.reports_path
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"{session_date.isoformat()}_research_report.pdf"

    # Build feature samples dict (convert FeatureRecord list → list of dicts)
    import dataclasses
    feature_samples: dict[str, list[dict]] = {
        sym: [dataclasses.asdict(f) for f in flist[:500]]  # sample for charts
        for sym, flist in features_by_symbol.items()
    }

    builder = PDFBuilder(settings)
    builder.build(
        session_date=session_date,
        manifest=manifest,
        all_results=all_results,
        candidates=all_candidates,
        windows_by_symbol=windows_by_symbol,
        feature_samples=feature_samples,
        output_path=pdf_path,
    )

    # JSON summary (lightweight machine-readable artifact)
    json_path = pdf_dir / f"{session_date.isoformat()}_summary.json"
    _write_json_summary(session_date, manifest, all_results, json_path)

    logger.info("stage_report_complete", pdf=str(pdf_path))
    return pdf_path


def _write_json_summary(
    session_date: datetime.date,
    manifest: ArchiveManifest,
    results: list[BacktestResult],
    path: Path,
) -> None:
    summary = {
        "session_date": session_date.isoformat(),
        "symbols": manifest.symbols,
        "total_ticks": manifest.total_tick_count,
        "validation_passed": manifest.validation_passed,
        "patterns": [
            {
                "pattern_id": r.pattern_id,
                "symbol": r.symbol,
                "direction": r.direction.value,
                "verdict": r.verdict,
                "win_rate": r.win_rate,
                "profit_factor": r.profit_factor,
                "sample_count": r.sample_count,
                "net_pnl": r.total_net_pnl,
                "rejection_reason": r.rejection_reason,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(summary, indent=2, default=str))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _compute_oos_start(
    windows: list[TickWindow], oos_fraction: float
) -> int:
    """Return the start timestamp of the OOS period."""
    if not windows:
        return 0
    split_idx = int(len(windows) * (1 - oos_fraction))
    split_idx = max(0, min(split_idx, len(windows) - 1))
    return windows[split_idx].start_t


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────


def run_daily_job(
    settings: Settings,
    archive_path: Optional[Path] = None,
    session_date: Optional[datetime.date] = None,
) -> bool:
    """
    Run the full daily pipeline for one archive.

    Parameters
    ----------
    settings:      Application settings.
    archive_path:  Path to the .tar.gz archive. If None, polls Telegram first.
    session_date:  Override session date. If None, inferred from archive filename.

    Returns True on full success, False if any stage failed.
    """
    t_start = time.monotonic()
    sender = TelegramSender(settings)

    # ── Determine archive ─────────────────────────────────────────────────────
    if archive_path is None:
        listener = TelegramListener(settings)
        archive_path = listener.poll_for_archive_sync(settings.archives_path)
        if archive_path is None:
            logger.info("no_archive_available_today")
            return False

    if session_date is None:
        # Extract date from filename: YYYY-MM-DD.tar.gz
        try:
            date_str = archive_path.stem.replace(".tar", "")
            session_date = datetime.date.fromisoformat(date_str)
        except ValueError:
            session_date = datetime.date.today()
            logger.warning(
                "could_not_parse_date_from_filename",
                filename=archive_path.name,
                using=str(session_date),
            )

    logger.info(
        "daily_job_start",
        session_date=str(session_date),
        archive=str(archive_path),
    )

    sender.send_status_alert_sync(
        f"📥 *Research Engine started processing archive:*\n`{archive_path.name}`"
    )

    # ── Storage setup ─────────────────────────────────────────────────────────
    features_dir = settings.ticks_path / "features"
    store = TickStore(
        db_path=settings.db_path,
        ticks_dir=settings.ticks_path,
        features_dir=features_dir,
        mongodb_uri=settings.mongodb_uri,
    )
    store.connect()

    try:
        # ── Stage 1: Ingest ───────────────────────────────────────────────────
        manifest = _stage_ingest(archive_path, settings, store, session_date)
        store.save_manifest(manifest)

        if not manifest.symbols:
            raise RuntimeError(
                "No valid symbols after ingestion. "
                f"Errors: {manifest.validation_errors}"
            )

        # ── Per-symbol stages ─────────────────────────────────────────────────
        windows_by_symbol: dict[str, list[TickWindow]] = {}
        features_by_symbol: dict[str, list[FeatureRecord]] = {}
        all_candidates: list[PatternCandidate] = []
        oos_by_symbol: dict[str, int] = {}

        for symbol in manifest.symbols:
            # Stage 2: Features
            features = _stage_features(session_date, symbol, settings, store)
            if not features:
                logger.warning("skipping_symbol_no_features", symbol=symbol)
                continue
            features_by_symbol[symbol] = features

            # Stage 3: Windows
            windows = _stage_windows(features, settings, symbol)
            if not windows:
                logger.warning("skipping_symbol_no_windows", symbol=symbol)
                continue
            windows_by_symbol[symbol] = windows

            # Stage 4: Pattern discovery
            candidates, oos_start_t = _stage_patterns(windows, settings, symbol)
            all_candidates.extend(candidates)
            oos_by_symbol[symbol] = oos_start_t

        store.save_pattern_candidates(session_date, all_candidates)

        # ── Stage 5: Backtest (across all symbols, unified) ───────────────────
        all_results: list[BacktestResult] = []
        for symbol in manifest.symbols:
            sym_candidates = [c for c in all_candidates if c.symbol == symbol]
            if not sym_candidates:
                continue
            sym_features = features_by_symbol.get(symbol, [])
            sym_windows = windows_by_symbol.get(symbol, [])
            oos_t = oos_by_symbol.get(symbol, 0)

            results = _stage_backtest(
                sym_candidates,
                sym_windows,
                sym_features,
                oos_t,
                settings,
                store,
                session_date,
            )
            all_results.extend(results)

        # ── Stage 6: Report ───────────────────────────────────────────────────
        pdf_path = _stage_report(
            session_date,
            manifest,
            all_results,
            all_candidates,
            windows_by_symbol,
            features_by_symbol,
            settings,
        )

        # ── Stage 7: Telegram send ────────────────────────────────────────────
        json_path = (
            settings.reports_path
            / f"{session_date.isoformat()}_summary.json"
        )
        send_ok = sender.send_report_sync(
            pdf_path=pdf_path,
            results=all_results,
            session_date_str=session_date.isoformat(),
            archive_filename=archive_path.name,
            json_summary_path=json_path if json_path.exists() else None,
        )

        elapsed = time.monotonic() - t_start
        logger.info(
            "daily_job_complete",
            session_date=str(session_date),
            elapsed_s=f"{elapsed:.1f}",
            telegram_sent=send_ok,
        )
        return True

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(
            "daily_job_failed",
            session_date=str(session_date),
            error=str(exc),
            traceback=tb,
        )
        asyncio.run(
            sender.send_error_alert(
                session_date_str=str(session_date),
                error=f"{exc}\n\n{tb[:400]}",
            )
        )
        return False
    finally:
        store.close()


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for `python -m app.jobs.daily_job`."""
    import argparse

    parser = argparse.ArgumentParser(description="Market Research Engine — Daily Job")
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Path to .tar.gz archive to process (skips Telegram poll)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Override session date YYYY-MM-DD",
    )
    args = parser.parse_args()

    settings = load_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    settings.ensure_directories()

    session_date = None
    if args.date:
        session_date = datetime.date.fromisoformat(args.date)

    success = run_daily_job(
        settings=settings,
        archive_path=args.archive,
        session_date=session_date,
    )
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
