"""
tests/conftest.py
-----------------
Shared pytest fixtures for the test suite.
"""

from __future__ import annotations

import datetime
import gzip
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from app.config import (
    BacktestConfig,
    FeaturesConfig,
    PatternsConfig,
    Settings,
    StorageConfig,
    WindowsConfig,
)
from app.models.session import FeatureRecord, PatternDirection, PatternRule, TickWindow
from app.models.tick import TickRecord


# ──────────────────────────────────────────────────────────────────────────────
# Tick factories
# ──────────────────────────────────────────────────────────────────────────────


def make_tick(
    t: int = 1_700_000_000_000,
    seq: int = 1,
    symbol: str = "NIFTY26JUNFUT",
    bid: float = 22000.0,
    ask: float = 22000.5,
    bq: float = 100.0,
    aq: float = 80.0,
    spread: float | None = None,
    imbalance: float = 0.11,
    **kwargs,
) -> TickRecord:
    """Create a valid TickRecord with sensible defaults."""
    if spread is None:
        spread = round(ask - bid, 6)
    data = dict(
        t=t,
        seq=seq,
        s=symbol,
        bid=bid,
        ask=ask,
        bq=bq,
        aq=aq,
        spread=spread,
        imbalance=imbalance,
        bp1=bid,
        bq1=bq,
        ap1=ask,
        aq1=aq,
        bp2=bid - 0.5,
        bq2=50.0,
        ap2=ask + 0.5,
        aq2=50.0,
        db=5.0,
        da=3.0,
    )
    data.update(kwargs)
    return TickRecord(**data)


def make_tick_sequence(
    n: int,
    symbol: str = "NIFTY26JUNFUT",
    start_t: int = 1_700_000_000_000,
    tick_interval_ms: int = 200,
    base_bid: float = 22000.0,
    bid_drift: float = 0.0,
) -> list[TickRecord]:
    """Create a list of n sequential TickRecords."""
    ticks = []
    for i in range(n):
        bid = base_bid + bid_drift * i
        ask = bid + 0.5
        spread = ask - bid
        bq = 100.0 + (i % 20) * 2
        aq = 100.0 - (i % 15) * 2
        aq = max(aq, 10.0)
        total = bq + aq
        imbalance = (bq - aq) / total
        ticks.append(make_tick(
            t=start_t + i * tick_interval_ms,
            seq=i + 1,
            symbol=symbol,
            bid=round(bid, 2),
            ask=round(ask, 2),
            bq=bq,
            aq=aq,
            spread=round(spread, 2),
            imbalance=round(imbalance, 6),
        ))
    return ticks


def make_feature_record(
    t: int = 1_700_000_000_000,
    seq: int = 1,
    symbol: str = "NIFTY26JUNFUT",
    imbalance: float = 0.20,
    microprice_slope: float = 0.001,
    aggression_score: float = 0.15,
    relative_spread: float = 0.000023,
    depth_ratio: float = 0.55,
    realised_vol: float = 0.05,
    microprice: float = 22000.25,
) -> FeatureRecord:
    return FeatureRecord(
        t=t,
        seq=seq,
        symbol=symbol,
        bid=22000.0,
        ask=22000.5,
        midprice=22000.25,
        spread=0.5,
        bq=100.0,
        aq=80.0,
        imbalance=imbalance,
        microprice=microprice,
        microprice_slope=microprice_slope,
        relative_spread=relative_spread,
        total_bid_depth=200.0,
        total_ask_depth=160.0,
        depth_ratio=depth_ratio,
        aggression_score=aggression_score,
        realised_vol=realised_vol,
        liquidity_thin=0.0,
        momentum=microprice_slope ** 2,
    )


def make_feature_sequence(
    n: int,
    symbol: str = "NIFTY26JUNFUT",
    start_t: int = 1_700_000_000_000,
    slope_trend: float = 0.0001,
) -> list[FeatureRecord]:
    return [
        make_feature_record(
            t=start_t + i * 200,
            seq=i + 1,
            symbol=symbol,
            microprice=22000.0 + i * 0.01,
            microprice_slope=slope_trend + i * 0.00001,
            imbalance=0.1 + (i % 10) * 0.02,
        )
        for i in range(n)
    ]


def make_tick_window(
    symbol: str = "NIFTY26JUNFUT",
    start_idx: int = 0,
    n_ticks: int = 50,
    start_t: int = 1_700_000_000_000,
    mean_imbalance: float = 0.30,
    mean_slope: float = 0.001,
) -> TickWindow:
    features = make_feature_sequence(n_ticks, symbol, start_t)
    return TickWindow(
        symbol=symbol,
        start_idx=start_idx,
        end_idx=start_idx + n_ticks,
        start_t=start_t,
        end_t=start_t + n_ticks * 200,
        ticks=n_ticks,
        features=features,
        mean_imbalance=mean_imbalance,
        mean_microprice_slope=mean_slope,
        mean_aggression=0.10,
        mean_relative_spread=0.000023,
        mean_depth_ratio=0.55,
        mean_realised_vol=0.05,
        entry_microprice=22000.0,
        exit_microprice=22000.5,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Archive factory
# ──────────────────────────────────────────────────────────────────────────────


def make_ndjson_gz_content(records: list[dict]) -> bytes:
    """Gzip-compress a list of dicts as NDJSON."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for record in records:
            gz.write((json.dumps(record) + "\n").encode())
    return buf.getvalue()


def make_test_archive(
    tmp_path: Path,
    ticks_per_symbol: int = 200,
    symbols: list[str] = None,
    include_system: bool = True,
) -> Path:
    """Create a real .tar.gz archive with valid NDJSON tick data for testing."""
    symbols = symbols or ["NIFTY26JUNFUT"]
    date_str = "2024-06-10"
    archive_path = tmp_path / f"{date_str}.tar.gz"

    tick_records = []
    for i in range(ticks_per_symbol):
        bid = 22000.0 + i * 0.01
        ask = bid + 0.5
        bq = 100.0
        aq = 80.0
        total = bq + aq
        imbalance = (bq - aq) / total
        tick_records.append({
            "t": 1_700_000_000_000 + i * 200,
            "seq": i + 1,
            "s": symbols[0],
            "bid": round(bid, 2),
            "ask": round(ask, 2),
            "bq": bq,
            "aq": aq,
            "spread": round(ask - bid, 2),
            "imbalance": round(imbalance, 6),
            "bp1": round(bid, 2),
            "bq1": bq,
            "ap1": round(ask, 2),
            "aq1": aq,
            "bp2": round(bid - 0.5, 2),
            "bq2": 50.0,
            "ap2": round(ask + 0.5, 2),
            "aq2": 50.0,
            "db": 5.0,
            "da": 3.0,
        })

    system_records = [
        {"event": "GAP", "t": 1_700_000_050_000, "duration": 5000},
        {"event": "SESSION_START", "t": 1_700_000_000_000},
    ]

    with tarfile.open(archive_path, "w:gz") as tar:
        for sym in symbols:
            sym_data = make_ndjson_gz_content(tick_records)
            info = tarfile.TarInfo(name=f"{sym}.ndjson.gz")
            info.size = len(sym_data)
            tar.addfile(info, io.BytesIO(sym_data))

        if include_system:
            sys_data = make_ndjson_gz_content(system_records)
            info = tarfile.TarInfo(name="SYSTEM.ndjson.gz")
            info.size = len(sys_data)
            tar.addfile(info, io.BytesIO(sys_data))

    return archive_path


# ──────────────────────────────────────────────────────────────────────────────
# Settings fixture
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Settings wired to a temp directory — safe for parallel tests."""
    settings = Settings(
        telegram_bot_token="test_token",
        telegram_channel_id=-1001234567890,
        data_root=str(tmp_path),
    )
    settings = settings.model_copy(update={
        "backtest": BacktestConfig(
            tick_size=0.05,
            lot_size=25,
            brokerage_per_lot=20.0,
            slippage_ticks=1,
            latency_ms=50,
            default_stop_ticks=5,
            default_target_ticks=10,
            max_hold_seconds=120,
            initial_capital=1_000_000.0,
        ),
        "patterns": PatternsConfig(
            min_samples=5,        # low for tests
            min_win_rate=0.40,    # relaxed for tests
            min_profit_factor=1.0,
            clustering_n_clusters=3,
            oos_split_fraction=0.30,
            stability_cv_threshold=0.50,
        ),
        "windows": WindowsConfig(
            tick_sizes=[20],
            time_sizes_seconds=[30],
            min_window_ticks=5,
        ),
    })
    settings.ensure_directories()
    return settings


@pytest.fixture
def sample_archive(tmp_path: Path) -> Path:
    return make_test_archive(tmp_path, ticks_per_symbol=300)


@pytest.fixture
def tick_sequence() -> list[TickRecord]:
    return make_tick_sequence(200)


@pytest.fixture
def feature_sequence() -> list[FeatureRecord]:
    return make_feature_sequence(200)
