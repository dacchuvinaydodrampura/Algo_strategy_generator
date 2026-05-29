# Market Microstructure Research Engine — Server 2

A production-grade Python research pipeline that receives daily market-data archives
from a Telegram channel, reconstructs tick data, discovers repeating microstructure
patterns, backtests them with realistic costs, and generates a transparent 20-section
PDF research report.

> **This is a research engine only.** No orders are placed. No live trading occurs.

---

## Architecture

```
Telegram Channel
    │
    ▼  (listener.py polls every 10s)
TelegramListener
    │  .tar.gz downloaded
    ▼
ArchiveReader          ← stream-extracts tar.gz, never loads full archive to RAM
    │  RawLine (symbol, filename, line_no, raw JSON)
    ▼
RecordValidator        ← strict schema validation, sequence/timestamp checks
    │  TickRecord | SystemEvent
    ▼
TickStore (DuckDB)     ← raw ticks → Parquet files per (date, symbol)
    │
    ▼
FeaturePipeline        ← stateful rolling-window microstructure features
    │  FeatureRecord (10 features per tick)
    ▼
WindowBuilder          ← fixed-tick or fixed-time rolling windows
    │  TickWindow (summary stats per window)
    ▼
RuleMiner + ClusterMiner  ← threshold rule scanning + k-means clustering
    │  PatternCandidate list
    ▼
OutcomeLabeler         ← forward-return labels, time-based IS/OOS split
    │
    ▼
BacktestEngine         ← realistic simulation: slippage, brokerage, latency
    │  BacktestResult (trade list + raw PnL)
    ▼
Analytics (metrics.py) ← WR, PF, Sharpe, drawdown, stability CV, verdict
    │
    ▼
PDFBuilder             ← 20-section ReportLab PDF with embedded matplotlib charts
    │
    ▼
TelegramSender         ← summary message + PDF attachment → channel
```

---

## Project Structure

```
market_research/
├── app/
│   ├── main.py                   # CLI entry point (run-once | watch)
│   ├── config.py                 # Pydantic settings, YAML + env merge
│   ├── telegram_io/
│   │   ├── listener.py           # Poll Telegram, download archive
│   │   └── sender.py             # Send PDF + summary back
│   ├── ingestion/
│   │   ├── archive_reader.py     # Streaming .tar.gz + .ndjson.gz reader
│   │   └── validator.py          # Schema validation, stats tracking
│   ├── models/
│   │   ├── tick.py               # TickRecord Pydantic model (all fields)
│   │   ├── system_event.py       # SystemEvent model (GAP, RECONNECT, etc.)
│   │   └── session.py            # ArchiveManifest, FeatureRecord, TickWindow,
│   │                             #   PatternCandidate, BacktestResult, TradeResult
│   ├── storage/
│   │   └── tick_store.py         # DuckDB + Parquet persistence layer
│   ├── features/
│   │   └── pipeline.py           # 10-feature microstructure pipeline
│   ├── windows/
│   │   └── tick_window.py        # Fixed-tick and fixed-time window builders
│   ├── patterns/
│   │   └── rule_miner.py         # Threshold rule mining + k-means clustering
│   ├── labeling/
│   │   └── outcome_labeler.py    # Forward-return labels, IS/OOS split
│   ├── backtest/
│   │   ├── engine.py             # Realistic trade simulation
│   │   └── costs.py              # Brokerage + slippage cost model
│   ├── analytics/
│   │   └── metrics.py            # All metrics, regime breakdown, verdict
│   ├── reports/
│   │   └── pdf_builder.py        # 20-section PDF (ReportLab + matplotlib)
│   └── jobs/
│       └── daily_job.py          # Pipeline orchestrator
├── tests/
│   ├── conftest.py               # Shared fixtures, factories
│   ├── test_ingestion.py         # Archive reader + validator tests
│   ├── test_features.py          # Feature pipeline + windowing tests
│   ├── test_backtest.py          # Cost model + backtest engine + metrics
│   └── test_patterns.py          # Rule miner + end-to-end tests
├── config/
│   ├── settings.yaml             # All non-secret configuration
│   └── .env.example              # Secret / override template
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp config/.env.example .env
# Edit .env and fill in TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID
```

### 3. Configure settings (optional)

Edit `config/settings.yaml` to adjust:
- Cost assumptions (brokerage, slippage, tick size, lot size)
- Pattern quality thresholds (min win rate, min profit factor)
- Window sizes
- Logging format

---

## Running

### Process a specific archive (skip Telegram)

```bash
python -m app.main run-once --archive /path/to/2024-06-10.tar.gz --date 2024-06-10
```

### Poll Telegram and process if a new archive is found

```bash
python -m app.main run-once
```

### Run as a daily daemon (polls continuously)

```bash
python -m app.main watch
```

The daemon polls every `telegram.poll_interval_seconds` seconds.
It will not re-process an archive it already handled today.

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

With coverage:

```bash
pytest tests/ --cov=app --cov-report=term-missing
```

---

## PDF Report Sections

| # | Section |
|---|---------|
| 1 | Cover Page |
| 2 | Session Summary |
| 3 | Archive and Validation Summary |
| 4 | Symbol Coverage |
| 5 | Data Quality / Gap Analysis |
| 6 | Strategy / Pattern Identity |
| 7 | Exact Pattern Definition |
| 8 | Feature Context |
| 9 | Sample Count and Match Distribution |
| 10 | Trade Rules |
| 11 | Backtest Assumptions |
| 12 | Backtest Results |
| 13 | Cost-Adjusted Results |
| 14 | Equity Curve |
| 15 | Drawdown Curve |
| 16 | Regime Breakdown |
| 17 | Failure Analysis |
| 18 | Out-of-Sample / Walk-Forward Results |
| 19 | Raw Matched Tick Examples |
| 20 | Final Verdict |

---

## Microstructure Features

All features are computed by `FeaturePipeline` (rolling, stateful, per-symbol):

| Feature | Description |
|---------|-------------|
| `imbalance` | (bq − aq) / (bq + aq) from tick, validated ∈ [−1, 1] |
| `microprice` | Depth-weighted midprice using best 2 bid/ask levels |
| `microprice_slope` | OLS slope of microprice over last 10 ticks |
| `relative_spread` | spread / midprice |
| `total_bid_depth` | Sum of bq1..bq5 |
| `total_ask_depth` | Sum of aq1..aq5 |
| `depth_ratio` | total_bid / (total_bid + total_ask) |
| `aggression_score` | EWMA of (db − da) / (\|db\| + \|da\| + ε) |
| `realised_vol` | Std of microprice changes over last 20 ticks |
| `liquidity_thin` | 1 if total depth < session 25th percentile |

---

## Backtest Cost Model

All costs are transparent and printed in every PDF:

| Component | Default | Configurable |
|-----------|---------|-------------|
| Brokerage | ₹20 per lot (round trip) | `brokerage_per_lot` |
| Slippage | 1 tick × ₹0.05 × lot\_size | `slippage_ticks`, `tick_size` |
| Latency | 50ms (entry tick skipped) | `latency_ms` |
| Exit fills | At exact target/stop price | (conservative assumption) |

---

## Data Contract

The archive format is fixed and strictly validated:

```
YYYY-MM-DD.tar.gz
├── SYMBOL.ndjson.gz        # One file per symbol
│   └── {t, seq, s, bid, ask, bq, aq, spread, imbalance,
│         bp1..bp5, bq1..bq5, ap1..ap5, aq1..aq5, db, da}
└── SYSTEM.ndjson.gz        # Feed health events
    └── {event, t, duration, t_start, t_end, ...}
```

No fields are invented beyond those in the contract.
All fields are validated; malformed records are rejected and counted.

---

## Engineering Decisions

1. **DuckDB + Parquet** over SQLite: Columnar storage is 10-100x faster for
   time-series feature queries. DuckDB's native Parquet scanning avoids full loads.

2. **Streaming ingestion**: `tarfile.open()` + `gzip.GzipFile` yield one line at a time.
   Peak memory for ingestion is ~2 × chunk_size_ticks × ~200 bytes ≈ 2 MB at defaults.

3. **Time-based IS/OOS split** (last 30%): Random splits would leak future data.
   Walk-forward integrity is preserved by never looking at OOS data during mining.

4. **Strict rejection > silent degradation**: Any record that fails validation is
   counted and logged. The PDF reports rejection rates explicitly.

5. **No overfitting safeguards**: Patterns requiring K>2 features are not mined
   on a single day of data (insufficient samples for reliable estimation).

6. **Verdict system**: ACCEPTED / MARGINAL / REJECTED with explicit reasons.
   MARGINAL patterns are reported but flagged; REJECTED patterns appear only in
   the Failure Analysis section.
