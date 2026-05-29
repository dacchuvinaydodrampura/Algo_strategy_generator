"""
app/storage/tick_store.py
--------------------------
DuckDB-backed or MongoDB-backed persistent storage for tick data, session manifests,
pattern candidates, and backtest results.

Design:
- Automatically detects MONGODB_URI. If present, connects to MongoDB Atlas.
- Otherwise, falls back to DuckDB + Parquet local file storage.
- Ticks and features in MongoDB are stored compactly as compressed Parquet binary blobs
  to minimize storage overhead and fit easily within the 512 MB free tier.
- Automatically monitors MongoDB database space and purges the oldest session data
  recursively if database utilization exceeds 90% of the 512 MB limit.
"""

from __future__ import annotations

import datetime
import io
import json
from pathlib import Path
from typing import Iterator, Optional

import duckdb
import pandas as pd
import pymongo
from bson.binary import Binary

from app.models.session import (
    ArchiveManifest,
    BacktestResult,
    FeatureRecord,
    PatternCandidate,
)
from app.models.tick import TickRecord
from app.utils.log_setup import get_logger

logger = get_logger(__name__)

_TICK_BATCH_SIZE = 2000
_FEATURE_BATCH_SIZE = 2000


class TickStore:
    """
    Persistent storage backed by MongoDB or DuckDB + Parquet.

    Maintains identical public APIs for both storage backends so the rest
    of the pipeline can execute transparently without knowing where data is saved.
    """

    def __init__(
        self,
        db_path: Path,
        ticks_dir: Path,
        features_dir: Path,
        mongodb_uri: str = "",
    ) -> None:
        self._db_path = db_path
        self._ticks_dir = ticks_dir
        self._features_dir = features_dir
        self._mongodb_uri = mongodb_uri
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._mongo_client: Optional[pymongo.MongoClient] = None
        self._mongo_db: Optional[pymongo.database.Database] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the database connection (MongoDB or DuckDB)."""
        if self._mongodb_uri:
            self._mongo_client = pymongo.MongoClient(self._mongodb_uri)
            try:
                # Use default database configured in URI or fallback to market_research
                self._mongo_db = self._mongo_client.get_default_database()
                if self._mongo_db is None or self._mongo_db.name == "test":
                    self._mongo_db = self._mongo_client["market_research"]
            except Exception:
                self._mongo_db = self._mongo_client["market_research"]
            
            self._create_mongo_indexes()
            logger.info("mongodb_connected", db=self._mongo_db.name)
            self.check_and_purge_storage()
        else:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(str(self._db_path))
            self._create_tables()
            logger.info("tickstore_connected", db=str(self._db_path))

    def close(self) -> None:
        if self._mongo_client:
            self._mongo_client.close()
            self._mongo_client = None
            self._mongo_db = None
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError("TickStore not connected — call connect() first")
        return self._conn

    def _create_tables(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS archive_manifests (
                session_date    DATE PRIMARY KEY,
                archive_path    TEXT NOT NULL,
                archive_size_bytes BIGINT,
                symbols         TEXT,       -- JSON array
                has_system_file BOOLEAN,
                total_ticks     TEXT,       -- JSON object
                rejected_ticks  TEXT,       -- JSON object
                gap_count       INTEGER,
                significant_gap_count INTEGER,
                total_gap_seconds DOUBLE,
                validation_passed BOOLEAN,
                validation_errors TEXT,     -- JSON array
                ingested_at     TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pattern_candidates (
                pattern_id      TEXT PRIMARY KEY,
                session_date    DATE,
                symbol          TEXT,
                direction       TEXT,
                discovery_method TEXT,
                rules_json      TEXT,
                sample_count    INTEGER,
                description     TEXT,
                created_at      TIMESTAMP DEFAULT current_timestamp
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                pattern_id      TEXT PRIMARY KEY,
                session_date    DATE,
                symbol          TEXT,
                direction       TEXT,
                sample_count    INTEGER,
                is_sample_count INTEGER,
                oos_sample_count INTEGER,
                win_rate        DOUBLE,
                oos_win_rate    DOUBLE,
                profit_factor   DOUBLE,
                oos_profit_factor DOUBLE,
                expectancy      DOUBLE,
                avg_win         DOUBLE,
                avg_loss        DOUBLE,
                max_drawdown    DOUBLE,
                sharpe_ratio    DOUBLE,
                total_net_pnl   DOUBLE,
                total_costs     DOUBLE,
                win_rate_cv     DOUBLE,
                is_stable       BOOLEAN,
                verdict         TEXT,
                rejection_reason TEXT,
                rules_json      TEXT,
                trades_json     TEXT,
                created_at      TIMESTAMP DEFAULT current_timestamp
            )
        """)

    def _create_mongo_indexes(self) -> None:
        """Create indexes in MongoDB collections to speed up queries."""
        if self._mongo_db is None:
            return
        self._mongo_db["archive_manifests"].create_index("session_date", unique=True)
        self._mongo_db["parquet_blobs"].create_index(
            [("session_date", 1), ("symbol", 1), ("data_type", 1)],
            unique=True
        )
        self._mongo_db["pattern_candidates"].create_index("pattern_id", unique=True)
        self._mongo_db["pattern_candidates"].create_index("session_date")
        self._mongo_db["backtest_results"].create_index("pattern_id", unique=True)
        self._mongo_db["backtest_results"].create_index("session_date")

    def check_and_purge_storage(self) -> None:
        """
        Monitor MongoDB storage usage. If it exceeds 90% of the 512 MB Atlas free limit,
        recursively delete the oldest session's data.
        """
        if self._mongo_db is None:
            return

        # 512 MB MongoDB Atlas free tier storage limit
        limit_bytes = 512 * 1024 * 1024
        threshold_bytes = int(0.90 * limit_bytes)  # 460.8 MB

        while True:
            try:
                stats = self._mongo_db.command("dbStats")
                # MongoDB Atlas M0 uses storageSize or dataSize to measure database usage
                current_bytes = stats.get("storageSize", stats.get("dataSize", 0))
            except Exception as e:
                logger.error("mongodb_dbstats_failed", error=str(e))
                break

            logger.info("mongodb_storage_check", current_bytes=current_bytes, limit_bytes=limit_bytes, threshold_bytes=threshold_bytes)
            if current_bytes < threshold_bytes:
                break

            # Find the oldest session date across all active collections
            oldest_dates = []
            for coll_name in ["archive_manifests", "parquet_blobs", "pattern_candidates", "backtest_results"]:
                doc = self._mongo_db[coll_name].find_one(sort=[("session_date", 1)])
                if doc and "session_date" in doc:
                    oldest_dates.append(doc["session_date"])

            if not oldest_dates:
                logger.warning("mongodb_storage_full_but_no_data_found_to_purge")
                break

            oldest_date = min(oldest_dates)
            logger.warning("mongodb_purging_oldest_data", session_date=oldest_date, current_bytes=current_bytes)

            # Purge the oldest date's records from all collections
            self._mongo_db["archive_manifests"].delete_many({"session_date": oldest_date})
            self._mongo_db["parquet_blobs"].delete_many({"session_date": oldest_date})
            self._mongo_db["pattern_candidates"].delete_many({"session_date": oldest_date})
            self._mongo_db["backtest_results"].delete_many({"session_date": oldest_date})

    # ──────────────────────────────────────────────────────────────────────────
    # Archive manifest
    # ──────────────────────────────────────────────────────────────────────────

    def save_manifest(self, manifest: ArchiveManifest) -> None:
        if self._mongo_db is not None:
            doc = {
                "session_date": manifest.session_date.isoformat(),
                "archive_path": manifest.archive_path,
                "archive_size_bytes": manifest.archive_size_bytes,
                "symbols": manifest.symbols,
                "has_system_file": manifest.has_system_file,
                "total_ticks": manifest.total_ticks,
                "rejected_ticks": manifest.rejected_ticks,
                "gap_count": manifest.gap_count,
                "significant_gap_count": manifest.significant_gap_count,
                "total_gap_seconds": manifest.total_gap_seconds,
                "validation_passed": manifest.validation_passed,
                "validation_errors": manifest.validation_errors,
                "ingested_at": manifest.ingestion_finished_at.isoformat()
                if manifest.ingestion_finished_at
                else None,
            }
            self._mongo_db["archive_manifests"].replace_one(
                {"session_date": manifest.session_date.isoformat()},
                doc,
                upsert=True
            )
            logger.info("manifest_saved_mongo", date=str(manifest.session_date))
            # Trigger storage check after every ingestion manifest save
            self.check_and_purge_storage()
        else:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO archive_manifests VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    manifest.session_date.isoformat(),
                    manifest.archive_path,
                    manifest.archive_size_bytes,
                    json.dumps(manifest.symbols),
                    manifest.has_system_file,
                    json.dumps(manifest.total_ticks),
                    json.dumps(manifest.rejected_ticks),
                    manifest.gap_count,
                    manifest.significant_gap_count,
                    manifest.total_gap_seconds,
                    manifest.validation_passed,
                    json.dumps(manifest.validation_errors),
                    manifest.ingestion_finished_at.isoformat()
                    if manifest.ingestion_finished_at
                    else None,
                ],
            )
            logger.info("manifest_saved", date=str(manifest.session_date))

    def load_manifest(self, session_date: datetime.date) -> Optional[ArchiveManifest]:
        if self._mongo_db is not None:
            row = self._mongo_db["archive_manifests"].find_one(
                {"session_date": session_date.isoformat()}
            )
            if not row:
                return None
            return ArchiveManifest(
                session_date=datetime.date.fromisoformat(str(row["session_date"])),
                archive_path=row["archive_path"],
                archive_size_bytes=row["archive_size_bytes"],
                symbols=row["symbols"],
                has_system_file=row["has_system_file"],
                total_ticks=row["total_ticks"],
                rejected_ticks=row["rejected_ticks"],
                gap_count=row["gap_count"],
                significant_gap_count=row["significant_gap_count"],
                total_gap_seconds=row["total_gap_seconds"],
                validation_passed=row["validation_passed"],
                validation_errors=row["validation_errors"],
            )
        else:
            rows = self.conn.execute(
                "SELECT * FROM archive_manifests WHERE session_date = ?",
                [session_date.isoformat()],
            ).fetchall()
            if not rows:
                return None
            r = rows[0]
            return ArchiveManifest(
                session_date=datetime.date.fromisoformat(str(r[0])),
                archive_path=r[1],
                archive_size_bytes=r[2],
                symbols=json.loads(r[3] or "[]"),
                has_system_file=r[4],
                total_ticks=json.loads(r[5] or "{}"),
                rejected_ticks=json.loads(r[6] or "{}"),
                gap_count=r[7],
                significant_gap_count=r[8],
                total_gap_seconds=r[9],
                validation_passed=r[10],
                validation_errors=json.loads(r[11] or "[]"),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Tick / feature storage (Parquet / MongoDB Parquet Blobs)
    # ──────────────────────────────────────────────────────────────────────────

    def _parquet_path(self, session_date: datetime.date, symbol: str) -> Path:
        date_str = session_date.isoformat()
        p = self._ticks_dir / date_str
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{symbol}.parquet"

    def _feature_parquet_path(self, session_date: datetime.date, symbol: str) -> Path:
        date_str = session_date.isoformat()
        p = self._features_dir / date_str
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{symbol}.parquet"

    def write_ticks(
        self,
        session_date: datetime.date,
        symbol: str,
        tick_iter: Iterator[TickRecord],
    ) -> int:
        """
        Write ticks for a symbol to Parquet. If MongoDB is configured, uploads the
        resulting file as a binary blob and cleans up the local temporary file.
        """
        parquet_path = self._parquet_path(session_date, symbol)
        total = 0
        batch: list[dict[str, object]] = []
        written_first = False

        def _flush() -> None:
            nonlocal written_first, total
            if not batch:
                return
            df = pd.DataFrame(batch)
            if not written_first:
                df.to_parquet(str(parquet_path), index=False, engine="pyarrow")
                written_first = True
            else:
                existing_df = pd.read_parquet(str(parquet_path), engine="pyarrow")
                df = pd.concat([existing_df, df], ignore_index=True)
                df.to_parquet(str(parquet_path), index=False, engine="pyarrow")
            total += len(batch)
            batch.clear()

        for tick in tick_iter:
            batch.append(tick.model_dump())
            if len(batch) >= _TICK_BATCH_SIZE:
                _flush()

        _flush()
        logger.info("ticks_written", symbol=symbol, count=total, path=str(parquet_path))

        # Upload Parquet file as binary block to MongoDB to maintain persistent state
        if self._mongo_db is not None and parquet_path.exists():
            try:
                parquet_bytes = parquet_path.read_bytes()
                self._mongo_db["parquet_blobs"].replace_one(
                    {
                        "session_date": session_date.isoformat(),
                        "symbol": symbol,
                        "data_type": "ticks",
                    },
                    {
                        "session_date": session_date.isoformat(),
                        "symbol": symbol,
                        "data_type": "ticks",
                        "data": Binary(parquet_bytes),
                        "created_at": datetime.datetime.utcnow(),
                    },
                    upsert=True
                )
                logger.info("ticks_uploaded_to_mongo", symbol=symbol, date=session_date.isoformat())
                # Unlink local file to save storage in Render's ephemeral environment
                parquet_path.unlink(missing_ok=True)
            except Exception as e:
                logger.error("ticks_mongo_upload_failed", symbol=symbol, error=str(e))

        return total

    def load_ticks(
        self, session_date: datetime.date, symbol: str
    ) -> pd.DataFrame:
        """Load raw ticks for a symbol as a DataFrame (from MongoDB or local Parquet)."""
        if self._mongo_db is not None:
            row = self._mongo_db["parquet_blobs"].find_one(
                {
                    "session_date": session_date.isoformat(),
                    "symbol": symbol,
                    "data_type": "ticks",
                }
            )
            if not row:
                return pd.DataFrame()
            return pd.read_parquet(io.BytesIO(row["data"]), engine="pyarrow")
        else:
            path = self._parquet_path(session_date, symbol)
            if not path.exists():
                return pd.DataFrame()
            return pd.read_parquet(str(path), engine="pyarrow")

    def write_features(
        self,
        session_date: datetime.date,
        symbol: str,
        features: list[FeatureRecord],
    ) -> None:
        """Write feature records to Parquet (local files or MongoDB blobs)."""
        if not features:
            return
        path = self._feature_parquet_path(session_date, symbol)
        import dataclasses
        rows = [dataclasses.asdict(f) for f in features]
        df = pd.DataFrame(rows)
        df.to_parquet(str(path), index=False, engine="pyarrow")
        logger.info("features_written", symbol=symbol, count=len(features))

        # Upload features Parquet file as binary block to MongoDB
        if self._mongo_db is not None and path.exists():
            try:
                parquet_bytes = path.read_bytes()
                self._mongo_db["parquet_blobs"].replace_one(
                    {
                        "session_date": session_date.isoformat(),
                        "symbol": symbol,
                        "data_type": "features",
                    },
                    {
                        "session_date": session_date.isoformat(),
                        "symbol": symbol,
                        "data_type": "features",
                        "data": Binary(parquet_bytes),
                        "created_at": datetime.datetime.utcnow(),
                    },
                    upsert=True
                )
                logger.info("features_uploaded_to_mongo", symbol=symbol, date=session_date.isoformat())
                path.unlink(missing_ok=True)
            except Exception as e:
                logger.error("features_mongo_upload_failed", symbol=symbol, error=str(e))

    def load_features(
        self, session_date: datetime.date, symbol: str
    ) -> pd.DataFrame:
        """Load feature records for a symbol as a DataFrame (from MongoDB or local Parquet)."""
        if self._mongo_db is not None:
            row = self._mongo_db["parquet_blobs"].find_one(
                {
                    "session_date": session_date.isoformat(),
                    "symbol": symbol,
                    "data_type": "features",
                }
            )
            if not row:
                return pd.DataFrame()
            return pd.read_parquet(io.BytesIO(row["data"]), engine="pyarrow")
        else:
            path = self._feature_parquet_path(session_date, symbol)
            if not path.exists():
                return pd.DataFrame()
            return pd.read_parquet(str(path), engine="pyarrow")

    # ──────────────────────────────────────────────────────────────────────────
    # Pattern / backtest persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save_pattern_candidates(
        self,
        session_date: datetime.date,
        candidates: list[PatternCandidate],
    ) -> None:
        import dataclasses

        if self._mongo_db is not None:
            for c in candidates:
                rules_data = [dataclasses.asdict(r) for r in c.rules]
                doc = {
                    "pattern_id": c.pattern_id,
                    "session_date": session_date.isoformat(),
                    "symbol": c.symbol,
                    "direction": c.direction.value,
                    "discovery_method": c.discovery_method,
                    "rules_json": json.dumps(rules_data),
                    "sample_count": c.sample_count,
                    "description": c.description,
                }
                self._mongo_db["pattern_candidates"].replace_one(
                    {"pattern_id": c.pattern_id},
                    doc,
                    upsert=True
                )
        else:
            for c in candidates:
                rules_data = [dataclasses.asdict(r) for r in c.rules]
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO pattern_candidates
                    (pattern_id, session_date, symbol, direction,
                     discovery_method, rules_json, sample_count, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        c.pattern_id,
                        session_date.isoformat(),
                        c.symbol,
                        c.direction.value,
                        c.discovery_method,
                        json.dumps(rules_data),
                        c.sample_count,
                        c.description,
                    ],
                )

    def save_backtest_result(
        self, session_date: datetime.date, result: BacktestResult
    ) -> None:
        import dataclasses

        rules_data = [dataclasses.asdict(r) for r in result.rules]
        trades_data = [dataclasses.asdict(t) for t in result.trades]

        if self._mongo_db is not None:
            doc = {
                "pattern_id": result.pattern_id,
                "session_date": session_date.isoformat(),
                "symbol": result.symbol,
                "direction": result.direction.value,
                "sample_count": result.sample_count,
                "is_sample_count": result.is_sample_count,
                "oos_sample_count": result.oos_sample_count,
                "win_rate": result.win_rate,
                "oos_win_rate": result.oos_win_rate,
                "profit_factor": result.profit_factor,
                "oos_profit_factor": result.oos_profit_factor,
                "expectancy": result.expectancy,
                "avg_win": result.avg_win,
                "avg_loss": result.avg_loss,
                "max_drawdown": result.max_drawdown,
                "sharpe_ratio": result.sharpe_ratio,
                "total_net_pnl": result.total_net_pnl,
                "total_costs": result.total_costs,
                "win_rate_cv": result.win_rate_cv,
                "is_stable": result.is_stable,
                "verdict": result.verdict,
                "rejection_reason": result.rejection_reason,
                "rules_json": json.dumps(rules_data),
                "trades_json": json.dumps(trades_data),
            }
            self._mongo_db["backtest_results"].replace_one(
                {"pattern_id": result.pattern_id},
                doc,
                upsert=True
            )
        else:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO backtest_results (
                    pattern_id, session_date, symbol, direction,
                    sample_count, is_sample_count, oos_sample_count,
                    win_rate, oos_win_rate, profit_factor, oos_profit_factor,
                    expectancy, avg_win, avg_loss, max_drawdown, sharpe_ratio,
                    total_net_pnl, total_costs, win_rate_cv, is_stable,
                    verdict, rejection_reason, rules_json, trades_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    result.pattern_id,
                    session_date.isoformat(),
                    result.symbol,
                    result.direction.value,
                    result.sample_count,
                    result.is_sample_count,
                    result.oos_sample_count,
                    result.win_rate,
                    result.oos_win_rate,
                    result.profit_factor,
                    result.oos_profit_factor,
                    result.expectancy,
                    result.avg_win,
                    result.avg_loss,
                    result.max_drawdown,
                    result.sharpe_ratio,
                    result.total_net_pnl,
                    result.total_costs,
                    result.win_rate_cv,
                    result.is_stable,
                    result.verdict,
                    result.rejection_reason,
                    json.dumps(rules_data),
                    json.dumps(trades_data),
                ],
            )

    def load_backtest_results(
        self, session_date: datetime.date
    ) -> list[BacktestResult]:
        """Load all backtest results for a session date."""
        if self._mongo_db is not None:
            cursor = self._mongo_db["backtest_results"].find(
                {"session_date": session_date.isoformat()}
            )
            rows = []
            for r in cursor:
                # Reconstruct row tuple format matching DuckDB's select results schema
                row = (
                    r.get("pattern_id"),
                    r.get("session_date"),
                    r.get("symbol"),
                    r.get("direction"),
                    r.get("sample_count"),
                    r.get("is_sample_count"),
                    r.get("oos_sample_count"),
                    r.get("win_rate"),
                    r.get("oos_win_rate"),
                    r.get("profit_factor"),
                    r.get("oos_profit_factor"),
                    r.get("expectancy"),
                    r.get("avg_win"),
                    r.get("avg_loss"),
                    r.get("max_drawdown"),
                    r.get("sharpe_ratio"),
                    r.get("total_net_pnl"),
                    r.get("total_costs"),
                    r.get("win_rate_cv"),
                    r.get("is_stable"),
                    r.get("verdict"),
                    r.get("rejection_reason"),
                    r.get("rules_json"),
                    r.get("trades_json"),
                    r.get("created_at"),
                )
                rows.append(row)
            return rows  # type: ignore[return-value]
        else:
            rows = self.conn.execute(
                "SELECT * FROM backtest_results WHERE session_date = ?",
                [session_date.isoformat()],
            ).fetchall()
            return rows  # type: ignore[return-value]
