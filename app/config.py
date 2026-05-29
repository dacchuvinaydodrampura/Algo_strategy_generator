"""
app/config.py
-------------
Centralised configuration for the Market Research Engine.

All settings are loaded from:
  1. config/settings.yaml  (base defaults)
  2. .env file or environment variables (secrets + overrides)

Secrets (TELEGRAM_BOT_TOKEN, etc.) must never live in YAML.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = parent of this file's parent
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_YAML = _PROJECT_ROOT / "config" / "settings.yaml"
_ENV_FILE = _PROJECT_ROOT / ".env"


def _load_yaml_defaults() -> dict[str, Any]:
    """Load base defaults from settings.yaml if it exists."""
    if _CONFIG_YAML.exists():
        with _CONFIG_YAML.open() as fh:
            return yaml.safe_load(fh) or {}
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Nested config dataclasses
# ──────────────────────────────────────────────────────────────────────────────

from pydantic import BaseModel


class TelegramConfig(BaseModel):
    poll_interval_seconds: int = 10
    download_timeout_seconds: int = 300


class StorageConfig(BaseModel):
    archives_dir: str = "data/archives"
    ticks_dir: str = "data/ticks"
    reports_dir: str = "data/reports"
    patterns_dir: str = "data/patterns"
    temp_dir: str = "data/temp"
    db_path: str = "data/research.duckdb"


class IngestionConfig(BaseModel):
    chunk_size_ticks: int = 5000
    max_gap_seconds: int = 300
    min_ticks_per_symbol: int = 100
    timestamp_field: str = "t"
    sequence_field: str = "seq"


class FeaturesConfig(BaseModel):
    imbalance_depth_levels: int = 5
    microprice_levels: int = 2
    slope_window_ticks: int = 10
    volatility_window_ticks: int = 20
    aggression_window_ticks: int = 15
    liquidity_window_ticks: int = 10


class WindowsConfig(BaseModel):
    tick_sizes: list[int] = [20, 50, 100]
    time_sizes_seconds: list[int] = [30, 60, 120]
    min_window_ticks: int = 10


class PatternsConfig(BaseModel):
    min_samples: int = 30
    min_win_rate: float = 0.52
    min_profit_factor: float = 1.25
    max_features_per_rule: int = 4
    clustering_n_clusters: int = 8
    stability_cv_threshold: float = 0.35
    oos_split_fraction: float = 0.30
    min_multi_day_sessions: int = 3
    max_multi_day_sessions: int = 5
    mc_trials: int = 50


class BacktestConfig(BaseModel):
    tick_size: float = 0.05
    lot_size: int = 25
    brokerage_per_lot: float = 20.0
    slippage_ticks: int = 1
    latency_ms: int = 50
    default_stop_ticks: int = 5
    default_target_ticks: int = 10
    max_hold_seconds: int = 120
    initial_capital: float = 1_000_000.0
    use_dynamic_stops: bool = False
    stop_vol_multiplier: float = 3.0
    target_vol_multiplier: float = 6.0
    entry_order_type: str = "market"  # limit | market
    queue_position_multiplier: float = 1.0
    limit_order_timeout_seconds: float = 10.0
    cooldown_ticks: int = 50


class ReportConfig(BaseModel):
    pdf_page_size: str = "A4"
    logo_path: str | None = None
    max_example_windows: int = 5
    author: str = "Market Research Engine v1.0"


# ──────────────────────────────────────────────────────────────────────────────
# Main settings
# ──────────────────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """
    Top-level application settings.

    Secrets come from environment variables (TELEGRAM_BOT_TOKEN, etc.).
    All other defaults are loaded from settings.yaml then overridable via env.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Secrets (env-only, never in YAML) ─────────────────────────────────────
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_channel_id: int = Field(default=0, alias="TELEGRAM_CHANNEL_ID")

    # ── Env-overridable scalars ────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")  # json | console
    data_root: str = Field(default=str(_PROJECT_ROOT), alias="DATA_ROOT")
    mongodb_uri: str = Field(default="", alias="MONGODB_URI")

    # ── Nested config blocks ───────────────────────────────────────────────────
    telegram: TelegramConfig = TelegramConfig()
    storage: StorageConfig = StorageConfig()
    ingestion: IngestionConfig = IngestionConfig()
    features: FeaturesConfig = FeaturesConfig()
    windows: WindowsConfig = WindowsConfig()
    patterns: PatternsConfig = PatternsConfig()
    backtest: BacktestConfig = BacktestConfig()
    report: ReportConfig = ReportConfig()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper

    def resolve_path(self, relative: str) -> Path:
        """Resolve a storage path relative to data_root."""
        p = Path(relative)
        if p.is_absolute():
            return p
        return Path(self.data_root) / p

    @property
    def archives_path(self) -> Path:
        return self.resolve_path(self.storage.archives_dir)

    @property
    def ticks_path(self) -> Path:
        return self.resolve_path(self.storage.ticks_dir)

    @property
    def reports_path(self) -> Path:
        return self.resolve_path(self.storage.reports_dir)

    @property
    def patterns_path(self) -> Path:
        return self.resolve_path(self.storage.patterns_dir)

    @property
    def temp_path(self) -> Path:
        return self.resolve_path(self.storage.temp_dir)

    @property
    def db_path(self) -> Path:
        return self.resolve_path(self.storage.db_path)

    def ensure_directories(self) -> None:
        """Create all required data directories."""
        for path in [
            self.archives_path,
            self.ticks_path,
            self.reports_path,
            self.patterns_path,
            self.temp_path,
            self.db_path.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def _merge_yaml_into_settings(yaml_data: dict[str, Any], settings: Settings) -> Settings:
    """
    Merge YAML config block into the nested Pydantic models.
    Environment variables always win; YAML provides structured defaults.
    """
    _d = yaml_data

    def _get(key: str) -> Any:
        return _d.get(key, {})

    if "telegram" in _d:
        settings = settings.model_copy(
            update={"telegram": TelegramConfig(**_get("telegram"))}
        )
    if "storage" in _d:
        settings = settings.model_copy(
            update={"storage": StorageConfig(**_get("storage"))}
        )
    if "ingestion" in _d:
        settings = settings.model_copy(
            update={"ingestion": IngestionConfig(**_get("ingestion"))}
        )
    if "features" in _d:
        settings = settings.model_copy(
            update={"features": FeaturesConfig(**_get("features"))}
        )
    if "windows" in _d:
        settings = settings.model_copy(
            update={"windows": WindowsConfig(**_get("windows"))}
        )
    if "patterns" in _d:
        settings = settings.model_copy(
            update={"patterns": PatternsConfig(**_get("patterns"))}
        )
    if "backtest" in _d:
        settings = settings.model_copy(
            update={"backtest": BacktestConfig(**_get("backtest"))}
        )
    if "report" in _d:
        settings = settings.model_copy(
            update={"report": ReportConfig(**_get("report"))}
        )
    return settings


def load_settings() -> Settings:
    """
    Load and return the fully merged Settings object.
    Call once at application startup and pass everywhere via DI.
    """
    yaml_defaults = _load_yaml_defaults()
    settings = Settings()
    if yaml_defaults:
        settings = _merge_yaml_into_settings(yaml_defaults, settings)
    return settings
