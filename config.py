"""
Configuration management for the Kalshi BTC trading bot.
All secrets loaded from environment variables. Never hard-code keys.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env file if present (development only)
load_dotenv()


def _env(key: str, default: Optional[str] = None, required: bool = False) -> str:
    val = os.environ.get(key, default)
    if required and val is None:
        raise EnvironmentError(f"Required environment variable {key} is not set")
    return val


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, str(default)).lower()
    return val in ("true", "1", "yes")


@dataclass(frozen=True)
class KalshiConfig:
    """Kalshi API connection settings."""

    # Production: https://api.elections.kalshi.com/trade-api/v2
    # Demo:       https://demo-api.kalshi.co/trade-api/v2
    base_url: str = field(
        default_factory=lambda: _env(
            "KALSHI_BASE_URL",
            "https://api.elections.kalshi.com/trade-api/v2",
        )
    )
    # WebSocket: wss://api.elections.kalshi.com  (production)
    #            wss://demo-api.kalshi.co         (demo)
    ws_url: str = field(
        default_factory=lambda: _env(
            "KALSHI_WS_URL",
            "wss://api.elections.kalshi.com",
        )
    )
    api_key_id: str = field(
        default_factory=lambda: _env("KALSHI_API_KEY_ID", required=True)
    )
    private_key_path: str = field(
        default_factory=lambda: _env("KALSHI_PRIVATE_KEY_PATH", required=True)
    )

    # Series ticker for BTC 15-minute Up/Down markets
    series_ticker: str = field(
        default_factory=lambda: _env("KALSHI_SERIES_TICKER", "KXBTC15M")
    )

    # Rate limits — Basic tier defaults (reads/sec, writes/sec)
    read_rate_limit: int = field(
        default_factory=lambda: _env_int("KALSHI_READ_RATE_LIMIT", 18)
    )
    write_rate_limit: int = field(
        default_factory=lambda: _env_int("KALSHI_WRITE_RATE_LIMIT", 8)
    )


@dataclass(frozen=True)
class PriceFeedConfig:
    """Multi-exchange BTC spot price feed settings."""

    # Coinbase Advanced Trade WebSocket (public)
    coinbase_ws_url: str = "wss://advanced-trade-ws.coinbase.com"
    coinbase_product_id: str = "BTC-USD"

    # Binance WebSocket (public)
    binance_ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@trade"

    # Kraken WebSocket (public v2)
    kraken_ws_url: str = "wss://ws.kraken.com/v2"
    kraken_pair: str = "BTC/USD"

    # Stale data threshold (seconds) — if no update from an exchange in this window, mark stale
    stale_threshold_sec: float = field(
        default_factory=lambda: _env_float("PRICE_STALE_THRESHOLD_SEC", 5.0)
    )
    # Minimum number of non-stale feeds required to produce a median price
    min_feeds: int = field(
        default_factory=lambda: _env_int("PRICE_MIN_FEEDS", 1)
    )


@dataclass(frozen=True)
class RiskConfig:
    """Risk management parameters."""

    # Maximum number of contracts per single order
    max_order_size: int = field(
        default_factory=lambda: _env_int("RISK_MAX_ORDER_SIZE", 25)
    )
    # Maximum total position (contracts) across all open BTC 15m markets
    max_position: int = field(
        default_factory=lambda: _env_int("RISK_MAX_POSITION", 100)
    )
    # Maximum number of open orders at any time
    max_open_orders: int = field(
        default_factory=lambda: _env_int("RISK_MAX_OPEN_ORDERS", 10)
    )
    # Maximum daily loss in cents before kill switch triggers
    max_daily_loss_cents: int = field(
        default_factory=lambda: _env_int("RISK_MAX_DAILY_LOSS_CENTS", 5000)
    )
    # Seconds before market close to stop placing new orders
    close_cutoff_sec: int = field(
        default_factory=lambda: _env_int("RISK_CLOSE_CUTOFF_SEC", 60)
    )
    # Kill switch — if True, no new orders are placed (can be toggled at runtime)
    kill_switch: bool = field(
        default_factory=lambda: _env_bool("RISK_KILL_SWITCH", False)
    )


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy / model parameters."""

    # Target entry YES price (cents) — buy YES when market price ≤ this
    entry_price_cents: int = field(
        default_factory=lambda: _env_int("STRATEGY_ENTRY_PRICE", 45)
    )
    # Target exit YES price (cents) — sell YES when market price ≥ this
    exit_price_cents: int = field(
        default_factory=lambda: _env_int("STRATEGY_EXIT_PRICE", 65)
    )
    # Order size in contracts
    order_size: int = field(
        default_factory=lambda: _env_int("STRATEGY_ORDER_SIZE", 5)
    )
    # Minimum edge (model_prob − market_price/100) to enter
    min_edge: float = field(
        default_factory=lambda: _env_float("STRATEGY_MIN_EDGE", 0.05)
    )
    # Poll interval for the main strategy loop (seconds)
    poll_interval_sec: float = field(
        default_factory=lambda: _env_float("STRATEGY_POLL_SEC", 2.0)
    )


@dataclass(frozen=True)
class StorageConfig:
    """SQLite storage configuration."""

    db_path: str = field(
        default_factory=lambda: _env("STORAGE_DB_PATH", "kalshi_bot.db")
    )


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    price_feed: PriceFeedConfig = field(default_factory=PriceFeedConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    dry_run: bool = field(default_factory=lambda: _env_bool("DRY_RUN", True))
    log_level: str = field(
        default_factory=lambda: _env("LOG_LEVEL", "INFO")
    )


def load_config() -> AppConfig:
    """Create and validate the application config from environment variables."""
    cfg = AppConfig()

    # Validate that the private key file exists
    pk_path = Path(cfg.kalshi.private_key_path)
    if not pk_path.exists():
        raise FileNotFoundError(
            f"Kalshi private key not found at {pk_path}. "
            "Set KALSHI_PRIVATE_KEY_PATH to your .pem key file."
        )

    # Validate price bounds
    if not (1 <= cfg.strategy.entry_price_cents <= 99):
        raise ValueError("STRATEGY_ENTRY_PRICE must be 1–99 cents")
    if not (1 <= cfg.strategy.exit_price_cents <= 99):
        raise ValueError("STRATEGY_EXIT_PRICE must be 1–99 cents")
    if cfg.strategy.exit_price_cents <= cfg.strategy.entry_price_cents:
        raise ValueError("STRATEGY_EXIT_PRICE must exceed STRATEGY_ENTRY_PRICE")

    return cfg
