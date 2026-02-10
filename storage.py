"""
SQLite storage for persisting ticks, predictions, orders, fills, and PnL snapshots.

All writes are asynchronous using aiosqlite. Schema is auto-created on first run.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import aiosqlite

from config import StorageConfig

logger = logging.getLogger("storage")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS btc_ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    exchange TEXT NOT NULL,
    price REAL NOT NULL,
    volume REAL,
    created_at REAL DEFAULT (strftime('%s','now'))
);

CREATE INDEX IF NOT EXISTS idx_ticks_ts ON btc_ticks(timestamp);
CREATE INDEX IF NOT EXISTS idx_ticks_exchange ON btc_ticks(exchange);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    ticker TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prob_yes REAL NOT NULL,
    btc_price REAL,
    market_yes_price INTEGER,
    features_json TEXT,
    created_at REAL DEFAULT (strftime('%s','now'))
);

CREATE INDEX IF NOT EXISTS idx_pred_ticker ON predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(timestamp);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_order_id TEXT UNIQUE NOT NULL,
    server_order_id TEXT,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    action TEXT NOT NULL,
    price_cents INTEGER NOT NULL,
    size INTEGER NOT NULL,
    filled_size INTEGER DEFAULT 0,
    state TEXT NOT NULL,
    is_entry INTEGER NOT NULL,
    edge REAL,
    model_prob REAL,
    market_prob REAL,
    reason TEXT,
    dry_run INTEGER DEFAULT 0,
    created_at REAL DEFAULT (strftime('%s','now')),
    updated_at REAL DEFAULT (strftime('%s','now'))
);

CREATE INDEX IF NOT EXISTS idx_orders_ticker ON orders(ticker);
CREATE INDEX IF NOT EXISTS idx_orders_state ON orders(state);

CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT,
    order_id TEXT,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    action TEXT NOT NULL,
    yes_price INTEGER,
    no_price INTEGER,
    count INTEGER NOT NULL,
    is_taker INTEGER DEFAULT 0,
    created_at REAL DEFAULT (strftime('%s','now'))
);

CREATE INDEX IF NOT EXISTS idx_fills_ticker ON fills(ticker);

CREATE TABLE IF NOT EXISTS pnl_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    realized_pnl_cents INTEGER NOT NULL,
    fees_cents INTEGER DEFAULT 0,
    num_trades INTEGER DEFAULT 0,
    balance_cents INTEGER,
    created_at REAL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    ticker TEXT NOT NULL,
    yes_price INTEGER,
    no_price INTEGER,
    volume INTEGER,
    status TEXT,
    orderbook_json TEXT,
    created_at REAL DEFAULT (strftime('%s','now'))
);

CREATE INDEX IF NOT EXISTS idx_mkt_snap_ticker ON market_snapshots(ticker);
CREATE INDEX IF NOT EXISTS idx_mkt_snap_ts ON market_snapshots(timestamp);

CREATE TABLE IF NOT EXISTS settlement_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT UNIQUE NOT NULL,
    result TEXT NOT NULL,
    settled_at REAL,
    created_at REAL DEFAULT (strftime('%s','now'))
);
"""


class Storage:
    """Async SQLite storage layer."""

    def __init__(self, config: StorageConfig):
        self._db_path = config.db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def init(self):
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()
        logger.info("Storage initialized at %s", self._db_path)

    async def close(self):
        if self._db:
            await self._db.close()

    # ── Ticks ────────────────────────────────────────────────────────

    async def save_tick(self, timestamp: float, exchange: str, price: float, volume: float = None):
        await self._db.execute(
            "INSERT INTO btc_ticks (timestamp, exchange, price, volume) VALUES (?, ?, ?, ?)",
            (timestamp, exchange, price, volume),
        )
        await self._db.commit()

    async def get_ticks(
        self, exchange: Optional[str] = None, since: Optional[float] = None, limit: int = 1000
    ) -> List[Dict]:
        query = "SELECT timestamp, exchange, price, volume FROM btc_ticks WHERE 1=1"
        params = []
        if exchange:
            query += " AND exchange = ?"
            params.append(exchange)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        return [
            {"timestamp": r[0], "exchange": r[1], "price": r[2], "volume": r[3]}
            for r in rows
        ]

    # ── Predictions ──────────────────────────────────────────────────

    async def save_prediction(
        self,
        timestamp: float,
        ticker: str,
        model_name: str,
        prob_yes: float,
        btc_price: float = None,
        market_yes_price: int = None,
        features: Dict = None,
    ):
        await self._db.execute(
            """INSERT INTO predictions
            (timestamp, ticker, model_name, prob_yes, btc_price, market_yes_price, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, ticker, model_name, prob_yes, btc_price, market_yes_price,
             json.dumps(features) if features else None),
        )
        await self._db.commit()

    async def get_predictions(self, ticker: Optional[str] = None, limit: int = 500) -> List[Dict]:
        query = "SELECT timestamp, ticker, model_name, prob_yes, btc_price, market_yes_price FROM predictions"
        params = []
        if ticker:
            query += " WHERE ticker = ?"
            params.append(ticker)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        return [
            {
                "timestamp": r[0], "ticker": r[1], "model_name": r[2],
                "prob_yes": r[3], "btc_price": r[4], "market_yes_price": r[5],
            }
            for r in rows
        ]

    # ── Orders ───────────────────────────────────────────────────────

    async def save_order(
        self,
        client_order_id: str,
        server_order_id: Optional[str],
        ticker: str,
        side: str,
        action: str,
        price_cents: int,
        size: int,
        state: str,
        is_entry: bool,
        edge: float = None,
        model_prob: float = None,
        market_prob: float = None,
        reason: str = None,
        dry_run: bool = False,
    ):
        await self._db.execute(
            """INSERT OR REPLACE INTO orders
            (client_order_id, server_order_id, ticker, side, action, price_cents, size,
             state, is_entry, edge, model_prob, market_prob, reason, dry_run, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (client_order_id, server_order_id, ticker, side, action, price_cents, size,
             state, int(is_entry), edge, model_prob, market_prob, reason, int(dry_run),
             time.time()),
        )
        await self._db.commit()

    async def update_order_state(self, client_order_id: str, state: str, filled_size: int = None):
        if filled_size is not None:
            await self._db.execute(
                "UPDATE orders SET state = ?, filled_size = ?, updated_at = ? WHERE client_order_id = ?",
                (state, filled_size, time.time(), client_order_id),
            )
        else:
            await self._db.execute(
                "UPDATE orders SET state = ?, updated_at = ? WHERE client_order_id = ?",
                (state, time.time(), client_order_id),
            )
        await self._db.commit()

    # ── Fills ────────────────────────────────────────────────────────

    async def save_fill(
        self, trade_id: str, order_id: str, ticker: str, side: str, action: str,
        yes_price: int, no_price: int, count: int, is_taker: bool = False,
    ):
        await self._db.execute(
            """INSERT INTO fills
            (trade_id, order_id, ticker, side, action, yes_price, no_price, count, is_taker)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trade_id, order_id, ticker, side, action, yes_price, no_price, count, int(is_taker)),
        )
        await self._db.commit()

    # ── PnL ──────────────────────────────────────────────────────────

    async def save_pnl_snapshot(
        self, date: str, realized_pnl_cents: int, fees_cents: int = 0,
        num_trades: int = 0, balance_cents: int = None,
    ):
        await self._db.execute(
            """INSERT INTO pnl_snapshots
            (date, realized_pnl_cents, fees_cents, num_trades, balance_cents)
            VALUES (?, ?, ?, ?, ?)""",
            (date, realized_pnl_cents, fees_cents, num_trades, balance_cents),
        )
        await self._db.commit()

    # ── Market snapshots ─────────────────────────────────────────────

    async def save_market_snapshot(
        self, timestamp: float, ticker: str, yes_price: int, no_price: int,
        volume: int, status: str, orderbook: Dict = None,
    ):
        await self._db.execute(
            """INSERT INTO market_snapshots
            (timestamp, ticker, yes_price, no_price, volume, status, orderbook_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, ticker, yes_price, no_price, volume, status,
             json.dumps(orderbook) if orderbook else None),
        )
        await self._db.commit()

    # ── Settlements ──────────────────────────────────────────────────

    async def save_settlement(self, ticker: str, result: str, settled_at: float = None):
        await self._db.execute(
            "INSERT OR REPLACE INTO settlement_outcomes (ticker, result, settled_at) VALUES (?, ?, ?)",
            (ticker, result, settled_at or time.time()),
        )
        await self._db.commit()

    async def get_settlements(self) -> List[Dict]:
        async with self._db.execute(
            "SELECT ticker, result, settled_at FROM settlement_outcomes ORDER BY settled_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
        return [{"ticker": r[0], "result": r[1], "settled_at": r[2]} for r in rows]

    # ── Replay / Backtest data ───────────────────────────────────────

    async def get_tick_replay(
        self, since: float, until: Optional[float] = None
    ) -> List[Dict]:
        """Get ticks for replay, ordered chronologically."""
        query = "SELECT timestamp, exchange, price, volume FROM btc_ticks WHERE timestamp >= ?"
        params: list = [since]
        if until:
            query += " AND timestamp <= ?"
            params.append(until)
        query += " ORDER BY timestamp ASC"
        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        return [
            {"timestamp": r[0], "exchange": r[1], "price": r[2], "volume": r[3]}
            for r in rows
        ]

    async def get_market_snapshot_replay(
        self, ticker: str, since: float, until: Optional[float] = None
    ) -> List[Dict]:
        query = """SELECT timestamp, ticker, yes_price, no_price, volume, status, orderbook_json
                   FROM market_snapshots WHERE ticker = ? AND timestamp >= ?"""
        params: list = [ticker, since]
        if until:
            query += " AND timestamp <= ?"
            params.append(until)
        query += " ORDER BY timestamp ASC"
        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        return [
            {
                "timestamp": r[0], "ticker": r[1], "yes_price": r[2],
                "no_price": r[3], "volume": r[4], "status": r[5],
                "orderbook": json.loads(r[6]) if r[6] else None,
            }
            for r in rows
        ]
