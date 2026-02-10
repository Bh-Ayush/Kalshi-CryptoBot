"""
Multi-exchange BTC spot price feed using WebSocket connections.

Connects to Coinbase, Binance, and Kraken simultaneously, takes the median
of non-stale prices to reduce outlier risk.

References:
  - Coinbase Advanced Trade WS: https://docs.cdp.coinbase.com/advanced-trade/docs/ws-overview
  - Binance WS: https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams
  - Kraken WS v2: https://docs.kraken.com/api/docs/websocket-v2/trade
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from config import PriceFeedConfig

logger = logging.getLogger("price_feed")


@dataclass
class TickData:
    """A single price observation from one exchange."""
    exchange: str
    price: float
    timestamp: float  # time.time()
    volume: Optional[float] = None


@dataclass
class AggregatedPrice:
    """Median price across non-stale exchange feeds."""
    price: float
    num_sources: int
    timestamp: float
    sources: Dict[str, float] = field(default_factory=dict)


class ExchangeFeed:
    """Base class for an exchange WebSocket feed."""

    def __init__(self, name: str, stale_threshold: float):
        self.name = name
        self.stale_threshold = stale_threshold
        self.last_price: Optional[float] = None
        self.last_update: float = 0.0
        self._running = False

    @property
    def is_stale(self) -> bool:
        if self.last_price is None:
            return True
        return (time.time() - self.last_update) > self.stale_threshold

    def _update(self, price: float, vol: Optional[float] = None) -> TickData:
        self.last_price = price
        self.last_update = time.time()
        tick = TickData(
            exchange=self.name,
            price=price,
            timestamp=self.last_update,
            volume=vol,
        )
        return tick

    async def run(self, on_tick: Callable[[TickData], None]):
        raise NotImplementedError


class CoinbaseFeed(ExchangeFeed):
    """Coinbase Advanced Trade WebSocket — BTC-USD trades."""

    def __init__(self, config: PriceFeedConfig):
        super().__init__("coinbase", config.stale_threshold_sec)
        self._ws_url = config.coinbase_ws_url
        self._product_id = config.coinbase_product_id

    async def run(self, on_tick: Callable[[TickData], None]):
        self._running = True
        while self._running:
            try:
                async with websockets.connect(self._ws_url) as ws:
                    # Subscribe to market_trades channel (public, no auth needed)
                    sub = {
                        "type": "subscribe",
                        "product_ids": [self._product_id],
                        "channel": "market_trades",
                    }
                    await ws.send(json.dumps(sub))
                    logger.info("[coinbase] Subscribed to %s trades", self._product_id)

                    async for raw_msg in ws:
                        try:
                            msg = json.loads(raw_msg)
                            if msg.get("channel") == "market_trades":
                                events = msg.get("events", [])
                                for event in events:
                                    for trade in event.get("trades", []):
                                        price = float(trade["price"])
                                        vol = float(trade.get("size", 0))
                                        tick = self._update(price, vol)
                                        on_tick(tick)
                        except (KeyError, ValueError) as e:
                            logger.debug("[coinbase] Parse error: %s", e)
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                logger.warning("[coinbase] Connection lost (%s), reconnecting in 2s", e)
                await asyncio.sleep(2)
            except Exception as e:
                logger.error("[coinbase] Unexpected error: %s", e, exc_info=True)
                await asyncio.sleep(5)


class BinanceFeed(ExchangeFeed):
    """Binance WebSocket — BTCUSDT trade stream."""

    def __init__(self, config: PriceFeedConfig):
        super().__init__("binance", config.stale_threshold_sec)
        self._ws_url = config.binance_ws_url

    async def run(self, on_tick: Callable[[TickData], None]):
        self._running = True
        while self._running:
            try:
                async with websockets.connect(self._ws_url) as ws:
                    logger.info("[binance] Connected to BTCUSDT trade stream")
                    async for raw_msg in ws:
                        try:
                            msg = json.loads(raw_msg)
                            # Binance trade stream: {"e":"trade","p":"97000.50","q":"0.001",...}
                            if msg.get("e") == "trade":
                                price = float(msg["p"])
                                vol = float(msg.get("q", 0))
                                tick = self._update(price, vol)
                                on_tick(tick)
                        except (KeyError, ValueError) as e:
                            logger.debug("[binance] Parse error: %s", e)
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                logger.warning("[binance] Connection lost (%s), reconnecting in 2s", e)
                await asyncio.sleep(2)
            except Exception as e:
                logger.error("[binance] Unexpected error: %s", e, exc_info=True)
                await asyncio.sleep(5)


class KrakenFeed(ExchangeFeed):
    """Kraken WebSocket v2 — BTC/USD trades."""

    def __init__(self, config: PriceFeedConfig):
        super().__init__("kraken", config.stale_threshold_sec)
        self._ws_url = config.kraken_ws_url
        self._pair = config.kraken_pair

    async def run(self, on_tick: Callable[[TickData], None]):
        self._running = True
        while self._running:
            try:
                async with websockets.connect(self._ws_url) as ws:
                    sub = {
                        "method": "subscribe",
                        "params": {
                            "channel": "trade",
                            "symbol": [self._pair],
                        },
                    }
                    await ws.send(json.dumps(sub))
                    logger.info("[kraken] Subscribed to %s trades", self._pair)

                    async for raw_msg in ws:
                        try:
                            msg = json.loads(raw_msg)
                            # Kraken v2 trade: {"channel":"trade","data":[{"price":"97000.5",...}]}
                            if msg.get("channel") == "trade":
                                for trade in msg.get("data", []):
                                    price = float(trade["price"])
                                    vol = float(trade.get("qty", 0))
                                    tick = self._update(price, vol)
                                    on_tick(tick)
                        except (KeyError, ValueError) as e:
                            logger.debug("[kraken] Parse error: %s", e)
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                logger.warning("[kraken] Connection lost (%s), reconnecting in 2s", e)
                await asyncio.sleep(2)
            except Exception as e:
                logger.error("[kraken] Unexpected error: %s", e, exc_info=True)
                await asyncio.sleep(5)


class PriceFeedAggregator:
    """
    Manages multiple exchange feeds and produces a median BTC price.

    Usage:
        agg = PriceFeedAggregator(config)
        await agg.start()
        price = agg.get_price()  # AggregatedPrice or None
        await agg.stop()
    """

    def __init__(self, config: PriceFeedConfig):
        self._config = config
        self._feeds: List[ExchangeFeed] = [
            CoinbaseFeed(config),
            BinanceFeed(config),
            KrakenFeed(config),
        ]
        self._tasks: List[asyncio.Task] = []
        self._tick_callbacks: List[Callable[[TickData], None]] = []
        self._latest_ticks: Dict[str, TickData] = {}

    def on_tick(self, callback: Callable[[TickData], None]):
        """Register a callback for each raw tick from any exchange."""
        self._tick_callbacks.append(callback)

    def _handle_tick(self, tick: TickData):
        self._latest_ticks[tick.exchange] = tick
        for cb in self._tick_callbacks:
            try:
                cb(tick)
            except Exception as e:
                logger.error("Tick callback error: %s", e)

    async def start(self):
        for feed in self._feeds:
            task = asyncio.create_task(feed.run(self._handle_tick))
            self._tasks.append(task)
        logger.info(
            "PriceFeedAggregator started with %d feeds: %s",
            len(self._feeds),
            [f.name for f in self._feeds],
        )

    async def stop(self):
        for feed in self._feeds:
            feed._running = False
        for task in self._tasks:
            task.cancel()
        logger.info("PriceFeedAggregator stopped")

    def get_price(self) -> Optional[AggregatedPrice]:
        """Return median price from non-stale feeds, or None if insufficient data."""
        active_prices: Dict[str, float] = {}
        for feed in self._feeds:
            if not feed.is_stale and feed.last_price is not None:
                active_prices[feed.name] = feed.last_price

        if len(active_prices) < self._config.min_feeds:
            logger.warning(
                "Insufficient price feeds: %d active (need %d). Stale: %s",
                len(active_prices),
                self._config.min_feeds,
                [f.name for f in self._feeds if f.is_stale],
            )
            return None

        median_price = statistics.median(active_prices.values())
        return AggregatedPrice(
            price=median_price,
            num_sources=len(active_prices),
            timestamp=time.time(),
            sources=active_prices,
        )

    def get_feed_status(self) -> Dict[str, Dict]:
        """Return diagnostic info about each feed."""
        status = {}
        for feed in self._feeds:
            status[feed.name] = {
                "last_price": feed.last_price,
                "last_update": feed.last_update,
                "is_stale": feed.is_stale,
                "age_sec": round(time.time() - feed.last_update, 2)
                if feed.last_update
                else None,
            }
        return status
