"""
Kalshi BTC 15-Minute Trading Bot — Main Entry Point.

Orchestrates:
  1. Multi-exchange BTC price feed (Coinbase, Binance, Kraken → median)
  2. Market discovery (find active KXBTC15M markets)
  3. Strategy evaluation (probabilistic model → trade decisions)
  4. Execution (limit order placement, exit management)
  5. Risk management (position/loss limits, kill switch)
  6. Persistence (SQLite logging of ticks, orders, PnL)
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timezone

from config import load_config, AppConfig
from execution import ExecutionManager
from kalshi_client import KalshiClient, KalshiAPIError, Market
from price_feed import PriceFeedAggregator, TickData
from risk import RiskManager
from storage import Storage
from strategy import StrategyEngine

# ── Logging setup ────────────────────────────────────────────────────

def setup_logging(level: str):
    """Structured logging with ISO timestamps."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("bot.log", mode="a"),
        ],
    )
    # Reduce noise from libraries
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


logger = logging.getLogger("main")


# ── Main bot class ───────────────────────────────────────────────────

class TradingBot:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = KalshiClient(config.kalshi)
        self.price_feed = PriceFeedAggregator(config.price_feed)
        self.strategy = StrategyEngine(config.strategy)
        self.execution = ExecutionManager(self.client, config)
        self.risk = RiskManager(config.risk)
        self.storage = Storage(config.storage)
        self._running = False
        self._active_markets: list[Market] = []
        self._tick_count = 0

    async def start(self):
        logger.info("=" * 60)
        logger.info("  Kalshi BTC 15-Min Trading Bot Starting")
        logger.info("  Mode: %s", "DRY RUN" if self.config.dry_run else "LIVE TRADING")
        logger.info("  Base URL: %s", self.config.kalshi.base_url)
        logger.info("  Series: %s", self.config.kalshi.series_ticker)
        logger.info("=" * 60)

        # Initialize components
        await self.storage.init()
        await self.client.start()
        self.price_feed.on_tick(self._on_btc_tick)
        await self.price_feed.start()

        self._running = True

        # Verify connectivity
        try:
            balance = await self.client.get_balance()
            logger.info(
                "Account balance: %d¢ ($%.2f) | Portfolio: %d¢ ($%.2f)",
                balance.available_balance, balance.available_balance / 100,
                balance.portfolio_value, balance.portfolio_value / 100,
            )
        except KalshiAPIError as e:
            logger.error("Failed to get balance — check API credentials: %s", e)
            if not self.config.dry_run:
                raise

        # Main loop
        try:
            await self._run_loop()
        except asyncio.CancelledError:
            logger.info("Bot shutdown requested")
        finally:
            await self._shutdown()

    async def _run_loop(self):
        """Core trading loop."""
        poll_sec = self.config.strategy.poll_interval_sec
        iteration = 0

        while self._running:
            iteration += 1
            try:
                # 1. Discover active markets
                await self._discover_markets()

                if not self._active_markets:
                    logger.debug("No active BTC 15m markets found, waiting...")
                    await asyncio.sleep(poll_sec * 2)
                    continue

                # 2. Get current BTC price
                btc_price = self.price_feed.get_price()
                if btc_price is None:
                    logger.warning("No BTC price available, waiting for feeds...")
                    await asyncio.sleep(poll_sec)
                    continue

                # 3. For each active market, evaluate and potentially trade
                for market in self._active_markets:
                    await self._process_market(market, btc_price)

                # 4. Manage existing orders (check fills, place exits)
                await self.execution.check_and_manage_orders()

                # 5. Update risk manager with current order count
                active_orders = self.execution.get_active_orders()
                self.risk.update_open_order_count(len(active_orders))

                # 6. Periodic status log
                if iteration % 30 == 0:
                    self._log_status(btc_price)

            except KalshiAPIError as e:
                logger.error("Kalshi API error in main loop: %s", e)
                if e.status == 401:
                    logger.critical("Authentication failure — stopping bot")
                    self._running = False
                    break
                await asyncio.sleep(poll_sec * 2)
            except Exception as e:
                logger.error("Unexpected error in main loop: %s", e, exc_info=True)
                await asyncio.sleep(poll_sec * 2)

            await asyncio.sleep(poll_sec)

    async def _discover_markets(self):
        """Find the currently active BTC 15-minute markets."""
        try:
            markets = await self.client.get_markets(
                series_ticker=self.config.kalshi.series_ticker,
                status="open",
            )
            self._active_markets = markets

            if markets and logger.isEnabledFor(logging.DEBUG):
                for m in markets:
                    logger.debug(
                        "Active market: %s | YES=%d¢ | close=%s",
                        m.ticker, m.yes_price, m.close_time,
                    )
        except KalshiAPIError as e:
            logger.error("Market discovery failed: %s", e)

    async def _process_market(self, market: Market, btc_price):
        """Evaluate one market and execute if there's a signal."""
        try:
            # Get orderbook
            orderbook = await self.client.get_orderbook(market.ticker)

            # Record market snapshot for replay
            await self.storage.save_market_snapshot(
                timestamp=time.time(),
                ticker=market.ticker,
                yes_price=market.yes_price,
                no_price=market.no_price,
                volume=market.volume,
                status=market.status,
                orderbook={
                    "yes_bids": [[l.price, l.quantity] for l in orderbook.yes_bids],
                    "no_bids": [[l.price, l.quantity] for l in orderbook.no_bids],
                },
            )

            # Get current position
            position = self.risk.get_position(market.ticker)

            # Run strategy
            decision = self.strategy.evaluate(
                market=market,
                orderbook=orderbook,
                btc_price=btc_price,
                current_position=position,
            )

            if decision is None:
                return  # HOLD

            # Save prediction
            await self.storage.save_prediction(
                timestamp=time.time(),
                ticker=market.ticker,
                model_name=self.strategy.model.name,
                prob_yes=decision.model_prob,
                btc_price=btc_price.price,
                market_yes_price=market.yes_price,
            )

            # Risk check
            allowed, reason = self.risk.check_order_allowed(
                ticker=decision.ticker,
                side=decision.side,
                action=decision.action,
                size=decision.size,
                market=market,
            )

            if not allowed:
                logger.info(
                    "Order blocked by risk manager: %s (signal=%s on %s)",
                    reason, decision.signal.value, market.ticker,
                )
                return

            # Execute
            managed = await self.execution.execute_decision(decision)
            if managed:
                await self.storage.save_order(
                    client_order_id=managed.client_order_id,
                    server_order_id=managed.server_order_id,
                    ticker=managed.ticker,
                    side=managed.side,
                    action=managed.action,
                    price_cents=managed.price_cents,
                    size=managed.requested_size,
                    state=managed.state.value,
                    is_entry=managed.is_entry,
                    edge=decision.edge,
                    model_prob=decision.model_prob,
                    market_prob=decision.market_prob,
                    reason=decision.reason,
                    dry_run=self.config.dry_run,
                )

        except KalshiAPIError as e:
            logger.error("Error processing market %s: %s", market.ticker, e)

    def _on_btc_tick(self, tick: TickData):
        """Callback for each raw BTC price tick from any exchange."""
        self._tick_count += 1
        self.strategy.record_tick(tick.price, tick.timestamp)

        # Persist every 10th tick to avoid write-thrashing
        if self._tick_count % 10 == 0:
            asyncio.create_task(
                self.storage.save_tick(tick.timestamp, tick.exchange, tick.price, tick.volume)
            )

    def _log_status(self, btc_price):
        """Periodic status summary."""
        feed_status = self.price_feed.get_feed_status()
        risk_status = self.risk.get_status()
        order_stats = self.execution.get_order_stats()

        logger.info(
            "STATUS | BTC=$%.2f (%d sources) | Markets=%d | Orders: %d active, %d filled | "
            "PnL=%d¢ | Kill=%s",
            btc_price.price, btc_price.num_sources,
            len(self._active_markets),
            order_stats["active"], order_stats["filled"],
            risk_status["daily_pnl_cents"],
            risk_status["kill_switch"],
        )
        for name, status in feed_status.items():
            logger.debug(
                "  Feed %s: price=%s, stale=%s, age=%ss",
                name, status["last_price"], status["is_stale"], status["age_sec"],
            )

    async def _shutdown(self):
        """Graceful shutdown: cancel orders, close connections."""
        logger.info("Shutting down...")

        # Cancel all open orders (safety)
        if not self.config.dry_run:
            try:
                await self.execution.cancel_all_orders()
            except Exception as e:
                logger.error("Error canceling orders during shutdown: %s", e)

        # Save final PnL snapshot
        risk_status = self.risk.get_status()
        try:
            await self.storage.save_pnl_snapshot(
                date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                realized_pnl_cents=risk_status["daily_pnl_cents"],
                num_trades=risk_status["daily_trades"],
            )
        except Exception:
            pass

        await self.price_feed.stop()
        await self.client.close()
        await self.storage.close()
        logger.info("Shutdown complete")

    def stop(self):
        self._running = False


# ── Entry point ──────────────────────────────────────────────────────

async def main():
    config = load_config()
    setup_logging(config.log_level)

    bot = TradingBot(config)

    # Handle SIGINT / SIGTERM gracefully
    #loop = asyncio.get_running_loop()
    #for sig in (signal.SIGINT, signal.SIGTERM):
        #loop.add_signal_handler(sig, bot.stop)

    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
