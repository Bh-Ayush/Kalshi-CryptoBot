"""
Replay / Backtest engine.

Reads recorded BTC ticks and Kalshi market snapshots from SQLite,
replays them through the strategy engine, and computes performance metrics.

Usage:
    python replay.py --since 2025-02-01 --until 2025-02-07

If no recorded data exists, uses a stub interface with synthetic data
to demonstrate the pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import load_config, AppConfig, StrategyConfig, StorageConfig
from kalshi_client import Market, OrderBook, OrderBookLevel
from price_feed import AggregatedPrice
from storage import Storage
from strategy import (
    StrategyEngine,
    BaselineHeuristicModel,
    Signal,
    brier_score,
    log_loss_score,
    hit_rate,
)

logger = logging.getLogger("replay")


@dataclass
class ReplayResult:
    """Summary of a backtest run."""
    total_markets: int = 0
    total_predictions: int = 0
    total_trades: int = 0
    total_entry_trades: int = 0
    total_exit_trades: int = 0

    # Calibration (requires settlement outcomes)
    predictions_with_outcome: int = 0
    brier: float = float("nan")
    log_loss: float = float("nan")
    accuracy: float = float("nan")

    # PnL simulation (simplified)
    simulated_pnl_cents: int = 0
    win_trades: int = 0
    loss_trades: int = 0
    total_contracts: int = 0
    avg_entry_price: float = 0.0
    avg_exit_price: float = 0.0

    # Fill ratio
    signals_generated: int = 0
    signals_executed: int = 0

    @property
    def fill_ratio(self) -> float:
        if self.signals_generated == 0:
            return 0.0
        return self.signals_executed / self.signals_generated

    @property
    def win_rate(self) -> float:
        total = self.win_trades + self.loss_trades
        if total == 0:
            return 0.0
        return self.win_trades / total

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  BACKTEST / REPLAY RESULTS",
            "=" * 60,
            f"  Markets evaluated:      {self.total_markets}",
            f"  Predictions made:       {self.total_predictions}",
            f"  Signals generated:      {self.signals_generated}",
            f"  Signals executed (sim):  {self.signals_executed}",
            f"  Fill ratio:             {self.fill_ratio:.1%}",
            "",
            "  --- Calibration ---",
            f"  Predictions w/ outcome: {self.predictions_with_outcome}",
            f"  Brier score:            {self.brier:.4f}" if not math.isnan(self.brier) else "  Brier score:            N/A (no outcomes)",
            f"  Log loss:               {self.log_loss:.4f}" if not math.isnan(self.log_loss) else "  Log loss:               N/A",
            f"  Accuracy (>0.5):        {self.accuracy:.1%}" if not math.isnan(self.accuracy) else "  Accuracy:               N/A",
            "",
            "  --- Simulated PnL ---",
            f"  Simulated PnL:          {self.simulated_pnl_cents}¢ (${self.simulated_pnl_cents/100:.2f})",
            f"  Wins / Losses:          {self.win_trades} / {self.loss_trades}",
            f"  Win rate:               {self.win_rate:.1%}",
            f"  Total contracts:        {self.total_contracts}",
            "=" * 60,
        ]
        return "\n".join(lines)


class ReplayEngine:
    """
    Replays historical data through the strategy to compute metrics.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.strategy = StrategyEngine(config.strategy)
        self.storage = Storage(config.storage)

    async def run(
        self,
        since_ts: float,
        until_ts: Optional[float] = None,
    ) -> ReplayResult:
        """Run replay over the specified time range."""
        await self.storage.init()

        result = ReplayResult()
        predictions_list: List[float] = []
        outcomes_list: List[int] = []

        # Load recorded ticks
        ticks = await self.storage.get_tick_replay(since_ts, until_ts)
        logger.info("Loaded %d ticks for replay", len(ticks))

        if not ticks:
            logger.warning("No tick data found. Generating synthetic replay...")
            return await self._synthetic_replay()

        # Feed ticks to strategy for feature computation
        for tick in ticks:
            self.strategy.record_tick(tick["price"], tick["timestamp"])

        # Load unique tickers from market snapshots
        all_snapshots = await self.storage.get_market_snapshot_replay(
            "%", since_ts, until_ts  # SQLite LIKE pattern
        )

        # Group snapshots by ticker
        ticker_snapshots: Dict[str, List[Dict]] = {}
        for snap in all_snapshots:
            tkr = snap["ticker"]
            if tkr not in ticker_snapshots:
                ticker_snapshots[tkr] = []
            ticker_snapshots[tkr].append(snap)

        result.total_markets = len(ticker_snapshots)

        # Load settlement outcomes
        settlements = await self.storage.get_settlements()
        settlement_map = {s["ticker"]: s["result"] for s in settlements}

        # Replay each market
        for ticker, snapshots in ticker_snapshots.items():
            for snap in snapshots:
                # Reconstruct market object
                market = Market(
                    ticker=ticker,
                    title="",
                    event_ticker="",
                    status=snap.get("status", "open"),
                    yes_price=snap.get("yes_price", 50),
                    no_price=snap.get("no_price", 50),
                    volume=snap.get("volume", 0),
                    close_time="",
                    open_time="",
                )

                # Reconstruct orderbook
                ob_data = snap.get("orderbook", {})
                orderbook = OrderBook(
                    yes_bids=[
                        OrderBookLevel(price=l[0], quantity=l[1])
                        for l in (ob_data or {}).get("yes_bids", [])
                    ],
                    no_bids=[
                        OrderBookLevel(price=l[0], quantity=l[1])
                        for l in (ob_data or {}).get("no_bids", [])
                    ],
                )

                # Find closest BTC price
                snap_ts = snap["timestamp"]
                closest_tick = min(ticks, key=lambda t: abs(t["timestamp"] - snap_ts))
                btc_price = AggregatedPrice(
                    price=closest_tick["price"],
                    num_sources=1,
                    timestamp=closest_tick["timestamp"],
                )

                # Run strategy
                decision = self.strategy.evaluate(
                    market=market,
                    orderbook=orderbook,
                    btc_price=btc_price,
                    current_position=0,
                )

                result.total_predictions += 1

                if decision and decision.signal != Signal.HOLD:
                    result.signals_generated += 1

                    # Simulate fill (optimistic: assume fill at limit price)
                    result.signals_executed += 1
                    result.total_contracts += decision.size

                    if decision.action == "buy":
                        result.total_entry_trades += 1
                    else:
                        result.total_exit_trades += 1

                    predictions_list.append(decision.model_prob)

                    # Check settlement if available
                    outcome = settlement_map.get(ticker)
                    if outcome is not None:
                        outcome_int = 1 if outcome == "yes" else 0
                        outcomes_list.append(outcome_int)

        # Compute calibration metrics
        if outcomes_list:
            matched_preds = predictions_list[: len(outcomes_list)]
            result.predictions_with_outcome = len(outcomes_list)
            result.brier = brier_score(matched_preds, outcomes_list)
            result.log_loss = log_loss_score(matched_preds, outcomes_list)
            result.accuracy = hit_rate(matched_preds, outcomes_list)

        await self.storage.close()
        return result

    async def _synthetic_replay(self) -> ReplayResult:
        """Generate a synthetic replay to demonstrate the pipeline."""
        import random
        random.seed(42)

        result = ReplayResult()
        predictions = []
        outcomes = []

        # Simulate 100 market events
        btc_base_price = 97000.0

        for i in range(100):
            # Random walk
            btc_price = btc_base_price + random.gauss(0, 200)
            ref_price = btc_base_price + random.gauss(0, 50)

            # True outcome: did BTC go up?
            true_up = btc_price > ref_price
            outcome = 1 if true_up else 0

            # Simulate market price (noisy version of truth)
            market_yes = int(max(1, min(99, 50 + random.gauss(0, 15))))

            market = Market(
                ticker=f"KXBTC15M-SYN{i:04d}",
                title="Synthetic",
                event_ticker="",
                status="open",
                yes_price=market_yes,
                no_price=100 - market_yes,
                volume=random.randint(10, 500),
                close_time="",
                open_time="",
            )

            agg_price = AggregatedPrice(
                price=btc_price, num_sources=3, timestamp=time.time()
            )
            self.strategy.set_reference_price(market.ticker, ref_price)
            self.strategy.record_tick(btc_price)

            orderbook = OrderBook(
                yes_bids=[OrderBookLevel(price=market_yes - 1, quantity=50)],
                no_bids=[OrderBookLevel(price=100 - market_yes - 1, quantity=50)],
            )

            decision = self.strategy.evaluate(
                market=market, orderbook=orderbook,
                btc_price=agg_price, current_position=0,
            )

            result.total_predictions += 1
            result.total_markets += 1

            if decision and decision.signal != Signal.HOLD:
                result.signals_generated += 1
                result.signals_executed += 1
                result.total_contracts += decision.size

                prob = decision.model_prob
                predictions.append(prob)
                outcomes.append(outcome)

                # Simulate PnL
                if decision.action == "buy":
                    entry_cost = decision.price_cents * decision.size
                    settlement_value = 100 * decision.size if outcome == 1 else 0
                    pnl = settlement_value - entry_cost
                    result.simulated_pnl_cents += pnl
                    if pnl > 0:
                        result.win_trades += 1
                    else:
                        result.loss_trades += 1

        if predictions and outcomes:
            result.predictions_with_outcome = len(outcomes)
            result.brier = brier_score(predictions, outcomes)
            result.log_loss = log_loss_score(predictions, outcomes)
            result.accuracy = hit_rate(predictions, outcomes)

        await self.storage.close()
        return result


# ── CLI ──────────────────────────────────────────────────────────────

async def run_replay(args):
    config = load_config()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    since_ts = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc).timestamp()
    until_ts = None
    if args.until:
        until_ts = datetime.fromisoformat(args.until).replace(tzinfo=timezone.utc).timestamp()

    engine = ReplayEngine(config)
    result = await engine.run(since_ts, until_ts)
    print(result.summary())


def main():
    parser = argparse.ArgumentParser(description="Replay / Backtest engine")
    parser.add_argument(
        "--since", default="2025-01-01",
        help="Start date (ISO format, default: 2025-01-01)"
    )
    parser.add_argument(
        "--until", default=None,
        help="End date (ISO format, default: now)"
    )
    args = parser.parse_args()
    asyncio.run(run_replay(args))


if __name__ == "__main__":
    main()
