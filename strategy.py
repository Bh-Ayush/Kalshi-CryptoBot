"""
Strategy module — probabilistic forecasting + trading decision logic.

Implements a pluggable model interface:
  1. BaselineHeuristicModel — simple rule-based model (no training data required)
  2. ModelInterface — abstract class for trained models (logistic regression, GBM, etc.)

Key concepts:
  - In event markets, "accuracy" alone is NOT enough. What matters is CALIBRATION
    and EDGE over the market price.
  - If the model says P(YES) = 0.60 and the market asks 0.45, the edge is +0.15.
  - Expected value per contract = edge × 100 cents = 15 cents (before fees).
  - A model that's 70% accurate but poorly calibrated (says 0.95 when true prob is 0.70)
    will systematically overpay and lose money.
  - Brier score measures calibration: lower is better. A perfectly calibrated model
    has Brier score = 0 for deterministic events.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from config import StrategyConfig
from kalshi_client import Market, OrderBook
from price_feed import AggregatedPrice

logger = logging.getLogger("strategy")


# ── Data structures ──────────────────────────────────────────────────

class Signal(Enum):
    BUY_YES = "buy_yes"
    BUY_NO = "buy_no"
    SELL_YES = "sell_yes"
    SELL_NO = "sell_no"
    HOLD = "hold"


@dataclass
class ModelFeatures:
    """Features extracted for the probabilistic model."""
    # BTC price features
    btc_price: float                    # current median BTC price
    btc_return_1m: float = 0.0          # 1-minute return
    btc_return_5m: float = 0.0          # 5-minute return
    btc_volatility_5m: float = 0.0      # 5-min realized vol (std of 1-min returns)

    # Market features
    market_yes_price: int = 50          # current YES price on Kalshi (cents)
    market_no_price: int = 50           # current NO price
    time_remaining_sec: float = 900.0   # seconds until market close
    time_remaining_frac: float = 1.0    # fraction of total market duration remaining

    # Orderbook features
    yes_bid_depth: int = 0              # total YES bid quantity
    no_bid_depth: int = 0               # total NO bid quantity
    spread: int = 0                     # yes_ask - yes_bid approximation

    # Reference price (the "strike" — BTC price at market open)
    reference_price: Optional[float] = None
    price_vs_reference: float = 0.0     # (current - reference) / reference


@dataclass
class ModelPrediction:
    """Output of the probabilistic model."""
    prob_yes: float         # P(YES settles at 100)  in [0, 1]
    prob_no: float          # 1 - prob_yes
    confidence: float       # model's self-assessed confidence (informational)
    features: ModelFeatures
    model_name: str = "unknown"
    timestamp: float = field(default_factory=time.time)


@dataclass
class TradeDecision:
    """Final decision: what to do."""
    signal: Signal
    ticker: str
    side: str               # "yes" or "no"
    action: str             # "buy" or "sell"
    price_cents: int        # limit price
    size: int               # number of contracts
    edge: float             # model_prob - market_implied_prob
    model_prob: float
    market_prob: float
    reason: str = ""


# ── Model interface ──────────────────────────────────────────────────

class ModelInterface(ABC):
    """Abstract interface for probabilistic settlement models."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def predict(self, features: ModelFeatures) -> float:
        """Return P(YES) given features. Must be in [0, 1]."""
        ...


class BaselineHeuristicModel(ModelInterface):
    """
    Simple heuristic model — no training data required.

    Logic:
      - If BTC is above the reference price AND has positive momentum,
        P(YES) is higher (price went up).
      - Adjusts for time remaining: less time → more certainty.
      - Adjusts for volatility: more vol → less certainty.

    This is a STARTING POINT. Replace with a trained model for real edge.
    """

    @property
    def name(self) -> str:
        return "baseline_heuristic_v1"

    def predict(self, features: ModelFeatures) -> float:
        # Base probability from price position relative to reference
        if features.reference_price is None or features.reference_price == 0:
            return 0.50  # no reference → no information

        pct_move = features.price_vs_reference

        # Sigmoid-like mapping: pct_move → probability
        # A 0.1% move in 15 minutes is meaningful for BTC
        # Scale factor controls sensitivity
        sensitivity = 500.0  # tunable
        raw_prob = 1.0 / (1.0 + math.exp(-sensitivity * pct_move))

        # Time adjustment: as time runs out, the current state is more likely final
        time_factor = 1.0 - features.time_remaining_frac  # 0 at start, 1 at end
        # Blend toward raw_prob as time passes
        prob = 0.5 + (raw_prob - 0.5) * (0.3 + 0.7 * time_factor)

        # Volatility dampening: high vol → regress toward 0.5
        if features.btc_volatility_5m > 0:
            vol_dampen = min(features.btc_volatility_5m * 100, 0.3)
            prob = prob * (1 - vol_dampen) + 0.5 * vol_dampen

        # Momentum bonus
        momentum = features.btc_return_1m
        prob += momentum * 50  # small nudge

        # Clamp to valid range
        prob = max(0.01, min(0.99, prob))
        return prob


# ── Feature extraction ───────────────────────────────────────────────

class FeatureExtractor:
    """Builds ModelFeatures from market data + BTC price history."""

    def __init__(self):
        self._price_history: List[Tuple[float, float]] = []  # (timestamp, price)
        self._max_history_sec = 600  # keep 10 minutes

    def record_price(self, price: float, timestamp: Optional[float] = None):
        ts = timestamp or time.time()
        self._price_history.append((ts, price))
        # Trim old data
        cutoff = ts - self._max_history_sec
        self._price_history = [
            (t, p) for t, p in self._price_history if t >= cutoff
        ]

    def extract(
        self,
        btc_price: AggregatedPrice,
        market: Market,
        orderbook: OrderBook,
        reference_price: Optional[float] = None,
    ) -> ModelFeatures:
        now = time.time()
        price = btc_price.price
        self.record_price(price, btc_price.timestamp)

        # Compute returns
        ret_1m = self._return_over(60)
        ret_5m = self._return_over(300)
        vol_5m = self._volatility(300, 60)

        # Time remaining
        from datetime import datetime, timezone
        try:
            close_dt = datetime.fromisoformat(market.close_time.replace("Z", "+00:00"))
            open_dt = datetime.fromisoformat(market.open_time.replace("Z", "+00:00"))
            now_dt = datetime.now(timezone.utc)
            remaining = max(0, (close_dt - now_dt).total_seconds())
            total_duration = max(1, (close_dt - open_dt).total_seconds())
            time_frac = remaining / total_duration
        except Exception:
            remaining = 900.0
            time_frac = 1.0

        # Orderbook depth
        yes_depth = sum(l.quantity for l in orderbook.yes_bids)
        no_depth = sum(l.quantity for l in orderbook.no_bids)

        # Price vs reference
        ref = reference_price
        pvr = 0.0
        if ref and ref > 0:
            pvr = (price - ref) / ref

        # Spread approximation (YES best bid vs implied ask from NO best bid)
        spread = 0
        if orderbook.yes_bids and orderbook.no_bids:
            best_yes_bid = max(l.price for l in orderbook.yes_bids)
            best_no_bid = max(l.price for l in orderbook.no_bids)
            yes_ask = 100 - best_no_bid
            spread = max(0, yes_ask - best_yes_bid)

        return ModelFeatures(
            btc_price=price,
            btc_return_1m=ret_1m,
            btc_return_5m=ret_5m,
            btc_volatility_5m=vol_5m,
            market_yes_price=market.yes_price,
            market_no_price=market.no_price,
            time_remaining_sec=remaining,
            time_remaining_frac=time_frac,
            yes_bid_depth=yes_depth,
            no_bid_depth=no_depth,
            spread=spread,
            reference_price=ref,
            price_vs_reference=pvr,
        )

    def _return_over(self, seconds: float) -> float:
        if len(self._price_history) < 2:
            return 0.0
        now = self._price_history[-1]
        target_time = now[0] - seconds
        # Find closest price to target_time
        past = min(self._price_history, key=lambda x: abs(x[0] - target_time))
        if past[1] == 0:
            return 0.0
        return (now[1] - past[1]) / past[1]

    def _volatility(self, window_sec: float, interval_sec: float) -> float:
        """Realized volatility: std of returns over window."""
        if len(self._price_history) < 3:
            return 0.0
        now_ts = self._price_history[-1][0]
        cutoff = now_ts - window_sec
        window = [(t, p) for t, p in self._price_history if t >= cutoff]
        if len(window) < 3:
            return 0.0

        # Sample at interval_sec intervals
        returns = []
        i = 0
        while i + 1 < len(window):
            j = i + 1
            while j < len(window) and (window[j][0] - window[i][0]) < interval_sec:
                j += 1
            if j < len(window) and window[i][1] > 0:
                ret = (window[j][1] - window[i][1]) / window[i][1]
                returns.append(ret)
            i = j

        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(var)


# ── Decision engine ──────────────────────────────────────────────────

class StrategyEngine:
    """
    Combines model predictions with market state to produce trade decisions.
    """

    def __init__(self, config: StrategyConfig, model: Optional[ModelInterface] = None):
        self._cfg = config
        self._model = model or BaselineHeuristicModel()
        self._feature_extractor = FeatureExtractor()
        self._reference_prices: Dict[str, float] = {}  # ticker → BTC ref price

    @property
    def model(self) -> ModelInterface:
        return self._model

    def set_model(self, model: ModelInterface):
        """Hot-swap the prediction model."""
        logger.info("Model swapped: %s → %s", self._model.name, model.name)
        self._model = model

    def record_tick(self, price: float, timestamp: Optional[float] = None):
        """Feed BTC price ticks for feature computation."""
        self._feature_extractor.record_price(price, timestamp)

    def set_reference_price(self, ticker: str, ref_price: float):
        """Set the BTC reference price for a market (price at market open)."""
        self._reference_prices[ticker] = ref_price

    def evaluate(
        self,
        market: Market,
        orderbook: OrderBook,
        btc_price: AggregatedPrice,
        current_position: int = 0,
    ) -> Optional[TradeDecision]:
        """
        Run the model and decide whether to trade.

        Returns a TradeDecision or None if HOLD.
        """
        ref = self._reference_prices.get(market.ticker)
        features = self._feature_extractor.extract(btc_price, market, orderbook, ref)

        # Get model prediction
        prob_yes = self._model.predict(features)
        prob_yes = max(0.01, min(0.99, prob_yes))

        prediction = ModelPrediction(
            prob_yes=prob_yes,
            prob_no=1.0 - prob_yes,
            confidence=abs(prob_yes - 0.5) * 2,
            features=features,
            model_name=self._model.name,
        )

        # Market implied probability
        market_yes_prob = market.yes_price / 100.0

        # Calculate edge
        edge_yes = prob_yes - market_yes_prob
        edge_no = (1.0 - prob_yes) - (market.no_price / 100.0)

        logger.debug(
            "[%s] Model P(YES)=%.3f, Market YES=%d¢, edge_yes=%.3f, edge_no=%.3f",
            market.ticker, prob_yes, market.yes_price, edge_yes, edge_no,
        )

        # Decision logic
        min_edge = self._cfg.min_edge

        # If we have no position, look for entry
        if current_position == 0:
            if edge_yes >= min_edge and market.yes_price <= self._cfg.entry_price_cents:
                return TradeDecision(
                    signal=Signal.BUY_YES,
                    ticker=market.ticker,
                    side="yes",
                    action="buy",
                    price_cents=market.yes_price,
                    size=self._cfg.order_size,
                    edge=edge_yes,
                    model_prob=prob_yes,
                    market_prob=market_yes_prob,
                    reason=f"Model P(YES)={prob_yes:.3f} > market {market_yes_prob:.2f} + min_edge {min_edge}",
                )
            elif edge_no >= min_edge and market.no_price <= (100 - self._cfg.exit_price_cents):
                return TradeDecision(
                    signal=Signal.BUY_NO,
                    ticker=market.ticker,
                    side="no",
                    action="buy",
                    price_cents=market.no_price,
                    size=self._cfg.order_size,
                    edge=edge_no,
                    model_prob=1.0 - prob_yes,
                    market_prob=market.no_price / 100.0,
                    reason=f"Model P(NO)={1-prob_yes:.3f} > market {market.no_price/100:.2f} + min_edge",
                )

        # If we have a YES position, look for exit
        elif current_position > 0:
            if market.yes_price >= self._cfg.exit_price_cents:
                return TradeDecision(
                    signal=Signal.SELL_YES,
                    ticker=market.ticker,
                    side="yes",
                    action="sell",
                    price_cents=self._cfg.exit_price_cents,
                    size=min(current_position, self._cfg.order_size),
                    edge=edge_yes,
                    model_prob=prob_yes,
                    market_prob=market_yes_prob,
                    reason=f"Exit target hit: market YES={market.yes_price}¢ >= {self._cfg.exit_price_cents}¢",
                )

        # If we have a NO position, look for exit
        elif current_position < 0:
            if market.no_price >= self._cfg.exit_price_cents:
                return TradeDecision(
                    signal=Signal.SELL_NO,
                    ticker=market.ticker,
                    side="no",
                    action="sell",
                    price_cents=self._cfg.exit_price_cents,
                    size=min(abs(current_position), self._cfg.order_size),
                    edge=edge_no,
                    model_prob=1.0 - prob_yes,
                    market_prob=market.no_price / 100.0,
                    reason="Exit target hit for NO position",
                )

        return None  # HOLD


# ── Calibration metrics ──────────────────────────────────────────────

def brier_score(predictions: List[float], outcomes: List[int]) -> float:
    """
    Brier Score = (1/N) * Σ (forecast_i - outcome_i)²
    Lower is better. 0 = perfect, 0.25 = coin flip.
    """
    if not predictions:
        return 0.0
    n = len(predictions)
    return sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / n


def log_loss_score(predictions: List[float], outcomes: List[int]) -> float:
    """
    Log loss = -(1/N) * Σ [y*log(p) + (1-y)*log(1-p)]
    Lower is better. Penalizes confident wrong predictions heavily.
    """
    if not predictions:
        return 0.0
    eps = 1e-7
    n = len(predictions)
    total = 0.0
    for p, y in zip(predictions, outcomes):
        p_clamped = max(eps, min(1 - eps, p))
        total += y * math.log(p_clamped) + (1 - y) * math.log(1 - p_clamped)
    return -total / n


def hit_rate(predictions: List[float], outcomes: List[int], threshold: float = 0.5) -> float:
    """Simple accuracy: fraction of correct binary predictions."""
    if not predictions:
        return 0.0
    correct = sum(
        1 for p, o in zip(predictions, outcomes)
        if (p >= threshold) == (o == 1)
    )
    return correct / len(predictions)
