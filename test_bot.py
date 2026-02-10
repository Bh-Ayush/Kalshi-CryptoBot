"""
Unit tests for risk and execution modules.

Run: pytest tests/ -v
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def risk_config():
    """Minimal RiskConfig for testing."""
    from config import RiskConfig
    return RiskConfig(
        max_order_size=25,
        max_position=100,
        max_open_orders=10,
        max_daily_loss_cents=5000,
        close_cutoff_sec=60,
        kill_switch=False,
    )


@pytest.fixture
def risk_manager(risk_config):
    from risk import RiskManager
    return RiskManager(risk_config)


@pytest.fixture
def mock_market():
    from kalshi_client import Market
    close_time = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    return Market(
        ticker="KXBTC15M-TEST001",
        title="Test BTC Up/Down",
        event_ticker="EVT-TEST",
        status="open",
        yes_price=50,
        no_price=50,
        volume=100,
        close_time=close_time,
        open_time=(datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
    )


@pytest.fixture
def market_near_close():
    """Market that closes in 30 seconds (within cutoff)."""
    from kalshi_client import Market
    close_time = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
    return Market(
        ticker="KXBTC15M-CLOSE001",
        title="Closing soon",
        event_ticker="EVT-CLOSE",
        status="open",
        yes_price=50,
        no_price=50,
        volume=100,
        close_time=close_time,
        open_time=(datetime.now(timezone.utc) - timedelta(minutes=14)).isoformat(),
    )


# ══════════════════════════════════════════════════════════════════════
#  RISK MODULE TESTS
# ══════════════════════════════════════════════════════════════════════

class TestRiskManagerBasicChecks:
    """Test basic order validation."""

    def test_order_allowed_basic(self, risk_manager, mock_market):
        allowed, reason = risk_manager.check_order_allowed(
            ticker="KXBTC15M-TEST001",
            side="yes",
            action="buy",
            size=5,
            market=mock_market,
        )
        assert allowed is True
        assert reason == "OK"

    def test_order_size_exceeded(self, risk_manager, mock_market):
        allowed, reason = risk_manager.check_order_allowed(
            ticker="KXBTC15M-TEST001",
            side="yes",
            action="buy",
            size=30,  # exceeds max_order_size=25
            market=mock_market,
        )
        assert allowed is False
        assert "Order size" in reason

    def test_position_limit(self, risk_manager, mock_market):
        # Set a large existing position
        risk_manager.update_position("KXBTC15M-TEST001", 95)
        allowed, reason = risk_manager.check_order_allowed(
            ticker="KXBTC15M-TEST001",
            side="yes",
            action="buy",
            size=10,  # would make position 105 > max 100
            market=mock_market,
        )
        assert allowed is False
        assert "Position limit" in reason

    def test_max_open_orders(self, risk_manager, mock_market):
        risk_manager.update_open_order_count(10)  # at max
        allowed, reason = risk_manager.check_order_allowed(
            ticker="KXBTC15M-TEST001",
            side="yes",
            action="buy",
            size=5,
            market=mock_market,
        )
        assert allowed is False
        assert "Max open orders" in reason

    def test_sell_bypasses_open_order_limit(self, risk_manager, mock_market):
        """Sell orders should not be blocked by open order count."""
        risk_manager.update_open_order_count(10)
        allowed, reason = risk_manager.check_order_allowed(
            ticker="KXBTC15M-TEST001",
            side="yes",
            action="sell",
            size=5,
            market=mock_market,
        )
        assert allowed is True


class TestRiskManagerKillSwitch:
    """Test kill switch triggering and behavior."""

    def test_manual_kill_switch(self, risk_manager, mock_market):
        risk_manager.activate_kill_switch("Manual test")
        assert risk_manager.is_kill_switch_active is True

        allowed, reason = risk_manager.check_order_allowed(
            ticker="KXBTC15M-TEST001",
            side="yes",
            action="buy",
            size=5,
            market=mock_market,
        )
        assert allowed is False
        assert "Kill switch" in reason

    def test_kill_switch_deactivation(self, risk_manager, mock_market):
        risk_manager.activate_kill_switch("Test")
        risk_manager.deactivate_kill_switch()
        assert risk_manager.is_kill_switch_active is False

        allowed, _ = risk_manager.check_order_allowed(
            ticker="KXBTC15M-TEST001",
            side="yes",
            action="buy",
            size=5,
            market=mock_market,
        )
        assert allowed is True

    def test_daily_loss_triggers_kill_switch(self, risk_manager, mock_market):
        # Record a huge loss
        risk_manager.record_pnl(-5001)  # exceeds max_daily_loss_cents=5000
        assert risk_manager.is_kill_switch_active is True

        allowed, reason = risk_manager.check_order_allowed(
            ticker="KXBTC15M-TEST001",
            side="yes",
            action="buy",
            size=5,
            market=mock_market,
        )
        assert allowed is False


class TestRiskManagerTimeCutoff:
    """Test time-based cutoff near market close."""

    def test_cutoff_blocks_new_orders(self, risk_manager, market_near_close):
        """Orders should be blocked within 60s of market close."""
        allowed, reason = risk_manager.check_order_allowed(
            ticker=market_near_close.ticker,
            side="yes",
            action="buy",
            size=5,
            market=market_near_close,
        )
        assert allowed is False
        assert "close" in reason.lower()

    def test_no_cutoff_when_time_remains(self, risk_manager, mock_market):
        """Orders should be allowed when market close is >60s away."""
        allowed, _ = risk_manager.check_order_allowed(
            ticker=mock_market.ticker,
            side="yes",
            action="buy",
            size=5,
            market=mock_market,
        )
        assert allowed is True


class TestRiskManagerFillTracking:
    """Test position tracking after fills."""

    def test_fill_updates_position(self, risk_manager):
        risk_manager.record_fill("KXBTC15M-TEST", "yes", "buy", 10, 45)
        assert risk_manager.get_position("KXBTC15M-TEST") == 10

        risk_manager.record_fill("KXBTC15M-TEST", "yes", "sell", 5, 65)
        assert risk_manager.get_position("KXBTC15M-TEST") == 5

    def test_no_side_position(self, risk_manager):
        risk_manager.record_fill("KXBTC15M-TEST", "no", "buy", 10, 55)
        assert risk_manager.get_position("KXBTC15M-TEST") == -10

    def test_total_position(self, risk_manager):
        risk_manager.record_fill("KXBTC15M-A", "yes", "buy", 10, 45)
        risk_manager.record_fill("KXBTC15M-B", "no", "buy", 20, 55)
        assert risk_manager.get_total_position() == 30  # abs(10) + abs(-20)


# ══════════════════════════════════════════════════════════════════════
#  EXECUTION MODULE TESTS
# ══════════════════════════════════════════════════════════════════════

class TestExecutionIdempotency:
    """Test that duplicate decisions don't create duplicate orders."""

    @pytest.fixture
    def app_config(self):
        from config import AppConfig, KalshiConfig, StrategyConfig
        # Use a dummy config for dry run
        with patch.dict("os.environ", {
            "KALSHI_API_KEY_ID": "test-key",
            "KALSHI_PRIVATE_KEY_PATH": "/dev/null",
        }):
            return AppConfig(
                dry_run=True,
                kalshi=KalshiConfig(
                    base_url="https://demo-api.kalshi.co/trade-api/v2",
                    ws_url="wss://demo-api.kalshi.co",
                    api_key_id="test-key",
                    private_key_path="/dev/null",
                ),
            )

    @pytest.fixture
    def execution_manager(self, app_config):
        from execution import ExecutionManager
        mock_client = MagicMock()
        return ExecutionManager(mock_client, app_config)

    @pytest.mark.asyncio
    async def test_duplicate_prevention(self, execution_manager):
        from strategy import TradeDecision, Signal

        decision = TradeDecision(
            signal=Signal.BUY_YES,
            ticker="KXBTC15M-TEST001",
            side="yes",
            action="buy",
            price_cents=45,
            size=5,
            edge=0.10,
            model_prob=0.60,
            market_prob=0.45,
        )

        # First execution should succeed
        result1 = await execution_manager.execute_decision(decision)
        assert result1 is not None

        # Second identical execution should be skipped
        result2 = await execution_manager.execute_decision(decision)
        assert result2 is None

    @pytest.mark.asyncio
    async def test_hold_returns_none(self, execution_manager):
        from strategy import TradeDecision, Signal

        decision = TradeDecision(
            signal=Signal.HOLD,
            ticker="KXBTC15M-TEST001",
            side="yes",
            action="buy",
            price_cents=45,
            size=5,
            edge=0.0,
            model_prob=0.50,
            market_prob=0.45,
        )

        result = await execution_manager.execute_decision(decision)
        assert result is None


class TestExecutionOrderStates:
    """Test order state transitions."""

    @pytest.fixture
    def app_config(self):
        from config import AppConfig, KalshiConfig
        with patch.dict("os.environ", {
            "KALSHI_API_KEY_ID": "test-key",
            "KALSHI_PRIVATE_KEY_PATH": "/dev/null",
        }):
            return AppConfig(
                dry_run=True,
                kalshi=KalshiConfig(
                    base_url="https://demo-api.kalshi.co/trade-api/v2",
                    ws_url="wss://demo-api.kalshi.co",
                    api_key_id="test-key",
                    private_key_path="/dev/null",
                ),
            )

    @pytest.fixture
    def execution_manager(self, app_config):
        from execution import ExecutionManager
        mock_client = MagicMock()
        return ExecutionManager(mock_client, app_config)

    @pytest.mark.asyncio
    async def test_dry_run_order_state(self, execution_manager):
        from execution import OrderState
        from strategy import TradeDecision, Signal

        decision = TradeDecision(
            signal=Signal.BUY_YES,
            ticker="KXBTC15M-DRY001",
            side="yes",
            action="buy",
            price_cents=45,
            size=5,
            edge=0.10,
            model_prob=0.60,
            market_prob=0.45,
        )

        result = await execution_manager.execute_decision(decision)
        assert result is not None
        assert result.state == OrderState.RESTING
        assert result.server_order_id.startswith("DRY-")

    @pytest.mark.asyncio
    async def test_cancel_order(self, execution_manager):
        from execution import OrderState
        from strategy import TradeDecision, Signal

        decision = TradeDecision(
            signal=Signal.BUY_YES,
            ticker="KXBTC15M-CANCEL001",
            side="yes",
            action="buy",
            price_cents=45,
            size=5,
            edge=0.10,
            model_prob=0.60,
            market_prob=0.45,
        )

        result = await execution_manager.execute_decision(decision)
        assert result is not None

        success = await execution_manager.cancel_order(result.client_order_id)
        assert success is True
        assert result.state == OrderState.CANCELED

    @pytest.mark.asyncio
    async def test_amend_price_dry_run(self, execution_manager):
        from strategy import TradeDecision, Signal

        decision = TradeDecision(
            signal=Signal.BUY_YES,
            ticker="KXBTC15M-AMEND001",
            side="yes",
            action="buy",
            price_cents=45,
            size=5,
            edge=0.10,
            model_prob=0.60,
            market_prob=0.45,
        )

        result = await execution_manager.execute_decision(decision)
        assert result is not None
        assert result.price_cents == 45

        success = await execution_manager.amend_order_price(
            result.client_order_id, 48
        )
        assert success is True
        assert result.price_cents == 48

    @pytest.mark.asyncio
    async def test_order_stats(self, execution_manager):
        from strategy import TradeDecision, Signal

        for i in range(3):
            decision = TradeDecision(
                signal=Signal.BUY_YES,
                ticker=f"KXBTC15M-STATS{i:03d}",
                side="yes",
                action="buy",
                price_cents=45,
                size=5,
                edge=0.10,
                model_prob=0.60,
                market_prob=0.45,
            )
            await execution_manager.execute_decision(decision)

        stats = execution_manager.get_order_stats()
        assert stats["total"] == 3
        assert stats["active"] == 3


# ══════════════════════════════════════════════════════════════════════
#  STRATEGY CALIBRATION TESTS
# ══════════════════════════════════════════════════════════════════════

class TestCalibrationMetrics:
    def test_brier_perfect(self):
        from strategy import brier_score
        assert brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == 0.0

    def test_brier_worst(self):
        from strategy import brier_score
        score = brier_score([0.0, 1.0], [1, 0])
        assert score == 1.0

    def test_brier_coin_flip(self):
        from strategy import brier_score
        score = brier_score([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0])
        assert abs(score - 0.25) < 0.001

    def test_hit_rate_perfect(self):
        from strategy import hit_rate
        assert hit_rate([0.9, 0.1, 0.8], [1, 0, 1]) == 1.0

    def test_hit_rate_zero(self):
        from strategy import hit_rate
        assert hit_rate([0.1, 0.9, 0.2], [1, 0, 1]) == 0.0

    def test_log_loss_empty(self):
        from strategy import log_loss_score
        assert log_loss_score([], []) == 0.0
