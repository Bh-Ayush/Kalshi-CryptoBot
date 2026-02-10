"""
Execution module — order lifecycle management.

Handles:
  - Placing entry limit orders with idempotency (client_order_id)
  - Managing exit orders (software-managed OCO-like behavior)
  - Cancel/replace logic within rate limits
  - Partial fill tracking
  - DRY_RUN mode (log decisions, simulate fills)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from config import AppConfig
from kalshi_client import KalshiClient, KalshiAPIError, Order
from strategy import TradeDecision, Signal

logger = logging.getLogger("execution")


class OrderState(Enum):
    PENDING = "pending"
    RESTING = "resting"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass
class ManagedOrder:
    """Internal order tracking with metadata."""
    client_order_id: str
    server_order_id: Optional[str] = None
    ticker: str = ""
    side: str = ""
    action: str = ""
    price_cents: int = 0
    requested_size: int = 0
    filled_size: int = 0
    remaining_size: int = 0
    state: OrderState = OrderState.PENDING
    is_entry: bool = True  # True=entry order, False=exit order
    linked_exit_id: Optional[str] = None  # client_order_id of paired exit
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    decision: Optional[TradeDecision] = None

    @property
    def is_active(self) -> bool:
        return self.state in (OrderState.PENDING, OrderState.RESTING, OrderState.PARTIALLY_FILLED)


class ExecutionManager:
    """
    Manages the full order lifecycle: placement, tracking, exit, cancellation.
    """

    def __init__(self, client: KalshiClient, config: AppConfig):
        self._client = client
        self._config = config
        self._orders: Dict[str, ManagedOrder] = {}  # client_order_id → ManagedOrder
        self._dry_run = config.dry_run

    # ── Public interface ─────────────────────────────────────────────

    async def execute_decision(self, decision: TradeDecision) -> Optional[ManagedOrder]:
        """
        Execute a trade decision by placing a limit order.
        Returns the ManagedOrder if successful.
        """
        if decision.signal == Signal.HOLD:
            return None

        # Check for duplicate orders (idempotency)
        for order in self._orders.values():
            if (
                order.is_active
                and order.ticker == decision.ticker
                and order.side == decision.side
                and order.action == decision.action
            ):
                logger.info(
                    "Skipping duplicate order: already have active %s %s %s on %s",
                    order.action, order.side, order.state.value, order.ticker,
                )
                return None

        client_order_id = str(uuid.uuid4())
        managed = ManagedOrder(
            client_order_id=client_order_id,
            ticker=decision.ticker,
            side=decision.side,
            action=decision.action,
            price_cents=decision.price_cents,
            requested_size=decision.size,
            remaining_size=decision.size,
            is_entry=(decision.action == "buy"),
            decision=decision,
        )

        if self._dry_run:
            return await self._simulate_order(managed)

        try:
            # Determine price field based on side
            kwargs = {}
            if decision.side == "yes":
                kwargs["yes_price"] = decision.price_cents
            else:
                kwargs["no_price"] = decision.price_cents

            order = await self._client.create_order(
                ticker=decision.ticker,
                side=decision.side,
                action=decision.action,
                count=decision.size,
                order_type="limit",
                client_order_id=client_order_id,
                **kwargs,
            )

            managed.server_order_id = order.order_id
            managed.state = OrderState.RESTING
            managed.filled_size = order.fill_count
            managed.remaining_size = order.remaining_count
            if managed.filled_size > 0 and managed.remaining_size > 0:
                managed.state = OrderState.PARTIALLY_FILLED
            elif managed.remaining_size == 0:
                managed.state = OrderState.FILLED

            self._orders[client_order_id] = managed
            logger.info(
                "ORDER PLACED: %s %s %s %d@%d¢ on %s (id=%s, state=%s)",
                decision.action, decision.side, decision.size,
                decision.price_cents, decision.price_cents,
                decision.ticker, order.order_id, managed.state.value,
            )

            # If entry was immediately filled, place exit order
            if managed.is_entry and managed.state == OrderState.FILLED:
                await self._place_exit_order(managed)

            return managed

        except KalshiAPIError as e:
            managed.state = OrderState.REJECTED
            self._orders[client_order_id] = managed
            logger.error(
                "ORDER REJECTED: %s %s on %s: %s",
                decision.action, decision.side, decision.ticker, e,
            )
            return managed

    async def check_and_manage_orders(self):
        """
        Poll active orders, update state, handle partial fills, place exits.
        Called periodically by the main loop.
        """
        active_orders = [o for o in self._orders.values() if o.is_active]

        for managed in active_orders:
            if self._dry_run:
                continue

            if managed.server_order_id is None:
                continue

            try:
                order = await self._client.get_order(managed.server_order_id)
                prev_state = managed.state

                # Update state
                managed.filled_size = order.fill_count
                managed.remaining_size = order.remaining_count
                managed.updated_at = time.time()

                if order.status == "canceled":
                    managed.state = OrderState.CANCELED
                elif order.remaining_count == 0:
                    managed.state = OrderState.FILLED
                elif order.fill_count > 0:
                    managed.state = OrderState.PARTIALLY_FILLED
                else:
                    managed.state = OrderState.RESTING

                if managed.state != prev_state:
                    logger.info(
                        "ORDER STATE CHANGE: %s → %s (id=%s, filled=%d/%d)",
                        prev_state.value, managed.state.value,
                        managed.server_order_id, managed.filled_size, managed.requested_size,
                    )

                # If entry just filled, place exit
                if (
                    managed.is_entry
                    and managed.state == OrderState.FILLED
                    and managed.linked_exit_id is None
                ):
                    await self._place_exit_order(managed)

            except KalshiAPIError as e:
                logger.error(
                    "Error polling order %s: %s", managed.server_order_id, e
                )

    async def cancel_all_orders(self, ticker: Optional[str] = None):
        """Cancel all active orders, optionally filtered by ticker."""
        active = [
            o for o in self._orders.values()
            if o.is_active and (ticker is None or o.ticker == ticker)
        ]
        for managed in active:
            await self._cancel_order(managed)

    async def cancel_order(self, client_order_id: str) -> bool:
        """Cancel a specific order by client_order_id."""
        managed = self._orders.get(client_order_id)
        if managed and managed.is_active:
            return await self._cancel_order(managed)
        return False

    async def amend_order_price(
        self, client_order_id: str, new_price_cents: int
    ) -> bool:
        """Amend an active order's price (cancel/replace)."""
        managed = self._orders.get(client_order_id)
        if not managed or not managed.is_active or not managed.server_order_id:
            return False

        if self._dry_run:
            logger.info(
                "[DRY_RUN] AMEND: %s from %d¢ to %d¢",
                managed.server_order_id, managed.price_cents, new_price_cents,
            )
            managed.price_cents = new_price_cents
            return True

        try:
            kwargs = {}
            if managed.side == "yes":
                kwargs["yes_price"] = new_price_cents
            else:
                kwargs["no_price"] = new_price_cents

            await self._client.amend_order(managed.server_order_id, **kwargs)
            managed.price_cents = new_price_cents
            managed.updated_at = time.time()
            logger.info(
                "ORDER AMENDED: %s price → %d¢", managed.server_order_id, new_price_cents
            )
            return True
        except KalshiAPIError as e:
            logger.error("Amend failed for %s: %s", managed.server_order_id, e)
            return False

    # ── Internal helpers ─────────────────────────────────────────────

    async def _place_exit_order(self, entry: ManagedOrder):
        """Place a software-managed exit order after an entry fills."""
        exit_price = self._config.strategy.exit_price_cents
        exit_action = "sell"  # sell to close

        exit_cid = str(uuid.uuid4())
        exit_managed = ManagedOrder(
            client_order_id=exit_cid,
            ticker=entry.ticker,
            side=entry.side,
            action=exit_action,
            price_cents=exit_price,
            requested_size=entry.filled_size,
            remaining_size=entry.filled_size,
            is_entry=False,
        )

        entry.linked_exit_id = exit_cid

        if self._dry_run:
            exit_managed.state = OrderState.RESTING
            self._orders[exit_cid] = exit_managed
            logger.info(
                "[DRY_RUN] EXIT ORDER: sell %s %d@%d¢ on %s",
                entry.side, entry.filled_size, exit_price, entry.ticker,
            )
            return

        try:
            kwargs = {}
            if entry.side == "yes":
                kwargs["yes_price"] = exit_price
            else:
                kwargs["no_price"] = exit_price

            order = await self._client.create_order(
                ticker=entry.ticker,
                side=entry.side,
                action=exit_action,
                count=entry.filled_size,
                order_type="limit",
                client_order_id=exit_cid,
                **kwargs,
            )
            exit_managed.server_order_id = order.order_id
            exit_managed.state = OrderState.RESTING
            exit_managed.filled_size = order.fill_count
            exit_managed.remaining_size = order.remaining_count

            self._orders[exit_cid] = exit_managed
            logger.info(
                "EXIT ORDER PLACED: sell %s %d@%d¢ on %s (id=%s)",
                entry.side, entry.filled_size, exit_price,
                entry.ticker, order.order_id,
            )
        except KalshiAPIError as e:
            logger.error("Failed to place exit for %s: %s", entry.ticker, e)

    async def _cancel_order(self, managed: ManagedOrder) -> bool:
        if self._dry_run:
            managed.state = OrderState.CANCELED
            managed.updated_at = time.time()
            logger.info("[DRY_RUN] CANCELED: %s", managed.client_order_id)
            return True

        if managed.server_order_id is None:
            managed.state = OrderState.CANCELED
            return True

        try:
            await self._client.cancel_order(managed.server_order_id)
            managed.state = OrderState.CANCELED
            managed.updated_at = time.time()
            logger.info("ORDER CANCELED: %s", managed.server_order_id)
            return True
        except KalshiAPIError as e:
            logger.error("Cancel failed for %s: %s", managed.server_order_id, e)
            return False

    async def _simulate_order(self, managed: ManagedOrder) -> ManagedOrder:
        """DRY_RUN mode: simulate order placement and immediate fill."""
        managed.state = OrderState.RESTING
        managed.server_order_id = f"DRY-{managed.client_order_id[:8]}"
        self._orders[managed.client_order_id] = managed

        logger.info(
            "[DRY_RUN] ORDER: %s %s %d@%d¢ on %s (edge=%.3f, model_p=%.3f)",
            managed.action, managed.side, managed.requested_size,
            managed.price_cents, managed.ticker,
            managed.decision.edge if managed.decision else 0,
            managed.decision.model_prob if managed.decision else 0,
        )
        return managed

    # ── Reporting ────────────────────────────────────────────────────

    def get_active_orders(self) -> List[ManagedOrder]:
        return [o for o in self._orders.values() if o.is_active]

    def get_all_orders(self) -> List[ManagedOrder]:
        return list(self._orders.values())

    def get_order_stats(self) -> Dict:
        orders = list(self._orders.values())
        return {
            "total": len(orders),
            "active": sum(1 for o in orders if o.is_active),
            "filled": sum(1 for o in orders if o.state == OrderState.FILLED),
            "canceled": sum(1 for o in orders if o.state == OrderState.CANCELED),
            "rejected": sum(1 for o in orders if o.state == OrderState.REJECTED),
            "total_filled_contracts": sum(o.filled_size for o in orders),
        }
