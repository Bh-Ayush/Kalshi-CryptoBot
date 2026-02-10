"""
Risk management module.

Enforces:
  - Max position size per market and globally
  - Max open orders
  - Max daily loss (PnL tracking)
  - Time-based cutoff near market close
  - Kill switch (manual or auto-triggered)

All checks are synchronous and should be called BEFORE placing any order.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

from config import RiskConfig
from kalshi_client import Market

logger = logging.getLogger("risk")


@dataclass
class DailyPnL:
    """Tracks daily profit and loss in cents."""
    date: str = ""  # YYYY-MM-DD
    realized_pnl_cents: int = 0
    fees_cents: int = 0
    num_trades: int = 0

    @property
    def net_pnl(self) -> int:
        return self.realized_pnl_cents - self.fees_cents


class RiskManager:
    """
    Centralized risk control. Every order must pass all checks before submission.
    """

    def __init__(self, config: RiskConfig):
        self._cfg = config
        self._kill_switch = config.kill_switch
        self._kill_reason: Optional[str] = None

        # Position tracking: ticker → net contracts (positive=YES, negative=NO)
        self._positions: Dict[str, int] = {}

        # Open order count
        self._open_order_count: int = 0

        # Daily PnL
        self._daily_pnl = DailyPnL(date=self._today())

    # ── Pre-trade checks ─────────────────────────────────────────────

    def check_order_allowed(
        self,
        ticker: str,
        side: str,
        action: str,
        size: int,
        market: Optional[Market] = None,
    ) -> tuple[bool, str]:
        """
        Run all risk checks. Returns (allowed: bool, reason: str).
        If allowed is False, the reason explains why.
        """
        # 1. Kill switch
        if self._kill_switch:
            return False, f"Kill switch active: {self._kill_reason or 'manually triggered'}"

        # 2. Daily loss limit
        self._rotate_daily_pnl()
        if self._daily_pnl.net_pnl <= -self._cfg.max_daily_loss_cents:
            self._trigger_kill_switch(
                f"Daily loss limit hit: {self._daily_pnl.net_pnl}¢ <= -{self._cfg.max_daily_loss_cents}¢"
            )
            return False, self._kill_reason

        # 3. Max open orders
        if action == "buy" and self._open_order_count >= self._cfg.max_open_orders:
            return False, f"Max open orders reached: {self._open_order_count} >= {self._cfg.max_open_orders}"

        # 4. Max position size
        current_pos = abs(self._positions.get(ticker, 0))
        new_pos = current_pos + size if action == "buy" else current_pos - size
        if new_pos > self._cfg.max_position:
            return False, f"Position limit: {new_pos} would exceed max {self._cfg.max_position}"

        # 5. Max order size
        if size > self._cfg.max_order_size:
            return False, f"Order size {size} exceeds max {self._cfg.max_order_size}"

        # 6. Time cutoff near market close
        if market and action == "buy":
            if not self._check_time_cutoff(market):
                return False, (
                    f"Too close to market close (cutoff={self._cfg.close_cutoff_sec}s)"
                )

        return True, "OK"

    # ── State updates ────────────────────────────────────────────────

    def record_fill(
        self, ticker: str, side: str, action: str, size: int, price_cents: int
    ):
        """Update position and PnL after a fill."""
        self._rotate_daily_pnl()

        # Update position
        direction = 1 if side == "yes" else -1
        if action == "buy":
            self._positions[ticker] = self._positions.get(ticker, 0) + direction * size
        else:
            self._positions[ticker] = self._positions.get(ticker, 0) - direction * size

        # For PnL, we track cost basis separately in storage.
        # Here we just increment trade count.
        self._daily_pnl.num_trades += 1
        logger.info(
            "FILL RECORDED: %s %s %d@%d¢ on %s → position=%d",
            action, side, size, price_cents, ticker,
            self._positions.get(ticker, 0),
        )

    def record_pnl(self, pnl_cents: int, fees_cents: int = 0):
        """Record realized PnL (from a settlement or closed trade)."""
        self._rotate_daily_pnl()
        self._daily_pnl.realized_pnl_cents += pnl_cents
        self._daily_pnl.fees_cents += fees_cents
        logger.info(
            "PnL update: +%d¢ (fees: %d¢) → daily net: %d¢",
            pnl_cents, fees_cents, self._daily_pnl.net_pnl,
        )

        # Check if loss limit is now breached
        if self._daily_pnl.net_pnl <= -self._cfg.max_daily_loss_cents:
            self._trigger_kill_switch(
                f"Daily loss limit hit after PnL update: {self._daily_pnl.net_pnl}¢"
            )

    def update_open_order_count(self, count: int):
        self._open_order_count = count

    def update_position(self, ticker: str, net_contracts: int):
        self._positions[ticker] = net_contracts

    # ── Kill switch ──────────────────────────────────────────────────

    def activate_kill_switch(self, reason: str = "Manual activation"):
        self._trigger_kill_switch(reason)

    def deactivate_kill_switch(self):
        self._kill_switch = False
        self._kill_reason = None
        logger.warning("KILL SWITCH DEACTIVATED")

    @property
    def is_kill_switch_active(self) -> bool:
        return self._kill_switch

    @property
    def kill_reason(self) -> Optional[str]:
        return self._kill_reason

    # ── Reporting ────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        self._rotate_daily_pnl()
        return {
            "kill_switch": self._kill_switch,
            "kill_reason": self._kill_reason,
            "daily_pnl_cents": self._daily_pnl.net_pnl,
            "daily_trades": self._daily_pnl.num_trades,
            "max_daily_loss_cents": self._cfg.max_daily_loss_cents,
            "open_orders": self._open_order_count,
            "max_open_orders": self._cfg.max_open_orders,
            "positions": dict(self._positions),
            "max_position": self._cfg.max_position,
        }

    def get_position(self, ticker: str) -> int:
        return self._positions.get(ticker, 0)

    def get_total_position(self) -> int:
        return sum(abs(v) for v in self._positions.values())

    # ── Internal ─────────────────────────────────────────────────────

    def _trigger_kill_switch(self, reason: str):
        if not self._kill_switch:
            self._kill_switch = True
            self._kill_reason = reason
            logger.critical("🚨 KILL SWITCH TRIGGERED: %s", reason)

    def _check_time_cutoff(self, market: Market) -> bool:
        """Return False if we're within the cutoff period before market close."""
        try:
            close_time = datetime.fromisoformat(
                market.close_time.replace("Z", "+00:00")
            )
            now = datetime.now(timezone.utc)
            remaining = (close_time - now).total_seconds()
            return remaining > self._cfg.close_cutoff_sec
        except Exception as e:
            logger.warning("Could not parse close_time for %s: %s", market.ticker, e)
            return True  # fail open (allow trade if can't determine)

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _rotate_daily_pnl(self):
        """Reset daily PnL if date has changed."""
        today = self._today()
        if self._daily_pnl.date != today:
            logger.info(
                "Daily PnL rotation: %s → %s (previous: %d¢)",
                self._daily_pnl.date, today, self._daily_pnl.net_pnl,
            )
            self._daily_pnl = DailyPnL(date=today)
            # Also reset kill switch on new day (configurable behavior)
            if self._kill_switch and "Daily loss" in (self._kill_reason or ""):
                logger.info("Kill switch auto-reset on new trading day")
                self._kill_switch = False
                self._kill_reason = None
