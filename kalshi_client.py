"""
Kalshi API client — authentication (RSA-PSS signing), REST calls, and WebSocket.

References:
  - Auth / API Keys: https://docs.kalshi.com/getting_started/api_keys
  - REST endpoints: https://docs.kalshi.com/api-reference/
  - WebSocket:      https://docs.kalshi.com/websockets/websocket-connection
  - Rate limits:    https://docs.kalshi.com/getting_started/rate_limits
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from config import KalshiConfig

logger = logging.getLogger("kalshi_client")

# ── Data classes for API responses ───────────────────────────────────

@dataclass
class Market:
    ticker: str
    title: str
    event_ticker: str
    status: str
    yes_price: int  # cents 0-100
    no_price: int
    volume: int
    close_time: str  # ISO-8601
    open_time: str
    result: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class OrderBookLevel:
    price: int  # cents
    quantity: int


@dataclass
class OrderBook:
    yes_bids: List[OrderBookLevel]
    no_bids: List[OrderBookLevel]
    timestamp: float = field(default_factory=time.time)


@dataclass
class Order:
    order_id: str
    client_order_id: str
    ticker: str
    side: str  # "yes" | "no"
    action: str  # "buy" | "sell"
    type: str  # "limit" | "market"
    status: str  # "resting" | "canceled" | "executed" | "pending"
    yes_price: int
    no_price: int
    initial_count: int
    remaining_count: int
    fill_count: int
    created_time: str
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class Fill:
    trade_id: str
    order_id: str
    ticker: str
    side: str
    action: str
    yes_price: int
    no_price: int
    count: int
    created_time: str
    is_taker: bool = False
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class Position:
    ticker: str
    market_exposure: int  # net contracts (positive=YES, negative=NO)
    realized_pnl: int  # cents
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class Balance:
    available_balance: int  # cents
    portfolio_value: int  # cents


# ── Rate limiter ─────────────────────────────────────────────────────

class AsyncRateLimiter:
    """Token-bucket rate limiter for async contexts."""

    def __init__(self, calls_per_second: int):
        self._rate = calls_per_second
        self._sem = asyncio.Semaphore(calls_per_second)
        self._refill_task: Optional[asyncio.Task] = None

    async def start(self):
        self._refill_task = asyncio.create_task(self._refill())

    async def _refill(self):
        while True:
            await asyncio.sleep(1.0)
            # Refill tokens up to max
            for _ in range(self._rate - self._sem._value):
                self._sem.release()

    async def acquire(self):
        await self._sem.acquire()

    def stop(self):
        if self._refill_task:
            self._refill_task.cancel()


# ── Main client ──────────────────────────────────────────────────────

class KalshiClient:
    """Async Kalshi API client with auth, rate limiting, and retries."""

    def __init__(self, config: KalshiConfig):
        self._cfg = config
        self._base_url = config.base_url.rstrip("/")
        self._ws_url = config.ws_url.rstrip("/")
        self._api_key_id = config.api_key_id
        self._private_key = self._load_private_key(config.private_key_path)
        self._session: Optional[aiohttp.ClientSession] = None
        self._read_limiter = AsyncRateLimiter(config.read_rate_limit)
        self._write_limiter = AsyncRateLimiter(config.write_rate_limit)

    # ── Key loading & signing ────────────────────────────────────────

    @staticmethod
    def _load_private_key(path: str) -> rsa.RSAPrivateKey:
        with open(path, "rb") as f:
            return serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        """Create RSA-PSS signature: sign(timestamp_ms + method + path_without_query)."""
        path_clean = path.split("?")[0]
        message = (timestamp_ms + method + path_clean).encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        import base64
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(self, method: str, path: str) -> Dict[str, str]:
        ts = str(int(time.time() * 1000))
        sig = self._sign(ts, method, path)
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
        }

    # ── Session lifecycle ────────────────────────────────────────────

    async def start(self):
        self._session = aiohttp.ClientSession()
        await self._read_limiter.start()
        await self._write_limiter.start()
        logger.info("KalshiClient started (base_url=%s)", self._base_url)

    async def close(self):
        self._read_limiter.stop()
        self._write_limiter.stop()
        if self._session:
            await self._session.close()
        logger.info("KalshiClient closed")

    # ── Generic HTTP helpers ─────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict] = None,
        params: Optional[Dict] = None,
        is_write: bool = False,
        retries: int = 3,
    ) -> Dict[str, Any]:
        limiter = self._write_limiter if is_write else self._read_limiter
        url = f"{self._base_url}{path}"
        headers = self._auth_headers(method.upper(), f"/trade-api/v2{path}")

        for attempt in range(1, retries + 1):
            await limiter.acquire()
            try:
                async with self._session.request(
                    method, url, headers=headers, json=json_body, params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 429:
                        wait = min(2 ** attempt, 8)
                        logger.warning(
                            "Rate limited (429) on %s %s, retry in %ss", method, path, wait
                        )
                        await asyncio.sleep(wait)
                        continue
                    if resp.status >= 500:
                        wait = min(2 ** attempt, 8)
                        logger.warning(
                            "Server error %s on %s %s, retry in %ss",
                            resp.status, method, path, wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    body = await resp.json()
                    if resp.status >= 400:
                        logger.error(
                            "API error %s on %s %s: %s",
                            resp.status, method, path, body,
                        )
                        raise KalshiAPIError(resp.status, body)
                    return body
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == retries:
                    raise
                wait = min(2 ** attempt, 8)
                logger.warning(
                    "Network error on %s %s (%s), retry in %ss",
                    method, path, exc, wait,
                )
                await asyncio.sleep(wait)

        raise KalshiAPIError(0, {"error": "Max retries exceeded"})

    async def _get(self, path: str, **kwargs) -> Dict:
        return await self._request("GET", path, **kwargs)

    async def _post(self, path: str, json_body: Dict, **kwargs) -> Dict:
        return await self._request("POST", path, json_body=json_body, is_write=True, **kwargs)

    async def _put(self, path: str, json_body: Dict, **kwargs) -> Dict:
        return await self._request("PUT", path, json_body=json_body, is_write=True, **kwargs)

    async def _delete(self, path: str, **kwargs) -> Dict:
        return await self._request("DELETE", path, is_write=True, **kwargs)

    # ── Market data ──────────────────────────────────────────────────

    async def get_markets(
        self,
        series_ticker: Optional[str] = None,
        status: str = "open",
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> List[Market]:
        """Fetch markets, optionally filtered by series ticker."""
        params = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        data = await self._get("/markets", params=params)
        markets = []
        for m in data.get("markets", []):
            markets.append(
                Market(
                    ticker=m["ticker"],
                    title=m.get("title", ""),
                    event_ticker=m.get("event_ticker", ""),
                    status=m.get("status", ""),
                    yes_price=m.get("yes_price", 0),
                    no_price=m.get("no_price", 0),
                    volume=m.get("volume", 0),
                    close_time=m.get("close_time", ""),
                    open_time=m.get("open_time", ""),
                    result=m.get("result"),
                    raw=m,
                )
            )
        return markets

    async def get_market(self, ticker: str) -> Market:
        data = await self._get(f"/markets/{ticker}")
        m = data["market"]
        return Market(
            ticker=m["ticker"],
            title=m.get("title", ""),
            event_ticker=m.get("event_ticker", ""),
            status=m.get("status", ""),
            yes_price=m.get("yes_price", 0),
            no_price=m.get("no_price", 0),
            volume=m.get("volume", 0),
            close_time=m.get("close_time", ""),
            open_time=m.get("open_time", ""),
            result=m.get("result"),
            raw=m,
        )

    async def get_orderbook(self, ticker: str) -> OrderBook:
        """Get the current order book for a market.

        Kalshi returns only bids for both YES and NO sides.
        In binary markets, a YES bid at X cents = NO ask at (100-X) cents.
        """
        data = await self._get(f"/markets/{ticker}/orderbook")
        ob = data.get("orderbook", {})

        yes_bids = [
            OrderBookLevel(price=level[0], quantity=level[1])
            for level in ob.get("yes", [])
            if len(level) >= 2
        ]
        no_bids = [
            OrderBookLevel(price=level[0], quantity=level[1])
            for level in ob.get("no", [])
            if len(level) >= 2
        ]
        return OrderBook(yes_bids=yes_bids, no_bids=no_bids)

    # ── Order management ─────────────────────────────────────────────

    async def create_order(
        self,
        ticker: str,
        side: str,
        action: str,
        count: int,
        *,
        order_type: str = "limit",
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        client_order_id: Optional[str] = None,
        time_in_force: Optional[str] = None,
        expiration_ts: Optional[int] = None,
        post_only: bool = False,
    ) -> Order:
        """Place an order on Kalshi.

        Args:
            ticker: Market ticker (e.g., "KXBTC15M-26FEB071500")
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            order_type: "limit" or "market"
            yes_price: Price in cents (1-99) for YES side
            no_price: Price in cents (1-99) for NO side
            client_order_id: Idempotency key (UUID recommended)
            time_in_force: "fill_or_kill" or None (GTC default)
            expiration_ts: Unix timestamp for order expiry
            post_only: If True, order rejected if it would cross
        """
        if client_order_id is None:
            client_order_id = str(uuid.uuid4())

        body: Dict[str, Any] = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": order_type,
            "client_order_id": client_order_id,
        }
        if yes_price is not None:
            if not (1 <= yes_price <= 99):
                raise ValueError(f"yes_price must be 1–99, got {yes_price}")
            body["yes_price"] = yes_price
        if no_price is not None:
            if not (1 <= no_price <= 99):
                raise ValueError(f"no_price must be 1–99, got {no_price}")
            body["no_price"] = no_price
        if time_in_force:
            body["time_in_force"] = time_in_force
        if expiration_ts:
            body["expiration_ts"] = expiration_ts
        if post_only:
            body["post_only"] = True

        data = await self._post("/portfolio/orders", body)
        o = data["order"]
        return self._parse_order(o)

    async def get_order(self, order_id: str) -> Order:
        data = await self._get(f"/portfolio/orders/{order_id}")
        return self._parse_order(data["order"])

    async def cancel_order(self, order_id: str) -> Order:
        data = await self._delete(f"/portfolio/orders/{order_id}")
        return self._parse_order(data["order"])

    async def amend_order(
        self, order_id: str, *, yes_price: Optional[int] = None, no_price: Optional[int] = None,
        count: Optional[int] = None,
    ) -> Order:
        body: Dict[str, Any] = {}
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        if count is not None:
            body["count"] = count
        data = await self._put(f"/portfolio/orders/{order_id}", body)
        return self._parse_order(data["order"])

    async def get_orders(
        self, ticker: Optional[str] = None, status: Optional[str] = None
    ) -> List[Order]:
        params: Dict[str, Any] = {}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        data = await self._get("/portfolio/orders", params=params)
        return [self._parse_order(o) for o in data.get("orders", [])]

    # ── Fills / Positions / Balance ──────────────────────────────────

    async def get_fills(
        self, ticker: Optional[str] = None, limit: int = 100
    ) -> List[Fill]:
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        data = await self._get("/portfolio/fills", params=params)
        fills = []
        for f in data.get("fills", []):
            fills.append(
                Fill(
                    trade_id=f.get("trade_id", ""),
                    order_id=f.get("order_id", ""),
                    ticker=f.get("ticker", ""),
                    side=f.get("side", ""),
                    action=f.get("action", ""),
                    yes_price=f.get("yes_price", 0),
                    no_price=f.get("no_price", 0),
                    count=f.get("count", 0),
                    created_time=f.get("created_time", ""),
                    is_taker=f.get("is_taker", False),
                    raw=f,
                )
            )
        return fills

    async def get_positions(
        self, ticker: Optional[str] = None, settlement_status: str = "unsettled"
    ) -> List[Position]:
        params: Dict[str, Any] = {"settlement_status": settlement_status}
        if ticker:
            params["ticker"] = ticker
        data = await self._get("/portfolio/positions", params=params)
        positions = []
        for p in data.get("market_positions", []):
            positions.append(
                Position(
                    ticker=p.get("ticker", ""),
                    market_exposure=p.get("market_exposure", 0),
                    realized_pnl=p.get("realized_pnl", 0),
                    raw=p,
                )
            )
        return positions

    async def get_balance(self) -> Balance:
        data = await self._get("/portfolio/balance")
        return Balance(
            available_balance=data.get("available_balance", 0),
            portfolio_value=data.get("portfolio_value", 0),
        )

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_order(o: Dict) -> Order:
        return Order(
            order_id=o.get("order_id", ""),
            client_order_id=o.get("client_order_id", ""),
            ticker=o.get("ticker", ""),
            side=o.get("side", ""),
            action=o.get("action", ""),
            type=o.get("type", ""),
            status=o.get("status", ""),
            yes_price=o.get("yes_price", 0),
            no_price=o.get("no_price", 0),
            initial_count=o.get("initial_count", 0),
            remaining_count=o.get("remaining_count", 0),
            fill_count=o.get("fill_count", 0),
            created_time=o.get("created_time", ""),
            raw=o,
        )

    @staticmethod
    def generate_client_order_id() -> str:
        return str(uuid.uuid4())


class KalshiAPIError(Exception):
    def __init__(self, status: int, body: Any):
        self.status = status
        self.body = body
        super().__init__(f"Kalshi API error {status}: {body}")
