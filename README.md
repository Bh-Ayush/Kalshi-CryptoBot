# Kalshi BTC 15-Minute Trading Bot

An automated trading bot for Kalshi's BTC "Up or Down — 15 minutes" prediction markets. Built with async Python, multi-exchange price feeds, and strict risk controls.

> **⚠️ DISCLAIMER:** This bot trades with real money. There is no guarantee of profit. Event markets have edge only when your model is well-calibrated AND the market price is mispriced relative to your forecast. Start with DRY_RUN=true and paper trade extensively before risking capital.

---

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                     main.py (Orchestrator)               │
│  - Async event loop                                      │
│  - Market discovery (KXBTC15M series)                    │
│  - Coordinates all modules                               │
└────┬──────────┬──────────┬──────────┬──────────┬────────┘
     │          │          │          │          │
┌────▼───┐ ┌───▼────┐ ┌───▼────┐ ┌───▼───┐ ┌───▼────┐
│ Price  │ │Strategy│ │Execut- │ │ Risk  │ │Storage │
│ Feed   │ │ Engine │ │ ion    │ │Manager│ │(SQLite)│
│        │ │        │ │        │ │       │ │        │
│Coinbase│ │Model   │ │Limit   │ │Max pos│ │Ticks   │
│Binance │ │→Predict│ │Orders  │ │Max    │ │Orders  │
│Kraken  │ │→Signal │ │Cancel/ │ │loss   │ │Fills   │
│        │ │→Decide │ │Replace │ │Kill   │ │PnL     │
│Median  │ │        │ │Exit    │ │switch │ │Snapshots│
│Price   │ │Features│ │Mgmt    │ │Time   │ │        │
└────────┘ └────────┘ └────────┘ │cutoff │ └────────┘
                                  └───────┘
```

### Module Responsibilities

| Module | File | Purpose |
|--------|------|---------|
| **Config** | `config.py` | All settings from env vars. Never hard-codes secrets. |
| **Kalshi Client** | `kalshi_client.py` | REST API client with RSA-PSS auth, rate limiting, retries. |
| **Price Feed** | `price_feed.py` | WebSocket connections to 3 exchanges, median price aggregation. |
| **Strategy** | `strategy.py` | Probabilistic model interface, feature extraction, decision logic. |
| **Execution** | `execution.py` | Order placement, cancel/replace, exit management, dry-run simulation. |
| **Risk** | `risk.py` | Position limits, daily loss limits, kill switch, time cutoffs. |
| **Storage** | `storage.py` | SQLite persistence for ticks, predictions, orders, fills, PnL. |
| **Replay** | `replay.py` | Backtest engine with Brier score, log loss, hit rate, simulated PnL. |
| **Main** | `main.py` | Entry point — wires everything together, runs the async loop. |

---

## Quick Start

### 1. Install

```bash
# Clone the repository
git clone <your-repo-url>
cd kalshi-btc-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the template
cp .env.example .env

# Edit with your Kalshi API credentials
# CRITICAL: Set your KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH
nano .env
```

**Getting Kalshi API keys:**
1. Log into [kalshi.com](https://kalshi.com)
2. Go to Settings → API
3. Create a new API key pair
4. Download the private key `.pem` file
5. Set `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH` in `.env`

### 3. Run in Dry-Run Mode (Recommended First)

```bash
# DRY_RUN=true is the default — no real orders placed
python main.py
```

The bot will:
- Connect to Coinbase, Binance, and Kraken for BTC prices
- Discover active KXBTC15M markets on Kalshi
- Run the strategy and LOG decisions without placing real orders
- Record ticks, predictions, and simulated orders to SQLite

### 4. Run Live

```bash
# Edit .env and set:
# DRY_RUN=false

python main.py
```

### 5. Run Tests

```bash
pytest tests/ -v
```

### 6. Run Replay / Backtest

```bash
# Uses recorded data from SQLite (or synthetic data if empty)
python replay.py --since 2025-01-01 --until 2025-02-01
```

---

## Safety and Operations

### Paper / Demo Mode

Kalshi provides a demo environment. To use it:

```bash
# In .env:
KALSHI_BASE_URL=https://demo-api.kalshi.co/trade-api/v2
KALSHI_WS_URL=wss://demo-api.kalshi.co
DRY_RUN=false  # safe to set false in demo
```

### Risk Controls

| Control | Default | Config Key |
|---------|---------|------------|
| Max order size | 25 contracts | `RISK_MAX_ORDER_SIZE` |
| Max position | 100 contracts | `RISK_MAX_POSITION` |
| Max open orders | 10 | `RISK_MAX_OPEN_ORDERS` |
| Max daily loss | $50 (5000¢) | `RISK_MAX_DAILY_LOSS_CENTS` |
| Close cutoff | 60 seconds | `RISK_CLOSE_CUTOFF_SEC` |
| Kill switch | Off | `RISK_KILL_SWITCH` |

The **kill switch** triggers automatically when daily loss exceeds the limit. It cancels all open orders and prevents new ones. It auto-resets at midnight UTC.

### Monitoring

**Log format:**
```
2025-02-07T14:30:00 | INFO     | main             | STATUS | BTC=$97000.50 (3 sources) | Markets=1 | Orders: 2 active, 5 filled | PnL=150¢ | Kill=False
```

**Key metrics to watch:**
- `BTC price` and `num_sources` — if sources drop to 0, the bot pauses
- `daily_pnl_cents` — approaching max_daily_loss triggers kill switch
- `active orders` — should stay below `RISK_MAX_OPEN_ORDERS`
- `Kill switch` — if True, investigate immediately

**Alert conditions (check bot.log):**
- `KILL SWITCH TRIGGERED` — daily loss limit or manual activation
- `Authentication failure` — API key issue, bot stops
- `Rate limited (429)` — hitting Kalshi rate limits, auto-retries
- `Insufficient price feeds` — exchange connections down

### Failure Modes

| Failure | Bot Behavior |
|---------|-------------|
| **Price feed stale** | Stops trading until feeds recover. Logs warning. |
| **Auth failure (401)** | Stops bot immediately. Check API keys. |
| **Rate limit (429)** | Exponential backoff retry (2s, 4s, 8s). |
| **Kalshi 500 error** | Retry with backoff, up to 3 attempts. |
| **Clock drift** | Uses API timestamps for auth signing. Ensure NTP sync. |
| **Network disconnect** | WebSocket auto-reconnect with 2s delay. REST retries. |
| **Partial fill** | Tracks remaining_count, adjusts exit order size. |

---

## Strategy: How It Works

### The Probabilistic Model

The bot uses a **pluggable model interface**. The default `BaselineHeuristicModel`:

1. Computes BTC price relative to the reference (strike) price
2. Applies a sigmoid function for P(YES) = P(BTC ends above reference)
3. Adjusts for time remaining (more certainty as expiry approaches)
4. Dampens for realized volatility (more vol → less certainty)
5. Adds a momentum factor from recent returns

### Why "Accuracy" Alone Is Not Enough

In event markets, your edge = `model_probability - market_price/100`.

- If the model says P(YES)=0.60 and the market asks 0.45¢, edge = +15¢/contract
- If the model says P(YES)=0.60 and the market asks 0.65¢, edge = -5¢/contract (do NOT trade)
- A model with 70% accuracy that's overconfident (predicts 0.95 when truth is 0.70) will **lose money** because it overpays

**Calibration** means: when you say 60%, it happens 60% of the time. Measured by **Brier score** (lower is better, 0.25 = coin flip).

### Plugging In a Trained Model

```python
from strategy import ModelInterface, ModelFeatures

class MyTrainedModel(ModelInterface):
    @property
    def name(self) -> str:
        return "gradient_boosting_v2"

    def predict(self, features: ModelFeatures) -> float:
        # Your trained model here
        # Must return P(YES) in [0, 1]
        X = [features.btc_return_1m, features.btc_volatility_5m,
             features.time_remaining_frac, features.price_vs_reference]
        return self.model.predict_proba([X])[0][1]

# In main.py or via config:
strategy.set_model(MyTrainedModel())
```

---

## Repo Structure

```
kalshi-btc-bot/
├── config.py            # Environment-based configuration
├── kalshi_client.py     # Kalshi REST API client (auth, orders, markets)
├── price_feed.py        # Multi-exchange BTC WebSocket feeds
├── strategy.py          # Model interface + baseline + decision logic
├── execution.py         # Order lifecycle management
├── risk.py              # Risk controls + kill switch
├── storage.py           # SQLite persistence
├── main.py              # Entry point
├── replay.py            # Backtest / replay engine
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore           # Excludes .env, keys, DB, caches
├── bot.log              # Runtime log (created on first run)
├── kalshi_bot.db        # SQLite database (created on first run)
└── tests/
    ├── conftest.py      # Pytest config
    └── test_bot.py      # Risk + execution + calibration tests
```

---

## Next Steps to Improve Performance Responsibly

### 1. Data Collection
- Run in DRY_RUN mode for 1–2 weeks to collect BTC ticks, market snapshots, and settlement outcomes
- Record reference prices at market open (the "strike" for Up/Down)
- Build a labeled dataset: features at decision time → settlement outcome

### 2. Model Training & Calibration
- Train a gradient boosting model (LightGBM/XGBoost) on collected features
- Apply **isotonic regression** or **Platt scaling** for calibration
- Validate with time-series cross-validation (no look-ahead bias)
- Target Brier score < 0.20 before deploying live

### 3. Slippage Analysis
- Compare intended fill price vs actual fill price
- Measure queue position and time-to-fill
- Consider `post_only=True` for maker orders (avoid taker fees)
- Analyze how many orders get partially filled or miss entirely

### 4. Fee-Aware Edge
- Kalshi charges fees that eat into your edge
- Ensure net edge after fees is positive: `edge - fees > 0`
- Track `taker_fees` vs `maker_fees` from fill data

### 5. Multi-Market Expansion
- Once calibrated on BTC 15-min, extend to ETH, SOL variants
- Same infrastructure, different series tickers
- Model may need retraining per asset (different vol profiles)

### 6. Production Hardening
- Deploy on a cloud VM with monitoring (Grafana/Prometheus)
- Add Slack/Discord webhook alerts for kill switch and errors
- Implement circuit breakers for sustained API failures
- Set up daily PnL reports emailed automatically
