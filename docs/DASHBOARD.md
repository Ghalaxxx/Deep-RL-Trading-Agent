# Product Dashboard

The dashboard turns the RL trading system into a deployable product surface. It is intentionally framed as an evaluation and trading-intelligence platform, not as a broker or market-prediction product.

## Runtime Modes

- `Demo mode`: uses local parquet data and generated benchmark reports. This mode is useful for product demos and engineering review, but it is not trained PPO/SAC performance.
- `Checkpoint-backed`: activates when trained model artifacts exist under `models/`. Evaluation and replay paths use saved `VecNormalize` statistics when available.

Feed labels are separate from runtime mode:

- `Historical Market Stream`: cached 2024 out-of-sample bars replayed through the market replay engine.
- `Cached Demo Mode`: static local cache/report state.
- `Delayed Market Snapshot`: optional yfinance snapshot mode, enabled with `RL_TRADING_FEED_MODE=snapshot`. `RL_TRADING_FEED_MODE=live` is accepted as a compatibility alias, but the UI still labels it as delayed snapshot mode.

## Backend

Run:

```bash
uvicorn backend.main:app --reload
```

Important endpoints:

- `GET /api/health`
- `GET /api/dashboard?ticker=AAPL`
- `GET /api/replay/AAPL?cursor=80`
- `GET /api/backtests`
- `GET /api/experiments`
- `WS /ws/replay/AAPL`

The API serves cached market data, benchmark reports, model readiness, paper trading state, risk state, regime detection, action explanations, inference replay metadata, and validation guardrails.

WebSocket events:

- `MARKET_REPLAY_UPDATE`
- `PAPER_TRADE_UPDATE`
- `RISK_UPDATE`
- `REGIME_UPDATE`
- `AGENT_SIGNAL`
- `EXPERIMENT_UPDATE`

The frontend subscribes to the replay WebSocket for smooth panel updates. If the socket is unavailable, the documented fallback is polling `/api/dashboard` or `/api/replay`.

To attempt delayed yfinance snapshots before falling back to replay:

```bash
$env:RL_TRADING_FEED_MODE="snapshot"
uvicorn backend.main:app --reload
```

The UI still displays the active feed mode. If yfinance is unavailable or rate-limited, the product should be run in the default historical replay mode.

## Frontend

Run:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Product Surfaces

- Historical Replay Engine panel with selected symbol, 2024 source bar, and replay update timestamp
- paper portfolio panel with cash, position, exposure, entry price, realized/unrealized PnL, total equity, and current action
- risk management panel with guardrails, drawdown, volatility, risk status, and simulated kill switch
- heuristic market regime card
- action explainability card
- portfolio/evaluation dashboard
- equity curve comparison
- drawdown visualization
- PPO vs SAC readiness panels
- strategy comparison heatmap
- backtesting result table
- paper trade timeline
- training session manager
- experiment tracking overview
- inference replay panel
- validation guardrails

## Quant Product Semantics

- Metrics shown in demo mode come from benchmark reports, not trained RL claims.
- Reward/training analytics are not simulated; the panel stays in an awaiting-telemetry state until real logs exist.
- PPO and SAC heatmap values stay empty until checkpoint-backed evaluations exist.
- Paper portfolio state is simulated and never routed to a broker.
- Risk rules can block additional paper exposure, but they are demo guardrails rather than a complete production risk engine.
- Market regime and action explanation panels are heuristic unless checkpoint inference is integrated.
- Buy & Hold trade win rate is shown as `N/A` because one open trade is not a meaningful trade-distribution statistic.
- The dashboard does not represent broker execution, order routing, brokerage connectivity, or unlabeled market data streaming.

## Financial Assumptions

- transaction cost: `0.1%`
- slippage: `0.05%`
- paper max exposure: `75%`
- leverage: disabled
- daily loss guardrail: `-3%`
- stop-loss guardrail: `-8%`
- max drawdown guardrail: `-12%`

These assumptions are intentionally visible in the UI so product demos remain realistic and inspectable.

## Future Deployment Placeholder

Deployment work is intentionally out of scope right now. A future production plan would separate market-data adapters, paper/broker execution adapters, model registry, experiment store, authentication, audit logging, and observability.
