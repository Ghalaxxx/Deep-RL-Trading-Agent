# Product Dashboard

The dashboard turns the RL trading system into a deployable product surface. It is intentionally framed as an evaluation and trading-intelligence platform, not as a live broker or market-prediction product.

## Runtime Modes

- `Demo mode`: uses local parquet data and generated benchmark reports. This mode is useful for product demos and engineering review, but it is not trained PPO/SAC performance.
- `Checkpoint-backed`: activates when trained model artifacts exist under `models/`. Evaluation and replay paths use saved `VecNormalize` statistics when available.

## Backend

Run:

```bash
uvicorn backend.main:app --reload
```

Important endpoints:

- `GET /api/health`
- `GET /api/dashboard?ticker=AAPL`
- `GET /api/backtests`
- `GET /api/experiments`

The API serves cached market data, benchmark reports, model readiness, inference replay metadata, and validation guardrails.

## Frontend

Run:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Product Surfaces

- portfolio/evaluation dashboard
- equity curve comparison
- drawdown visualization
- PPO vs SAC readiness panels
- backtesting result table
- trade timeline
- experiment tracking overview
- inference replay panel
- validation guardrails

## Quant Product Semantics

- Metrics shown in demo mode come from benchmark reports, not trained RL claims.
- Reward/training analytics are not simulated; the panel stays in an awaiting-telemetry state until real logs exist.
- Buy & Hold trade win rate is shown as `N/A` because one open trade is not a meaningful trade-distribution statistic.
- The dashboard does not represent live trading, order routing, brokerage connectivity, or market data streaming.
