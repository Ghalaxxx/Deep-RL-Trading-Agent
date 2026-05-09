# Demo Guide

This guide gives you a clean path for presenting the system in an interview, investor-style demo, or portfolio walkthrough.

## 1. Prepare Data

```bash
python train.py --prepare-data
```

Show `reports/data_manifest.csv` to prove the dataset exists locally, contains all indicators, and has no NaNs.

## 2. Run Tests

```bash
pytest
```

The tests cover reward functions, metrics, Gymnasium environment compatibility, random-agent stepping, replay export, and action sensitivity.

## 3. Generate Baselines

Before training, generate a baseline leaderboard:

```bash
python train.py --benchmark-data
```

This creates `reports/baseline_benchmarks.csv`, which is a useful comparison table for judging whether PPO/SAC adds value beyond simple strategies.

The dashboard can be run before model training, but that mode is a benchmark/data demo. It should be described as an evaluation product shell, not as current-market execution or trained policy performance.

## 4. Run the Product Dashboard

Start the API:

```bash
uvicorn backend.main:app --reload
```

Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

Demo the product in this order:

1. Select a symbol and show the Historical Replay Engine. Call out the feed label, usually `Historical Market Stream`, and the source bar date from the 2024 out-of-sample test split. Optional yfinance delayed snapshot mode can be enabled with `RL_TRADING_FEED_MODE=snapshot`.
2. Show the paper portfolio panel: cash, exposure, PnL, equity curve, and paper execution assumptions.
3. Show risk management: drawdown, volatility, limits, risk status, and guardrail pass/fail states.
4. Show market regime and `Why this action?`, explicitly describing them as heuristic unless checkpoint inference is available.
5. Show the strategy heatmap and point out that PPO/SAC are `Awaiting checkpoint` until trained artifacts exist.
6. Show the training session manager and experiment tracking overview.

The dashboard uses replay WebSocket updates through `/ws/replay/{ticker}`. If WebSocket is unavailable, use `/api/replay/{ticker}` or `/api/dashboard` as a polling fallback.

## 5. Train a Short Demo Model

For a quick smoke test:

```bash
python train.py --algorithm ppo --ticker AAPL --timesteps 10000 --reward sharpe --wandb-mode disabled
```

For a serious run, increase timesteps to `1000000`.

## 6. Evaluate and Compare

```bash
python train.py --compare --model-path models/best_model.zip --ticker AAPL
```

Use the benchmark table to discuss why an RL agent must beat simple baselines, not just produce a positive return.
If `models/vecnormalize.pkl` exists, the CLI uses it during evaluation so normalized training and normalized inference match.

## 7. Export Replay

```bash
python train.py --evaluate --model-path models/best_model.zip --ticker AAPL --export-replay reports/aapl_replay.csv
```

The replay CSV records each decision, action, position, portfolio value, return, reward, drawdown, and trading costs. It is useful for debugging and for explaining what the policy did during the test period.

## What To Say Clearly

- The default market panel is historical replay of 2024 out-of-sample bars, not current market data.
- `Delayed Market Snapshot` mode is optional and should be described as experimental yfinance snapshot data, not broker-grade streaming.
- The portfolio panel is paper trading only.
- There is no broker connection and no real-money execution.
- PPO/SAC performance should not be claimed until checkpoint-backed out-of-sample evaluation exists.
- Risk and regime panels are product guardrails/explainability aids, not profitability guarantees.
