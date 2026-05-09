# Deep RL Trading Agent

Deep RL Trading Agent is a production-minded experimentation platform for training PPO and SAC policies on adjusted market data with a custom Gymnasium trading environment. The system learns continuous position sizing from market observations, portfolio state, transaction costs, slippage, and risk-adjusted rewards. No hardcoded trading rules are used by the RL policy.

The project is designed like an applied AI/quant engineering system: reproducible data preparation, leakage-safe feature construction, vectorized RL training, benchmark leaderboards, policy replay, live-feeling paper trading telemetry, and out-of-sample evaluation are all first-class workflows.

## Design

Figma FigJam generation is supported through the Figma plugin, but this session did not expose a plan key lookup tool. The same architecture design is included here and can be regenerated in FigJam once a Figma team/org plan key is available.

```mermaid
flowchart LR
  A["Market Data: yfinance OHLCV"] --> B["Preprocessor: TA indicators"]
  B --> C["Observation Builder: 30 day window"]
  C --> D["TradingEnv: Gymnasium MDP"]
  D --> E["Vectorized Envs: SubprocVecEnv / DummyVecEnv"]
  E --> F["RL Agents: PPO / SAC"]
  F --> G["Eval Callback: validation Sharpe"]
  G --> H["Artifacts: best model and VecNormalize stats"]
  F --> I["W&B Tracking"]
  H --> J["Walk-Forward Backtester"]
  L["Baselines: Buy & Hold, Random, SMA, Momentum"] --> J
  J --> K["Reports: metrics and plots"]
  K --> O["FastAPI Product API"]
  O --> P["React Quant Dashboard"]
  Q["Simulated live replay"] --> O
  O --> R["Paper trading + risk controls"]
  M["Optuna Hyperopt"] --> F
  N["Curriculum Scheduler"] --> E
```

## Features

- Custom `TradingEnv` with continuous actions from `-1` full short to `+1` full long.
- Observation space `(window_size, 18)` with normalized OHLCV, technical indicators, and portfolio state.
- Switchable reward functions: rolling Sharpe, rolling Sortino, and risk-adjusted PnL.
- Stable-Baselines3 PPO and SAC wrappers.
- yfinance data downloader with adjusted prices and parquet cache.
- Reproducible data manifest for all default tickers.
- Walk-forward backtesting with no future data in train/test windows.
- Benchmarks: Buy & Hold, random agent, SMA crossover, and momentum.
- Baseline leaderboard export across the full ticker universe.
- Policy replay export for decision-level audit trails.
- Lightweight action-sensitivity explainability by feature group.
- Optuna hyperparameter search and W&B experiment tracking.
- FastAPI + React product dashboard with WebSocket updates.
- Live-feeling market feed labeled as live, simulated replay, or cached demo mode.
- Paper trading engine with cash, exposure, entry price, realized/unrealized PnL, costs, slippage, and no broker execution.
- Risk management panel with drawdown, volatility, exposure limits, stop-loss threshold, risk status, and simulated kill switch.
- Heuristic market regime detection and action explainability when no trained PPO/SAC checkpoint is available.
- Strategy comparison heatmap that keeps PPO/SAC values empty until checkpoint-backed evaluations exist.

## Engineering Safeguards

- Indicator warm-up rows are dropped instead of backward-filled, preventing early observations from receiving future indicator values.
- Chronological train/validation/test splits keep 2024 reserved for final out-of-sample evaluation.
- Walk-forward splits use non-overlapping train/test boundaries.
- Policies trained with `VecNormalize` are evaluated and replayed with saved normalization statistics when `models/vecnormalize.pkl` is present.
- RL trades include transaction costs and slippage; SMA and momentum baselines also pay position-change costs in benchmark reports.
- Training reward is treated as an optimization signal, while final quality is judged with out-of-sample return, Sharpe, drawdown, trade statistics, and benchmark comparisons.
- Dashboard demo mode uses benchmark reports and cached market data; trained PPO/SAC performance appears only after checkpoint artifacts are available.
- Live-feeling dashboard updates are paper/demo telemetry only. The system does not connect to broker APIs and does not route real-money orders.
- Heuristic regime and action-explanation panels are labeled as heuristic when no trained checkpoint is loaded.

## Product Dashboard

The repository includes a FastAPI + React dashboard for a product-grade trading intelligence experience.

Run the API:

```bash
uvicorn backend.main:app --reload
```

Run the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

![Dashboard overview](docs/assets/dashboard-overview.png)

Mobile-responsive view:

![Dashboard mobile](docs/assets/dashboard-mobile.png)

The dashboard includes a live-status market feed, paper portfolio, equity curves, drawdown visualization, risk management guardrails, market regime detection, action explainability, PPO/SAC readiness, training session manager, strategy heatmap, experiment tracking, trade timeline, validation guardrails, and inference demo mode. In demo mode, panels are explicitly based on benchmark reports and local cached data rather than trained model claims.

The product layer is deliberately not a real-money trading system:

- `Simulated live replay` streams recent cached market bars through the same dashboard flow used for live-style updates.
- `Live market data` can be attempted with yfinance by setting `RL_TRADING_FEED_MODE=live`; cached replay remains the default for reproducible demos.
- `Paper trading simulation` applies transaction costs and slippage, tracks exposure and PnL, and blocks additional exposure when risk limits are breached.
- PPO/SAC performance panels show `Awaiting checkpoint` until trained model artifacts and evaluation outputs exist.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

Prepare and validate the dataset:

```bash
python train.py --prepare-data
```

Generate a 2024 baseline leaderboard:

```bash
python train.py --benchmark-data
```

Train PPO:

```bash
python train.py --algorithm ppo --ticker AAPL --timesteps 1000000 --reward sharpe
```

Train SAC with hyperparameter search:

```bash
python train.py --algorithm sac --ticker SPY --optimize --n-trials 50
```

Evaluate a trained model:

```bash
python train.py --evaluate --model-path models/best_ppo_AAPL.zip --ticker AAPL
```

Compare against benchmarks:

```bash
python train.py --compare --model-path models/best_ppo_AAPL.zip --ticker AAPL
```

Export a decision replay:

```bash
python train.py --evaluate --model-path models/best_ppo_AAPL.zip --ticker AAPL --export-replay reports/aapl_replay.csv
```

## Project Layout

```text
envs/                 Gymnasium environment, observations, rewards
agents/               PPO/SAC factories and benchmark strategies
training/             Trainer, curriculum scheduler, Optuna search
evaluation/           Metrics, walk-forward backtester, visualizations
data/                 yfinance downloader and preprocessing
config/               Environment, PPO, and SAC YAML configs
tests/                Focused unit tests for env and metrics
notebooks/            Analysis notebook placeholder
docs/                 Architecture, data card, and demo guide
```

## Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Data Card](docs/DATA_CARD.md)
- [Evaluation Methodology](docs/EVALUATION.md)
- [Product Dashboard](docs/DASHBOARD.md)
- [Demo Guide](docs/DEMO_GUIDE.md)

## Results Workflow

Use the generated baseline and evaluation reports as the source of truth for model claims. A trained PPO/SAC run should be reported only after it is evaluated out-of-sample and compared against simple baselines after costs.

| Strategy     | Annual Return | Sharpe | Max DD  | Win Rate |
|--------------|---------------|--------|---------|----------|
| PPO Agent    | Run evaluation | From report | From report | From report |
| SAC Agent    | Run evaluation | From report | From report | From report |
| Buy & Hold   | `reports/baseline_benchmarks.csv` | From report | From report | From report |
| SMA Crossover | `reports/baseline_benchmarks.csv` | From report | From report | From report |
| Random Agent | `reports/baseline_benchmarks.csv` | From report | From report | From report |

## Engineering Talking Points

- Reward shaping balances risk-adjusted returns, turnover penalties, and drawdown control without using handcrafted entry/exit rules.
- Evaluation avoids leakage by fitting on past windows and testing only on future windows.
- PPO provides a stable on-policy baseline; SAC adds an off-policy continuous-control comparison.
- Curriculum learning can start on lower-volatility regimes before exposing the policy to broader market stress.
- Replay exports make policy behavior inspectable at the decision level, which helps debug failure modes before trusting aggregate metrics.
