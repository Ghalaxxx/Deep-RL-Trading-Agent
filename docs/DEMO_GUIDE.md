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

The dashboard can be run before model training, but that mode is a benchmark/data demo. It should be described as an evaluation product shell, not as live trading or trained policy performance.

## 4. Train a Short Demo Model

For a quick smoke test:

```bash
python train.py --algorithm ppo --ticker AAPL --timesteps 10000 --reward sharpe --wandb-mode disabled
```

For a serious run, increase timesteps to `1000000`.

## 5. Evaluate and Compare

```bash
python train.py --compare --model-path models/best_model.zip --ticker AAPL
```

Use the benchmark table to discuss why an RL agent must beat simple baselines, not just produce a positive return.
If `models/vecnormalize.pkl` exists, the CLI uses it during evaluation so normalized training and normalized inference match.

## 6. Export Replay

```bash
python train.py --evaluate --model-path models/best_model.zip --ticker AAPL --export-replay reports/aapl_replay.csv
```

The replay CSV records each decision, action, position, portfolio value, return, reward, drawdown, and trading costs. It is useful for debugging and for explaining what the policy did during the test period.
