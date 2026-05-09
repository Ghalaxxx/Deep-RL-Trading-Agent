# Evaluation Methodology

The evaluation layer is built to answer a practical systems question: did the learned policy add value beyond simple, cheaper strategies after realistic frictions?

## Backtesting Flow

- Training, validation, and test windows are chronological.
- Walk-forward splits train on past windows and test on future windows with non-overlapping boundaries.
- Metrics are computed from the equity curve and realized trade segments.
- Benchmark reports cover Buy & Hold, random continuous actions, SMA crossover, and momentum.
- Replay export records each model decision so aggregate metrics can be inspected step by step.

## Cost Model

The RL environment applies:

- transaction cost: `0.1%` per position change
- slippage: `0.05%` per position change

Costs are applied when exposure changes, so high-turnover policies are penalized directly in the simulation. This is still a simplified execution model: it does not model intraday liquidity, partial fills, borrow fees, short rebates, market impact, latency, or exchange-specific order book depth.

SMA and momentum reports include the same combined `0.15%` position-change cost. Buy & Hold is represented as one open trade over the test period, so its trade-level win rate is intentionally shown as `N/A` in the product dashboard. Return, Sharpe, volatility, and drawdown are more meaningful for one-trade benchmarks.

The paper trading dashboard uses the same default assumptions: `0.1%` transaction cost and `0.05%` slippage per simulated fill. It should be read as a product/inference demonstration, not as a replacement for the formal backtester.

## Exposure And Leverage

The trading environment uses continuous target exposure in `[-1, 1]`:

- `-1` means fully short
- `0` means flat
- `+1` means fully long

The default setup does not use leverage above one-times notional exposure. This keeps early experiments focused on policy quality rather than leverage-amplified returns. Any future leverage support should add explicit margin, financing, borrow, and liquidation constraints before being used in reported results.

The paper engine currently caps target exposure at `75%` notional, does not enable leverage, and prevents long purchases that would require negative cash. Short exposure is simulated for product realism, but the current model does not include borrow fees, locate constraints, margin interest, or forced buy-ins.

## Risk Management Semantics

The dashboard risk panel is a paper-trading guardrail layer:

- daily loss limit: `-3%`
- max exposure: `75%`
- position limit: `100%`
- stop-loss threshold: `-8%`
- max drawdown guardrail: `-12%`

If a halt-level rule is breached, the paper engine flattens target exposure and the dashboard shows a halted risk state. This is a realistic control pattern for demos, but it is not a complete institutional risk system. A production system would require independent limit services, durable audit logs, exchange calendars, borrow/margin accounting, and approval workflows.

## Normalization Consistency

Stable-Baselines3 policies trained with `VecNormalize` expect normalized observations at inference time. The CLI evaluation and replay paths load `models/vecnormalize.pkl` when it exists, then run the model with frozen normalization statistics.

This prevents a common deployment bug: evaluating a normalized policy on raw observations and reporting invalid performance.

## Metrics

The backtester reports:

- annualized return and volatility
- Sharpe and Sortino ratios
- maximum drawdown and Calmar ratio
- win rate and profit factor
- average win/loss
- number of trades
- average holding period

Return and risk metrics come from the full equity curve. Trade metrics use realized trade segments, which makes turnover-heavy policies easier to diagnose.

Sharpe and Sortino are annualized from daily returns with a 252-period convention. They should be read together with maximum drawdown, volatility, turnover, and average holding period rather than treated as standalone proof of strategy quality.

## Product Dashboard Semantics

The dashboard has two runtime modes:

- `Demo mode`: uses local benchmark reports and cached data. It does not imply a trained PPO/SAC model or live trading.
- `Checkpoint-backed`: becomes available when trained model artifacts are present under `models/`.

Reward/training analytics are not simulated. If no checkpoint or training telemetry is available, the panel shows an awaiting-telemetry state instead of drawing artificial learning curves.

The strategy comparison heatmap follows the same rule: PPO/SAC cells show `Awaiting checkpoint` until local model artifacts and evaluation outputs exist. Baseline rows come from generated benchmark reports.

## Regime and Explainability Semantics

Market regime detection is heuristic, not predictive modeling. It combines moving-average slope, rolling volatility, price-vs-moving-average state, and drawdown behavior to produce labels such as trending up, trending down, sideways, volatility expansion, and volatility compression.

The action explanation panel is also heuristic unless a trained checkpoint is integrated. It summarizes momentum, RSI, volatility, risk status, and target exposure. This helps demo the decision interface while avoiding claims that a trained PPO/SAC policy produced the signal.

## Responsible Reporting

The project should make measured, engineering-grounded claims:

- Compare PPO/SAC against all simple baselines after costs.
- Do not tune hyperparameters on the 2024 test split.
- Do not use training reward as a substitute for portfolio metrics.
- Prefer multi-ticker and walk-forward results over a single cherry-picked run.
- Treat replay inspection as part of model validation, especially when a model has high Sharpe but unusual turnover or drawdown behavior.
