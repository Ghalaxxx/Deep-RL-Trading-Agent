# Research Notes

This project should be presented as an RL engineering system, not as financial advice or a claim of market edge. The most important research risks are data leakage, unstable reward optimization, benchmark validity, and overfitting to one historical period.

## Leakage Controls

- Technical indicators are not backward-filled. Warm-up rows are dropped so early observations do not receive future indicator values.
- Static train/validation/test splits are chronological: train through 2022, validation in 2023, and test in 2024.
- Walk-forward splits use non-overlapping train/test boundaries.
- Evaluation can load saved `VecNormalize` statistics through `models/vecnormalize.pkl`; this matters because policies trained with normalized observations should not be evaluated on raw observations.

## Reward Design

The environment supports three reward families:

- `sharpe`: rolling 30-step Sharpe, clipped to control reward explosions.
- `sortino`: rolling downside-risk-adjusted reward.
- `pnl`: step return with turnover and drawdown penalties.

Sharpe and Sortino rewards can be noisy over short windows. They are useful for shaping behavior but should not be the only evidence of a good strategy. Final reporting should use out-of-sample equity curves and benchmark comparisons.

## Backtesting Assumptions

- RL environment trades include transaction costs and slippage.
- SMA and momentum baselines also pay a simple 0.15% round-trip-style position-change cost in the benchmark report.
- Buy & Hold is treated as one open trade across the test period. Trade metrics for Buy & Hold are less informative than return, Sharpe, and drawdown metrics.
- The random agent is a stress benchmark for continuous action turnover; it is expected to perform poorly after costs.

## Claims To Avoid

- Do not claim the trained agent is profitable until a completed out-of-sample run beats all simple baselines after costs.
- Do not compare training reward directly to benchmark returns.
- Do not tune hyperparameters on the 2024 test split.
- Do not report a single ticker as conclusive evidence. Use a multi-ticker table and walk-forward results.
