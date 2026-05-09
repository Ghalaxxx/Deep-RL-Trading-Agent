# Data Card

## Source

Market data comes from `yfinance` with `auto_adjust=True`, so splits and dividends are reflected in adjusted OHLC prices.

The dashboard product layer uses the local parquet cache and generated reports as its default source of truth. When the dashboard shows `Simulated live replay`, it is replaying recent cached bars through a live-style update loop. It is not claiming exchange-grade real-time data.

## Universe

The default universe is:

```text
AAPL, MSFT, GOOGL, AMZN, TSLA, SPY, QQQ, GLD, BTC-USD
```

The project downloads daily data for `2015-01-01` through `2025-01-01`, which gives train/validation/test coverage through the full 2024 test year.

## Splits

```text
Train: 2015-01-01 to 2022-12-31
Val:   2023-01-01 to 2023-12-31
Test:  2024-01-01 to 2024-12-31
```

The split design is intentionally chronological. Validation is used for model selection and tuning; the 2024 test period is reserved for final out-of-sample evaluation.

## Generated Features

The observation builder produces 18 features per timestep:

- 5 normalized OHLCV features
- 10 normalized technical indicators
- 3 portfolio state features

Indicator warm-up rows are dropped instead of backward-filled. This is intentional: backward-filling early RSI/MACD/Bollinger/ADX values would leak future information into the first observations.

Run this to reproduce the cache and write a validation manifest:

```bash
python train.py --prepare-data
```

The manifest is written to `reports/data_manifest.csv` and includes row counts, split counts, NaN counts, total return, and cache file paths.

## Leakage Controls

- Feature windows are rolling and use historical observations.
- Indicator warm-up rows are removed rather than imputed from future values.
- Backtesting uses train/test windows where the test period begins after the training period.
- `VecNormalize` statistics are saved with the trained model so inference uses training-time normalization.
- Live-style replay uses test-period bars only for dashboard demonstration. It does not feed back into training or hyperparameter selection.
- Paper trading state is derived from replayed bars and dashboard heuristics; it is not mixed into benchmark reports unless an explicit report-generation workflow is run.

## Product Data Modes

The product API exposes feed mode labels so demos remain honest:

- `Live market data`: optional yfinance near-live snapshot mode, enabled with `RL_TRADING_FEED_MODE=live`.
- `Simulated live replay`: cached historical bars replayed through WebSocket/live endpoints.
- `Cached demo mode`: static local cache/report state without live-style streaming.

The current implementation defaults to `Simulated live replay` for selected symbols in the default universe. Live yfinance mode is intentionally opt-in so tests and demos remain reproducible when network data is unavailable or rate-limited.

## Reliability Checks

`python train.py --prepare-data` validates the local cache before experiments:

- all default tickers are present
- train/validation/test splits are nonempty
- required indicator columns exist
- NaN counts are zero after warm-up trimming
- cache paths are recorded for reproducibility

Dashboard endpoints also validate supported symbols and report checkpoint availability instead of inventing unavailable PPO/SAC results.
