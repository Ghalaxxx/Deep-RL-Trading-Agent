# Data Card

## Source

Market data comes from `yfinance` with `auto_adjust=True`, so splits and dividends are reflected in adjusted OHLC prices.

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

## Reliability Checks

`python train.py --prepare-data` validates the local cache before experiments:

- all default tickers are present
- train/validation/test splits are nonempty
- required indicator columns exist
- NaN counts are zero after warm-up trimming
- cache paths are recorded for reproducibility
