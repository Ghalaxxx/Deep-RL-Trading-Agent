"""Report builders for benchmark leaderboards and experiment summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.downloader import DEFAULT_TICKERS, load_ticker_data
from data.preprocessor import split_data
from evaluation.backtester import WalkForwardBacktester


def build_baseline_leaderboard(
    tickers: list[str] | None = None,
    cache_dir: Path | str = Path("cache"),
    start: str = "2015-01-01",
    end: str = "2025-01-01",
) -> pd.DataFrame:
    """Evaluate all benchmark strategies on the configured 2024 test split."""

    requested = tickers or DEFAULT_TICKERS
    rows: list[dict[str, float | str]] = []
    for ticker in requested:
        df = load_ticker_data(ticker=ticker, start=start, end=end, cache_dir=cache_dir)
        _, _, test_df = split_data(df)
        backtester = WalkForwardBacktester(test_df)
        for strategy, metrics in backtester.benchmark_statistics(test_df).items():
            rows.append({"ticker": ticker, "strategy": strategy, **metrics})

    leaderboard = pd.DataFrame(rows)
    return leaderboard.sort_values(["sharpe", "annualized_return"], ascending=False).reset_index(drop=True)


def write_report(df: pd.DataFrame, output_path: Path | str) -> Path:
    """Write a report dataframe as CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
