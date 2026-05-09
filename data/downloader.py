"""Market data download and cache pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

from data.preprocessor import add_technical_indicators, clean_market_data

DEFAULT_TICKERS: list[str] = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "SPY",
    "QQQ",
    "GLD",
    "BTC-USD",
]


@dataclass(slots=True)
class MarketDataDownloader:
    """Downloads OHLCV data with yfinance and stores a local parquet cache."""

    cache_dir: Path | str = Path("cache")
    auto_adjust: bool = True

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, ticker: str, start: str, end: str) -> Path:
        safe_ticker = ticker.replace("/", "_").replace("-", "_")
        return self.cache_dir / f"{safe_ticker}_{start}_{end}.parquet"

    def download(
        self,
        ticker: str,
        start: str = "2015-01-01",
        end: str = "2025-01-01",
        force: bool = False,
        add_indicators: bool = True,
    ) -> pd.DataFrame:
        """Download or load cached daily data for one ticker."""

        path = self.cache_path(ticker, start, end)
        if path.exists() and not force:
            return pd.read_parquet(path)

        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=self.auto_adjust,
            progress=False,
            group_by="column",
        )
        if df.empty:
            raise ValueError(f"No data returned for ticker {ticker!r}.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(col[0]).lower() for col in df.columns]
        else:
            df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]

        df = clean_market_data(df)
        if add_indicators:
            df = add_technical_indicators(df)

        try:
            df.to_parquet(path)
        except ImportError:
            # Pandas needs pyarrow or fastparquet for parquet. Keep a CSV fallback
            # so the data pipeline still works in lean Python environments.
            df.to_csv(path.with_suffix(".csv"), index=True)
        return df

    def download_many(
        self,
        tickers: list[str] | None = None,
        start: str = "2015-01-01",
        end: str = "2025-01-01",
        force: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Download a dictionary of preprocessed ticker dataframes."""

        requested = tickers or DEFAULT_TICKERS
        return {
            ticker: self.download(ticker=ticker, start=start, end=end, force=force)
            for ticker in requested
        }


def load_ticker_data(
    ticker: str,
    start: str = "2015-01-01",
    end: str = "2025-01-01",
    cache_dir: Path | str = Path("cache"),
    force: bool = False,
) -> pd.DataFrame:
    """Convenience function used by the trainer and CLI."""

    return MarketDataDownloader(cache_dir=cache_dir).download(
        ticker=ticker,
        start=start,
        end=end,
        force=force,
        add_indicators=True,
    )
