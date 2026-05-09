"""Dataset manifest creation and validation utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.downloader import DEFAULT_TICKERS, MarketDataDownloader
from data.preprocessor import INDICATOR_COLUMNS, split_data


def build_data_manifest(
    tickers: list[str] | None = None,
    start: str = "2015-01-01",
    end: str = "2025-01-01",
    cache_dir: Path | str = Path("cache"),
    force: bool = False,
) -> pd.DataFrame:
    """Download/validate all requested tickers and return a manifest dataframe."""

    requested = tickers or DEFAULT_TICKERS
    downloader = MarketDataDownloader(cache_dir=cache_dir)
    rows: list[dict[str, object]] = []

    for ticker in requested:
        df = downloader.download(ticker=ticker, start=start, end=end, force=force, add_indicators=True)
        train, val, test = split_data(df)
        missing_indicators = [col for col in INDICATOR_COLUMNS if col not in df.columns]
        cache_path = downloader.cache_path(ticker, start, end)
        actual_path = cache_path if cache_path.exists() else cache_path.with_suffix(".csv")

        first_close = float(df["close"].iloc[0])
        last_close = float(df["close"].iloc[-1])
        rows.append(
            {
                "ticker": ticker,
                "rows": int(len(df)),
                "start_date": str(df.index.min().date()),
                "end_date": str(df.index.max().date()),
                "train_rows": int(len(train)),
                "val_rows": int(len(val)),
                "test_rows": int(len(test)),
                "nan_count": int(df.isna().sum().sum()),
                "missing_indicators": ",".join(missing_indicators),
                "first_close": first_close,
                "last_close": last_close,
                "total_return": (last_close / first_close) - 1.0,
                "cache_file": str(actual_path),
            },
        )

    manifest = pd.DataFrame(rows)
    _assert_manifest_valid(manifest)
    return manifest


def write_data_manifest(
    manifest: pd.DataFrame,
    output_path: Path | str = Path("reports") / "data_manifest.csv",
) -> Path:
    """Write a manifest CSV and return its path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    return path


def _assert_manifest_valid(manifest: pd.DataFrame) -> None:
    if manifest.empty:
        raise ValueError("Data manifest is empty.")
    invalid_nan = manifest[manifest["nan_count"] != 0]
    if not invalid_nan.empty:
        tickers = invalid_nan["ticker"].to_list()
        raise ValueError(f"Downloaded data contains NaNs for: {tickers}")
    missing = manifest[manifest["missing_indicators"] != ""]
    if not missing.empty:
        details = missing[["ticker", "missing_indicators"]].to_dict(orient="records")
        raise ValueError(f"Downloaded data is missing indicators: {details}")
    empty_splits = manifest[
        (manifest["train_rows"] == 0) | (manifest["val_rows"] == 0) | (manifest["test_rows"] == 0)
    ]
    if not empty_splits.empty:
        tickers = empty_splits["ticker"].to_list()
        raise ValueError(f"One or more date splits are empty for: {tickers}")
