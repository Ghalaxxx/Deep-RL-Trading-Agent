"""Data loading and preprocessing utilities for the RL trading agent."""

from data.preprocessor import add_technical_indicators, clean_market_data

__all__ = [
    "DEFAULT_TICKERS",
    "MarketDataDownloader",
    "add_technical_indicators",
    "build_data_manifest",
    "clean_market_data",
    "write_data_manifest",
]


def __getattr__(name: str) -> object:
    """Lazy-load yfinance-backed data symbols only when requested."""

    if name in {"DEFAULT_TICKERS", "MarketDataDownloader"}:
        from data.downloader import DEFAULT_TICKERS, MarketDataDownloader

        return {"DEFAULT_TICKERS": DEFAULT_TICKERS, "MarketDataDownloader": MarketDataDownloader}[name]
    if name in {"build_data_manifest", "write_data_manifest"}:
        from data.manifest import build_data_manifest, write_data_manifest

        return {
            "build_data_manifest": build_data_manifest,
            "write_data_manifest": write_data_manifest,
        }[name]
    raise AttributeError(name)
