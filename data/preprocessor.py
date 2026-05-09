"""Market data cleaning and technical indicator creation."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import talib
except ImportError:  # pragma: no cover - exercised only when TA-Lib is absent
    talib = None

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
INDICATOR_COLUMNS: tuple[str, ...] = (
    "rsi_14",
    "macd_histogram",
    "bb_lower",
    "bb_middle",
    "bb_upper",
    "atr_14",
    "obv",
    "ema_9",
    "ema_21",
    "adx_14",
    "stoch_k_14",
    "willr_14",
    "cci_14",
)


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, validate OHLCV data, and remove unusable rows."""

    cleaned = df.copy()
    cleaned.columns = [str(col).lower().replace(" ", "_") for col in cleaned.columns]
    missing = [col for col in OHLCV_COLUMNS if col not in cleaned.columns]
    if missing:
        raise ValueError(f"Market data is missing required columns: {missing}")

    cleaned = cleaned.loc[:, list(OHLCV_COLUMNS)]
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(subset=list(OHLCV_COLUMNS))
    cleaned = cleaned[cleaned["close"] > 0]
    cleaned = cleaned[cleaned["volume"] >= 0]
    cleaned = cleaned.sort_index()
    return cleaned.astype(float)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add the full indicator set required by the observation builder."""

    data = clean_market_data(df)
    if talib is not None:
        return _add_talib_indicators(data)
    return _add_pandas_indicators(data)


def _add_talib_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    open_ = data["open"].to_numpy(dtype=float)
    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)
    close = data["close"].to_numpy(dtype=float)
    volume = data["volume"].to_numpy(dtype=float)

    data["rsi_14"] = talib.RSI(close, timeperiod=14)
    _, _, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    data["macd_histogram"] = macd_hist
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    data["bb_lower"] = lower
    data["bb_middle"] = middle
    data["bb_upper"] = upper
    data["atr_14"] = talib.ATR(high, low, close, timeperiod=14)
    data["obv"] = talib.OBV(close, volume)
    data["ema_9"] = talib.EMA(close, timeperiod=9)
    data["ema_21"] = talib.EMA(close, timeperiod=21)
    data["adx_14"] = talib.ADX(high, low, close, timeperiod=14)
    slow_k, _ = talib.STOCH(high, low, close, fastk_period=14)
    data["stoch_k_14"] = slow_k
    data["willr_14"] = talib.WILLR(high, low, close, timeperiod=14)
    data["cci_14"] = talib.CCI(high, low, close, timeperiod=14)
    _ = open_  # Keeps the TA-Lib path explicit about the OHLCV inputs.
    return _finalize_indicators(data)


def _add_pandas_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Pandas fallback mirroring the required TA-Lib indicators."""

    data = df.copy()
    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    data["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    ema_12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()
    data["macd_histogram"] = macd - signal

    rolling_mean = close.rolling(20, min_periods=20).mean()
    rolling_std = close.rolling(20, min_periods=20).std(ddof=0)
    data["bb_middle"] = rolling_mean
    data["bb_upper"] = rolling_mean + 2.0 * rolling_std
    data["bb_lower"] = rolling_mean - 2.0 * rolling_std

    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    data["atr_14"] = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    direction = np.sign(close.diff()).fillna(0.0)
    data["obv"] = (direction * volume).cumsum()
    data["ema_9"] = close.ewm(span=9, min_periods=9, adjust=False).mean()
    data["ema_21"] = close.ewm(span=21, min_periods=21, adjust=False).mean()

    plus_dm = (high.diff()).where((high.diff() > -low.diff()) & (high.diff() > 0), 0.0)
    minus_dm = (-low.diff()).where((-low.diff() > high.diff()) & (-low.diff() > 0), 0.0)
    atr = data["atr_14"].replace(0.0, np.nan)
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    data["adx_14"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    lowest_low = low.rolling(14, min_periods=14).min()
    highest_high = high.rolling(14, min_periods=14).max()
    range_14 = (highest_high - lowest_low).replace(0.0, np.nan)
    data["stoch_k_14"] = 100.0 * (close - lowest_low) / range_14
    data["willr_14"] = -100.0 * (highest_high - close) / range_14

    typical_price = (high + low + close) / 3.0
    tp_mean = typical_price.rolling(14, min_periods=14).mean()
    mean_dev = (typical_price - tp_mean).abs().rolling(14, min_periods=14).mean()
    data["cci_14"] = (typical_price - tp_mean) / (0.015 * mean_dev.replace(0.0, np.nan))
    return _finalize_indicators(data)


def _finalize_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data[list(INDICATOR_COLUMNS)] = data[list(INDICATOR_COLUMNS)].ffill()
    data = data.dropna(subset=list(OHLCV_COLUMNS) + list(INDICATOR_COLUMNS))
    return data


def split_data(
    df: pd.DataFrame,
    train_end: str = "2022-12-31",
    val_start: str = "2023-01-01",
    val_end: str = "2023-12-31",
    test_start: str = "2024-01-01",
    test_end: str = "2024-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test windows."""

    train = df.loc[:train_end].copy()
    val = df.loc[val_start:val_end].copy()
    test = df.loc[test_start:test_end].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("One or more data splits are empty. Check date coverage.")
    return train, val, test
