"""Feature engineering for Gymnasium observations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.preprocessor import add_technical_indicators
from data.preprocessor import INDICATOR_COLUMNS, OHLCV_COLUMNS

MARKET_FEATURE_COLUMNS: tuple[str, ...] = (
    "open_pct_change",
    "high_pct_change",
    "low_pct_change",
    "close_pct_change",
    "volume_normalized",
    "rsi_14_norm",
    "macd_histogram_norm",
    "bb_position",
    "atr_14_normalized",
    "obv_slope_norm",
    "ema_ratio_9_21",
    "adx_14_norm",
    "stoch_k_14_norm",
    "willr_14_norm",
    "cci_14_norm",
)

PORTFOLIO_FEATURE_COLUMNS: tuple[str, ...] = (
    "current_position",
    "unrealized_pnl_pct",
    "cash_ratio",
)

TOTAL_FEATURES = len(MARKET_FEATURE_COLUMNS) + len(PORTFOLIO_FEATURE_COLUMNS)


@dataclass(slots=True)
class ObservationBuilder:
    """Builds fixed-width observation windows with market and portfolio state."""

    df: pd.DataFrame
    window_size: int = 30
    feature_frame: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.df = add_technical_indicators(self.df)
        self.feature_frame = build_market_features(self.df)

    @property
    def n_features(self) -> int:
        return TOTAL_FEATURES

    def build_observation(
        self,
        current_step: int,
        current_position: float,
        unrealized_pnl_pct: float,
        cash_ratio: float,
    ) -> np.ndarray:
        """Return a `(window_size, 18)` observation ending at current_step."""

        start = current_step - self.window_size + 1
        if start < 0:
            pad_count = abs(start)
            start = 0
        else:
            pad_count = 0

        market_window = self.feature_frame.iloc[start : current_step + 1].to_numpy(
            dtype=np.float32,
        )
        if pad_count:
            pad = np.repeat(market_window[:1], pad_count, axis=0)
            market_window = np.vstack([pad, market_window])

        portfolio_state = np.array(
            [
                np.clip(current_position, -1.0, 1.0),
                np.clip(unrealized_pnl_pct, -1.0, 1.0),
                np.clip(cash_ratio, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        portfolio_window = np.repeat(
            portfolio_state.reshape(1, -1),
            repeats=self.window_size,
            axis=0,
        )
        observation = np.concatenate([market_window, portfolio_window], axis=1)
        return np.nan_to_num(np.clip(observation, -1.0, 1.0), nan=0.0).astype(np.float32)


def build_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the 15 normalized market features required by the spec."""

    required_columns = set(OHLCV_COLUMNS) | set(INDICATOR_COLUMNS)
    if required_columns.issubset(df.columns) and not df.loc[:, list(required_columns)].isna().any().any():
        data = df.copy()
    else:
        data = add_technical_indicators(df)
    features = pd.DataFrame(index=data.index)
    for col in ("open", "high", "low", "close"):
        pct = data[col].pct_change().replace([np.inf, -np.inf], np.nan)
        features[f"{col}_pct_change"] = _clip_scale(pct, limit=0.10)

    volume_mean = data["volume"].rolling(30, min_periods=5).mean()
    volume_std = data["volume"].rolling(30, min_periods=5).std(ddof=0).replace(0.0, np.nan)
    features["volume_normalized"] = ((data["volume"] - volume_mean) / volume_std).clip(-3, 3) / 3

    features["rsi_14_norm"] = (data["rsi_14"] / 100.0).clip(0.0, 1.0)
    macd_std = data["macd_histogram"].rolling(30, min_periods=5).std(ddof=0).replace(0.0, np.nan)
    features["macd_histogram_norm"] = (data["macd_histogram"] / macd_std).clip(-3, 3) / 3

    bb_range = (data["bb_upper"] - data["bb_lower"]).replace(0.0, np.nan)
    features["bb_position"] = ((data["close"] - data["bb_lower"]) / bb_range).clip(0.0, 1.0)
    features["atr_14_normalized"] = (data["atr_14"] / data["close"]).clip(0.0, 0.20) / 0.20

    obv_slope = data["obv"].diff(5) / 5.0
    obv_std = obv_slope.rolling(30, min_periods=5).std(ddof=0).replace(0.0, np.nan)
    features["obv_slope_norm"] = (obv_slope / obv_std).clip(-3, 3) / 3

    ema_ratio = data["ema_9"] / data["ema_21"].replace(0.0, np.nan) - 1.0
    features["ema_ratio_9_21"] = _clip_scale(ema_ratio, limit=0.10)
    features["adx_14_norm"] = (data["adx_14"] / 100.0).clip(0.0, 1.0)
    features["stoch_k_14_norm"] = (data["stoch_k_14"] / 100.0).clip(0.0, 1.0)
    features["willr_14_norm"] = ((data["willr_14"] + 100.0) / 100.0).clip(0.0, 1.0)
    features["cci_14_norm"] = ((data["cci_14"] / 100.0).clip(-3.0, 3.0) / 3.0).clip(-1.0, 1.0)

    features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return features.loc[:, list(MARKET_FEATURE_COLUMNS)].astype(np.float32)


def _clip_scale(series: pd.Series, limit: float) -> pd.Series:
    return (series.clip(-limit, limit) / limit).astype(float)
