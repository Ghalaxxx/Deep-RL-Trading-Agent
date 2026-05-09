from __future__ import annotations

import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

from data.preprocessor import add_technical_indicators
from envs.observation_builder import TOTAL_FEATURES
from envs.trading_env import TradingEnv


def make_synthetic_ohlcv(n_rows: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    index = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    returns = rng.normal(loc=0.0005, scale=0.015, size=n_rows)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = close * (1.0 + rng.normal(0.0, 0.002, size=n_rows))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.01, size=n_rows))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.01, size=n_rows))
    volume = rng.integers(1_000_000, 5_000_000, size=n_rows)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_preprocessor_adds_required_indicators_without_nan() -> None:
    df = add_technical_indicators(make_synthetic_ohlcv())
    required = {
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
    }
    assert required.issubset(df.columns)
    assert not df.loc[:, list(required)].isna().any().any()


def test_trading_env_check_env_passes() -> None:
    env = TradingEnv(make_synthetic_ohlcv(), reward_function="pnl", random_start=False)
    check_env(env, skip_render_check=True)


def test_random_agent_can_step_1000_times_or_until_done() -> None:
    env = TradingEnv(make_synthetic_ohlcv(1300), reward_function="sharpe")
    observation, info = env.reset(seed=123)
    assert observation.shape == (env.window_size, TOTAL_FEATURES)
    assert np.isfinite(observation).all()
    assert info["portfolio_value"] == env.initial_cash

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        assert observation.shape == (env.window_size, TOTAL_FEATURES)
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        assert info["portfolio_value"] > 0.0
        if terminated or truncated:
            observation, _ = env.reset()
    assert env.observation_space.contains(observation)
