"""Benchmark trading strategies used during evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from envs.trading_env import TradingEnv


def buy_and_hold(df: pd.DataFrame, initial_cash: float = 100_000) -> pd.DataFrame:
    """Long-only benchmark that buys at the first close and holds."""

    close = df["close"].astype(float)
    returns = close.pct_change().fillna(0.0)
    equity = initial_cash * (1.0 + returns).cumprod()
    position = pd.Series(1.0, index=df.index)
    position.iloc[0] = 0.0
    realized_trade_pnl = pd.Series(0.0, index=df.index)
    if len(realized_trade_pnl) > 0:
        realized_trade_pnl.iloc[-1] = float(equity.iloc[-1] / initial_cash - 1.0)
    return pd.DataFrame(
        {
            "returns": returns,
            "equity": equity,
            "position": position,
            "realized_trade_pnl": realized_trade_pnl,
        },
        index=df.index,
    )


def sma_crossover(
    df: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 50,
    initial_cash: float = 100_000,
) -> pd.DataFrame:
    """SMA 20/50 benchmark with long/flat exposure."""

    close = df["close"].astype(float)
    fast = close.rolling(fast_window, min_periods=fast_window).mean()
    slow = close.rolling(slow_window, min_periods=slow_window).mean()
    position = (fast > slow).astype(float).shift(1).fillna(0.0)
    position_change = position.diff().fillna(position)
    trade_cost = position_change.abs() * 0.0015
    returns = close.pct_change().fillna(0.0) * position - trade_cost
    equity = initial_cash * (1.0 + returns).cumprod()
    return pd.DataFrame(
        {
            "returns": returns,
            "equity": equity,
            "position": position,
            "realized_trade_pnl": _realized_trade_pnl(returns, position),
        },
        index=df.index,
    )


def momentum_strategy(
    df: pd.DataFrame,
    lookback: int = 20,
    initial_cash: float = 100_000,
) -> pd.DataFrame:
    """Simple time-series momentum benchmark with long/short exposure."""

    close = df["close"].astype(float)
    momentum = close.pct_change(lookback)
    position = np.sign(momentum).shift(1).fillna(0.0)
    position_change = position.diff().fillna(position)
    trade_cost = position_change.abs() * 0.0015
    returns = close.pct_change().fillna(0.0) * position - trade_cost
    equity = initial_cash * (1.0 + returns).cumprod()
    return pd.DataFrame(
        {
            "returns": returns,
            "equity": equity,
            "position": position,
            "realized_trade_pnl": _realized_trade_pnl(returns, position),
        },
        index=df.index,
    )


def random_agent(
    df: pd.DataFrame,
    n_steps: int | None = None,
    seed: int = 42,
    initial_cash: float = 100_000,
) -> pd.DataFrame:
    """Random continuous-action agent benchmark run inside the trading env."""

    env = TradingEnv(df=df, initial_cash=initial_cash, random_start=False, reward_function="pnl")
    observation, _ = env.reset(seed=seed)
    del observation
    rng = np.random.default_rng(seed)
    returns: list[float] = [0.0]
    equity: list[float] = [initial_cash]
    positions: list[float] = [0.0]
    max_steps = n_steps or env.episode_length

    for _ in range(max_steps):
        action = rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
        _, _, terminated, truncated, info = env.step(action)
        returns.append(float(info["step_return"]))
        equity.append(float(info["portfolio_value"]))
        positions.append(float(info["position"]))
        if terminated or truncated:
            break

    start = env.start_step
    index = df.index[start : start + len(returns)]
    return pd.DataFrame(
        {
            "returns": returns,
            "equity": equity,
            "position": positions,
            "realized_trade_pnl": _realized_trade_pnl(pd.Series(returns), pd.Series(positions)).to_numpy(),
        },
        index=index,
    )


def _realized_trade_pnl(returns: pd.Series, position: pd.Series) -> pd.Series:
    """Approximate per-trade PnL by accumulating returns until exposure changes."""

    realized = pd.Series(0.0, index=returns.index)
    cumulative = 0.0
    previous_position = 0.0
    for idx, step_return in returns.items():
        current_position = float(position.loc[idx])
        if abs(current_position - previous_position) > 1e-9 and abs(cumulative) > 0.0:
            realized.loc[idx] = cumulative
            cumulative = 0.0
        if abs(current_position) > 1e-9:
            cumulative += float(step_return)
        previous_position = current_position
    if len(realized) > 0 and abs(cumulative) > 0.0:
        realized.iloc[-1] += cumulative
    return realized
