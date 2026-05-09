"""Walk-forward backtesting utilities with leakage-aware splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.baselines import buy_and_hold, momentum_strategy, random_agent, sma_crossover
from envs.trading_env import TradingEnv
from evaluation.metrics import (
    annualized_return,
    annualized_volatility,
    average_win_loss,
    calmar_ratio,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


@dataclass(slots=True)
class WalkForwardBacktester:
    """Leakage-aware walk-forward backtester with benchmark strategies."""

    df: pd.DataFrame
    train_years: int = 2
    test_months: int = 6
    step_months: int = 6
    initial_cash: float = 100_000

    def generate_splits(self) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Return rolling train/test windows without using future data."""

        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("Walk-forward backtesting requires a DatetimeIndex.")
        splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        start = self.df.index.min()
        final = self.df.index.max()

        train_offset = pd.DateOffset(years=self.train_years)
        test_offset = pd.DateOffset(months=self.test_months)
        step_offset = pd.DateOffset(months=self.step_months)
        cursor = start
        while cursor + train_offset + test_offset <= final:
            train_start = cursor
            train_end = cursor + train_offset
            test_end = train_end + test_offset
            train = self.df[(self.df.index >= train_start) & (self.df.index < train_end)].copy()
            test = self.df[(self.df.index >= train_end) & (self.df.index < test_end)].copy()
            if len(train) > 60 and len(test) > 20:
                splits.append((train, test))
            cursor += step_offset
        return splits

    def run_model(
        self,
        model: Any,
        test_df: pd.DataFrame,
        vecnormalize_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Evaluate an SB3-style model on one deterministic test episode."""

        if vecnormalize_path is not None and Path(vecnormalize_path).exists():
            return self._run_normalized_model(model, test_df, Path(vecnormalize_path))

        env = TradingEnv(
            test_df,
            initial_cash=self.initial_cash,
            reward_function="pnl",
            random_start=False,
            episode_length=max(2, len(test_df) - 32),
        )
        observation, _ = env.reset()
        done = False
        positions: list[float] = []
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            positions.append(float(info["position"]))
            done = terminated or truncated
        returns = np.asarray(env.portfolio_returns, dtype=float)
        position_values = np.asarray(positions, dtype=float)
        return self.compute_statistics(
            returns=returns,
            equity_curve=np.asarray(env.equity_curve, dtype=float),
            trades=_realized_trade_pnl(returns, position_values),
            positions=position_values,
        )

    def _run_normalized_model(
        self,
        model: Any,
        test_df: pd.DataFrame,
        vecnormalize_path: Path,
    ) -> dict[str, Any]:
        try:
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        except ImportError as exc:  # pragma: no cover
            raise ImportError("stable-baselines3 is required to evaluate with VecNormalize stats.") from exc

        raw_env = TradingEnv(
            test_df,
            initial_cash=self.initial_cash,
            reward_function="pnl",
            random_start=False,
            episode_length=max(2, len(test_df) - 32),
        )
        vec_env = DummyVecEnv([lambda: raw_env])
        normalized_env = VecNormalize.load(str(vecnormalize_path), vec_env)
        normalized_env.training = False
        normalized_env.norm_reward = False

        observation = normalized_env.reset()
        done = np.array([False])
        returns: list[float] = []
        positions: list[float] = []
        equity_curve: list[float] = [self.initial_cash]
        while not bool(done[0]):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, done, infos = normalized_env.step(action)
            info = infos[0]
            returns.append(float(info["step_return"]))
            positions.append(float(info["position"]))
            equity_curve.append(float(info["portfolio_value"]))

        return self.compute_statistics(
            returns=np.asarray(returns, dtype=float),
            equity_curve=np.asarray(equity_curve, dtype=float),
            trades=_realized_trade_pnl(np.asarray(returns, dtype=float), np.asarray(positions, dtype=float)),
            positions=np.asarray(positions, dtype=float),
        )

    def benchmark_statistics(self, test_df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Compute metrics for all required benchmark strategies."""

        strategies = {
            "buy_hold": buy_and_hold(test_df, initial_cash=self.initial_cash),
            "random": random_agent(test_df, initial_cash=self.initial_cash),
            "sma_crossover": sma_crossover(test_df, initial_cash=self.initial_cash),
            "momentum": momentum_strategy(test_df, initial_cash=self.initial_cash),
        }
        return {
            name: self.compute_statistics(
                returns=result["returns"].to_numpy(dtype=float),
                equity_curve=result["equity"].to_numpy(dtype=float),
                trades=result.get("realized_trade_pnl", pd.Series(dtype=float)).to_numpy(dtype=float),
                positions=result.get("position", pd.Series(dtype=float)).to_numpy(dtype=float),
            )
            for name, result in strategies.items()
        }

    def compute_statistics(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray | None = None,
        trades: np.ndarray | None = None,
        positions: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute the complete evaluation metric set."""

        clean_returns = np.asarray(returns, dtype=float)
        clean_returns = clean_returns[np.isfinite(clean_returns)]
        if equity_curve is None:
            equity_curve = self.initial_cash * np.cumprod(1.0 + clean_returns)
        clean_equity = np.asarray(equity_curve, dtype=float)
        trade_values = np.asarray([] if trades is None else trades, dtype=float)
        trade_values = trade_values[np.isfinite(trade_values)]
        trade_values = trade_values[np.abs(trade_values) > 1e-12]
        avg_win, avg_loss = average_win_loss(trade_values)

        position_values = np.asarray([] if positions is None else positions, dtype=float)
        num_trades = _count_position_changes(position_values)
        avg_holding_period = _average_holding_period(position_values)

        return {
            "annualized_return": annualized_return(clean_returns),
            "volatility": annualized_volatility(clean_returns),
            "sharpe": sharpe_ratio(clean_returns, risk_free_rate=0.03),
            "sortino": sortino_ratio(clean_returns, risk_free_rate=0.03),
            "max_drawdown": max_drawdown(clean_equity),
            "calmar": calmar_ratio(clean_returns, clean_equity),
            "win_rate": win_rate(trade_values),
            "profit_factor": profit_factor(trade_values),
            "average_win": avg_win,
            "average_loss": avg_loss,
            "number_of_trades": float(num_trades),
            "average_holding_period": avg_holding_period,
        }


def _average_holding_period(positions: np.ndarray) -> float:
    if positions.size == 0:
        return 0.0
    active = np.abs(positions) > 1e-6
    if not np.any(active):
        return 0.0
    changes = np.abs(np.diff(positions, prepend=0.0)) > 1e-6
    lengths: list[int] = []
    current = 0
    for is_active, changed in zip(active, changes):
        if changed and current:
            lengths.append(current)
            current = 0
        if is_active:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return float(np.mean(lengths)) if lengths else 0.0


def _count_position_changes(positions: np.ndarray) -> int:
    if positions.size == 0:
        return 0
    changes = np.abs(np.diff(positions, prepend=0.0)) > 1e-6
    return int(np.sum(changes))


def _realized_trade_pnl(returns: np.ndarray, positions: np.ndarray) -> np.ndarray:
    if returns.size == 0 or positions.size == 0:
        return np.asarray([], dtype=float)
    realized = np.zeros_like(returns, dtype=float)
    cumulative = 0.0
    previous_position = 0.0
    for idx, (step_return, position) in enumerate(zip(returns, positions)):
        if abs(position - previous_position) > 1e-9 and abs(cumulative) > 0.0:
            realized[idx] = cumulative
            cumulative = 0.0
        if abs(position) > 1e-9:
            cumulative += float(step_return)
        previous_position = float(position)
    if abs(cumulative) > 0.0:
        realized[-1] += cumulative
    return realized
