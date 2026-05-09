"""Custom Gymnasium trading environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces

from envs.observation_builder import ObservationBuilder, TOTAL_FEATURES
from envs.reward_functions import (
    risk_adjusted_pnl_reward,
    sharpe_reward,
    sortino_reward,
)


class TradingEnv(gym.Env[np.ndarray, np.ndarray]):
    """
    Custom trading environment following the Gymnasium interface.

    Observation shape is `(window_size, 18)`. The continuous action in [-1, 1]
    represents target portfolio exposure: -1 full short, 0 flat, +1 full long.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        initial_cash: float = 100_000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        reward_function: str = "sharpe",
        episode_length: int = 252,
        random_start: bool = True,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if reward_function not in {"sharpe", "sortino", "pnl"}:
            raise ValueError("reward_function must be one of: sharpe, sortino, pnl")
        if len(df) < window_size + 2:
            raise ValueError("Dataframe is too short for the requested window size.")

        self.df = df.copy()
        self.window_size = window_size
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self.slippage = float(slippage)
        self.reward_function = reward_function
        self.episode_length = int(min(episode_length, len(df) - window_size - 1))
        self.random_start = random_start
        self.render_mode = render_mode

        self.observation_builder = ObservationBuilder(self.df, window_size=window_size)
        self.prices = self.observation_builder.df["close"].to_numpy(dtype=float)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(window_size, TOTAL_FEATURES),
            dtype=np.float32,
        )

        self.start_step = window_size
        self.end_step = self.start_step + self.episode_length
        self.current_step = self.start_step
        self.position = 0.0
        self.entry_price = float(self.prices[self.current_step])
        self.portfolio_value = self.initial_cash
        self.peak_portfolio_value = self.initial_cash
        self.portfolio_returns: list[float] = []
        self.equity_curve: list[float] = [self.initial_cash]
        self.buy_hold_curve: list[float] = [self.initial_cash]
        self.trades: list[dict[str, float]] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        max_start = len(self.prices) - self.episode_length - 1
        if self.random_start and max_start > self.window_size:
            self.start_step = int(self.np_random.integers(self.window_size, max_start + 1))
        else:
            self.start_step = self.window_size
        self.end_step = min(self.start_step + self.episode_length, len(self.prices) - 1)
        self.current_step = self.start_step

        self.position = 0.0
        self.entry_price = float(self.prices[self.current_step])
        self.portfolio_value = self.initial_cash
        self.peak_portfolio_value = self.initial_cash
        self.portfolio_returns = []
        self.equity_curve = [self.initial_cash]
        self.buy_hold_curve = [self.initial_cash]
        self.trades = []

        observation = self._get_observation()
        info = self._get_info(step_return=0.0, transaction_cost_paid=0.0, slippage_paid=0.0)
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        target_position = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -1.0, 1.0))
        current_price = float(self.prices[self.current_step])
        next_price = float(self.prices[self.current_step + 1])
        old_value = self.portfolio_value

        position_change = target_position - self.position
        turnover = abs(position_change)
        transaction_cost_paid = old_value * turnover * self.transaction_cost
        slippage_paid = old_value * turnover * self.slippage
        tradable_value = max(old_value - transaction_cost_paid - slippage_paid, 0.0)

        if turnover > 1e-6:
            self.trades.append(
                {
                    "step": float(self.current_step),
                    "position_change": float(position_change),
                    "price": current_price,
                    "cost": float(transaction_cost_paid + slippage_paid),
                },
            )
            self.entry_price = current_price

        market_return = (next_price / current_price) - 1.0
        self.portfolio_value = tradable_value * (1.0 + target_position * market_return)
        self.position = target_position
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)

        step_return = (self.portfolio_value / old_value) - 1.0 if old_value > 0.0 else 0.0
        self.portfolio_returns.append(float(step_return))
        self.equity_curve.append(float(self.portfolio_value))

        buy_hold_return = next_price / float(self.prices[self.start_step])
        self.buy_hold_curve.append(float(self.initial_cash * buy_hold_return))

        drawdown = self._current_drawdown()
        reward = self._calculate_reward(
            step_return=step_return,
            transaction_cost_paid=(transaction_cost_paid + slippage_paid) / max(old_value, 1e-9),
            position_change=position_change,
            drawdown=drawdown,
        )

        self.current_step += 1
        terminated = self.current_step >= self.end_step
        truncated = self.portfolio_value <= self.initial_cash * 0.05
        observation = self._get_observation()
        info = self._get_info(
            step_return=step_return,
            transaction_cost_paid=transaction_cost_paid,
            slippage_paid=slippage_paid,
        )
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.equity_curve, label="Agent")
        ax.plot(self.buy_hold_curve, label="Buy & Hold")
        ax.set_title("Trading Environment Equity Curve")
        ax.set_xlabel("Step")
        ax.set_ylabel("Portfolio Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            plt.close(fig)
            return image.reshape(height, width, 3)

        plt.show()
        return None

    def _calculate_reward(
        self,
        step_return: float,
        transaction_cost_paid: float,
        position_change: float,
        drawdown: float,
    ) -> float:
        if self.reward_function == "sharpe":
            return sharpe_reward(self.portfolio_returns)
        if self.reward_function == "sortino":
            return sortino_reward(self.portfolio_returns)
        return risk_adjusted_pnl_reward(
            step_pnl=step_return,
            transaction_cost=transaction_cost_paid,
            position_change=position_change,
            max_drawdown_current=drawdown,
        )

    def _get_observation(self) -> np.ndarray:
        current_price = float(self.prices[self.current_step])
        unrealized_pnl_pct = self.position * ((current_price / self.entry_price) - 1.0)
        cash_ratio = max(0.0, 1.0 - abs(self.position))
        return self.observation_builder.build_observation(
            current_step=self.current_step,
            current_position=self.position,
            unrealized_pnl_pct=unrealized_pnl_pct,
            cash_ratio=cash_ratio,
        )

    def _current_drawdown(self) -> float:
        if self.peak_portfolio_value <= 0.0:
            return 0.0
        return max(0.0, 1.0 - (self.portfolio_value / self.peak_portfolio_value))

    def _get_info(
        self,
        step_return: float,
        transaction_cost_paid: float,
        slippage_paid: float,
    ) -> dict[str, Any]:
        return {
            "portfolio_value": float(self.portfolio_value),
            "position": float(self.position),
            "step_return": float(step_return),
            "cash_ratio": float(max(0.0, 1.0 - abs(self.position))),
            "drawdown": float(self._current_drawdown()),
            "transaction_cost_paid": float(transaction_cost_paid),
            "slippage_paid": float(slippage_paid),
            "num_trades": len(self.trades),
        }
