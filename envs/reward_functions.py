"""Reward functions for the trading environment."""

from __future__ import annotations

from math import sqrt

import numpy as np


def sharpe_reward(portfolio_returns: list[float], risk_free_rate: float = 0.0) -> float:
    """Rolling annualized Sharpe ratio over the last 30 steps, clipped to [-3, 3]."""

    returns = np.asarray(portfolio_returns[-30:], dtype=float)
    if returns.size < 2:
        return 0.0
    excess_returns = returns - (risk_free_rate / 252.0)
    std = float(np.std(excess_returns, ddof=1))
    if std < 1e-9:
        return 0.0
    reward = float(np.mean(excess_returns) / (std + 1e-9) * sqrt(252.0))
    return float(np.clip(reward, -3.0, 3.0))


def sortino_reward(portfolio_returns: list[float]) -> float:
    """Rolling annualized Sortino ratio over the last 30 steps, clipped to [-3, 3]."""

    returns = np.asarray(portfolio_returns[-30:], dtype=float)
    if returns.size < 2:
        return 0.0
    downside = returns[returns < 0.0]
    if downside.size < 2:
        return float(np.clip(np.mean(returns) * sqrt(252.0), -3.0, 3.0))
    downside_std = float(np.std(downside, ddof=1))
    if downside_std < 1e-9:
        return 0.0
    reward = float(np.mean(returns) / (downside_std + 1e-9) * sqrt(252.0))
    return float(np.clip(reward, -3.0, 3.0))


def risk_adjusted_pnl_reward(
    step_pnl: float,
    transaction_cost: float,
    position_change: float,
    max_drawdown_current: float,
) -> float:
    """Composite PnL reward with turnover and drawdown penalties."""

    transaction_cost_penalty = transaction_cost * abs(position_change)
    drawdown_penalty = 2.0 * max(0.0, max_drawdown_current - 0.10)
    reward = step_pnl - transaction_cost_penalty - drawdown_penalty
    return float(np.clip(reward, -3.0, 3.0))
