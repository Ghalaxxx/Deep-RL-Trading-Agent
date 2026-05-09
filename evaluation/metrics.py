"""Evaluation metrics for strategy returns and trade lists."""

from __future__ import annotations

from math import sqrt
from typing import Sequence

import numpy as np


def sharpe_ratio(
    returns: Sequence[float] | np.ndarray,
    risk_free_rate: float = 0.03,
    periods: int = 252,
) -> float:
    daily_returns = _to_returns(returns)
    if daily_returns.size < 2:
        return 0.0
    excess = daily_returns - risk_free_rate / periods
    std = float(np.std(excess, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(excess) / std * sqrt(periods))


def sortino_ratio(
    returns: Sequence[float] | np.ndarray,
    risk_free_rate: float = 0.03,
    periods: int = 252,
) -> float:
    daily_returns = _to_returns(returns)
    if daily_returns.size < 2:
        return 0.0
    excess = daily_returns - risk_free_rate / periods
    downside = excess[excess < 0.0]
    if downside.size < 2:
        return 0.0
    downside_std = float(np.std(downside, ddof=1))
    if downside_std < 1e-12:
        return 0.0
    return float(np.mean(excess) / downside_std * sqrt(periods))


def max_drawdown(equity_curve: Sequence[float] | np.ndarray) -> float:
    equity = np.asarray(equity_curve, dtype=float)
    if equity.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / np.maximum(running_peak, 1e-12) - 1.0
    return float(np.min(drawdowns))


def calmar_ratio(
    returns: Sequence[float] | np.ndarray,
    equity_curve: Sequence[float] | np.ndarray,
    periods: int = 252,
) -> float:
    ann_return = annualized_return(returns, periods=periods)
    max_dd = abs(max_drawdown(equity_curve))
    if max_dd < 1e-12:
        return 0.0
    return float(ann_return / max_dd)


def win_rate(trades: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(trades, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.mean(values > 0.0))


def profit_factor(trades: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(trades, dtype=float)
    wins = values[values > 0.0].sum()
    losses = abs(values[values < 0.0].sum())
    if losses < 1e-12:
        return float("inf") if wins > 0.0 else 0.0
    return float(wins / losses)


def annualized_return(returns: Sequence[float] | np.ndarray, periods: int = 252) -> float:
    daily_returns = _to_returns(returns)
    if daily_returns.size == 0:
        return 0.0
    compounded = float(np.prod(1.0 + daily_returns))
    years = daily_returns.size / periods
    if years <= 0.0 or compounded <= 0.0:
        return 0.0
    return float(compounded ** (1.0 / years) - 1.0)


def annualized_volatility(returns: Sequence[float] | np.ndarray, periods: int = 252) -> float:
    daily_returns = _to_returns(returns)
    if daily_returns.size < 2:
        return 0.0
    return float(np.std(daily_returns, ddof=1) * sqrt(periods))


def average_win_loss(trades: Sequence[float] | np.ndarray) -> tuple[float, float]:
    values = np.asarray(trades, dtype=float)
    wins = values[values > 0.0]
    losses = values[values < 0.0]
    avg_win = float(np.mean(wins)) if wins.size else 0.0
    avg_loss = float(np.mean(losses)) if losses.size else 0.0
    return avg_win, avg_loss


def _to_returns(values: Sequence[float] | np.ndarray) -> np.ndarray:
    returns = np.asarray(values, dtype=float)
    returns = returns[np.isfinite(returns)]
    return returns
