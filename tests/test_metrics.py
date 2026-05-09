from __future__ import annotations

import math

import numpy as np

from envs.reward_functions import (
    risk_adjusted_pnl_reward,
    sharpe_reward,
    sortino_reward,
)
from evaluation.metrics import max_drawdown, profit_factor, sharpe_ratio, win_rate
from evaluation.backtester import WalkForwardBacktester


def test_max_drawdown_matches_manual_calculation() -> None:
    equity = np.array([100.0, 110.0, 90.0, 120.0])
    assert math.isclose(max_drawdown(equity), -20.0 / 110.0, rel_tol=1e-9)


def test_sharpe_ratio_is_finite_for_regular_returns() -> None:
    returns = np.array([0.01, -0.005, 0.004, 0.003, -0.002])
    assert math.isfinite(sharpe_ratio(returns, risk_free_rate=0.0))


def test_trade_metrics() -> None:
    trades = np.array([10.0, -5.0, 15.0, -10.0])
    assert win_rate(trades) == 0.5
    assert profit_factor(trades) == 25.0 / 15.0


def test_holding_period_uses_position_change_segments() -> None:
    backtester = WalkForwardBacktester(df=None)
    stats = backtester.compute_statistics(
        returns=np.array([0.01, 0.02, -0.01, 0.0]),
        equity_curve=np.array([100.0, 101.0, 103.02, 101.9898]),
        trades=np.array([0.02, -0.01]),
        positions=np.array([1.0, 1.0, -1.0, -1.0]),
    )
    assert stats["number_of_trades"] == 2.0
    assert stats["average_holding_period"] == 2.0


def test_reward_functions_are_clipped_and_finite() -> None:
    returns = [0.01, -0.005, 0.002, 0.004, -0.003] * 10
    assert -3.0 <= sharpe_reward(returns) <= 3.0
    assert -3.0 <= sortino_reward(returns) <= 3.0
    pnl_reward = risk_adjusted_pnl_reward(
        step_pnl=0.01,
        transaction_cost=0.001,
        position_change=0.5,
        max_drawdown_current=0.12,
    )
    assert -3.0 <= pnl_reward <= 3.0
