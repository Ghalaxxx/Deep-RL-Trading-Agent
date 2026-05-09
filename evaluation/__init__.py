"""Backtesting, metrics, and visualization."""

from evaluation.metrics import (
    calmar_ratio,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)
from evaluation.replay import action_sensitivity, feature_schema, run_policy_replay

__all__ = [
    "action_sensitivity",
    "build_baseline_leaderboard",
    "calmar_ratio",
    "feature_schema",
    "max_drawdown",
    "profit_factor",
    "run_policy_replay",
    "sharpe_ratio",
    "sortino_ratio",
    "win_rate",
    "write_report",
]


def __getattr__(name: str) -> object:
    """Lazy-load data-backed reporting helpers only when requested."""

    if name in {"build_baseline_leaderboard", "write_report"}:
        from evaluation.reporting import build_baseline_leaderboard, write_report

        return {
            "build_baseline_leaderboard": build_baseline_leaderboard,
            "write_report": write_report,
        }[name]
    raise AttributeError(name)
