"""Policy replay and lightweight action-sensitivity explainability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from envs.observation_builder import MARKET_FEATURE_COLUMNS, PORTFOLIO_FEATURE_COLUMNS
from envs.trading_env import TradingEnv

FEATURE_GROUPS: dict[str, list[int]] = {
    "ohlcv_returns": [0, 1, 2, 3, 4],
    "momentum": [5, 6, 10, 12, 13, 14],
    "volatility": [7, 8, 11],
    "volume_flow": [9],
    "portfolio_state": [15, 16, 17],
}


def run_policy_replay(
    model: Any,
    df: pd.DataFrame,
    output_path: Path | str | None = None,
    deterministic: bool = True,
    reward_function: str = "pnl",
    initial_cash: float = 100_000,
    vecnormalize_path: Path | str | None = None,
) -> pd.DataFrame:
    """Run one deterministic policy episode and return an auditable replay table."""

    if vecnormalize_path is not None and Path(vecnormalize_path).exists():
        replay = _run_normalized_policy_replay(
            model=model,
            df=df,
            vecnormalize_path=Path(vecnormalize_path),
            deterministic=deterministic,
            reward_function=reward_function,
            initial_cash=initial_cash,
        )
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            replay.to_csv(path, index=False)
        return replay

    env = TradingEnv(
        df=df,
        initial_cash=initial_cash,
        reward_function=reward_function,
        random_start=False,
        episode_length=max(2, len(df) - 32),
    )
    observation, _ = env.reset()
    rows: list[dict[str, float | int | str]] = []
    done = False

    while not done:
        action, _ = model.predict(observation, deterministic=deterministic)
        scalar_action = float(np.asarray(action, dtype=float).reshape(-1)[0])
        step_before = env.current_step
        close = float(env.prices[step_before])
        next_observation, reward, terminated, truncated, info = env.step(action)
        timestamp = df.index[min(step_before, len(df.index) - 1)]
        rows.append(
            {
                "date": str(timestamp.date()) if hasattr(timestamp, "date") else str(timestamp),
                "step": int(step_before),
                "close": close,
                "action": scalar_action,
                "position": float(info["position"]),
                "portfolio_value": float(info["portfolio_value"]),
                "step_return": float(info["step_return"]),
                "reward": float(reward),
                "drawdown": float(info["drawdown"]),
                "transaction_cost_paid": float(info["transaction_cost_paid"]),
                "slippage_paid": float(info["slippage_paid"]),
                "num_trades": int(info["num_trades"]),
            },
        )
        observation = next_observation
        done = terminated or truncated

    replay = pd.DataFrame(rows)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        replay.to_csv(path, index=False)
    return replay


def _run_normalized_policy_replay(
    model: Any,
    df: pd.DataFrame,
    vecnormalize_path: Path,
    deterministic: bool,
    reward_function: str,
    initial_cash: float,
) -> pd.DataFrame:
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as exc:  # pragma: no cover
        raise ImportError("stable-baselines3 is required to replay with VecNormalize stats.") from exc

    raw_env = TradingEnv(
        df=df,
        initial_cash=initial_cash,
        reward_function=reward_function,
        random_start=False,
        episode_length=max(2, len(df) - 32),
    )
    vec_env = DummyVecEnv([lambda: raw_env])
    normalized_env = VecNormalize.load(str(vecnormalize_path), vec_env)
    normalized_env.training = False
    normalized_env.norm_reward = False

    observation = normalized_env.reset()
    done = np.array([False])
    rows: list[dict[str, float | int | str]] = []

    while not bool(done[0]):
        step_before = raw_env.current_step
        close = float(raw_env.prices[step_before])
        action, _ = model.predict(observation, deterministic=deterministic)
        scalar_action = float(np.asarray(action, dtype=float).reshape(-1)[0])
        observation, reward, done, infos = normalized_env.step(action)
        info = infos[0]
        timestamp = df.index[min(step_before, len(df.index) - 1)]
        rows.append(
            {
                "date": str(timestamp.date()) if hasattr(timestamp, "date") else str(timestamp),
                "step": int(step_before),
                "close": close,
                "action": scalar_action,
                "position": float(info["position"]),
                "portfolio_value": float(info["portfolio_value"]),
                "step_return": float(info["step_return"]),
                "reward": float(np.asarray(reward, dtype=float).reshape(-1)[0]),
                "drawdown": float(info["drawdown"]),
                "transaction_cost_paid": float(info["transaction_cost_paid"]),
                "slippage_paid": float(info["slippage_paid"]),
                "num_trades": int(info["num_trades"]),
            },
        )

    return pd.DataFrame(rows)


def action_sensitivity(
    model: Any,
    observation: np.ndarray,
    feature_groups: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    """
    Estimate action sensitivity by zeroing feature groups in the current observation.

    This is not a causal explanation. It is a fast sanity check that helps inspect
    which observation groups the trained policy reacts to at one decision point.
    """

    groups = feature_groups or FEATURE_GROUPS
    base_action = _predict_scalar_action(model, observation)
    rows: list[dict[str, float | str]] = []

    for group_name, indices in groups.items():
        perturbed = np.array(observation, copy=True)
        perturbed[..., indices] = 0.0
        perturbed_action = _predict_scalar_action(model, perturbed)
        rows.append(
            {
                "feature_group": group_name,
                "base_action": base_action,
                "perturbed_action": perturbed_action,
                "absolute_delta": abs(base_action - perturbed_action),
            },
        )

    return pd.DataFrame(rows).sort_values("absolute_delta", ascending=False).reset_index(drop=True)


def feature_schema() -> pd.DataFrame:
    """Return the ordered observation schema for docs, reports, and notebooks."""

    names = list(MARKET_FEATURE_COLUMNS) + list(PORTFOLIO_FEATURE_COLUMNS)
    return pd.DataFrame({"index": list(range(len(names))), "feature": names})


def _predict_scalar_action(model: Any, observation: np.ndarray) -> float:
    action, _ = model.predict(observation, deterministic=True)
    return float(np.asarray(action, dtype=float).reshape(-1)[0])
