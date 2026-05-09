"""Stable-Baselines3 SAC model factory."""

from __future__ import annotations

from typing import Any

try:
    import torch.nn as nn
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import VecEnv
except ImportError:  # pragma: no cover
    SAC = None
    VecEnv = Any
    nn = None


def create_sac_model(
    env: VecEnv,
    config: dict[str, Any],
    tensorboard_log: str | None = "runs/sac",
) -> Any:
    """Create a SAC model from a project config dictionary."""

    if SAC is None:
        raise ImportError("stable-baselines3 and torch are required to train SAC.")

    policy = config.get("policy", "MlpPolicy")
    policy_kwargs = _normalize_policy_kwargs(config.get("policy_kwargs", {}))
    hyperparameters = config.get("hyperparameters", {}).copy()
    return SAC(
        policy=policy,
        env=env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=1,
        **hyperparameters,
    )


def _normalize_policy_kwargs(policy_kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized = policy_kwargs.copy()
    activation = normalized.get("activation_fn")
    if isinstance(activation, str):
        activation_map = {
            "ReLU": nn.ReLU,
            "Tanh": nn.Tanh,
            "ELU": nn.ELU,
            "LeakyReLU": nn.LeakyReLU,
        }
        normalized["activation_fn"] = activation_map[activation]
    return normalized
