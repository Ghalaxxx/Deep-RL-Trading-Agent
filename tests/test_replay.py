from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.replay import action_sensitivity, feature_schema, run_policy_replay


class MeanActionModel:
    """Tiny deterministic model used to test replay utilities without SB3."""

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        del deterministic
        action = float(np.clip(np.mean(observation), -1.0, 1.0))
        return np.array([action], dtype=np.float32), None


def make_synthetic_ohlcv(n_rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    index = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    returns = rng.normal(loc=0.0004, scale=0.01, size=n_rows)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = close * (1.0 + rng.normal(0.0, 0.001, size=n_rows))
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    volume = rng.integers(500_000, 2_000_000, size=n_rows)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def test_run_policy_replay_returns_auditable_table() -> None:
    replay = run_policy_replay(MeanActionModel(), make_synthetic_ohlcv())
    required = {
        "date",
        "step",
        "close",
        "action",
        "position",
        "portfolio_value",
        "step_return",
        "reward",
        "drawdown",
        "num_trades",
    }
    assert required.issubset(replay.columns)
    assert len(replay) > 0
    assert replay["portfolio_value"].gt(0).all()


def test_action_sensitivity_ranks_feature_groups() -> None:
    observation = np.ones((30, 18), dtype=np.float32) * 0.25
    sensitivity = action_sensitivity(MeanActionModel(), observation)
    assert set(sensitivity.columns) == {
        "feature_group",
        "base_action",
        "perturbed_action",
        "absolute_delta",
    }
    assert sensitivity["absolute_delta"].is_monotonic_decreasing


def test_feature_schema_matches_observation_width() -> None:
    schema = feature_schema()
    assert len(schema) == 18
    assert schema["index"].tolist() == list(range(18))
