"""Training orchestration for PPO and SAC agents."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

try:
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize
except ImportError:  # pragma: no cover
    BaseCallback = object
    EvalCallback = None
    DummyVecEnv = None
    SubprocVecEnv = None
    VecEnv = Any
    VecNormalize = None

from agents.ppo_agent import create_ppo_model
from agents.sac_agent import create_sac_model
from data.downloader import load_ticker_data
from data.preprocessor import split_data
from envs.trading_env import TradingEnv
from evaluation.backtester import WalkForwardBacktester


class WandbRiskMetricsCallback(BaseCallback):
    """Logs environment risk metrics from SB3 infos to Weights & Biases."""

    def __init__(self) -> None:
        super().__init__()

    def _on_step(self) -> bool:
        if wandb is None or wandb.run is None:
            return True
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        portfolio_values = [info.get("portfolio_value") for info in infos if "portfolio_value" in info]
        drawdowns = [info.get("drawdown") for info in infos if "drawdown" in info]
        if portfolio_values:
            wandb.log({"portfolio_value": float(np.mean(portfolio_values))}, step=self.num_timesteps)
        if drawdowns:
            wandb.log({"drawdown": float(np.mean(drawdowns))}, step=self.num_timesteps)
        return True


class RLTrainer:
    """End-to-end trainer for PPO/SAC trading experiments."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.wandb_run: Any | None = None
        self.model: Any | None = None
        self.vecnormalize_path = Path(config.get("paths", {}).get("vecnormalize", "models/vecnormalize.pkl"))
        self.model_dir = Path(config.get("paths", {}).get("models", "models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RLTrainer":
        with Path(path).open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return cls(config)

    def setup_wandb(self) -> None:
        if wandb is None:
            return
        mode = self.config.get("wandb", {}).get("mode", "online")
        self.wandb_run = wandb.init(
            project="rl-trading-agent",
            config=self.config,
            mode=mode,
            name=f"{self.config['algorithm']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    def create_env(self, data: pd.DataFrame, is_eval: bool = False) -> VecEnv:
        """Create vectorized envs and apply VecNormalize."""

        if DummyVecEnv is None or SubprocVecEnv is None or VecNormalize is None:
            raise ImportError("stable-baselines3 is required for vectorized training environments.")

        env_config = self.config.get("environment", {})
        training_config = self.config.get("training", {})
        n_envs = 1 if is_eval else int(training_config.get("n_envs", 8))

        def make_env(rank: int) -> Any:
            def _init() -> TradingEnv:
                return TradingEnv(
                    df=data,
                    window_size=int(env_config.get("window_size", 30)),
                    initial_cash=float(env_config.get("initial_cash", 100_000)),
                    transaction_cost=float(env_config.get("transaction_cost", 0.001)),
                    slippage=float(env_config.get("slippage", 0.0005)),
                    reward_function=str(env_config.get("reward_function", "sharpe")),
                    random_start=not is_eval,
                    episode_length=int(env_config.get("episode_length", 252)),
                )

            _ = rank
            return _init

        vec_env = DummyVecEnv([make_env(0)]) if is_eval else SubprocVecEnv([make_env(i) for i in range(n_envs)])
        normalized = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=not is_eval,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        normalized.training = not is_eval
        normalized.norm_reward = not is_eval
        return normalized

    def train(self) -> Any:
        """Download data, train the selected algorithm, and save artifacts."""

        self.setup_wandb()
        ticker = self.config.get("ticker", "AAPL")
        start = self.config.get("data", {}).get("start", "2015-01-01")
        end = self.config.get("data", {}).get("end", "2025-01-01")
        data = load_ticker_data(ticker=ticker, start=start, end=end)
        train_df, val_df, _ = split_data(data)

        train_env = self.create_env(train_df, is_eval=False)
        eval_env = self.create_env(val_df, is_eval=True)
        self.model = self._create_model(train_env)

        callbacks: list[Any] = [WandbRiskMetricsCallback()]
        if EvalCallback is not None:
            callbacks.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(self.model_dir),
                    log_path=str(Path("runs") / "eval"),
                    eval_freq=int(self.config.get("training", {}).get("eval_freq", 10_000)),
                    n_eval_episodes=int(self.config.get("training", {}).get("n_eval_episodes", 10)),
                    deterministic=True,
                ),
            )

        total_timesteps = int(self.config.get("training", {}).get("total_timesteps", 1_000_000))
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks)

        model_path = self.model_dir / f"final_{self.config['algorithm'].lower()}_{ticker}.zip"
        self.model.save(model_path)
        train_env.save(self.vecnormalize_path)
        if self.wandb_run is not None:
            self.wandb_run.finish()
        return self.model

    def evaluate(self, model: Any, env: VecEnv | None = None) -> dict[str, float]:
        """Run evaluation and return risk/return metrics."""

        if env is None:
            ticker = self.config.get("ticker", "AAPL")
            data = load_ticker_data(ticker=ticker, start="2015-01-01", end="2025-01-01")
            _, _, test_df = split_data(data)
            backtester = WalkForwardBacktester(test_df)
            return backtester.run_model(model, test_df, vecnormalize_path=self.vecnormalize_path)

        rewards: list[float] = []
        portfolio_values: list[float] = []
        n_eval_episodes = int(self.config.get("training", {}).get("n_eval_episodes", 10))
        for _ in range(n_eval_episodes):
            observation = env.reset()
            done = np.array([False])
            while not bool(done[0]):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, done, infos = env.step(action)
                rewards.append(float(np.mean(reward)))
                if infos and "portfolio_value" in infos[0]:
                    portfolio_values.append(float(infos[0]["portfolio_value"]))
        returns = np.diff(portfolio_values) / np.maximum(portfolio_values[:-1], 1e-9) if len(portfolio_values) > 1 else np.array([])
        return WalkForwardBacktester(pd.DataFrame()).compute_statistics(
            returns=np.asarray(returns, dtype=float),
            equity_curve=np.asarray(portfolio_values, dtype=float),
        ) | {"mean_reward": float(np.mean(rewards)) if rewards else 0.0}

    def _create_model(self, env: VecEnv) -> Any:
        algorithm = str(self.config.get("algorithm", "PPO")).upper()
        if algorithm == "PPO":
            return create_ppo_model(env=env, config=self.config)
        if algorithm == "SAC":
            return create_sac_model(env=env, config=self.config)
        raise ValueError(f"Unsupported algorithm: {algorithm}")
