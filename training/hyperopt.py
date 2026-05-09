"""Optuna hyperparameter search for PPO and SAC."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import optuna
import yaml
from optuna.pruners import MedianPruner

from training.trainer import RLTrainer


def optimize_hyperparameters(
    algorithm: str,
    n_trials: int = 100,
    config_path: Path | str | None = None,
) -> optuna.Study:
    """Run an Optuna study with validation Sharpe ratio as objective."""

    normalized_algorithm = algorithm.upper()
    base_config = _load_base_config(normalized_algorithm, config_path)

    def objective(trial: optuna.Trial) -> float:
        config = deepcopy(base_config)
        config["algorithm"] = normalized_algorithm
        config["hyperparameters"] = _sample_hyperparameters(trial, normalized_algorithm)
        config.setdefault("training", {})["total_timesteps"] = 500_000
        trainer = RLTrainer(config)
        model = trainer.train()
        metrics = trainer.evaluate(model)
        trial.report(float(metrics["sharpe"]), step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return float(metrics["sharpe"])

    study = optuna.create_study(direction="maximize", pruner=MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    return study


def _load_base_config(algorithm: str, config_path: Path | str | None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else Path("config") / f"{algorithm.lower()}_config.yaml"
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _sample_hyperparameters(trial: optuna.Trial, algorithm: str) -> dict[str, Any]:
    if algorithm == "PPO":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [512, 2048, 4096]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "n_epochs": trial.suggest_categorical("n_epochs", [3, 5, 10]),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 0.99),
            "clip_range": trial.suggest_float("clip_range", 0.10, 0.30),
            "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.01),
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantage": True,
        }
    if algorithm == "SAC":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "buffer_size": trial.suggest_categorical("buffer_size", [100_000, 300_000, 500_000]),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999),
            "tau": trial.suggest_float("tau", 0.005, 0.05),
            "ent_coef": trial.suggest_categorical("ent_coef", ["auto", 0.01, 0.05]),
            "train_freq": 1,
            "gradient_steps": 1,
        }
    raise ValueError(f"Unsupported algorithm: {algorithm}")
