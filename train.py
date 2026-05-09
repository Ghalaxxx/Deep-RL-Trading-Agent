"""Command-line interface for training, optimizing, and evaluating agents."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep RL Trading Agent")
    parser.add_argument("--algorithm", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--reward", choices=["sharpe", "sortino", "pnl"], default="sharpe")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--benchmark-data", action="store_true")
    parser.add_argument("--export-replay", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.algorithm, args.config_dir)
    config["ticker"] = args.ticker
    config.setdefault("environment", {})["reward_function"] = args.reward
    if args.timesteps is not None:
        config.setdefault("training", {})["total_timesteps"] = args.timesteps
    if args.wandb_mode is not None:
        config.setdefault("wandb", {})["mode"] = args.wandb_mode

    if args.prepare_data:
        from data.manifest import build_data_manifest, write_data_manifest

        data_config = config.get("data", {})
        paths_config = config.get("paths", {})
        manifest = build_data_manifest(
            tickers=data_config.get("tickers"),
            start=data_config.get("start", "2015-01-01"),
            end=data_config.get("end", "2025-01-01"),
            cache_dir=paths_config.get("cache", "cache"),
            force=args.force_download,
        )
        output_path = write_data_manifest(manifest, args.reports_dir / "data_manifest.csv")
        print(f"Data manifest written to {output_path}")
        print(manifest.to_string(index=False))
        return

    if args.benchmark_data:
        from evaluation.reporting import build_baseline_leaderboard, write_report

        data_config = config.get("data", {})
        paths_config = config.get("paths", {})
        leaderboard = build_baseline_leaderboard(
            tickers=data_config.get("tickers"),
            start=data_config.get("start", "2015-01-01"),
            end=data_config.get("end", "2025-01-01"),
            cache_dir=paths_config.get("cache", "cache"),
        )
        output_path = write_report(leaderboard, args.reports_dir / "baseline_benchmarks.csv")
        print(f"Baseline benchmark report written to {output_path}")
        print(leaderboard.to_string(index=False))
        return

    if args.optimize:
        from training.hyperopt import optimize_hyperparameters

        study = optimize_hyperparameters(
            algorithm=args.algorithm,
            n_trials=args.n_trials,
            config_path=args.config_dir / f"{args.algorithm}_config.yaml",
        )
        print(f"Best Sharpe: {study.best_value:.4f}")
        print(study.best_params)
        return

    if args.evaluate or args.compare:
        from data.downloader import load_ticker_data
        from data.preprocessor import split_data
        from evaluation.backtester import WalkForwardBacktester

        if args.model_path is None:
            raise SystemExit("--model-path is required for --evaluate and --compare")
        model = load_model(args.algorithm, args.model_path)
        data = load_ticker_data(ticker=args.ticker)
        _, _, test_df = split_data(data)
        backtester = WalkForwardBacktester(test_df)
        vecnormalize_path = Path(config.get("paths", {}).get("vecnormalize", "models/vecnormalize.pkl"))
        model_stats = backtester.run_model(model, test_df, vecnormalize_path=vecnormalize_path)
        print_metrics("RL Agent", model_stats)
        if args.export_replay is not None:
            from evaluation.replay import run_policy_replay

            replay = run_policy_replay(
                model,
                test_df,
                output_path=args.export_replay,
                vecnormalize_path=vecnormalize_path,
            )
            print(f"\nReplay exported to {args.export_replay} ({len(replay)} decisions)")
        if args.compare:
            for name, stats in backtester.benchmark_statistics(test_df).items():
                print_metrics(name, stats)
        return

    from training.trainer import RLTrainer

    trainer = RLTrainer(config)
    trainer.train()


def load_config(algorithm: str, config_dir: Path) -> dict[str, Any]:
    env_config = _read_yaml(config_dir / "env_config.yaml")
    agent_config = _read_yaml(config_dir / f"{algorithm}_config.yaml")
    return _deep_merge(env_config, agent_config)


def load_model(algorithm: str, path: Path) -> Any:
    if algorithm == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(path)
    from stable_baselines3 import SAC

    return SAC.load(path)


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(f"\n{name}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


if __name__ == "__main__":
    main()
