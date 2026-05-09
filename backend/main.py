"""FastAPI service for the Deep RL Trading Agent dashboard."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from agents.baselines import buy_and_hold, momentum_strategy, random_agent, sma_crossover
from data.downloader import DEFAULT_TICKERS
from evaluation.metrics import max_drawdown
from backend.quant_product import (
    build_agent_signal,
    build_market_feed,
    build_product_context,
    build_strategy_heatmap,
    build_training_sessions,
    detect_market_regime,
    evaluate_risk,
    market_replay_event_bundle,
    normalize_ticker,
    simulate_paper_portfolio,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

app = FastAPI(
    title="Deep RL Trading Agent API",
    description="Dashboard API for model evaluation, benchmark reports, and demo inference telemetry.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "rl-trading-dashboard"}


@app.get("/api/dashboard")
def dashboard(ticker: str = Query(default="AAPL"), cursor: int | None = Query(default=None)) -> dict[str, Any]:
    normalized_ticker = normalize_ticker(ticker)
    if normalized_ticker not in DEFAULT_TICKERS:
        raise HTTPException(status_code=404, detail=f"Unsupported ticker: {ticker}")

    manifest = _load_manifest()
    benchmarks = _load_benchmarks()
    ticker_benchmarks = benchmarks[benchmarks["ticker"] == normalized_ticker].copy()
    if ticker_benchmarks.empty:
        raise HTTPException(status_code=404, detail=f"No benchmark report for ticker: {normalized_ticker}")

    context = build_product_context(normalized_ticker, ROOT, cursor=cursor)
    test_df = context.test_data
    strategy_curves = _strategy_curves(test_df)
    best_row = ticker_benchmarks.sort_values(["sharpe", "annualized_return"], ascending=False).iloc[0]
    selected_strategy = str(best_row["strategy"])
    selected_curve = strategy_curves.get(selected_strategy, strategy_curves["buy_hold"])
    market_feed = build_market_feed(context)
    preliminary_signal = build_agent_signal(context)
    paper_portfolio = simulate_paper_portfolio(context, preliminary_signal)
    risk = evaluate_risk(paper_portfolio, context)
    agent_signal = build_agent_signal(context, risk["status"])
    if agent_signal["action"] != preliminary_signal["action"]:
        paper_portfolio = simulate_paper_portfolio(context, agent_signal)
        risk = evaluate_risk(paper_portfolio, context)

    return {
        "mode": _runtime_mode(),
        "ticker": normalized_ticker,
        "tickers": DEFAULT_TICKERS,
        "marketFeed": market_feed,
        "paperPortfolio": paper_portfolio,
        "risk": risk,
        "marketRegime": detect_market_regime(context),
        "agentExplainability": agent_signal,
        "portfolio": _portfolio_summary(selected_curve, best_row),
        "metrics": _metric_cards(best_row),
        "equityCurve": _serialize_curves(strategy_curves),
        "drawdownCurve": _drawdown_curve(selected_curve),
        "modelComparison": _model_comparison(),
        "backtests": _records(ticker_benchmarks),
        "tradeTimeline": _trade_timeline(strategy_curves[selected_strategy], selected_strategy),
        "trainingAnalytics": _training_analytics(),
        "experiments": _experiment_overview(),
        "trainingSessions": build_training_sessions(ROOT),
        "strategyHeatmap": build_strategy_heatmap(benchmarks, normalized_ticker, MODELS_DIR),
        "inferenceDemo": _inference_demo(test_df, selected_curve, selected_strategy),
        "dataManifest": _records(manifest),
        "marketTransport": {
            "transport": "websocket replay",
            "endpoint": f"/ws/replay/{normalized_ticker}",
            "fallback": "polling /api/replay",
            "eventTypes": [
                "MARKET_REPLAY_UPDATE",
                "PAPER_TRADE_UPDATE",
                "RISK_UPDATE",
                "REGIME_UPDATE",
                "AGENT_SIGNAL",
                "EXPERIMENT_UPDATE",
            ],
        },
        "safeguards": [
            "Indicator warm-up rows are dropped, not backward-filled.",
            "2024 is reserved for out-of-sample evaluation.",
            "VecNormalize stats are reused at evaluation when available.",
            "Transaction costs and slippage are included in trading simulation.",
            "Benchmark reports use realized trade segments for trade metrics.",
            "Dashboard values are evaluation/demo telemetry unless checkpoint-backed mode is active.",
            "Paper trading simulation never routes real-money orders.",
        ],
    }


@app.get("/api/backtests")
def backtests() -> dict[str, Any]:
    benchmarks = _load_benchmarks()
    return {"rows": _records(benchmarks)}


@app.get("/api/experiments")
def experiments() -> dict[str, Any]:
    return {
        "models": _model_comparison(),
        "experiments": _experiment_overview(),
        "sessions": build_training_sessions(ROOT),
    }


@app.get("/api/replay/{ticker}")
def replay_snapshot(ticker: str, cursor: int | None = Query(default=None)) -> dict[str, Any]:
    try:
        context = build_product_context(ticker, ROOT, cursor=cursor)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    bundle = market_replay_event_bundle(context.ticker, ROOT, cursor=context.cursor)
    return {
        "ticker": context.ticker,
        "cursor": bundle["cursor"],
        "events": bundle["events"],
    }


@app.websocket("/ws/replay/{ticker}")
async def replay_websocket(websocket: WebSocket, ticker: str) -> None:
    await websocket.accept()
    try:
        normalized_ticker = normalize_ticker(ticker)
        cursor = None
        while True:
            bundle = market_replay_event_bundle(normalized_ticker, ROOT, cursor=cursor)
            await websocket.send_json(
                {
                    "ticker": normalized_ticker,
                    "cursor": bundle["cursor"],
                    "events": bundle["events"],
                    "transport": "websocket replay",
                },
            )
            cursor = bundle["cursor"] + 1
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        return
    except ValueError as exc:
        await websocket.send_json({"type": "ERROR", "detail": str(exc)})
        await websocket.close()


def _load_manifest() -> pd.DataFrame:
    path = REPORTS_DIR / "data_manifest.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run `python train.py --prepare-data` first.")
    return pd.read_csv(path)


def _load_benchmarks() -> pd.DataFrame:
    path = REPORTS_DIR / "baseline_benchmarks.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run `python train.py --benchmark-data` first.")
    data = pd.read_csv(path)
    return data.replace([np.inf, -np.inf], np.nan)


def _strategy_curves(test_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "buy_hold": buy_and_hold(test_df),
        "sma_crossover": sma_crossover(test_df),
        "momentum": momentum_strategy(test_df),
        "random": random_agent(test_df),
    }


def _portfolio_summary(curve: pd.DataFrame, best_row: pd.Series) -> dict[str, Any]:
    initial_value = float(curve["equity"].iloc[0])
    ending_value = float(curve["equity"].iloc[-1])
    return {
        "label": "2024 out-of-sample benchmark view",
        "strategy": str(best_row["strategy"]),
        "initialValue": initial_value,
        "endingValue": ending_value,
        "totalReturn": (ending_value / initial_value) - 1.0 if initial_value else 0.0,
        "sharpe": _safe_float(best_row["sharpe"]),
        "maxDrawdown": _safe_float(best_row["max_drawdown"]),
        "status": "Benchmark demo, no broker execution" if not _has_trained_checkpoint() else "Checkpoint-backed evaluation",
    }


def _metric_cards(row: pd.Series) -> list[dict[str, Any]]:
    strategy = str(row["strategy"])
    trade_win_rate = None if strategy == "buy_hold" else _safe_float(row["win_rate"])
    return [
        {"label": "Annual Return", "value": _safe_float(row["annualized_return"]), "format": "percent"},
        {"label": "Sharpe", "value": _safe_float(row["sharpe"]), "format": "number"},
        {"label": "Max Drawdown", "value": _safe_float(row["max_drawdown"]), "format": "percent"},
        {"label": "Trade Win Rate", "value": trade_win_rate, "format": "percent"},
        {"label": "Trades", "value": _safe_float(row["number_of_trades"]), "format": "integer"},
        {"label": "Avg Hold", "value": _safe_float(row["average_holding_period"]), "format": "days"},
    ]


def _serialize_curves(curves: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    for strategy, curve in curves.items():
        for timestamp, value in curve["equity"].items():
            key = str(timestamp.date()) if hasattr(timestamp, "date") else str(timestamp)
            combined.setdefault(key, {"date": key})
            combined[key][strategy] = round(float(value), 2)
    return list(combined.values())


def _drawdown_curve(curve: pd.DataFrame) -> list[dict[str, Any]]:
    equity = curve["equity"].astype(float)
    running_peak = equity.cummax()
    drawdown = equity / running_peak.replace(0.0, np.nan) - 1.0
    return [
        {
            "date": str(timestamp.date()) if hasattr(timestamp, "date") else str(timestamp),
            "drawdown": round(float(value), 5),
        }
        for timestamp, value in drawdown.fillna(0.0).items()
    ]


def _model_comparison() -> list[dict[str, Any]]:
    ppo_config = _load_yaml(ROOT / "config" / "ppo_config.yaml")
    sac_config = _load_yaml(ROOT / "config" / "sac_config.yaml")
    return [
        _model_status("PPO", ppo_config, ["best_model.zip", "final_ppo_AAPL.zip", "best_ppo_AAPL.zip"]),
        _model_status("SAC", sac_config, ["final_sac_SPY.zip", "best_sac_SPY.zip"]),
    ]


def _model_status(name: str, config: dict[str, Any], candidates: list[str]) -> dict[str, Any]:
    existing = [str(MODELS_DIR / candidate) for candidate in candidates if (MODELS_DIR / candidate).exists()]
    hyperparameters = config.get("hyperparameters", {})
    return {
        "name": name,
        "status": "ready" if existing else "awaiting checkpoint",
        "policy": config.get("policy", "MlpPolicy"),
        "reward": config.get("environment", {}).get("reward_function", "configurable"),
        "timesteps": config.get("training", {}).get("total_timesteps", 1_000_000),
        "learningRate": hyperparameters.get("learning_rate"),
        "checkpoint": existing[0] if existing else None,
        "notes": "Uses saved VecNormalize stats for evaluation." if existing else "Train model to populate checkpoint-backed metrics.",
    }


def _training_analytics() -> dict[str, Any]:
    return {
        "source": "awaiting_training_run",
        "label": "No checkpoint-backed reward telemetry loaded",
        "isDemo": False,
        "points": [],
        "channels": [
            "policy reward",
            "validation Sharpe",
            "drawdown",
            "turnover",
            "evaluation return",
        ],
    }


def _experiment_overview() -> list[dict[str, Any]]:
    return [
        {"name": "Data cache", "status": "ready", "detail": "9 tickers, adjusted OHLCV, parquet cache"},
        {"name": "Leakage checks", "status": "ready", "detail": "No backward-filled indicators; chronological splits"},
        {"name": "PPO training", "status": "pending" if not _has_model("ppo") else "ready", "detail": "On-policy baseline; no checkpoint in demo mode"},
        {"name": "SAC training", "status": "pending" if not _has_model("sac") else "ready", "detail": "Off-policy continuous control; no checkpoint in demo mode"},
        {"name": "W&B telemetry", "status": "optional", "detail": "Enabled by training config"},
    ]


def _inference_demo(test_df: pd.DataFrame, curve: pd.DataFrame, strategy: str) -> dict[str, Any]:
    last_rows = curve.tail(8)
    actions = last_rows.get("position", pd.Series(0.0, index=last_rows.index)).astype(float)
    close = test_df["close"].reindex(last_rows.index).ffill()
    return {
        "mode": "checkpoint inference" if _has_trained_checkpoint() else "baseline demo",
        "strategy": strategy,
        "message": "Current sequence is a baseline exposure replay. Load a trained PPO/SAC checkpoint to run policy inference.",
        "steps": [
            {
                "date": str(timestamp.date()) if hasattr(timestamp, "date") else str(timestamp),
                "close": round(float(close.loc[timestamp]), 2),
                "action": round(float(actions.loc[timestamp]), 3),
                "equity": round(float(row["equity"]), 2),
                "drawdown": round(float(max_drawdown(last_rows.loc[:timestamp, "equity"].to_numpy())), 4),
            }
            for timestamp, row in last_rows.iterrows()
        ],
    }


def _trade_timeline(curve: pd.DataFrame, strategy: str) -> list[dict[str, Any]]:
    position = curve.get("position", pd.Series(0.0, index=curve.index)).astype(float)
    returns = curve["returns"].astype(float)
    changes = position.diff().fillna(position).abs() > 1e-6
    rows = curve[changes].tail(8)
    if rows.empty:
        rows = curve.tail(4)
    events: list[dict[str, Any]] = []
    for timestamp, row in rows.iterrows():
        pos = float(position.loc[timestamp])
        side = "Long" if pos > 0 else "Short" if pos < 0 else "Flat"
        events.append(
            {
                "date": str(timestamp.date()) if hasattr(timestamp, "date") else str(timestamp),
                "strategy": strategy,
                "side": side,
                "position": round(pos, 3),
                "return": round(float(returns.loc[timestamp]), 5),
                "equity": round(float(row["equity"]), 2),
            },
        )
    return events


def _runtime_mode() -> dict[str, Any]:
    return {
        "label": "Checkpoint-backed" if _has_trained_checkpoint() else "Demo mode",
        "hasCheckpoint": _has_trained_checkpoint(),
        "vecNormalize": (MODELS_DIR / "vecnormalize.pkl").exists(),
        "dataSource": "Local yfinance parquet cache + generated reports",
    }


def _has_trained_checkpoint() -> bool:
    return _has_model("ppo") or _has_model("sac") or (MODELS_DIR / "best_model.zip").exists()


def _has_model(kind: str) -> bool:
    if not MODELS_DIR.exists():
        return False
    return any(path.suffix == ".zip" and kind.lower() in path.name.lower() for path in MODELS_DIR.iterdir())


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    sanitized = df.replace([np.inf, -np.inf], np.nan).where(pd.notna(df), None)
    return sanitized.to_dict(orient="records")


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result
