"""Quant product services for live-feeling paper trading dashboard telemetry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

from data.downloader import DEFAULT_TICKERS, load_ticker_data
from data.preprocessor import split_data
from evaluation.metrics import annualized_volatility, max_drawdown

TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005
TOTAL_EXECUTION_COST = TRANSACTION_COST + SLIPPAGE
LIVE_CACHE_TTL = timedelta(seconds=60)
_LIVE_FEED_CACHE: dict[str, tuple[datetime, dict[str, Any]]] = {}

HEATMAP_METRICS: tuple[str, ...] = (
    "annualized_return",
    "sharpe",
    "sortino",
    "max_drawdown",
    "volatility",
    "win_rate",
    "number_of_trades",
    "profit_factor",
    "calmar",
)


@dataclass(frozen=True, slots=True)
class ProductContext:
    ticker: str
    full_data: pd.DataFrame
    test_data: pd.DataFrame
    cursor: int


def normalize_ticker(ticker: str) -> str:
    normalized = ticker.upper()
    if normalized == "BTC_USD":
        return "BTC-USD"
    return normalized


def build_product_context(ticker: str, root: Path, cursor: int | None = None) -> ProductContext:
    normalized = normalize_ticker(ticker)
    if normalized not in DEFAULT_TICKERS:
        raise ValueError(f"Unsupported ticker: {ticker}")

    df = load_ticker_data(ticker=normalized, cache_dir=root / "cache")
    _, _, test_df = split_data(df)
    if len(test_df) < 80:
        raise ValueError(f"Not enough test data for product telemetry: {ticker}")

    replay_cursor = cursor if cursor is not None else _time_based_cursor(len(test_df))
    replay_cursor = int(np.clip(replay_cursor, 30, len(test_df) - 1))
    return ProductContext(
        ticker=normalized,
        full_data=df,
        test_data=test_df,
        cursor=replay_cursor,
    )


def build_live_feed(context: ProductContext) -> dict[str, Any]:
    """Return a clearly-labeled market snapshot.

    The default is a deterministic simulated replay from cached bars. Set
    RL_TRADING_FEED_MODE=live to attempt a yfinance near-live snapshot first.
    """

    feed_mode = os.environ.get("RL_TRADING_FEED_MODE", "replay").lower()
    if feed_mode in {"live", "auto"}:
        live_snapshot = _fetch_live_market_snapshot(context.ticker)
        if live_snapshot is not None:
            return live_snapshot

    df = context.test_data
    row = df.iloc[context.cursor]
    previous = df.iloc[context.cursor - 1]
    price = float(row["close"])
    previous_close = float(previous["close"])
    change = price - previous_close
    pct_change = change / previous_close if previous_close else 0.0
    history = df.iloc[max(0, context.cursor - 60) : context.cursor + 1]

    return {
        "mode": "Simulated live replay",
        "symbol": context.ticker,
        "price": round(price, 4),
        "previousClose": round(previous_close, 4),
        "change": round(change, 4),
        "percentChange": round(pct_change, 6),
        "asOf": datetime.now(timezone.utc).isoformat(),
        "sourceTimestamp": _date_string(row.name),
        "latencyMs": 0,
        "history": [
            {
                "time": _date_string(index),
                "price": round(float(value), 4),
            }
            for index, value in history["close"].items()
        ],
    }


def _fetch_live_market_snapshot(ticker: str) -> dict[str, Any] | None:
    now = datetime.now(timezone.utc)
    cached = _LIVE_FEED_CACHE.get(ticker)
    if cached is not None and now - cached[0] < LIVE_CACHE_TTL:
        return cached[1]

    started = now
    try:
        history = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=True)
    except Exception:
        return None
    if history.empty or "Close" not in history.columns:
        return None

    close = history["Close"].dropna()
    if len(close) < 2:
        return None

    latest = float(close.iloc[-1])
    previous = float(close.iloc[-2])
    change = latest - previous
    pct_change = change / previous if previous else 0.0
    latency_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    payload = {
        "mode": "Live market data",
        "symbol": ticker,
        "price": round(latest, 4),
        "previousClose": round(previous, 4),
        "change": round(change, 4),
        "percentChange": round(pct_change, 6),
        "asOf": datetime.now(timezone.utc).isoformat(),
        "sourceTimestamp": _date_string(close.index[-1]),
        "latencyMs": latency_ms,
        "history": [
            {
                "time": _date_string(index),
                "price": round(float(value), 4),
            }
            for index, value in close.items()
        ],
    }
    _LIVE_FEED_CACHE[ticker] = (datetime.now(timezone.utc), payload)
    return payload


def detect_market_regime(context: ProductContext) -> dict[str, Any]:
    """Heuristic market regime classifier based on trend, volatility, and drawdown."""

    window = context.test_data.iloc[: context.cursor + 1].copy()
    close = window["close"].astype(float)
    returns = close.pct_change().dropna()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    vol20 = returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else returns.std()
    vol60 = returns.rolling(60).std().iloc[-1] if len(returns) >= 60 else returns.std()
    recent_vol = _safe_float(vol20, 0.0)
    baseline_vol = max(_safe_float(vol60, recent_vol), 1e-9)
    slope20 = (sma20.iloc[-1] / sma20.iloc[-10] - 1.0) if len(sma20.dropna()) >= 10 else 0.0
    above_sma = close.iloc[-1] > sma50.iloc[-1] if not np.isnan(sma50.iloc[-1]) else close.iloc[-1] > close.iloc[0]
    drawdown = max_drawdown(close.to_numpy())

    trend_score = float(np.clip(slope20 / 0.04, -1.0, 1.0))
    vol_ratio = recent_vol / baseline_vol
    high_vol = vol_ratio > 1.15 or recent_vol * np.sqrt(252) > 0.30
    low_vol = vol_ratio < 0.85

    if high_vol and trend_score > 0.15 and above_sma:
        regime = "High Volatility Uptrend"
    elif high_vol and (trend_score < -0.15 or drawdown < -0.12):
        regime = "High Volatility Downtrend"
    elif trend_score > 0.20 and above_sma:
        regime = "Trending Up"
    elif trend_score < -0.20 and not above_sma:
        regime = "Trending Down"
    elif high_vol:
        regime = "Volatility Expansion"
    elif low_vol:
        regime = "Volatility Compression"
    else:
        regime = "Sideways"

    confidence = float(np.clip(0.45 + abs(trend_score) * 0.30 + abs(vol_ratio - 1.0) * 0.20, 0.35, 0.92))
    explanation = _regime_explanation(regime, above_sma, recent_vol, baseline_vol, slope20, drawdown)
    return {
        "label": regime,
        "confidence": round(confidence, 3),
        "method": "Heuristic regime detection",
        "explanation": explanation,
        "factors": [
            {"name": "20-day MA slope", "value": round(float(slope20), 5), "status": "up" if slope20 > 0 else "down"},
            {"name": "Volatility ratio", "value": round(float(vol_ratio), 3), "status": "elevated" if vol_ratio > 1 else "compressed"},
            {"name": "Price vs 50-day MA", "value": 1.0 if above_sma else -1.0, "status": "above" if above_sma else "below"},
            {"name": "Close drawdown", "value": round(float(drawdown), 4), "status": "risk" if drawdown < -0.10 else "normal"},
        ],
    }


def build_agent_signal(context: ProductContext, risk_status: str | None = None) -> dict[str, Any]:
    """Return heuristic action explanation when no trained checkpoint signal exists."""

    window = context.test_data.iloc[: context.cursor + 1].copy()
    latest = window.iloc[-1]
    close = window["close"].astype(float)
    returns = close.pct_change().dropna()
    momentum_20 = close.iloc[-1] / close.iloc[max(0, len(close) - 21)] - 1.0
    momentum_5 = close.iloc[-1] / close.iloc[max(0, len(close) - 6)] - 1.0
    rsi = float(latest.get("rsi_14", 50.0))
    vol20 = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 5 else 0.0

    raw_score = 0.0
    raw_score += np.clip(momentum_20 / 0.10, -1.0, 1.0) * 0.38
    raw_score += np.clip(momentum_5 / 0.04, -1.0, 1.0) * 0.24
    raw_score += np.clip((rsi - 50.0) / 25.0, -1.0, 1.0) * 0.18
    raw_score -= np.clip((vol20 - 0.35) / 0.40, 0.0, 1.0) * 0.20

    if risk_status == "Halted":
        action = "Flat"
        target_exposure = 0.0
        raw_score = 0.0
        rationale = "Risk status is halted, so the paper engine blocks additional exposure."
    elif raw_score > 0.18:
        action = "Long"
        target_exposure = min(0.75, max(0.15, raw_score))
        rationale = "Momentum and trend inputs favor long exposure, sized down by volatility."
    elif raw_score < -0.18:
        action = "Short"
        target_exposure = max(-0.75, min(-0.15, raw_score))
        rationale = "Momentum inputs are negative enough to justify short paper exposure."
    else:
        action = "Hold"
        target_exposure = 0.0
        rationale = "Signals are mixed or weak; the heuristic avoids unnecessary turnover."

    confidence = float(np.clip(0.42 + abs(raw_score) * 0.55, 0.35, 0.88))
    return {
        "mode": "Heuristic explanation",
        "action": action,
        "targetExposure": round(float(target_exposure), 3),
        "confidence": round(confidence, 3),
        "rationale": rationale,
        "signals": [
            _signal("20-day momentum", momentum_20, "positive" if momentum_20 > 0 else "negative"),
            _signal("5-day momentum", momentum_5, "positive" if momentum_5 > 0 else "negative"),
            _signal("RSI level", rsi, "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"),
            _signal("Annualized volatility", vol20, "elevated" if vol20 > 0.35 else "normal"),
            _signal("Risk status", 1.0 if risk_status == "Halted" else 0.0, risk_status or "Normal"),
        ],
    }


def simulate_paper_portfolio(context: ProductContext, signal: dict[str, Any]) -> dict[str, Any]:
    """Simulate paper execution over the replay window with cost/slippage realism."""

    df = context.test_data.iloc[max(0, context.cursor - 90) : context.cursor + 1].copy()
    cash = 100_000.0
    shares = 0.0
    entry_price = 0.0
    realized_pnl = 0.0
    equity_peak = cash
    equity_curve: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    risk_state = {
        "halted": False,
        "daily_loss_limit": -0.03,
        "max_exposure": 0.75,
        "position_limit": 1.0,
        "stop_loss": -0.08,
    }

    for idx, (timestamp, row) in enumerate(df.iterrows()):
        price = float(row["close"])
        equity = cash + shares * price
        equity_peak = max(equity_peak, equity)
        drawdown = equity / max(equity_peak, 1e-9) - 1.0

        if drawdown <= risk_state["stop_loss"]:
            risk_state["halted"] = True

        desired_exposure = _target_exposure_for_step(df, idx, signal)
        if risk_state["halted"]:
            desired_exposure = 0.0
        desired_exposure = float(np.clip(desired_exposure, -risk_state["max_exposure"], risk_state["max_exposure"]))
        target_notional = desired_exposure * equity
        current_notional = shares * price
        trade_notional = target_notional - current_notional

        if abs(trade_notional) > max(50.0, equity * 0.005):
            execution_price = price * (1.0 + np.sign(trade_notional) * SLIPPAGE)
            trade_shares = trade_notional / execution_price
            trade_value = trade_shares * execution_price
            cost = abs(trade_value) * TRANSACTION_COST

            if trade_value > 0 and trade_value + cost > cash:
                affordable_value = max(cash - cost, 0.0)
                trade_shares = affordable_value / execution_price
                trade_value = trade_shares * execution_price
                cost = abs(trade_value) * TRANSACTION_COST

            previous_shares = shares
            cash -= trade_value + cost
            shares += trade_shares
            if abs(shares) > 1e-9 and abs(previous_shares) < 1e-9:
                entry_price = execution_price
            if np.sign(previous_shares) != np.sign(shares) and abs(previous_shares) > 1e-9:
                realized_pnl += previous_shares * (execution_price - entry_price) - cost
                entry_price = execution_price if abs(shares) > 1e-9 else 0.0

            trades.append(
                {
                    "timestamp": _date_string(timestamp),
                    "side": "BUY" if trade_shares > 0 else "SELL",
                    "price": round(execution_price, 4),
                    "shares": round(float(trade_shares), 5),
                    "notional": round(float(trade_value), 2),
                    "cost": round(float(cost), 2),
                    "mode": "paper",
                },
            )

        equity = cash + shares * price
        unrealized = shares * (price - entry_price) if abs(shares) > 1e-9 and entry_price > 0 else 0.0
        equity_peak = max(equity_peak, equity)
        equity_curve.append(
            {
                "time": _date_string(timestamp),
                "equity": round(equity, 2),
                "cash": round(cash, 2),
                "exposure": round((shares * price) / max(equity, 1e-9), 4),
                "drawdown": round(equity / max(equity_peak, 1e-9) - 1.0, 5),
                "unrealizedPnl": round(unrealized, 2),
            },
        )

    latest_price = float(df["close"].iloc[-1])
    final_equity = cash + shares * latest_price
    unrealized_pnl = shares * (latest_price - entry_price) if abs(shares) > 1e-9 and entry_price > 0 else 0.0
    exposure = shares * latest_price / max(final_equity, 1e-9)
    action = signal["action"]
    if abs(shares) > 1e-9 and action == "Hold":
        action = "Hold"

    return {
        "mode": "Paper trading simulation",
        "cash": round(cash, 2),
        "positionShares": round(float(shares), 5),
        "positionMarketValue": round(float(shares * latest_price), 2),
        "exposure": round(float(exposure), 4),
        "entryPrice": round(float(entry_price), 4) if entry_price else None,
        "realizedPnl": round(float(realized_pnl), 2),
        "unrealizedPnl": round(float(unrealized_pnl), 2),
        "totalEquity": round(float(final_equity), 2),
        "currentAction": action,
        "allowLeverage": False,
        "equityCurve": equity_curve,
        "trades": trades[-12:],
        "executionAssumptions": {
            "transactionCost": TRANSACTION_COST,
            "slippage": SLIPPAGE,
            "realMoney": False,
            "brokerConnected": False,
        },
    }


def evaluate_risk(paper: dict[str, Any], context: ProductContext) -> dict[str, Any]:
    equity_curve = pd.DataFrame(paper["equityCurve"])
    equity = equity_curve["equity"].to_numpy(dtype=float)
    drawdown_series = equity_curve["drawdown"].to_numpy(dtype=float)
    current_drawdown = float(drawdown_series[-1]) if len(drawdown_series) else 0.0
    max_dd = max_drawdown(equity)
    returns = pd.Series(equity).pct_change().dropna()
    volatility = annualized_volatility(returns.to_numpy()) if len(returns) > 1 else 0.0
    exposure = abs(float(paper["exposure"]))

    guardrails = [
        _guardrail("Daily loss limit", current_drawdown > -0.03, current_drawdown, -0.03),
        _guardrail("Max drawdown", max_dd > -0.12, max_dd, -0.12),
        _guardrail("Max exposure", exposure <= 0.75, exposure, 0.75),
        _guardrail("Position limit", exposure <= 1.0, exposure, 1.0),
        _guardrail("Stop-loss threshold", current_drawdown > -0.08, current_drawdown, -0.08),
    ]
    violations = [item for item in guardrails if not item["passing"]]
    status = "Normal"
    if violations:
        status = "Halted" if any(item["severity"] == "halt" for item in violations) else "Warning"

    return {
        "status": status,
        "currentDrawdown": round(current_drawdown, 5),
        "maxDrawdown": round(float(max_dd), 5),
        "volatility": round(float(volatility), 5),
        "dailyLossLimit": -0.03,
        "maxExposure": 0.75,
        "positionLimit": 1.0,
        "stopLossThreshold": -0.08,
        "killSwitchActive": status == "Halted",
        "violations": violations,
        "guardrails": guardrails,
        "mode": "Paper risk controls",
    }


def build_training_sessions(root: Path) -> list[dict[str, Any]]:
    configs = [
        ("PPO", root / "config" / "ppo_config.yaml"),
        ("SAC", root / "config" / "sac_config.yaml"),
    ]
    rows: list[dict[str, Any]] = []
    for algorithm, path in configs:
        config = _load_yaml(path)
        checkpoint = _find_checkpoint(root / "models", algorithm.lower())
        rows.append(
            {
                "id": f"{algorithm.lower()}-{config.get('ticker', 'AAPL')}",
                "algorithm": algorithm,
                "ticker": config.get("ticker", "AAPL"),
                "timesteps": config.get("training", {}).get("total_timesteps", 0),
                "rewardFunction": config.get("environment", {}).get("reward_function", "configurable"),
                "status": "completed" if checkpoint else "pending",
                "bestValidationMetric": None,
                "checkpointAvailable": checkpoint is not None,
                "checkpointPath": str(checkpoint) if checkpoint else None,
                "notes": "Ready for checkpoint-backed dashboard telemetry." if checkpoint else "Configured but no local checkpoint artifact found.",
            },
        )
    return rows


def build_strategy_heatmap(benchmarks: pd.DataFrame, ticker: str, models_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    ticker_rows = benchmarks[benchmarks["ticker"] == ticker].copy()
    for strategy in ["PPO", "SAC"]:
        checkpoint = _find_checkpoint(models_dir, strategy.lower())
        rows.append(
            {
                "strategy": strategy,
                "status": "Awaiting checkpoint" if checkpoint is None else "Checkpoint available",
                "metrics": {metric: None for metric in HEATMAP_METRICS},
            },
        )

    for _, row in ticker_rows.iterrows():
        rows.append(
            {
                "strategy": _label_strategy(str(row["strategy"])),
                "status": "Benchmark",
                "metrics": {metric: _safe_float(row.get(metric)) for metric in HEATMAP_METRICS},
            },
        )
    return {"metrics": list(HEATMAP_METRICS), "rows": rows}


def websocket_event_bundle(ticker: str, root: Path, cursor: int | None = None) -> dict[str, Any]:
    context = build_product_context(ticker, root, cursor)
    live_feed = build_live_feed(context)
    preliminary_signal = build_agent_signal(context)
    paper = simulate_paper_portfolio(context, preliminary_signal)
    risk = evaluate_risk(paper, context)
    signal = build_agent_signal(context, risk["status"])
    if signal["action"] != preliminary_signal["action"]:
        paper = simulate_paper_portfolio(context, signal)
        risk = evaluate_risk(paper, context)
    return {
        "cursor": context.cursor,
        "events": [
            {"type": "PRICE_UPDATE", "payload": live_feed},
            {"type": "PAPER_TRADE_UPDATE", "payload": paper},
            {"type": "RISK_UPDATE", "payload": risk},
            {"type": "REGIME_UPDATE", "payload": detect_market_regime(context)},
            {"type": "AGENT_SIGNAL", "payload": signal},
            {"type": "EXPERIMENT_UPDATE", "payload": build_training_sessions(root)},
        ],
    }


def _target_exposure_for_step(df: pd.DataFrame, idx: int, terminal_signal: dict[str, Any]) -> float:
    if idx < 20:
        return 0.0
    close = df["close"].astype(float)
    momentum_10 = close.iloc[idx] / close.iloc[max(0, idx - 10)] - 1.0
    vol = close.pct_change().iloc[max(0, idx - 20) : idx + 1].std() * np.sqrt(252)
    exposure = float(np.clip(momentum_10 / 0.08, -0.6, 0.6))
    if vol > 0.35:
        exposure *= 0.55
    if idx == len(df) - 1:
        exposure = float(terminal_signal["targetExposure"])
    return exposure


def _regime_explanation(regime: str, above_sma: bool, vol: float, baseline_vol: float, slope: float, drawdown: float) -> str:
    relation = "above" if above_sma else "below"
    vol_state = "elevated" if vol > baseline_vol else "subdued"
    return (
        f"{regime} based on price trading {relation} its medium moving average, "
        f"{vol_state} rolling volatility, 20-day slope of {slope:.2%}, and drawdown of {drawdown:.2%}."
    )


def _guardrail(name: str, passing: bool, current: float, limit: float) -> dict[str, Any]:
    severity = "halt" if name in {"Max drawdown", "Stop-loss threshold"} else "warning"
    return {
        "name": name,
        "passing": bool(passing),
        "current": round(float(current), 5),
        "limit": round(float(limit), 5),
        "severity": severity,
    }


def _signal(name: str, value: float, interpretation: str) -> dict[str, Any]:
    return {
        "name": name,
        "value": round(float(value), 5),
        "interpretation": interpretation,
    }


def _time_based_cursor(length: int) -> int:
    usable = max(length - 30, 1)
    seconds = int(datetime.now(timezone.utc).timestamp())
    return 30 + seconds % usable


def _date_string(value: Any) -> str:
    if hasattr(value, "date"):
        return str(value.date())
    return str(value)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _find_checkpoint(models_dir: Path, algorithm: str) -> Path | None:
    if not models_dir.exists():
        return None
    candidates = sorted(path for path in models_dir.glob("*.zip") if algorithm in path.name.lower())
    return candidates[0] if candidates else None


def _label_strategy(strategy: str) -> str:
    return {
        "buy_hold": "Buy & Hold",
        "sma_crossover": "SMA Crossover",
        "momentum": "Momentum",
        "random": "Random",
    }.get(strategy, strategy)
