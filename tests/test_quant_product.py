from __future__ import annotations

import numpy as np
import pandas as pd

from backend.quant_product import (
    ProductContext,
    build_agent_signal,
    build_market_feed,
    detect_market_regime,
    evaluate_risk,
    simulate_paper_portfolio,
)


def _synthetic_context() -> ProductContext:
    index = pd.date_range("2024-01-02", periods=160, freq="B")
    trend = np.linspace(100.0, 124.0, len(index))
    cycle = np.sin(np.linspace(0, 8, len(index))) * 2.5
    close = trend + cycle
    df = pd.DataFrame(
        {
            "close": close,
            "rsi_14": np.clip(50 + np.gradient(close) * 10, 20, 80),
        },
        index=index,
    )
    return ProductContext(ticker="AAPL", full_data=df, test_data=df, cursor=120)


def test_product_feed_regime_signal_and_paper_portfolio_are_labeled() -> None:
    context = _synthetic_context()
    feed = build_market_feed(context)
    regime = detect_market_regime(context)
    signal = build_agent_signal(context)
    paper = simulate_paper_portfolio(context, signal)
    risk = evaluate_risk(paper, context)

    assert feed["mode"] == "Historical Market Stream"
    assert feed["sourceKind"] == "historical_replay"
    assert "2024 out-of-sample" in feed["dataBasis"]
    assert regime["method"] == "Heuristic regime detection"
    assert signal["mode"] == "Heuristic explanation"
    assert paper["mode"] == "Paper trading simulation"
    assert paper["cash"] >= 0.0
    assert abs(paper["exposure"]) <= 0.75 + 1e-6
    assert paper["executionAssumptions"]["realMoney"] is False
    assert paper["executionAssumptions"]["brokerConnected"] is False
    assert risk["status"] in {"Normal", "Warning", "Halted"}
    assert len(risk["guardrails"]) >= 5
