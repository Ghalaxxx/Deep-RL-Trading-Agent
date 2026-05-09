from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


def test_dashboard_api_serves_product_payload() -> None:
    client = TestClient(app)
    response = client.get("/api/dashboard?ticker=AAPL")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "AAPL"
    assert payload["liveFeed"]["mode"] in {"Live market data", "Simulated live replay", "Cached demo mode"}
    assert payload["paperPortfolio"]["mode"] == "Paper trading simulation"
    assert payload["paperPortfolio"]["executionAssumptions"]["realMoney"] is False
    assert payload["paperPortfolio"]["executionAssumptions"]["brokerConnected"] is False
    assert payload["risk"]["status"] in {"Normal", "Warning", "Halted"}
    assert payload["marketRegime"]["method"] == "Heuristic regime detection"
    assert payload["agentExplainability"]["mode"] == "Heuristic explanation"
    assert payload["portfolio"]["endingValue"] > 0
    assert len(payload["equityCurve"]) > 0
    assert len(payload["modelComparison"]) == 2
    assert len(payload["trainingSessions"]) >= 2
    assert {"PPO", "SAC"}.issubset({row["strategy"] for row in payload["strategyHeatmap"]["rows"]})
    assert payload["realtime"]["transport"] == "websocket"
    assert len(payload["safeguards"]) >= 4


def test_dashboard_api_rejects_unknown_ticker() -> None:
    client = TestClient(app)
    response = client.get("/api/dashboard?ticker=NOPE")
    assert response.status_code == 404


def test_live_snapshot_serves_realtime_event_contract() -> None:
    client = TestClient(app)
    response = client.get("/api/live/AAPL?cursor=80")
    assert response.status_code == 200
    payload = response.json()
    event_types = {event["type"] for event in payload["events"]}
    assert {
        "PRICE_UPDATE",
        "PAPER_TRADE_UPDATE",
        "RISK_UPDATE",
        "REGIME_UPDATE",
        "AGENT_SIGNAL",
        "EXPERIMENT_UPDATE",
    }.issubset(event_types)
