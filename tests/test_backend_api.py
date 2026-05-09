from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


def test_dashboard_api_serves_product_payload() -> None:
    client = TestClient(app)
    response = client.get("/api/dashboard?ticker=AAPL")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "AAPL"
    assert payload["portfolio"]["endingValue"] > 0
    assert len(payload["equityCurve"]) > 0
    assert len(payload["modelComparison"]) == 2
    assert len(payload["safeguards"]) >= 4


def test_dashboard_api_rejects_unknown_ticker() -> None:
    client = TestClient(app)
    response = client.get("/api/dashboard?ticker=NOPE")
    assert response.status_code == 404
