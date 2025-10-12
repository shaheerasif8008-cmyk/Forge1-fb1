from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from forge1.billing import usage_meter

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover - FastAPI not installed
    TestClient = None  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _reset_meter() -> None:
    usage_meter.reset()
    yield
    usage_meter.reset()


def _sample_timestamp(day: int) -> datetime:
    return datetime(2024, 5, day, 12, 0, tzinfo=timezone.utc)


def test_usage_meter_summary_and_reconciliation() -> None:
    usage_meter.record_model_call(
        tenant_id="tenant-a",
        employee_id="user-1",
        model="gpt-4o-mini",
        tokens_input=120,
        tokens_output=40,
        latency_ms=210.5,
        cost_estimate=0.0123,
        request_id="req-model-1",
        timestamp=_sample_timestamp(10),
    )
    usage_meter.record_tool_call(
        tenant_id="tenant-a",
        employee_id="user-1",
        tool="kb.search",
        latency_ms=45.2,
        cost_estimate=0.0,
        request_id="req-tool-1",
        metadata={"backend": "local"},
        tokens_input=3,
        timestamp=_sample_timestamp(11),
    )

    summary = usage_meter.month_summary("2024-05")
    assert summary["summary"]["total_events"] == 2
    assert summary["summary"]["model_calls"] == 1
    assert summary["summary"]["tool_calls"] == 1
    assert summary["tenants"][0]["employees"][0]["total_events"] == 2

    reconciliation = usage_meter.reconcile_month("2024-05")
    assert reconciliation["variance"] == 0
    assert reconciliation["variance_pct"] == 0.0


@pytest.mark.skipif(TestClient is None, reason="FastAPI test client not available")
def test_usage_report_endpoint_supports_json_and_csv() -> None:
    pytest.importorskip("jwt")
    from forge1.auth.jwt import UserRole, issue_token  # type: ignore import-not-found
    from forge1.main import app, jwt_secret  # type: ignore import-not-found

    usage_meter.record_model_call(
        tenant_id="tenant-b",
        employee_id="user-9",
        model="gpt-4o-mini",
        tokens_input=50,
        tokens_output=20,
        latency_ms=150.0,
        cost_estimate=0.008,
        request_id="req-model-9",
        timestamp=_sample_timestamp(15),
    )

    token = issue_token(
        secret=jwt_secret,
        user_id="admin",
        tenant_id="tenant-b",
        role=UserRole.ADMIN,
        expires_in=timedelta(minutes=15),
    )

    client = TestClient(app)
    headers = {"Authorization": f"Bearer {token}"}

    json_response = client.get("/reports/usage", params={"month": "2024-05"}, headers=headers)
    assert json_response.status_code == 200
    body = json_response.json()
    assert body["month"] == "2024-05"
    assert body["summary"]["total_events"] == 1

    csv_response = client.get(
        "/reports/usage",
        params={"month": "2024-05", "format": "csv"},
        headers=headers,
    )
    assert csv_response.status_code == 200
    assert "text/csv" in csv_response.headers["content-type"]
    assert "timestamp" in csv_response.text.splitlines()[0]
