from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from forge1.core.security import SecretManager
from forge1.middleware.policy_enforcer import PolicyEnforcementMiddleware
from forge1.middleware.tenant_context import TenantContextMiddleware
from forge1.policy.opa_client import DecisionLogger, OPAAdapter


@pytest.fixture()
def policy_app(tmp_path: Path) -> TestClient:
    decision_log = tmp_path / "decisions.jsonl"
    opa = OPAAdapter(decision_logger=DecisionLogger(decision_log))

    app = FastAPI()
    secret_manager = SecretManager()
    app.add_middleware(PolicyEnforcementMiddleware, opa_adapter=opa, bypass_paths={"/health"})
    app.add_middleware(
        TenantContextMiddleware,
        secret_manager=secret_manager,
        allow_anonymous_paths={"/health"},
    )

    @app.post("/tools/run")
    async def run_tool() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/documents/{doc_id}")
    async def read_document(doc_id: str) -> Dict[str, str]:
        return {"document": doc_id}

    client = TestClient(app)
    client.decision_log = decision_log  # type: ignore[attr-defined]
    return client


def _headers(tenant: str, user: str, role: str) -> Dict[str, str]:
    return {
        "X-Tenant-ID": tenant,
        "X-User-ID": user,
        "X-Role": role,
        "Content-Type": "application/json",
    }


def _read_log(decision_log: Path) -> Dict[str, object]:
    data = decision_log.read_text(encoding="utf-8").strip().splitlines()
    return json.loads(data[-1])


def test_cross_tenant_denied(policy_app: TestClient) -> None:
    response = policy_app.post(
        "/tools/run",
        headers={
            **_headers("tenant-a", "user-1", "user"),
            "X-Policy-Resource": "tool",
            "X-Resource-Tenant": "tenant-b",
            "X-Tool-Name": "slack_post",
        },
        json={"payload": "noop"},
    )

    assert response.status_code == 403
    entry = _read_log(policy_app.decision_log)  # type: ignore[arg-type]
    assert entry["decision"]["decision"] == "deny"
    assert entry["decision"]["policy"] == "forge1.policies.tool_access"
    assert entry["context"]["resource"]["tenant_id"] == "tenant-b"


def test_high_sensitivity_denied_for_user(policy_app: TestClient) -> None:
    response = policy_app.post(
        "/tools/run",
        headers={
            **_headers("tenant-a", "user-1", "user"),
            "X-Policy-Resource": "tool",
            "X-Resource-Tenant": "tenant-a",
            "X-Resource-Sensitivity": "high",
            "X-Tool-Name": "drive_fetch",
        },
        json={"payload": "noop"},
    )

    assert response.status_code == 403

    entry = _read_log(policy_app.decision_log)  # type: ignore[arg-type]
    assert entry["decision"]["reason"] == "insufficient_role"


def test_admin_allowed(policy_app: TestClient) -> None:
    response = policy_app.post(
        "/tools/run",
        headers={
            **_headers("tenant-a", "admin-1", "admin"),
            "X-Policy-Resource": "tool",
            "X-Resource-Tenant": "tenant-a",
            "X-Resource-Sensitivity": "high",
            "X-Tool-Name": "drive_fetch",
        },
        json={"payload": "noop"},
    )

    assert response.status_code == 200

    entry = _read_log(policy_app.decision_log)  # type: ignore[arg-type]
    assert entry["decision"]["decision"] == "allow"
    assert entry["context"]["resource"]["sensitivity"] == "high"
