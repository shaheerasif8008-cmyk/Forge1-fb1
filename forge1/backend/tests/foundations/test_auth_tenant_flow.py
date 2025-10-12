import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("FORGE1_JWT_SECRET", "phase1-secret")
    monkeypatch.setenv("DATABASE_DSN", "postgresql+asyncpg://forge1:forge1@localhost:5432/forge1")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    module = importlib.import_module("forge1.main")
    return module.app


def test_jwt_token_round_trip(app):
    client = TestClient(app)
    response = client.post(
        "/auth/token",
        json={"user_id": "alice", "tenant_id": "tenantA", "role": "admin", "expires_in_seconds": 60},
    )
    assert response.status_code == 200
    token = response.json()["token"]

    whoami = client.get("/whoami", headers={"Authorization": f"Bearer {token}"})
    assert whoami.status_code == 200
    assert whoami.json() == {"tenant_id": "tenantA", "user_id": "alice", "role": "admin"}


def test_header_based_authentication(app):
    client = TestClient(app)
    response = client.get(
        "/whoami",
        headers={"X-Tenant-ID": "tenantB", "X-User-ID": "service-account", "X-Role": "service"},
    )
    assert response.status_code == 200
    assert response.json()["tenant_id"] == "tenantB"
    assert response.json()["user_id"] == "service-account"
    assert response.json()["role"] == "service"
