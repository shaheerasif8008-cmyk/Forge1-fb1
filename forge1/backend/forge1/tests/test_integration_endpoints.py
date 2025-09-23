# forge1/backend/forge1/tests/test_integration_endpoints.py
"""Integration tests for key API endpoints (Task 10.1)"""

import pytest
from httpx import AsyncClient

from forge1.core.app_kernel import Forge1App


@pytest.mark.asyncio
async def test_detailed_health_and_models_and_compliance_overview():
    app = Forge1App().app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/api/v1/forge1/health/detailed")
        assert r.status_code == 200
        data = r.json()
        assert data.get("components", {}).get("model_router") is not None

        r2 = await ac.get("/api/v1/forge1/models/available")
        assert r2.status_code == 200
        assert "models" in r2.json()

        r3 = await ac.get("/api/v1/compliance/overview")
        assert r3.status_code == 200
        ov = r3.json()
        assert ov.get("status") in ["healthy", "degraded", "unhealthy"]

