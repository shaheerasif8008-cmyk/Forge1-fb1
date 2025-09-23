# forge1/backend/forge1/tests/test_performance_benchmark.py
"""Automated performance tests for core endpoints (Task 9.2)"""

import asyncio
import time
import pytest
from httpx import AsyncClient

from forge1.core.app_kernel import Forge1App


@pytest.mark.asyncio
async def test_health_endpoint_latency():
    app = Forge1App().app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        t0 = time.perf_counter()
        resp = await ac.get("/health")
        dt = time.perf_counter() - t0
        assert resp.status_code == 200
        # Soft latency ceiling for basic health
        assert dt < 0.25


@pytest.mark.asyncio
async def test_models_available_endpoint_concurrency():
    app = Forge1App().app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        async def call_once():
            r = await ac.get("/api/v1/forge1/models/available")
            assert r.status_code == 200

        tasks = [asyncio.create_task(call_once()) for _ in range(25)]
        t0 = time.perf_counter()
        await asyncio.gather(*tasks)
        dt = time.perf_counter() - t0
        # Concurrency batch should finish reasonably fast in mock mode
        assert dt < 2.5

