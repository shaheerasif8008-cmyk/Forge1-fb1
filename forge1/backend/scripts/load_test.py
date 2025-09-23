#!/usr/bin/env python3
"""Simple async load test against in-process app (Task 9.2)

Runs bursts of requests and prints timing stats.
"""

import asyncio
import time
import statistics
from httpx import AsyncClient
from forge1.core.app_kernel import Forge1App


async def main():
    app = Forge1App().app
    latencies = []
    n = 100

    async with AsyncClient(app=app, base_url="http://test") as ac:
        async def call_once():
            t0 = time.perf_counter()
            r = await ac.get("/api/v1/forge1/models/available")
            r.raise_for_status()
            latencies.append(time.perf_counter() - t0)

        tasks = [asyncio.create_task(call_once()) for _ in range(n)]
        await asyncio.gather(*tasks)

    print(f"Requests: {n}")
    print(f"Mean: {statistics.mean(latencies):.4f}s | 95p: {statistics.quantiles(latencies, n=20)[18]:.4f}s | Max: {max(latencies):.4f}s")


if __name__ == "__main__":
    asyncio.run(main())

