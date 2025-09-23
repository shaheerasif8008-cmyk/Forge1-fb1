#!/usr/bin/env python3
"""
Superhuman Performance Validation (Task 12.2)

Runs synthetic load against in-process API and validates latency thresholds
that represent "superhuman" responsiveness for common operations.
"""

import asyncio
import time
import statistics
from httpx import AsyncClient

from forge1.core.app_kernel import Forge1App


THRESHOLDS = {
    "mean": 0.20,   # seconds
    "p95": 0.50,    # seconds
    "max": 1.00     # seconds
}


async def run_validation(requests: int = 60) -> int:
    app_wrapper = Forge1App()
    app = app_wrapper.app
    latencies = []

    async with AsyncClient(app=app, base_url="http://test") as ac:
        async def call_once():
            t0 = time.perf_counter()
            r = await ac.get("/api/v1/forge1/models/available")
            r.raise_for_status()
            latencies.append(time.perf_counter() - t0)

        tasks = [asyncio.create_task(call_once()) for _ in range(requests)]
        await asyncio.gather(*tasks)

    mean = statistics.mean(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
    mx = max(latencies)

    print("Superhuman Performance Validation")
    print("-" * 40)
    print(f"Requests: {requests}")
    print(f"Mean: {mean:.4f}s  P95: {p95:.4f}s  Max: {mx:.4f}s")
    print(f"Targets -> Mean <= {THRESHOLDS['mean']:.2f}s, P95 <= {THRESHOLDS['p95']:.2f}s, Max <= {THRESHOLDS['max']:.2f}s")

    ok = (mean <= THRESHOLDS['mean']) and (p95 <= THRESHOLDS['p95']) and (mx <= THRESHOLDS['max'])
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    exit(asyncio.run(run_validation()))

