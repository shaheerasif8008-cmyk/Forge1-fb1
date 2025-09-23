# Tests for Durable Orchestration (A.1)

import asyncio
import json
import pytest

from forge1.core.orchestration import DurableQueue, RetryPolicy, call_with_retry, CircuitBreaker, outbox_record
from forge1.core.tenancy import set_current_tenant
from forge1.core.database_config import get_database_manager


@pytest.mark.asyncio
async def test_queue_enqueue_dequeue_idempotent():
    set_current_tenant("testco")
    q = DurableQueue("jobs")
    job_type = "demo"
    payload = {"x": 1}
    job_id_1 = await q.enqueue(job_type, payload, idempotency_key="abc")
    job_id_2 = await q.enqueue(job_type, payload, idempotency_key="abc")
    assert job_id_1 == job_id_2  # idempotent
    item = await q.dequeue(timeout=1)
    assert item is not None
    assert item["type"] == job_type


@pytest.mark.asyncio
async def test_call_with_retry_and_circuit_breaker():
    set_current_tenant("testco")
    attempts = {"n": 0}

    async def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("fail")
        return "ok"

    cb = CircuitBreaker("flaky", failure_threshold=10, window_sec=60)
    rp = RetryPolicy(retries=5, base_delay=0.01, max_delay=0.02)
    res = await call_with_retry(flaky, rp, cb)
    assert res == "ok"


@pytest.mark.asyncio
async def test_outbox_record():
    set_current_tenant("testco")
    await outbox_record("https://api.example.com", {"a": 1}, {"b": 2}, None)
    db = await get_database_manager()
    # Just assert that an entry exists
    entries = await db.redis.lrange("t:testco:outbox", 0, -1)
    assert len(entries) >= 1
    data = json.loads(entries[-1])
    assert data["target"].startswith("http")

