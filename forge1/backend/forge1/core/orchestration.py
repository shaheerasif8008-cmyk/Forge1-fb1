"""
Durable Orchestration Layer (Phase A.1)

Features:
- Durable queue (Redis) with idempotency keys
- Retry policy with exponential backoff
- Simple circuit breaker using Redis counters
- Outbox recording for external calls
- Minimal saga runner (steps + compensations)
"""

import json
import time
import asyncio
import hashlib
from typing import Any, Awaitable, Callable, Dict, Optional, List, Tuple

from forge1.core.database_config import get_database_manager
from forge1.core.tenancy import tenant_prefix, get_current_tenant


class RetryPolicy:
    def __init__(self, retries: int = 5, base_delay: float = 0.2, max_delay: float = 5.0):
        self.retries = retries
        self.base = base_delay
        self.max = max_delay

    def backoff(self, attempt: int) -> float:
        return min(self.base * (2 ** attempt), self.max)


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5, window_sec: int = 60):
        self.name = name
        self.failure_threshold = failure_threshold
        self.window = window_sec

    async def allow(self) -> bool:
        db = await get_database_manager()
        key = tenant_prefix(f"cb:{self.name}:{int(time.time() // self.window)}")
        val = await db.redis.get(key)
        if not val:
            return True
        return int(val) < self.failure_threshold

    async def record_failure(self):
        db = await get_database_manager()
        key = tenant_prefix(f"cb:{self.name}:{int(time.time() // self.window)}")
        await db.redis.incr(key)
        await db.redis.expire(key, self.window)


class DurableQueue:
    def __init__(self, name: str = "jobs"):
        self.name = name

    async def enqueue(self, job_type: str, payload: Dict[str, Any], idempotency_key: Optional[str] = None) -> str:
        db = await get_database_manager()
        key = tenant_prefix(f"queue:{self.name}")
        idem_key = tenant_prefix(f"idem:{self.name}")
        job_id = hashlib.sha256(json.dumps([job_type, payload, idempotency_key], sort_keys=True).encode()).hexdigest()[:16]
        if idempotency_key:
            stored = await db.redis.hsetnx(idem_key, idempotency_key, job_id)
            if stored == 0:
                # already processed/enqueued
                return await db.redis.hget(idem_key, idempotency_key)
        await db.redis.rpush(key, json.dumps({"id": job_id, "type": job_type, "payload": payload}))
        return job_id

    async def dequeue(self, timeout: int = 2) -> Optional[Dict[str, Any]]:
        db = await get_database_manager()
        key = tenant_prefix(f"queue:{self.name}")
        item = await db.redis.blpop(key, timeout=timeout)
        if not item:
            return None
        _, data = item
        return json.loads(data)


async def outbox_record(target: str, request: Dict[str, Any], response: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
    db = await get_database_manager()
    key = tenant_prefix("outbox")
    entry = {
        "ts": time.time(),
        "target": target,
        "request": request,
        "response": response,
        "error": error,
    }
    await db.redis.rpush(key, json.dumps(entry))


async def call_with_retry(fn: Callable[[], Awaitable[Any]], retry: RetryPolicy, cb: Optional[CircuitBreaker] = None) -> Any:
    last_err = None
    for attempt in range(retry.retries + 1):
        if cb and not await cb.allow():
            raise RuntimeError("Circuit open for target")
        try:
            return await fn()
        except Exception as e:
            last_err = e
            if cb:
                await cb.record_failure()
            if attempt == retry.retries:
                break
            await asyncio.sleep(retry.backoff(attempt))
    raise last_err


class SagaStep:
    def __init__(self, action: Callable[[], Awaitable[Any]], compensate: Optional[Callable[[], Awaitable[Any]]] = None):
        self.action = action
        self.compensate = compensate


class Saga:
    def __init__(self, steps: List[SagaStep]):
        self.steps = steps

    async def run(self) -> None:
        executed: List[SagaStep] = []
        try:
            for step in self.steps:
                await step.action()
                executed.append(step)
        except Exception:
            # compensate in reverse
            for step in reversed(executed):
                if step.compensate:
                    try:
                        await step.compensate()
                    except Exception:
                        # best effort
                        pass
            raise


__all__ = [
    "RetryPolicy",
    "CircuitBreaker",
    "DurableQueue",
    "outbox_record",
    "call_with_retry",
    "SagaStep",
    "Saga",
]

