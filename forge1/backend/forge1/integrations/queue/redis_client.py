"""Redis integration placeholder.

Provides a simplified Redis adapter so the backend can start without a live
Redis connection. The adapter reports degraded health but allows dependent
components to continue operating in a mocked mode.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from forge1.integrations.base_adapter import (
    AdapterStatus,
    BaseAdapter,
    HealthCheckResult,
)


logger = logging.getLogger(__name__)


class RedisAdapter(BaseAdapter):
    """Minimal Redis adapter implementation."""

    def __init__(self) -> None:
        super().__init__("redis")
        self._cache: Dict[str, Any] = {}

    async def initialize(self) -> bool:
        self._initialized = True
        logger.warning("Redis adapter running in in-memory mock mode")
        return True

    async def health_check(self) -> HealthCheckResult:
        status = AdapterStatus.DEGRADED if self._cache else AdapterStatus.HEALTHY
        message = (
            "Redis adapter in mock mode"
            if status is AdapterStatus.DEGRADED
            else "Redis adapter initialized"
        )
        return HealthCheckResult(
            status=status,
            message=message,
            details={"cache_entries": len(self._cache)},
            timestamp=time.time(),
            response_time_ms=0.0,
        )

    async def cleanup(self) -> bool:
        self._cache.clear()
        self._initialized = False
        return True

    async def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    async def set(self, key: str, value: Any, expire_seconds: Optional[int] = None) -> None:
        expiry = time.time() + expire_seconds if expire_seconds else None
        self._cache[key] = {"value": value, "expiry": expiry}

    async def delete(self, key: str) -> None:
        self._cache.pop(key, None)


redis_adapter = RedisAdapter()
