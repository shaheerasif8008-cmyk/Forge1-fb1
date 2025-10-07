"""Celery integration placeholder.

This module provides a lightweight stand-in for the production Celery adapter so
that the Forge1 backend can start without the full queue stack. The adapter
implements the expected interface but records that it is running in degraded
mode. Replace with the real Celery implementation for full functionality.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from forge1.integrations.base_adapter import (
    AdapterStatus,
    BaseAdapter,
    HealthCheckResult,
)


logger = logging.getLogger(__name__)


class CeleryAdapter(BaseAdapter):
    """Minimal Celery adapter implementation used for bootstrapping."""

    def __init__(self) -> None:
        super().__init__("celery")
        self._degraded_reason = "Celery backend not configured; running in noop mode"

    async def initialize(self) -> bool:
        self._initialized = True
        logger.warning(self._degraded_reason)
        return True

    async def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(
            status=AdapterStatus.DEGRADED,
            message=self._degraded_reason,
            details={"initialized": self.initialized},
            timestamp=time.time(),
            response_time_ms=0.0,
        )

    async def cleanup(self) -> bool:
        self._initialized = False
        return True

    # Convenience API used by other modules expecting Celery features
    async def enqueue_task(self, task_name: str, payload: Dict[str, Any]) -> None:
        logger.info(
            "Celery adapter noop enqueue", extra={"task": task_name, "payload": payload}
        )


celery_adapter = CeleryAdapter()
