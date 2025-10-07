"""Base adapter abstractions for Forge1 integrations."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, List


logger = logging.getLogger(__name__)


class AdapterStatus(Enum):
    """Represents the health status of an integration adapter."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Structured health check response for adapters."""

    status: AdapterStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    response_time_ms: float = 0.0


@dataclass
class TenantContext:
    """Represents tenant-scoped identity and authorization metadata."""

    tenant_id: str
    user_id: Optional[str] = None
    employee_id: Optional[str] = None
    role: Optional[str] = None
    security_level: Optional[str] = None
    department: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    plan: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "employee_id": self.employee_id,
            "role": self.role,
            "security_level": self.security_level,
            "department": self.department,
            "permissions": list(self.permissions) if self.permissions else [],
            "features": list(self.features) if self.features else [],
            "plan": self.plan,
            "extra": dict(self.extra) if self.extra else {},
        }


@dataclass
class ExecutionContext:
    """Context propagated across integrations and tools."""

    tenant_id: str
    employee_id: Optional[str]
    case_id: Optional[str]
    request_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tenant_context: Optional[TenantContext] = None

    def __post_init__(self) -> None:
        if self.tenant_context is None:
            self.tenant_context = TenantContext(
                tenant_id=self.tenant_id,
                user_id=self.user_id,
                employee_id=self.employee_id,
                role=self.metadata.get("role"),
                security_level=self.metadata.get("security_level"),
            )
        else:
            # Backfill missing identifiers
            if not getattr(self.tenant_context, "tenant_id", None):
                self.tenant_context.tenant_id = self.tenant_id
            if self.tenant_context.employee_id is None:
                self.tenant_context.employee_id = self.employee_id
            if self.tenant_context.user_id is None:
                self.tenant_context.user_id = self.user_id

    @property
    def tenant(self) -> str:
        return self.tenant_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "employee_id": self.employee_id,
            "case_id": self.case_id,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "operation": self.operation,
            "metadata": dict(self.metadata),
            "tenant_context": self.tenant_context.to_dict() if self.tenant_context else None,
        }


class BaseAdapter:
    """Base integration adapter with lifecycle helpers."""

    def __init__(self, name: str, config: Optional[Any] = None):
        self.name = name
        self.config = config
        self._initialized = False
        self.logger = logging.getLogger(f"forge1.integrations.{name}")

    @property
    def initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> bool:
        """Initialize underlying resources. Override in subclasses."""
        self.logger.info("Adapter %s initialized (default noop)", self.name)
        self._initialized = True
        return True

    async def health_check(self) -> HealthCheckResult:
        """Return a basic health check result. Override in subclasses."""
        status = AdapterStatus.HEALTHY if self._initialized else AdapterStatus.UNHEALTHY
        message = "Adapter initialized" if self._initialized else "Adapter not initialized"
        return HealthCheckResult(status=status, message=message)

    async def cleanup(self) -> bool:
        """Cleanup resources. Override in subclasses."""
        self.logger.info("Adapter %s cleanup (default noop)", self.name)
        self._initialized = False
        return True


class CachedAdapter(BaseAdapter):
    """Adapter base class adding simple TTL caching semantics."""

    def __init__(self, name: str, config: Optional[Any] = None, cache_ttl: int = 300):
        super().__init__(name, config)
        self._cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._cache_ttl = cache_ttl

    def _tenant_bucket(self, tenant_id: Optional[str]) -> Dict[str, Dict[str, Any]]:
        key = tenant_id or "__global__"
        return self._cache.setdefault(key, {})

    def get_from_cache(self, key: str, tenant_id: Optional[str] = None) -> Optional[Any]:
        bucket = self._tenant_bucket(tenant_id)
        entry = bucket.get(key)
        if not entry:
            return None

        if (time.time() - entry["timestamp"]) > self._cache_ttl:
            bucket.pop(key, None)
            return None
        return entry["value"]

    def set_in_cache(
        self,
        key: str,
        value: Any,
        tenant_context: Optional[TenantContext] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        tid = tenant_id or (tenant_context.tenant_id if tenant_context else None)
        bucket = self._tenant_bucket(tid)
        bucket[key] = {"value": value, "timestamp": time.time()}

    def invalidate_cache(self, key: Optional[str] = None, tenant_id: Optional[str] = None) -> None:
        if tenant_id:
            bucket = self._cache.get(tenant_id)
            if not bucket:
                return
            if key:
                bucket.pop(key, None)
            else:
                bucket.clear()
            return

        if key:
            for bucket in self._cache.values():
                bucket.pop(key, None)
        else:
            self._cache.clear()


class AdapterInitializationError(RuntimeError):
    """Raised when an adapter fails to initialize."""


def now_ms() -> float:
    return time.time() * 1000
