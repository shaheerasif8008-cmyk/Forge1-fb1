"""
Secret Manager with caching and audit (Phase A.3)

Wraps SecurityManager.get_secret with in-memory caching, optional TTL, and audit logging via SecurityManager.
"""

import time
from typing import Optional

from forge1.core.security_manager import SecurityManager


class SecretManager:
    def __init__(self, security_manager: Optional[SecurityManager] = None, ttl_seconds: int = 300):
        self.security = security_manager or SecurityManager()
        self.ttl = ttl_seconds
        self._cache = {}  # name -> (value, fetched_at)

    async def get(self, name: str, audit_user: str = "system") -> Optional[str]:
        now = time.time()
        if name in self._cache:
            value, ts = self._cache[name]
            if now - ts < self.ttl:
                return value
        value = await self.security.get_secret(name)
        # Audit secret access without leaking value
        await self.security._log_security_event(
            user_id=audit_user,
            action="secret_access",
            resource=name,
            ip_address="internal",
            user_agent="secret_manager",
            success=value is not None,
            details={"cached": False}
        )
        if value is not None:
            self._cache[name] = (value, now)
        return value


__all__ = ["SecretManager"]

