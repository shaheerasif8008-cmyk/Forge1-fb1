"""Database bootstrap utilities for Phase 1."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import asyncpg
import redis.asyncio as redis

from forge1.core.logging_config import init_logger
from forge1.core.tenancy import TenantContext, get_tenant_context, tenant_prefix

logger = init_logger("forge1.core.database")


@dataclass
class DatabaseSettings:
    dsn: str = os.getenv("DATABASE_DSN", "postgresql+asyncpg://forge1:forge1@localhost:5432/forge1")
    redis_url: str = os.getenv("REDIS_URL", "redis://:forge1_redis_pass@localhost:6379/0")

    @property
    def asyncpg_dsn(self) -> str:
        if self.dsn.startswith("postgresql+asyncpg://"):
            return self.dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
        return self.dsn


class TenantRedis:
    """Redis wrapper that prefixes keys with the active tenant namespace."""

    def __init__(self, client: redis.Redis):
        self._client = client

    def _namespaced(self, key: str) -> str:
        return f"{tenant_prefix(None)}:{key}"

    def __getattr__(self, item):  # pragma: no cover - passthrough convenience
        return getattr(self._client, item)

    async def get(self, key: str):
        return await self._client.get(self._namespaced(key))

    async def set(self, key: str, value, *args, **kwargs):
        return await self._client.set(self._namespaced(key), value, *args, **kwargs)


class DatabaseManager:
    def __init__(self, settings: Optional[DatabaseSettings] = None) -> None:
        self.settings = settings or DatabaseSettings()
        self._pool: Optional[asyncpg.pool.Pool] = None
        self._redis: Optional[redis.Redis] = None

    async def start(self) -> None:
        if self._pool is None:
            logger.info("Creating asyncpg pool", extra={"dsn": self.settings.asyncpg_dsn})
            self._pool = await asyncpg.create_pool(self.settings.asyncpg_dsn, min_size=1, max_size=5)
        if self._redis is None:
            logger.info("Connecting to Redis", extra={"url": self.settings.redis_url})
            self._redis = redis.from_url(self.settings.redis_url, encoding="utf-8", decode_responses=True)
            await self._redis.ping()

    async def stop(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
        if self._redis:
            await self._redis.close()
            self._redis = None

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[asyncpg.Connection]:
        if self._pool is None:
            raise RuntimeError("DatabaseManager.start() must be called before acquiring connections")
        async with self._pool.acquire() as conn:
            await self._prepare_connection(conn)
            yield conn

    def redis(self) -> TenantRedis:
        if self._redis is None:
            raise RuntimeError("DatabaseManager.start() must be called before accessing Redis")
        return TenantRedis(self._redis)

    async def health(self) -> dict:
        status = {"postgres": False, "redis": False}
        try:
            async with self.connection() as conn:
                await conn.fetchval("SELECT 1")
            status["postgres"] = True
        except Exception as exc:  # pragma: no cover - exercised via health endpoint
            logger.warning("Postgres health check failed", extra={"error": str(exc)})

        if self._redis:
            try:
                await self._redis.ping()
                status["redis"] = True
            except Exception as exc:  # pragma: no cover
                logger.warning("Redis health check failed", extra={"error": str(exc)})

        status["overall"] = all(status.values())
        return status

    async def _prepare_connection(self, conn: asyncpg.Connection) -> None:
        context: Optional[TenantContext] = get_tenant_context()
        schema = context.tenant_id if context else "public"
        try:
            await conn.execute(f"SET LOCAL search_path TO {schema}, public")
        except Exception:
            # Fall back to public schema when tenant specific schemas are missing.
            await conn.execute("SET LOCAL search_path TO public")


__all__ = ["DatabaseManager", "DatabaseSettings", "TenantRedis"]


# Global database manager instance
_database_manager: Optional[DatabaseManager] = None

def get_database_manager() -> DatabaseManager:
    """Get or create the global database manager instance"""
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
    return _database_manager

async def init_database_manager(settings: Optional[DatabaseSettings] = None) -> DatabaseManager:
    """Initialize and start the database manager"""
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager(settings)
        await _database_manager.start()
    return _database_manager

async def close_database_manager() -> None:
    """Close the global database manager"""
    global _database_manager
    if _database_manager is not None:
        await _database_manager.stop()
        _database_manager = None
