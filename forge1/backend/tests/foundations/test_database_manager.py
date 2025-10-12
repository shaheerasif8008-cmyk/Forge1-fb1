import pytest

from forge1.core.database_config import DatabaseManager
from forge1.core.tenancy import clear_tenant_context, set_current_tenant


@pytest.mark.asyncio
async def test_database_health_and_namespacing(monkeypatch):
    monkeypatch.setenv("DATABASE_DSN", "postgresql+asyncpg://forge1:forge1@localhost:5432/forge1")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    manager = DatabaseManager()
    await manager.start()

    try:
        health = await manager.health()
        assert health["postgres"] is True
        assert health["redis"] is True

        set_current_tenant("tenant_test", user_id="tester")
        redis_client = manager.redis()
        await redis_client.set("welcome", "hello")
        assert await redis_client.get("welcome") == "hello"
    finally:
        clear_tenant_context()
        await manager.stop()
