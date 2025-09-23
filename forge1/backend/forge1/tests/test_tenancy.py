# Tests for Multi-Tenant Isolation (A.2)

import pytest
from unittest.mock import AsyncMock

from forge1.core.memory_manager import MemoryManager
from forge1.core.memory_models import MemoryContext, MemoryType, SecurityLevel
from forge1.core.tenancy import set_current_tenant


@pytest.mark.asyncio
async def test_memory_tags_include_tenant_and_search_filters_by_tenant():
    mm = MemoryManager()
    await mm.initialize()

    # Tenant A
    set_current_tenant("tenantA")
    m1 = MemoryContext(
        employee_id="e1",
        session_id=None,
        memory_type=MemoryType.KNOWLEDGE,
        content={"text": "Alpha"},
        owner_id="userA",
        security_level=SecurityLevel.INTERNAL,
    )
    await mm.store_memory(m1)

    # Tenant B
    set_current_tenant("tenantB")
    m2 = MemoryContext(
        employee_id="e2",
        session_id=None,
        memory_type=MemoryType.KNOWLEDGE,
        content={"text": "Beta"},
        owner_id="userB",
        security_level=SecurityLevel.INTERNAL,
    )
    await mm.store_memory(m2)

    # Search under A should not see B's memory
    from forge1.core.memory_models import MemoryQuery
    set_current_tenant("tenantA")
    resA = await mm.search_memories(MemoryQuery(query_text="Alpha", limit=10), user_id="userA")
    assert any(r.memory.id == m1.id for r in resA.results)
    assert not any(r.memory.id == m2.id for r in resA.results)

