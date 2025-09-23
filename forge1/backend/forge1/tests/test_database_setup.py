# forge1/backend/forge1/tests/test_database_setup.py
"""
Test database setup and connectivity for Forge 1 memory system
"""

import pytest
import asyncio
import os
from typing import Dict, Any

from forge1.core.database_config import (
    DatabaseManager, 
    DatabaseSettings, 
    VectorDBType,
    get_database_manager,
    cleanup_database_manager
)
from forge1.core.memory_models import MemoryContext, MemoryType, SecurityLevel

class TestDatabaseSetup:
    """Test database setup and basic operations"""
    
    @pytest.fixture
    async def db_manager(self):
        """Create database manager for testing"""
        # Use test database settings
        settings = DatabaseSettings(
            postgres_db="forge1_test",
            postgres_user="forge1_dev_user", 
            postgres_password="forge1_dev_pass",
            redis_password=None,  # No password for dev
            vector_db_type=VectorDBType.PINECONE,
            pinecone_api_key=None  # Will use mock client
        )
        
        manager = DatabaseManager(settings)
        await manager.initialize()
        yield manager
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_database_initialization(self, db_manager):
        """Test that all database connections initialize properly"""
        assert db_manager._initialized
        assert db_manager.postgres is not None
        assert db_manager.redis is not None
        assert db_manager.vector_db is not None
    
    @pytest.mark.asyncio
    async def test_postgres_connectivity(self, db_manager):
        """Test PostgreSQL connectivity and basic operations"""
        async with db_manager.postgres.acquire() as conn:
            # Test basic query
            result = await conn.fetchval("SELECT 1")
            assert result == 1
            
            # Test schema exists
            schemas = await conn.fetch("""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name LIKE 'forge1_%'
            """)
            schema_names = [row['schema_name'] for row in schemas]
            assert 'forge1_core' in schema_names
            assert 'forge1_memory' in schema_names
            assert 'forge1_audit' in schema_names
    
    @pytest.mark.asyncio
    async def test_redis_connectivity(self, db_manager):
        """Test Redis connectivity and basic operations"""
        # Test ping
        pong = await db_manager.redis.ping()
        assert pong is True
        
        # Test set/get
        await db_manager.redis.set("test_key", "test_value")
        value = await db_manager.redis.get("test_key")
        assert value == "test_value"
        
        # Cleanup
        await db_manager.redis.delete("test_key")
    
    @pytest.mark.asyncio
    async def test_vector_db_connectivity(self, db_manager):
        """Test vector database connectivity"""
        # Since we're using mock client, just test it exists
        assert db_manager.vector_db is not None
        
        # Test mock operations
        if hasattr(db_manager.vector_db, 'describe_index_stats'):
            # Pinecone mock
            stats = db_manager.vector_db.describe_index_stats()
            assert 'total_vector_count' in stats
        elif hasattr(db_manager.vector_db, 'is_ready'):
            # Weaviate mock
            assert db_manager.vector_db.is_ready()
    
    @pytest.mark.asyncio
    async def test_memory_table_structure(self, db_manager):
        """Test that memory tables are created with correct structure"""
        async with db_manager.postgres.acquire() as conn:
            # Check memory_contexts table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'forge1_memory' 
                    AND table_name = 'memory_contexts'
                )
            """)
            assert table_exists
            
            # Check key columns exist
            columns = await conn.fetch("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'forge1_memory' 
                AND table_name = 'memory_contexts'
            """)
            
            column_names = [row['column_name'] for row in columns]
            required_columns = [
                'id', 'employee_id', 'memory_type', 'content',
                'embeddings', 'overall_relevance_score', 'security_level',
                'owner_id', 'created_at', 'updated_at'
            ]
            
            for col in required_columns:
                assert col in column_names, f"Missing column: {col}"
    
    @pytest.mark.asyncio
    async def test_memory_indexes(self, db_manager):
        """Test that memory indexes are created"""
        async with db_manager.postgres.acquire() as conn:
            indexes = await conn.fetch("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE schemaname = 'forge1_memory' 
                AND tablename = 'memory_contexts'
            """)
            
            index_names = [row['indexname'] for row in indexes]
            required_indexes = [
                'idx_memory_contexts_employee_id',
                'idx_memory_contexts_type',
                'idx_memory_contexts_relevance',
                'idx_memory_contexts_owner'
            ]
            
            for idx in required_indexes:
                assert idx in index_names, f"Missing index: {idx}"
    
    @pytest.mark.asyncio
    async def test_database_health_check(self, db_manager):
        """Test database health check functionality"""
        health = await db_manager.health_check()
        
        assert health.postgres is True
        assert health.redis is True
        assert health.vector_db is True
        assert health.overall is True
    
    @pytest.mark.asyncio
    async def test_global_database_manager(self):
        """Test global database manager singleton"""
        # Get manager
        manager1 = await get_database_manager()
        assert manager1 is not None
        
        # Get again - should be same instance
        manager2 = await get_database_manager()
        assert manager1 is manager2
        
        # Cleanup
        await cleanup_database_manager()
        
        # Get new manager - should be different instance
        manager3 = await get_database_manager()
        assert manager3 is not manager1
        
        # Final cleanup
        await cleanup_database_manager()

class TestMemoryModels:
    """Test memory model validation and serialization"""
    
    def test_memory_context_creation(self):
        """Test creating a memory context"""
        memory = MemoryContext(
            employee_id="test-employee-123",
            memory_type=MemoryType.CONVERSATION,
            content={"message": "Hello world", "response": "Hi there!"},
            owner_id="test-user-456",
            security_level=SecurityLevel.INTERNAL
        )
        
        assert memory.id is not None
        assert memory.employee_id == "test-employee-123"
        assert memory.memory_type == MemoryType.CONVERSATION
        assert memory.security_level == SecurityLevel.INTERNAL
        assert memory.access_count == 0
        assert not memory.is_expired()
    
    def test_memory_context_access_tracking(self):
        """Test memory access tracking"""
        memory = MemoryContext(
            employee_id="test-employee-123",
            memory_type=MemoryType.TASK_EXECUTION,
            content={"task": "test task"},
            owner_id="test-user-456"
        )
        
        initial_access_count = memory.access_count
        initial_last_accessed = memory.last_accessed
        
        # Update access
        memory.update_access()
        
        assert memory.access_count == initial_access_count + 1
        assert memory.last_accessed > initial_last_accessed
    
    def test_memory_context_permissions(self):
        """Test memory access permissions"""
        memory = MemoryContext(
            employee_id="test-employee-123",
            memory_type=MemoryType.KNOWLEDGE,
            content={"fact": "important information"},
            owner_id="owner-123",
            security_level=SecurityLevel.CONFIDENTIAL,
            shared_with=["user-456"]
        )
        
        # Owner can access
        assert memory.can_access("owner-123")
        
        # Shared user can access
        assert memory.can_access("user-456")
        
        # Other user cannot access
        assert not memory.can_access("other-user-789")
        
        # Public memory can be accessed by anyone
        memory.security_level = SecurityLevel.PUBLIC
        assert memory.can_access("anyone")

# Integration test for full database setup
@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests requiring actual database connections"""
    
    @pytest.mark.asyncio
    async def test_full_database_setup(self):
        """Test complete database setup process"""
        # This test requires actual database connections
        # Skip if not in integration test environment
        if not os.getenv("RUN_INTEGRATION_TESTS"):
            pytest.skip("Integration tests not enabled")
        
        try:
            manager = await get_database_manager()
            
            # Test all connections
            health = await manager.health_check()
            assert health.overall, f"Database health check failed: {health}"
            
            # Test basic operations
            async with manager.postgres.acquire() as conn:
                result = await conn.fetchval("SELECT COUNT(*) FROM forge1_memory.memory_contexts")
                assert result >= 0  # Should be 0 or more
            
            await manager.redis.set("integration_test", "success")
            value = await manager.redis.get("integration_test")
            assert value == "success"
            
            await manager.redis.delete("integration_test")
            
        finally:
            await cleanup_database_manager()

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])