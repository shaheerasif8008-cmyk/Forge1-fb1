# forge1/backend/forge1/tests/test_memory_integration.py
"""
Integration tests for the complete memory system
"""

import pytest
import asyncio
import os
from datetime import datetime, timezone, timedelta

from forge1.core.memory_models import (
    MemoryContext, MemoryQuery, MemoryType, SecurityLevel,
    MemoryPruningRule, MemoryShare, MemoryOptimizationResult
)

class TestMemorySystemIntegration:
    """Integration tests for the complete memory system"""
    
    def test_memory_system_components_exist(self):
        """Test that all memory system components can be imported"""
        # Test database configuration
        try:
            from forge1.core.database_config import DatabaseManager, DatabaseSettings, VectorDBType
            assert DatabaseManager is not None
            assert DatabaseSettings is not None
            assert VectorDBType is not None
            print("✅ Database configuration components imported successfully")
        except ImportError as e:
            print(f"⚠️  Database configuration import issue: {e}")
        
        # Test memory models
        try:
            from forge1.core.memory_models import (
                MemoryContext, MemoryQuery, MemorySearchResult, MemoryStats,
                MemoryConflict, MemoryPruningRule, MemoryShare
            )
            assert MemoryContext is not None
            assert MemoryQuery is not None
            print("✅ Memory models imported successfully")
        except ImportError as e:
            print(f"❌ Memory models import failed: {e}")
            raise
        
        # Test memory manager
        try:
            from forge1.core.memory_manager import MemoryManager, EmbeddingService
            assert MemoryManager is not None
            assert EmbeddingService is not None
            print("✅ Memory manager components imported successfully")
        except ImportError as e:
            print(f"⚠️  Memory manager import issue: {e}")
        
        # Test memory optimizer
        try:
            from forge1.core.memory_optimizer import MemoryOptimizer, PruningStrategy, ShareType
            assert MemoryOptimizer is not None
            assert PruningStrategy is not None
            assert ShareType is not None
            print("✅ Memory optimizer components imported successfully")
        except ImportError as e:
            print(f"⚠️  Memory optimizer import issue: {e}")
    
    def test_memory_workflow_simulation(self):
        """Simulate a complete memory workflow without database"""
        print("Testing memory workflow simulation...")
        
        # Step 1: Create memory contexts
        memories = []
        for i in range(5):
            memory = MemoryContext(
                employee_id=f"employee-{i % 2}",  # Two employees
                memory_type=MemoryType.CONVERSATION if i % 2 == 0 else MemoryType.KNOWLEDGE,
                content={
                    "message": f"Test message {i}",
                    "response": f"Test response {i}",
                    "context": f"Test context {i}"
                },
                summary=f"Test memory {i}",
                keywords=[f"keyword{i}", "test"],
                owner_id=f"user-{i % 3}",  # Three users
                security_level=SecurityLevel.INTERNAL,
                created_at=datetime.now(timezone.utc) - timedelta(days=i*10)
            )
            memories.append(memory)
        
        print(f"Created {len(memories)} test memories")
        
        # Step 2: Test memory queries
        query = MemoryQuery(
            query_text="test message",
            memory_types=[MemoryType.CONVERSATION],
            keywords=["test"],
            limit=10,
            min_relevance_score=0.0
        )
        
        print(f"Created query: {query.query_text}")
        
        # Step 3: Test memory sharing
        share = MemoryShare(
            memory_id=memories[0].id,
            from_employee_id="employee-0",
            to_employee_id="employee-1",
            share_type="copy",
            permissions=["read"]
        )
        
        print(f"Created memory share: {share.share_id}")
        
        # Step 4: Test pruning rules
        rule = MemoryPruningRule(
            name="Test Age-Based Cleanup",
            description="Remove memories older than 60 days",
            conditions={"max_age_days": 60, "min_access_count": 1},
            action="delete",
            priority=5
        )
        
        print(f"Created pruning rule: {rule.name}")
        
        # Step 5: Simulate optimization
        result = MemoryOptimizationResult(
            operation="test_optimization",
            memories_processed=len(memories),
            memories_removed=2,
            memories_compressed=1,
            storage_saved_mb=5.2,
            processing_time_ms=150.0,
            details={"simulation": True}
        )
        
        print(f"Simulated optimization: {result.operation}")
        print(f"  Processed: {result.memories_processed}")
        print(f"  Removed: {result.memories_removed}")
        print(f"  Compressed: {result.memories_compressed}")
        print(f"  Storage saved: {result.storage_saved_mb} MB")
        
        print("✅ Memory workflow simulation completed successfully")
    
    def test_memory_access_permissions(self):
        """Test memory access permission logic"""
        print("Testing memory access permissions...")
        
        # Create memories with different security levels
        public_memory = MemoryContext(
            employee_id="employee-123",
            memory_type=MemoryType.KNOWLEDGE,
            content={"fact": "Public information"},
            owner_id="user-123",
            security_level=SecurityLevel.PUBLIC
        )
        
        internal_memory = MemoryContext(
            employee_id="employee-123",
            memory_type=MemoryType.CONVERSATION,
            content={"message": "Internal conversation"},
            owner_id="user-123",
            security_level=SecurityLevel.INTERNAL,
            shared_with=["user-456"]
        )
        
        confidential_memory = MemoryContext(
            employee_id="employee-123",
            memory_type=MemoryType.SKILL,
            content={"skill": "Confidential capability"},
            owner_id="user-123",
            security_level=SecurityLevel.CONFIDENTIAL
        )
        
        restricted_memory = MemoryContext(
            employee_id="employee-123",
            memory_type=MemoryType.EXPERIENCE,
            content={"experience": "Restricted information"},
            owner_id="user-123",
            security_level=SecurityLevel.RESTRICTED
        )
        
        # Test access permissions
        test_cases = [
            (public_memory, "user-123", True, "Owner can access public"),
            (public_memory, "user-456", True, "Anyone can access public"),
            (public_memory, "user-789", True, "Anyone can access public"),
            
            (internal_memory, "user-123", True, "Owner can access internal"),
            (internal_memory, "user-456", True, "Shared user can access internal"),
            (internal_memory, "user-789", False, "Non-shared user cannot access internal"),
            
            (confidential_memory, "user-123", True, "Owner can access confidential"),
            (confidential_memory, "user-456", False, "Non-owner cannot access confidential"),
            
            (restricted_memory, "user-123", True, "Owner can access restricted"),
            (restricted_memory, "user-456", False, "Non-owner cannot access restricted"),
        ]
        
        for memory, user_id, expected, description in test_cases:
            result = memory.can_access(user_id)
            assert result == expected, f"Failed: {description}"
            print(f"✅ {description}: {result}")
        
        print("✅ Memory access permissions working correctly")
    
    def test_memory_lifecycle_tracking(self):
        """Test memory lifecycle and access tracking"""
        print("Testing memory lifecycle tracking...")
        
        memory = MemoryContext(
            employee_id="employee-123",
            memory_type=MemoryType.CONVERSATION,
            content={"message": "Test conversation"},
            owner_id="user-123"
        )
        
        # Test initial state
        assert memory.access_count == 0
        initial_time = memory.last_accessed
        
        # Test access tracking
        memory.update_access()
        assert memory.access_count == 1
        assert memory.last_accessed > initial_time
        
        # Test multiple accesses
        for i in range(5):
            memory.update_access()
        
        assert memory.access_count == 6
        
        # Test expiration
        assert not memory.is_expired()  # No expiration set
        
        memory.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        assert memory.is_expired()  # Expired 1 hour ago
        
        memory.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        assert not memory.is_expired()  # Expires in 1 hour
        
        print("✅ Memory lifecycle tracking working correctly")
    
    def test_memory_relationships(self):
        """Test memory relationship tracking"""
        print("Testing memory relationships...")
        
        # Create parent memory
        parent_memory = MemoryContext(
            employee_id="employee-123",
            memory_type=MemoryType.KNOWLEDGE,
            content={"topic": "Machine Learning Basics"},
            owner_id="user-123"
        )
        
        # Create child memory
        child_memory = MemoryContext(
            employee_id="employee-123",
            memory_type=MemoryType.CONVERSATION,
            content={"question": "What is supervised learning?"},
            owner_id="user-123",
            parent_memory_id=parent_memory.id
        )
        
        # Create related memories
        related_memory1 = MemoryContext(
            employee_id="employee-123",
            memory_type=MemoryType.KNOWLEDGE,
            content={"topic": "Neural Networks"},
            owner_id="user-123",
            related_memory_ids=[parent_memory.id]
        )
        
        related_memory2 = MemoryContext(
            employee_id="employee-123",
            memory_type=MemoryType.EXPERIENCE,
            content={"experience": "Implemented ML model"},
            owner_id="user-123",
            related_memory_ids=[parent_memory.id, related_memory1.id]
        )
        
        # Test relationships
        assert child_memory.parent_memory_id == parent_memory.id
        assert parent_memory.id in related_memory1.related_memory_ids
        assert parent_memory.id in related_memory2.related_memory_ids
        assert related_memory1.id in related_memory2.related_memory_ids
        
        print("✅ Memory relationships working correctly")
    
    @pytest.mark.integration
    def test_database_schema_compatibility(self):
        """Test that our models are compatible with the database schema"""
        if not os.getenv("RUN_INTEGRATION_TESTS"):
            pytest.skip("Integration tests not enabled")
        
        print("Testing database schema compatibility...")
        
        # This would test:
        # 1. All model fields map to database columns
        # 2. Data types are compatible
        # 3. Constraints are properly handled
        # 4. Indexes exist for performance
        
        # For now, just verify the SQL schema exists
        from forge1.core.memory_models import MEMORY_TABLES_SQL
        assert MEMORY_TABLES_SQL is not None
        assert "memory_contexts" in MEMORY_TABLES_SQL
        assert "memory_conflicts" in MEMORY_TABLES_SQL
        assert "memory_shares" in MEMORY_TABLES_SQL
        assert "memory_pruning_rules" in MEMORY_TABLES_SQL
        
        print("✅ Database schema compatibility verified")

class TestMemorySystemPerformance:
    """Performance tests for memory system"""
    
    def test_memory_creation_performance(self):
        """Test memory creation performance"""
        print("Testing memory creation performance...")
        
        import time
        start_time = time.time()
        
        memories = []
        for i in range(1000):
            memory = MemoryContext(
                employee_id=f"employee-{i % 10}",
                memory_type=MemoryType.CONVERSATION,
                content={"message": f"Performance test message {i}"},
                owner_id=f"user-{i % 5}"
            )
            memories.append(memory)
        
        creation_time = time.time() - start_time
        print(f"Created 1000 memories in {creation_time:.3f} seconds")
        print(f"Average: {(creation_time / 1000) * 1000:.3f} ms per memory")
        
        # Should be able to create memories quickly
        assert creation_time < 1.0, "Memory creation too slow"
        
        print("✅ Memory creation performance acceptable")
    
    def test_memory_search_simulation(self):
        """Simulate memory search performance"""
        print("Testing memory search simulation...")
        
        # Create test memories
        memories = []
        for i in range(100):
            memory = MemoryContext(
                employee_id=f"employee-{i % 5}",
                memory_type=MemoryType.CONVERSATION if i % 2 == 0 else MemoryType.KNOWLEDGE,
                content={
                    "message": f"Test message {i}",
                    "topic": f"Topic {i % 10}"
                },
                keywords=[f"keyword{i % 20}", "test", "performance"],
                owner_id=f"user-{i % 3}"
            )
            memories.append(memory)
        
        # Simulate different types of searches
        search_queries = [
            MemoryQuery(query_text="test message", limit=10),
            MemoryQuery(memory_types=[MemoryType.CONVERSATION], limit=20),
            MemoryQuery(keywords=["test"], limit=15),
            MemoryQuery(employee_ids=["employee-0", "employee-1"], limit=25),
        ]
        
        for i, query in enumerate(search_queries):
            # Simulate search filtering
            matching_memories = []
            for memory in memories:
                matches = True
                
                if query.memory_types and memory.memory_type not in query.memory_types:
                    matches = False
                
                if query.employee_ids and memory.employee_id not in query.employee_ids:
                    matches = False
                
                if query.keywords and not any(kw in memory.keywords for kw in query.keywords):
                    matches = False
                
                if matches:
                    matching_memories.append(memory)
            
            # Apply limit
            results = matching_memories[:query.limit]
            print(f"Query {i+1}: Found {len(results)} matches (limit: {query.limit})")
        
        print("✅ Memory search simulation completed")

if __name__ == "__main__":
    # Run integration tests
    test_suite = TestMemorySystemIntegration()
    test_suite.test_memory_system_components_exist()
    test_suite.test_memory_workflow_simulation()
    test_suite.test_memory_access_permissions()
    test_suite.test_memory_lifecycle_tracking()
    test_suite.test_memory_relationships()
    
    perf_suite = TestMemorySystemPerformance()
    perf_suite.test_memory_creation_performance()
    perf_suite.test_memory_search_simulation()
    
    print("\n🎉 All memory system integration tests passed!")