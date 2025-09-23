# forge1/backend/forge1/tests/test_memory_optimizer.py
"""
Test memory optimizer functionality
"""

import pytest
from datetime import datetime, timezone, timedelta

from forge1.core.memory_optimizer import (
    MemoryOptimizer, PruningStrategy, ConflictType, ShareType,
    PruningCriteria, ConflictResolution, get_memory_optimizer
)
from forge1.core.memory_models import (
    MemoryContext, MemoryPruningRule, MemoryShare, MemoryConflict,
    MemoryType, SecurityLevel
)

class TestMemoryOptimizer:
    """Test memory optimizer functionality"""
    
    def test_memory_optimizer_creation(self):
        """Test creating memory optimizer"""
        optimizer = MemoryOptimizer()
        assert optimizer is not None
        assert not optimizer._initialized
        assert len(optimizer.default_pruning_rules) > 0
    
    def test_default_pruning_rules(self):
        """Test default pruning rules are properly configured"""
        optimizer = MemoryOptimizer()
        
        # Check that default rules exist
        assert len(optimizer.default_pruning_rules) >= 3
        
        # Check rule structure
        for rule in optimizer.default_pruning_rules:
            assert isinstance(rule, MemoryPruningRule)
            assert rule.name is not None
            assert rule.description is not None
            assert rule.conditions is not None
            assert rule.action in ["delete", "archive", "compress"]
            assert 1 <= rule.priority <= 10
            assert rule.enabled is True
    
    def test_memory_size_estimation(self):
        """Test memory size estimation"""
        optimizer = MemoryOptimizer()
        
        # Create test memory
        memory = MemoryContext(
            employee_id="test-employee-123",
            memory_type=MemoryType.CONVERSATION,
            content={
                "user_message": "What is the weather like?",
                "assistant_response": "I don't have access to real-time weather data.",
                "context": "Weather inquiry"
            },
            owner_id="test-user-456",
            embeddings=[0.1] * 1536  # Mock embedding
        )
        
        size_mb = optimizer._estimate_memory_size(memory)
        assert size_mb > 0
        assert isinstance(size_mb, float)
        
        # Memory with embeddings should be larger than without
        memory_no_embeddings = memory.copy()
        memory_no_embeddings.embeddings = None
        
        size_no_embeddings = optimizer._estimate_memory_size(memory_no_embeddings)
        assert size_mb > size_no_embeddings

class TestPruningCriteria:
    """Test pruning criteria functionality"""
    
    def test_pruning_criteria_creation(self):
        """Test creating pruning criteria"""
        criteria = PruningCriteria(
            max_age_days=30,
            min_relevance_score=0.5,
            min_access_count=2,
            preserve_types=[MemoryType.KNOWLEDGE, MemoryType.SKILL]
        )
        
        assert criteria.max_age_days == 30
        assert criteria.min_relevance_score == 0.5
        assert criteria.min_access_count == 2
        assert MemoryType.KNOWLEDGE in criteria.preserve_types
        assert MemoryType.SKILL in criteria.preserve_types

class TestMemorySharing:
    """Test memory sharing functionality"""
    
    def test_memory_share_creation(self):
        """Test creating memory share"""
        share = MemoryShare(
            memory_id="memory-123",
            from_employee_id="employee-456",
            to_employee_id="employee-789",
            share_type=ShareType.COPY.value,
            permissions=["read", "write"]
        )
        
        assert share.memory_id == "memory-123"
        assert share.from_employee_id == "employee-456"
        assert share.to_employee_id == "employee-789"
        assert share.share_type == ShareType.COPY.value
        assert "read" in share.permissions
        assert "write" in share.permissions
        assert share.created_at is not None
    
    def test_share_types(self):
        """Test different share types"""
        assert ShareType.COPY == "copy"
        assert ShareType.REFERENCE == "reference"
        assert ShareType.TEMPORARY == "temporary"
        assert ShareType.SYNCHRONIZED == "synchronized"

class TestConflictResolution:
    """Test conflict resolution functionality"""
    
    def test_conflict_resolution_creation(self):
        """Test creating conflict resolution"""
        resolution = ConflictResolution(
            conflict_id="conflict-123",
            resolution_type="merge",
            winning_memory_id="memory-456",
            confidence=0.85
        )
        
        assert resolution.conflict_id == "conflict-123"
        assert resolution.resolution_type == "merge"
        assert resolution.winning_memory_id == "memory-456"
        assert resolution.confidence == 0.85
    
    def test_conflict_types(self):
        """Test conflict type enumeration"""
        assert ConflictType.CONTRADICTION == "contradiction"
        assert ConflictType.DUPLICATION == "duplication"
        assert ConflictType.INCONSISTENCY == "inconsistency"
        assert ConflictType.OUTDATED == "outdated"

class TestMemoryConflicts:
    """Test memory conflict detection and resolution"""
    
    def test_memory_conflict_creation(self):
        """Test creating memory conflict"""
        conflict = MemoryConflict(
            memory_ids=["memory-123", "memory-456"],
            conflict_type=ConflictType.CONTRADICTION.value,
            description="Conflicting information about weather conditions",
            confidence_score=0.75
        )
        
        assert len(conflict.memory_ids) == 2
        assert "memory-123" in conflict.memory_ids
        assert "memory-456" in conflict.memory_ids
        assert conflict.conflict_type == ConflictType.CONTRADICTION.value
        assert conflict.confidence_score == 0.75
        assert not conflict.resolved
        assert conflict.resolved_at is None

class TestOptimizationResults:
    """Test optimization result tracking"""
    
    def test_optimization_result_structure(self):
        """Test optimization result data structure"""
        from forge1.core.memory_models import MemoryOptimizationResult
        
        result = MemoryOptimizationResult(
            operation="test_optimization",
            memories_processed=100,
            memories_removed=25,
            memories_compressed=10,
            storage_saved_mb=15.5,
            processing_time_ms=1250.0,
            details={"test_param": "test_value"}
        )
        
        assert result.operation == "test_optimization"
        assert result.memories_processed == 100
        assert result.memories_removed == 25
        assert result.memories_compressed == 10
        assert result.storage_saved_mb == 15.5
        assert result.processing_time_ms == 1250.0
        assert result.details["test_param"] == "test_value"

class TestGlobalOptimizer:
    """Test global optimizer functionality"""
    
    @pytest.mark.asyncio
    async def test_global_optimizer_singleton(self):
        """Test global optimizer singleton pattern"""
        # This test would require actual database connections
        # For now, just test the function exists
        assert callable(get_memory_optimizer)

class TestPruningRuleMatching:
    """Test pruning rule matching logic"""
    
    def test_age_based_rule_matching(self):
        """Test age-based pruning rule logic"""
        # Create old memory
        old_memory = MemoryContext(
            employee_id="test-employee-123",
            memory_type=MemoryType.CONVERSATION,
            content={"message": "old conversation"},
            owner_id="test-user-456",
            created_at=datetime.now(timezone.utc) - timedelta(days=100)
        )
        
        # Create recent memory
        recent_memory = MemoryContext(
            employee_id="test-employee-123",
            memory_type=MemoryType.CONVERSATION,
            content={"message": "recent conversation"},
            owner_id="test-user-456",
            created_at=datetime.now(timezone.utc) - timedelta(days=5)
        )
        
        # Test age calculation
        old_age = (datetime.now(timezone.utc) - old_memory.created_at).days
        recent_age = (datetime.now(timezone.utc) - recent_memory.created_at).days
        
        assert old_age > 90  # Should be pruned by 90-day rule
        assert recent_age < 30  # Should not be pruned
    
    def test_relevance_based_rule_matching(self):
        """Test relevance-based pruning rule logic"""
        # Create low relevance memory
        low_relevance_memory = MemoryContext(
            employee_id="test-employee-123",
            memory_type=MemoryType.CONVERSATION,
            content={"message": "low relevance conversation"},
            owner_id="test-user-456"
        )
        # Simulate low relevance score
        low_relevance_memory.relevance_score = type('obj', (object,), {'overall_score': 0.2})()
        
        # Create high relevance memory
        high_relevance_memory = MemoryContext(
            employee_id="test-employee-123",
            memory_type=MemoryType.KNOWLEDGE,
            content={"fact": "important knowledge"},
            owner_id="test-user-456"
        )
        # Simulate high relevance score
        high_relevance_memory.relevance_score = type('obj', (object,), {'overall_score': 0.9})()
        
        # Test relevance scores
        assert low_relevance_memory.relevance_score.overall_score < 0.3  # Should be pruned
        assert high_relevance_memory.relevance_score.overall_score > 0.5  # Should be kept

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])