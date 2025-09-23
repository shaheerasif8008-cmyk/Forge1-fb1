# forge1/backend/forge1/tests/test_memory_manager.py
"""
Test memory manager functionality
"""

import pytest
import asyncio
from datetime import datetime, timezone

from forge1.core.memory_manager import MemoryManager, EmbeddingService, get_memory_manager
from forge1.core.memory_models import (
    MemoryContext, MemoryQuery, MemoryType, SecurityLevel,
    EmbeddingRequest, EmbeddingResponse
)

class TestEmbeddingService:
    """Test embedding service functionality"""
    
    def test_embedding_service_creation(self):
        """Test creating embedding service"""
        service = EmbeddingService()
        assert service is not None
        assert service.model == "text-embedding-ada-002"
    
    @pytest.mark.asyncio
    async def test_generate_embedding(self):
        """Test embedding generation (using mock)"""
        service = EmbeddingService()
        
        request = EmbeddingRequest(
            text="This is a test message for embedding generation",
            user_id="test-user-123"
        )
        
        response = await service.generate_embedding(request)
        
        assert isinstance(response, EmbeddingResponse)
        assert len(response.embedding) == 1536  # OpenAI embedding dimension
        assert response.model == "text-embedding-ada-002"
        assert response.token_count > 0
        assert response.processing_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_embedding_caching(self):
        """Test embedding caching functionality"""
        service = EmbeddingService()
        
        request = EmbeddingRequest(
            text="Cached test message",
            cache_key="test-cache-key"
        )
        
        # First request - should not be cached
        response1 = await service.generate_embedding(request)
        assert not response1.cached
        
        # Second request - should be cached
        response2 = await service.generate_embedding(request)
        assert response2.cached
        assert response1.embedding == response2.embedding
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        service = EmbeddingService()
        
        # Test identical vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = service.cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001
        
        # Test orthogonal vectors
        vec3 = [1.0, 0.0, 0.0]
        vec4 = [0.0, 1.0, 0.0]
        similarity = service.cosine_similarity(vec3, vec4)
        assert abs(similarity - 0.0) < 0.001
        
        # Test opposite vectors
        vec5 = [1.0, 0.0, 0.0]
        vec6 = [-1.0, 0.0, 0.0]
        similarity = service.cosine_similarity(vec5, vec6)
        assert abs(similarity - (-1.0)) < 0.001

class TestMemoryManager:
    """Test memory manager functionality"""
    
    @pytest.fixture
    def sample_memory(self):
        """Create a sample memory context for testing"""
        return MemoryContext(
            employee_id="test-employee-123",
            memory_type=MemoryType.CONVERSATION,
            content={
                "user_message": "What is the weather like today?",
                "assistant_response": "I don't have access to real-time weather data.",
                "context": "User asking about weather information"
            },
            summary="Weather inquiry conversation",
            keywords=["weather", "inquiry", "information"],
            owner_id="test-user-456",
            security_level=SecurityLevel.INTERNAL,
            tags=["conversation", "weather"]
        )
    
    def test_memory_manager_creation(self):
        """Test creating memory manager"""
        manager = MemoryManager()
        assert manager is not None
        assert not manager._initialized
    
    def test_extract_text_from_content(self):
        """Test text extraction from memory content"""
        manager = MemoryManager()
        
        # Test simple string
        text = manager._extract_text_from_content("Simple text")
        assert text == "Simple text"
        
        # Test dictionary content
        content = {
            "message": "Hello world",
            "response": "Hi there",
            "metadata": {
                "timestamp": "2024-01-01",
                "user": "test"
            }
        }
        text = manager._extract_text_from_content(content)
        assert "Hello world" in text
        assert "Hi there" in text
        assert "2024-01-01" in text
        assert "test" in text
        
        # Test list content
        content = ["First item", "Second item", {"nested": "value"}]
        text = manager._extract_text_from_content(content)
        assert "First item" in text
        assert "Second item" in text
        assert "value" in text
    
    def test_relevance_score_calculation(self, sample_memory):
        """Test relevance score calculation"""
        manager = MemoryManager()
        
        # Test with recent memory
        recent_memory = sample_memory.copy()
        recent_memory.created_at = datetime.now(timezone.utc)
        recent_memory.access_count = 5
        recent_memory.last_accessed = datetime.now(timezone.utc)
        
        # This would be tested with actual async method in integration tests
        # For now, just test the logic components exist
        assert hasattr(manager, '_calculate_relevance_score')
    
    def test_search_explanation_generation(self, sample_memory):
        """Test search explanation generation"""
        manager = MemoryManager()
        
        query = MemoryQuery(
            query_text="weather information",
            keywords=["weather"],
            tags=["conversation"]
        )
        
        explanation = manager._generate_search_explanation(sample_memory, query, 0.85)
        
        assert "High semantic similarity" in explanation
        assert "Matching keywords" in explanation
        assert "Matching tags" in explanation

class TestMemoryQueries:
    """Test memory query functionality"""
    
    def test_memory_query_creation(self):
        """Test creating memory queries"""
        query = MemoryQuery(
            query_text="Find conversations about weather",
            memory_types=[MemoryType.CONVERSATION],
            keywords=["weather", "forecast"],
            limit=20,
            min_relevance_score=0.5
        )
        
        assert query.query_text == "Find conversations about weather"
        assert MemoryType.CONVERSATION in query.memory_types
        assert "weather" in query.keywords
        assert query.limit == 20
        assert query.min_relevance_score == 0.5
    
    def test_memory_query_validation(self):
        """Test memory query validation"""
        # Test limit validation
        with pytest.raises(ValueError):
            MemoryQuery(limit=0)  # Should be >= 1
        
        with pytest.raises(ValueError):
            MemoryQuery(limit=200)  # Should be <= 100
        
        # Test offset validation
        with pytest.raises(ValueError):
            MemoryQuery(offset=-1)  # Should be >= 0

class TestMemoryIntegration:
    """Integration tests for memory system"""
    
    @pytest.mark.asyncio
    async def test_global_memory_manager(self):
        """Test global memory manager singleton"""
        # This test would require actual database connections
        # For now, just test the function exists
        assert callable(get_memory_manager)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_memory_workflow(self):
        """Test complete memory workflow - store, retrieve, search, update, delete"""
        # This would be a full integration test requiring database setup
        # Skip if not in integration test environment
        import os
        if not os.getenv("RUN_INTEGRATION_TESTS"):
            pytest.skip("Integration tests not enabled")
        
        # Would test:
        # 1. Store memory with embeddings
        # 2. Retrieve memory by ID
        # 3. Search memories with various filters
        # 4. Update memory content
        # 5. Delete memory
        # 6. Verify all operations work correctly
        pass

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])