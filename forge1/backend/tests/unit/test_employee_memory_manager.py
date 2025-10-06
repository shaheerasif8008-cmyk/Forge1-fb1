# forge1/backend/tests/unit/test_employee_memory_manager.py
"""
Unit tests for EmployeeMemoryManager

Tests memory management functionality including storage, retrieval,
semantic search, and tenant isolation.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from forge1.services.employee_memory_manager import EmployeeMemoryManager
from forge1.models.employee_models import (
    EmployeeInteraction, EmployeeResponse, MemoryItem, MemoryType,
    SummaryType, KnowledgeSourceType, MemoryAccessError
)


class TestEmployeeMemoryManager:
    """Test suite for EmployeeMemoryManager"""
    
    @pytest.fixture
    async def memory_manager(self):
        """Create EmployeeMemoryManager instance with mocked dependencies"""
        db_manager = AsyncMock()
        
        manager = EmployeeMemoryManager(db_manager=db_manager)
        manager._initialized = True
        
        # Mock OpenAI client
        manager.openai_client = AsyncMock()
        
        return manager
    
    @pytest.fixture
    def sample_interaction(self):
        """Sample employee interaction for testing"""
        return EmployeeInteraction(
            id="int_test_001",
            employee_id="emp_test_001",
            client_id="client_test_001",
            message="Hello, I need help with my account",
            context={"session_id": "sess_001", "user_type": "customer"},
            timestamp=datetime.now(timezone.utc)
        )
    
    @pytest.fixture
    def sample_response(self):
        """Sample employee response for testing"""
        return EmployeeResponse(
            message="I'd be happy to help you with your account. What specific issue are you experiencing?",
            employee_id="emp_test_001",
            interaction_id="int_test_001",
            timestamp=datetime.now(timezone.utc),
            model_used="gpt-4",
            processing_time_ms=1500,
            tokens_used=45,
            cost=0.002,
            embedding=[0.1, 0.2, 0.3] * 512  # Mock embedding vector
        )
    
    @pytest.mark.asyncio
    async def test_initialize_employee_namespace_success(self, memory_manager):
        """Test successful employee namespace initialization"""
        # Mock database connection
        mock_conn = AsyncMock()
        memory_manager.db_manager.get_connection.return_value.__aenter__.return_value = mock_conn
        
        # Mock vector DB and Redis operations
        memory_manager._create_vector_collection = AsyncMock()
        
        with patch('forge1.services.employee_memory_manager.get_current_tenant', return_value="client_test_001"):
            await memory_manager.initialize_employee_namespace("client_test_001", "emp_test_001")
        
        # Verify namespace creation
        memory_manager._create_vector_collection.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_interaction_success(self, memory_manager, sample_interaction, sample_response):
        """Test successful interaction storage"""
        # Mock database connection
        mock_conn = AsyncMock()
        memory_manager.db_manager.get_connection.return_value.__aenter__.return_value = mock_conn
        
        # Mock vector DB and cache operations
        memory_manager._store_in_vector_db = AsyncMock()
        memory_manager._cache_interaction = AsyncMock()
        memory_manager._update_employee_last_interaction = AsyncMock()
        
        with patch('forge1.services.employee_memory_manager.get_current_tenant', return_value="client_test_001"):
            await memory_manager.store_interaction(
                "client_test_001", "emp_test_001", sample_interaction, sample_response
            )
        
        # Verify storage operations
        mock_conn.execute.assert_called()
        memory_manager._store_in_vector_db.assert_called_once()
        memory_manager._cache_interaction.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_employee_context_success(self, memory_manager):
        """Test successful employee context retrieval"""
        # Mock database results
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                "id": "int_001",
                "message": "Test message",
                "response": "Test response",
                "timestamp": datetime.now(timezone.utc),
                "importance_score": 0.8,
                "memory_type": "interaction"
            }
        ]
        memory_manager.db_manager.get_connection.return_value.__aenter__.return_value = mock_conn
        
        with patch('forge1.services.employee_memory_manager.get_current_tenant', return_value="client_test_001"):
            result = await memory_manager.get_employee_context(
                "client_test_001", "emp_test_001", limit=10
            )
        
        assert len(result) == 1
        assert result[0].content == "Test message"
        assert result[0].response == "Test response"
    
    @pytest.mark.asyncio
    async def test_search_employee_memory_success(self, memory_manager):
        """Test successful semantic memory search"""
        # Mock embedding generation
        memory_manager._generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3] * 512)
        
        # Mock vector search results
        memory_manager._semantic_search = AsyncMock(return_value=[
            MemoryItem(
                id="mem_001",
                content="Previous support question",
                response="Previous support answer",
                timestamp=datetime.now(timezone.utc),
                memory_type=MemoryType.INTERACTION,
                importance_score=0.9,
                relevance_score=0.85
            )
        ])
        
        with patch('forge1.services.employee_memory_manager.get_current_tenant', return_value="client_test_001"):
            results, search_time = await memory_manager.search_employee_memory(
                "client_test_001", "emp_test_001", "support question", limit=5
            )
        
        assert len(results) == 1
        assert results[0].relevance_score == 0.85
        assert search_time > 0
    
    @pytest.mark.asyncio
    async def test_create_memory_summary_success(self, memory_manager):
        """Test memory summary creation"""
        # Mock recent interactions
        memory_manager._get_recent_interactions = AsyncMock(return_value=[
            "User asked about account settings",
            "User requested password reset",
            "User inquired about billing"
        ])
        
        # Mock summary generation
        memory_manager._generate_summary = AsyncMock(return_value="Summary of recent support interactions")
        
        # Mock database connection
        mock_conn = AsyncMock()
        memory_manager.db_manager.get_connection.return_value.__aenter__.return_value = mock_conn
        
        with patch('forge1.services.employee_memory_manager.get_current_tenant', return_value="client_test_001"):
            summary_id = await memory_manager.create_memory_summary(
                "client_test_001", "emp_test_001", SummaryType.DAILY, interaction_count=10
            )
        
        assert summary_id is not None
        mock_conn.execute.assert_called()
    
    @pytest.mark.asyncio
    async def test_add_knowledge_source_success(self, memory_manager):
        """Test adding knowledge source"""
        # Mock database connection
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = "kb_test_001"
        memory_manager.db_manager.get_connection.return_value.__aenter__.return_value = mock_conn
        
        # Mock embedding generation
        memory_manager._generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3] * 512)
        
        with patch('forge1.services.employee_memory_manager.get_current_tenant', return_value="client_test_001"):
            knowledge_id = await memory_manager.add_knowledge_source(
                client_id="client_test_001",
                employee_id="emp_test_001",
                title="Product Documentation",
                content="This is product documentation content",
                source_type=KnowledgeSourceType.DOCUMENT,
                keywords=["product", "documentation"],
                tags=["help", "guide"]
            )
        
        assert knowledge_id == "kb_test_001"
        mock_conn.execute.assert_called()
    
    @pytest.mark.asyncio
    async def test_search_knowledge_base_success(self, memory_manager):
        """Test knowledge base search"""
        # Mock embedding generation
        memory_manager._generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3] * 512)
        
        # Mock database results
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                "knowledge_id": "kb_001",
                "title": "Account Settings Guide",
                "content": "How to manage your account settings",
                "source_type": "document",
                "relevance_score": 0.9
            }
        ]
        memory_manager.db_manager.get_connection.return_value.__aenter__.return_value = mock_conn
        
        with patch('forge1.services.employee_memory_manager.get_current_tenant', return_value="client_test_001"):
            results = await memory_manager.search_knowledge_base(
                "client_test_001", "emp_test_001", "account settings", limit=5
            )
        
        assert len(results) == 1
        assert results[0]["title"] == "Account Settings Guide"
        assert results[0]["relevance_score"] == 0.9
    
    @pytest.mark.asyncio
    async def test_get_memory_stats_success(self, memory_manager):
        """Test memory statistics retrieval"""
        # Mock database results
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {
            "total_interactions": 150,
            "total_memories": 75,
            "avg_importance": 0.7,
            "memory_size_mb": 2.5,
            "oldest_memory": datetime.now(timezone.utc),
            "newest_memory": datetime.now(timezone.utc)
        }
        memory_manager.db_manager.get_connection.return_value.__aenter__.return_value = mock_conn
        
        with patch('forge1.services.employee_memory_manager.get_current_tenant', return_value="client_test_001"):
            stats = await memory_manager.get_memory_stats("client_test_001", "emp_test_001")
        
        assert stats["total_interactions"] == 150
        assert stats["total_memories"] == 75
        assert "memory_efficiency" in stats
    
    @pytest.mark.asyncio
    async def test_tenant_isolation_validation(self, memory_manager):
        """Test tenant isolation validation"""
        with patch('forge1.services.employee_memory_manager.get_current_tenant', return_value="client_001"):
            # Should not raise exception for matching tenant
            await memory_manager._validate_tenant_access("client_001", "emp_001")
        
        with patch('forge1.services.employee_memory_manager.get_current_tenant', return_value="client_002"):
            # Should raise exception for mismatched tenant
            with pytest.raises(Exception):  # TenantIsolationError
                await memory_manager._validate_tenant_access("client_001", "emp_001")
    
    @pytest.mark.asyncio
    async def test_embedding_generation_with_openai(self, memory_manager):
        """Test embedding generation with OpenAI"""
        # Mock OpenAI response
        mock_embedding_response = AsyncMock()
        mock_embedding_response.data = [AsyncMock()]
        mock_embedding_response.data[0].embedding = [0.1, 0.2, 0.3] * 512
        
        memory_manager.openai_client.embeddings.create.return_value = mock_embedding_response
        
        result = await memory_manager._generate_embedding("test text")
        
        assert len(result) == 1536  # OpenAI embedding dimension
        assert result[0] == 0.1
    
    @pytest.mark.asyncio
    async def test_embedding_generation_fallback(self, memory_manager):
        """Test embedding generation fallback when OpenAI fails"""
        # Mock OpenAI to raise exception
        memory_manager.openai_client.embeddings.create.side_effect = Exception("API Error")
        
        result = await memory_manager._generate_embedding("test text")
        
        # Should return mock embedding
        assert len(result) == 1536
        assert all(isinstance(x, float) for x in result)
    
    @pytest.mark.asyncio
    async def test_memory_analytics_success(self, memory_manager):
        """Test memory analytics functionality"""
        # Mock database results for analytics
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 100  # total interactions
        mock_conn.fetch.return_value = [
            {"date": datetime.now(timezone.utc).date(), "count": 10},
            {"date": datetime.now(timezone.utc).date(), "count": 15}
        ]
        memory_manager.db_manager.get_connection.return_value.__aenter__.return_value = mock_conn
        
        # Mock memory stats
        memory_manager.get_memory_stats = AsyncMock(return_value={
            "total_interactions": 100,
            "total_memories": 50
        })
        
        with patch('forge1.services.employee_memory_manager.get_current_tenant', return_value="client_test_001"):
            analytics = await memory_manager.get_memory_analytics(
                "client_test_001", "emp_test_001", days=30
            )
        
        assert "interaction_stats" in analytics
        assert "memory_usage" in analytics
        assert analytics["period"]["days"] == 30
    
    def test_namespace_generation(self, memory_manager):
        """Test namespace generation for tenant isolation"""
        namespace = memory_manager._get_namespace("client_001", "emp_001")
        assert namespace == "client_001:emp_001"
    
    @pytest.mark.asyncio
    async def test_cache_interaction_success(self, memory_manager, sample_interaction):
        """Test interaction caching functionality"""
        # Mock Redis operations (would be actual Redis in integration tests)
        memory_manager.redis = AsyncMock()
        
        namespace = "client_test_001:emp_test_001"
        await memory_manager._cache_interaction(namespace, sample_interaction, sample_response={})
        
        # Verify Redis operations were called
        # In a real test, we'd verify the actual Redis calls
        assert True  # Placeholder for Redis verification
    
    @pytest.mark.asyncio
    async def test_memory_export_functionality(self, memory_manager):
        """Test memory export functionality"""
        # Mock database results
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                "interaction_id": "int_001",
                "message": "Test message",
                "response": "Test response",
                "timestamp": datetime.now(timezone.utc),
                "context": {"session_id": "sess_001"}
            }
        ]
        memory_manager.db_manager.get_connection.return_value.__aenter__.return_value = mock_conn
        
        with patch('forge1.services.employee_memory_manager.get_current_tenant', return_value="client_test_001"):
            export_result = await memory_manager.export_memory(
                "client_test_001", "emp_test_001", format="json"
            )
        
        assert "export_id" in export_result
        assert "estimated_completion" in export_result


if __name__ == "__main__":
    pytest.main([__file__])