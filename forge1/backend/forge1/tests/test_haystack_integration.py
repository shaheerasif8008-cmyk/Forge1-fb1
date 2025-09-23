# forge1/backend/forge1/tests/test_haystack_integration.py
"""
Tests for Haystack/LlamaIndex Integration

Comprehensive tests for the Haystack/LlamaIndex adapter and enterprise enhancements.
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from forge1.integrations.haystack_adapter import (
    HaystackLlamaIndexAdapter,
    ForgeDocumentProcessor,
    ForgeLlamaIndexProcessor,
    DocumentProcessingType,
    DocumentFormat
)
from forge1.core.quality_assurance import QualityLevel


class TestHaystackLlamaIndexAdapter:
    """Test cases for Haystack/LlamaIndex Adapter"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for adapter"""
        return {
            "memory_manager": Mock(),
            "model_router": Mock(),
            "performance_monitor": Mock(),
            "quality_assurance": Mock(),
            "security_manager": Mock()
        }
    
    @pytest.fixture
    def haystack_adapter(self, mock_dependencies):
        """Create Haystack/LlamaIndex adapter for testing"""
        return HaystackLlamaIndexAdapter(**mock_dependencies)
    
    @pytest.mark.asyncio
    async def test_adapter_initialization(self, haystack_adapter):
        """Test adapter initialization"""
        
        assert haystack_adapter.memory_manager is not None
        assert haystack_adapter.model_router is not None
        assert haystack_adapter.performance_monitor is not None
        assert haystack_adapter.quality_assurance is not None
        assert haystack_adapter.security_manager is not None
        
        assert haystack_adapter.active_processors == {}
        assert haystack_adapter.document_stores == {}
        
        # Check initial metrics
        metrics = haystack_adapter.integration_metrics
        assert metrics["processors_created"] == 0
        assert metrics["documents_processed"] == 0
        assert metrics["queries_executed"] == 0
    
    @pytest.mark.asyncio
    async def test_create_document_processor(self, haystack_adapter):
        """Test creating document processor"""
        
        processor = await haystack_adapter.create_document_processor(
            processor_type=DocumentProcessingType.EXTRACTION,
            processor_config={"session_id": "test_session"}
        )
        
        assert isinstance(processor, ForgeDocumentProcessor)
        assert processor.processor_type == DocumentProcessingType.EXTRACTION
        assert processor.session_id == "test_session"
        assert processor.processor_id.startswith("doc_proc_")
        
        # Check that processor was stored
        assert processor.processor_id in haystack_adapter.active_processors
        
        # Check metrics updated
        assert haystack_adapter.integration_metrics["processors_created"] == 1
    
    @pytest.mark.asyncio
    async def test_create_llamaindex_processor(self, haystack_adapter):
        """Test creating LlamaIndex processor"""
        
        processor = await haystack_adapter.create_llamaindex_processor(
            processor_config={"session_id": "test_session"}
        )
        
        assert isinstance(processor, ForgeLlamaIndexProcessor)
        assert processor.session_id == "test_session"
        assert processor.processor_id.startswith("llama_proc_")
        
        # Check that processor was stored
        assert processor.processor_id in haystack_adapter.active_processors
        
        # Check metrics updated
        assert haystack_adapter.integration_metrics["processors_created"] == 1
    
    def test_get_integration_metrics(self, haystack_adapter):
        """Test getting integration metrics"""
        
        metrics = haystack_adapter.get_integration_metrics()
        
        assert "integration_metrics" in metrics
        assert "active_processors" in metrics
        assert "haystack_available" in metrics
        assert "llamaindex_available" in metrics
        assert "capabilities" in metrics
        
        # Check capabilities
        capabilities = metrics["capabilities"]
        expected_capabilities = [
            "document_processing",
            "enterprise_security",
            "quality_assurance",
            "performance_monitoring",
            "advanced_search",
            "content_extraction"
        ]
        
        for capability in expected_capabilities:
            assert capability in capabilities
    
    @pytest.mark.asyncio
    async def test_cleanup_processor(self, haystack_adapter):
        """Test processor cleanup"""
        
        # Create a processor
        processor = await haystack_adapter.create_document_processor(
            DocumentProcessingType.EXTRACTION
        )
        
        processor_id = processor.processor_id
        
        # Verify processor exists
        assert processor_id in haystack_adapter.active_processors
        
        # Cleanup processor
        cleanup_result = await haystack_adapter.cleanup_processor(processor_id)
        
        assert cleanup_result == True
        assert processor_id not in haystack_adapter.active_processors
        
        # Test cleanup of non-existent processor
        cleanup_result = await haystack_adapter.cleanup_processor("non_existent")
        assert cleanup_result == False


class TestForgeDocumentProcessor:
    """Test cases for Forge Document Processor"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for processor"""
        return {
            "memory_manager": Mock(),
            "performance_monitor": Mock(),
            "security_manager": Mock(),
            "quality_assurance": Mock()
        }
    
    @pytest.fixture
    def document_processor(self, mock_dependencies):
        """Create document processor for testing"""
        mock_dependencies["security_manager"].validate_document_access = AsyncMock()
        mock_dependencies["memory_manager"].store_memory = AsyncMock()
        mock_dependencies["performance_monitor"].track_document_processing = AsyncMock()
        mock_dependencies["quality_assurance"].conduct_quality_review = AsyncMock(return_value={
            "quality_decision": {"approved": True, "overall_score": 0.95}
        })
        
        return ForgeDocumentProcessor(
            processor_type=DocumentProcessingType.EXTRACTION,
            **mock_dependencies
        )
    
    @pytest.fixture
    def temp_text_file(self):
        """Create temporary text file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document with sample content for processing.")
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_processor_initialization(self, document_processor):
        """Test processor initialization"""
        
        assert document_processor.processor_type == DocumentProcessingType.EXTRACTION
        assert document_processor.processor_id.startswith("doc_proc_")
        assert document_processor.processing_metrics["documents_processed"] == 0
    
    @pytest.mark.asyncio
    async def test_process_document_extraction(self, document_processor, temp_text_file, mock_dependencies):
        """Test document content extraction"""
        
        result = await document_processor.process_document(
            document_path=temp_text_file,
            document_format=DocumentFormat.TXT,
            context={"extraction_type": "full_text"}
        )
        
        assert "processing_id" in result
        assert result["processor_id"] == document_processor.processor_id
        assert result["document_path"] == temp_text_file
        assert result["document_format"] == "txt"
        assert result["processor_type"] == "extraction"
        assert "result" in result
        assert "processing_time" in result
        assert result["status"] == "completed"
        
        # Verify security validation was called
        mock_dependencies["security_manager"].validate_document_access.assert_called_once()
        
        # Verify memory storage was called
        mock_dependencies["memory_manager"].store_memory.assert_called_once()
        
        # Verify performance tracking was called
        mock_dependencies["performance_monitor"].track_document_processing.assert_called_once()
        
        # Verify quality assurance was called
        mock_dependencies["quality_assurance"].conduct_quality_review.assert_called_once()
        
        # Check metrics updated
        assert document_processor.processing_metrics["documents_processed"] == 1
    
    @pytest.mark.asyncio
    async def test_process_document_indexing(self, mock_dependencies):
        """Test document indexing"""
        
        # Create indexing processor
        mock_dependencies["security_manager"].validate_document_access = AsyncMock()
        mock_dependencies["memory_manager"].store_memory = AsyncMock()
        mock_dependencies["performance_monitor"].track_document_processing = AsyncMock()
        mock_dependencies["quality_assurance"].conduct_quality_review = AsyncMock(return_value={
            "quality_decision": {"approved": True, "overall_score": 0.95}
        })
        
        indexing_processor = ForgeDocumentProcessor(
            processor_type=DocumentProcessingType.INDEXING,
            **mock_dependencies
        )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Document content for indexing test.")
            temp_path = f.name
        
        try:
            result = await indexing_processor.process_document(
                document_path=temp_path,
                document_format=DocumentFormat.TXT
            )
            
            assert result["processor_type"] == "indexing"
            assert result["status"] == "completed"
            assert "result" in result
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_process_document_search(self, mock_dependencies):
        """Test document search"""
        
        # Create search processor
        mock_dependencies["security_manager"].validate_document_access = AsyncMock()
        mock_dependencies["memory_manager"].store_memory = AsyncMock()
        mock_dependencies["performance_monitor"].track_document_processing = AsyncMock()
        mock_dependencies["quality_assurance"].conduct_quality_review = AsyncMock(return_value={
            "quality_decision": {"approved": True, "overall_score": 0.95}
        })
        
        search_processor = ForgeDocumentProcessor(
            processor_type=DocumentProcessingType.SEARCH,
            **mock_dependencies
        )
        
        result = await search_processor.process_document(
            document_path="",  # Not needed for search
            document_format=DocumentFormat.TXT,
            context={"query": "test search query"}
        )
        
        assert result["processor_type"] == "search"
        assert result["status"] == "completed"
        assert "result" in result
        
        # Check that search result contains expected fields
        search_result = result["result"]
        assert "query" in search_result
        assert "results" in search_result
        assert search_result["query"] == "test search query"
    
    def test_get_processor_info(self, document_processor):
        """Test getting processor information"""
        
        info = document_processor.get_processor_info()
        
        assert info["processor_id"] == document_processor.processor_id
        assert info["processor_type"] == "extraction"
        assert "processing_metrics" in info
        assert "haystack_available" in info
        assert "llamaindex_available" in info


class TestForgeLlamaIndexProcessor:
    """Test cases for Forge LlamaIndex Processor"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for processor"""
        return {
            "memory_manager": Mock(),
            "performance_monitor": Mock(),
            "security_manager": Mock(),
            "quality_assurance": Mock()
        }
    
    @pytest.fixture
    def llamaindex_processor(self, mock_dependencies):
        """Create LlamaIndex processor for testing"""
        mock_dependencies["security_manager"].validate_directory_access = AsyncMock()
        mock_dependencies["security_manager"].validate_query = AsyncMock()
        mock_dependencies["memory_manager"].store_memory = AsyncMock()
        mock_dependencies["performance_monitor"].track_index_creation = AsyncMock()
        mock_dependencies["performance_monitor"].track_query_execution = AsyncMock()
        
        return ForgeLlamaIndexProcessor(**mock_dependencies)
    
    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory with test files"""
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        
        # Create test files
        with open(os.path.join(temp_dir, "doc1.txt"), 'w') as f:
            f.write("This is the first test document with important information.")
        
        with open(os.path.join(temp_dir, "doc2.txt"), 'w') as f:
            f.write("This is the second test document with different content.")
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_processor_initialization(self, llamaindex_processor):
        """Test processor initialization"""
        
        assert llamaindex_processor.processor_id.startswith("llama_proc_")
        assert llamaindex_processor.processing_metrics["documents_indexed"] == 0
        assert llamaindex_processor.processing_metrics["queries_processed"] == 0
    
    @pytest.mark.asyncio
    async def test_create_index_from_directory(self, llamaindex_processor, temp_directory, mock_dependencies):
        """Test creating index from directory"""
        
        result = await llamaindex_processor.create_index_from_directory(temp_directory)
        
        assert "index_created" in result
        assert result["index_created"] == True
        assert "documents_processed" in result
        assert "processing_time" in result
        
        # Verify security validation was called
        mock_dependencies["security_manager"].validate_directory_access.assert_called_once()
        
        # Verify performance tracking was called
        mock_dependencies["performance_monitor"].track_index_creation.assert_called_once()
        
        # Check metrics updated
        assert llamaindex_processor.processing_metrics["documents_indexed"] >= 0
    
    @pytest.mark.asyncio
    async def test_query_index(self, llamaindex_processor, mock_dependencies):
        """Test querying the index"""
        
        query = "What information is available about test documents?"
        context = {"query_type": "information_retrieval"}
        
        result = await llamaindex_processor.query_index(query, context)
        
        assert "query_id" in result
        assert result["query"] == query
        assert "response" in result or "error" in result
        assert "query_time" in result
        assert "status" in result
        
        # Verify security validation was called
        mock_dependencies["security_manager"].validate_query.assert_called_once()
        
        # If successful, verify other calls
        if result["status"] != "failed":
            # Verify memory storage was called
            mock_dependencies["memory_manager"].store_memory.assert_called_once()
            
            # Verify performance tracking was called
            mock_dependencies["performance_monitor"].track_query_execution.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_without_index(self, llamaindex_processor):
        """Test querying without an index"""
        
        result = await llamaindex_processor.query_index("test query")
        
        assert result["status"] == "failed"
        assert "No index available" in result["error"]
    
    def test_get_processor_info(self, llamaindex_processor):
        """Test getting processor information"""
        
        info = llamaindex_processor.get_processor_info()
        
        assert info["processor_id"] == llamaindex_processor.processor_id
        assert "processing_metrics" in info
        assert "has_index" in info
        assert "has_query_engine" in info
        assert "llamaindex_available" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])