# forge1/backend/forge1/integrations/haystack_adapter.py
"""
Haystack/LlamaIndex Adapter for Forge 1

Comprehensive integration of Haystack and LlamaIndex frameworks with Forge 1 enterprise enhancements.
Extends document processing capabilities with:
- Enterprise security and compliance for document handling
- Advanced memory layer integration
- Performance monitoring and optimization
- Quality assurance and validation
- Superhuman document processing standards
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import json
import uuid
import os

# Haystack imports (with fallback for testing)
try:
    from haystack import Document, Pipeline
    from haystack.components.readers import TextFileReader, PDFReader
    from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
    from haystack.components.writers import DocumentWriter
    from haystack.components.retrievers import InMemoryEmbeddingRetriever
    from haystack.document_stores import InMemoryDocumentStore
    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False
    Document = dict
    Pipeline = object

# LlamaIndex imports (with fallback for testing)
try:
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.vector_stores import VectorStore
    from llama_index.core.query_engine import BaseQueryEngine
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    VectorStoreIndex = object
    SimpleDirectoryReader = object

from forge1.core.memory_manager import MemoryManager
from forge1.core.model_router import ModelRouter
from forge1.core.performance_monitor import PerformanceMonitor
from forge1.core.quality_assurance import QualityAssuranceSystem, QualityLevel
from forge1.core.security_manager import SecurityManager

logger = logging.getLogger(__name__)

class DocumentProcessingType(Enum):
    """Types of document processing"""
    EXTRACTION = "extraction"
    INDEXING = "indexing"
    SEARCH = "search"
    SUMMARIZATION = "summarization"
    QA = "question_answering"
    CLASSIFICATION = "classification"
clas
s DocumentFormat(Enum):
    """Supported document formats"""
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "md"
    JSON = "json"
    CSV = "csv"

class ForgeDocumentProcessor:
    """Enhanced document processor with Forge 1 capabilities"""
    
    def __init__(
        self,
        processor_type: DocumentProcessingType,
        memory_manager: MemoryManager,
        performance_monitor: PerformanceMonitor,
        security_manager: SecurityManager,
        quality_assurance: QualityAssuranceSystem,
        **kwargs
    ):
        """Initialize enhanced document processor
        
        Args:
            processor_type: Type of document processing
            memory_manager: Forge 1 memory manager
            performance_monitor: Performance monitoring system
            security_manager: Security management system
            quality_assurance: Quality assurance system
            **kwargs: Additional processor parameters
        """
        self.processor_type = processor_type
        self.memory_manager = memory_manager
        self.performance_monitor = performance_monitor
        self.security_manager = security_manager
        self.quality_assurance = quality_assurance
        
        # Forge 1 enhancements
        self.processor_id = f"doc_proc_{uuid.uuid4().hex[:8]}"
        self.session_id = kwargs.get("session_id", str(uuid.uuid4()))
        
        # Processing metrics
        self.processing_metrics = {
            "documents_processed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "success_rate": 0.0,
            "quality_score": 0.0,
            "security_violations": 0
        }
        
        # Initialize document store and pipeline
        if HAYSTACK_AVAILABLE:
            self.document_store = InMemoryDocumentStore()
            self.pipeline = self._create_processing_pipeline(kwargs)
        else:
            self.document_store = None
            self.pipeline = None
        
        logger.info(f"Created enhanced document processor {self.processor_id} ({processor_type.value})")
    
    def _create_processing_pipeline(self, config: Dict[str, Any]) -> Pipeline:
        """Create Haystack processing pipeline"""
        
        pipeline = Pipeline()
        
        if self.processor_type == DocumentProcessingType.EXTRACTION:
            # Text extraction pipeline
            pipeline.add_component("reader", TextFileReader())
            pipeline.add_component("cleaner", DocumentCleaner())
            pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=3))
            
            pipeline.connect("reader", "cleaner")
            pipeline.connect("cleaner", "splitter")
            
        elif self.processor_type == DocumentProcessingType.INDEXING:
            # Document indexing pipeline
            pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
            pipeline.add_component("writer", DocumentWriter(document_store=self.document_store))
            
            pipeline.connect("embedder", "writer")
            
        elif self.processor_type == DocumentProcessingType.SEARCH:
            # Document search pipeline
            pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
            pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=self.document_store))
            
            pipeline.connect("text_embedder", "retriever")
        
        return pipeline
    
    async def process_document(
        self,
        document_path: str,
        document_format: DocumentFormat,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process document with Forge 1 enhancements"""
        
        processing_id = f"proc_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now(timezone.utc)
        
        try:
            # Security validation
            await self.security_manager.validate_document_access(
                document_path=document_path,
                processor_id=self.processor_id,
                context=context or {}
            )
            
            # Process document based on type
            if self.processor_type == DocumentProcessingType.EXTRACTION:
                result = await self._extract_document_content(document_path, document_format)
            elif self.processor_type == DocumentProcessingType.INDEXING:
                result = await self._index_document(document_path, document_format)
            elif self.processor_type == DocumentProcessingType.SEARCH:
                query = context.get("query", "") if context else ""
                result = await self._search_documents(query)
            else:
                result = await self._extract_document_content(document_path, document_format)
            
            # Store processing result in memory
            await self.memory_manager.store_memory(
                content=json.dumps(result),
                memory_type="document_processing",
                metadata={
                    "processing_id": processing_id,
                    "processor_id": self.processor_id,
                    "document_path": document_path,
                    "document_format": document_format.value,
                    "processor_type": self.processor_type.value,
                    "session_id": self.session_id
                }
            )
            
            # Quality assurance validation
            qa_result = await self.quality_assurance.conduct_quality_review(
                {"content": str(result), "confidence": 0.9},
                {
                    "processing_type": self.processor_type.value,
                    "document_format": document_format.value,
                    "context": context or {}
                },
                [self.processor_id],
                QualityLevel.SUPERHUMAN
            )
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Track performance
            await self.performance_monitor.track_document_processing(
                processor_id=self.processor_id,
                processing_id=processing_id,
                processing_time=processing_time,
                success=True,
                document_size=os.path.getsize(document_path) if os.path.exists(document_path) else 0
            )
            
            # Update metrics
            await self._update_processing_metrics(processing_time, True)
            
            return {
                "processing_id": processing_id,
                "processor_id": self.processor_id,
                "document_path": document_path,
                "document_format": document_format.value,
                "processor_type": self.processor_type.value,
                "result": result,
                "processing_time": processing_time,
                "quality_assessment": qa_result,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Document processing failed for {document_path}: {e}")
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            await self.performance_monitor.track_document_processing(
                processor_id=self.processor_id,
                processing_id=processing_id,
                processing_time=processing_time,
                success=False,
                error=str(e)
            )
            
            await self._update_processing_metrics(processing_time, False)
            
            return {
                "processing_id": processing_id,
                "processor_id": self.processor_id,
                "document_path": document_path,
                "error": str(e),
                "processing_time": processing_time,
                "status": "failed"
            }
    
    async def _extract_document_content(self, document_path: str, document_format: DocumentFormat) -> Dict[str, Any]:
        """Extract content from document"""
        
        if not HAYSTACK_AVAILABLE:
            # Mock extraction for testing
            return {
                "content": f"Mock extracted content from {document_path}",
                "metadata": {"format": document_format.value, "pages": 1}
            }
        
        try:
            # Use appropriate reader based on format
            if document_format == DocumentFormat.PDF:
                reader = PDFReader()
            else:
                reader = TextFileReader()
            
            # Run extraction pipeline
            result = self.pipeline.run({"reader": {"file_paths": [document_path]}})
            
            # Extract documents from result
            documents = result.get("splitter", {}).get("documents", [])
            
            extracted_content = {
                "content": "\n".join([doc.content for doc in documents]),
                "metadata": {
                    "format": document_format.value,
                    "chunks": len(documents),
                    "total_length": sum(len(doc.content) for doc in documents)
                },
                "documents": [{"content": doc.content, "metadata": doc.meta} for doc in documents]
            }
            
            return extracted_content
            
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return {"error": str(e), "content": "", "metadata": {}}
    
    async def _index_document(self, document_path: str, document_format: DocumentFormat) -> Dict[str, Any]:
        """Index document for search"""
        
        if not HAYSTACK_AVAILABLE:
            return {
                "indexed": True,
                "document_id": f"mock_doc_{uuid.uuid4().hex[:8]}",
                "chunks": 5
            }
        
        try:
            # First extract content
            extraction_result = await self._extract_document_content(document_path, document_format)
            
            if "error" in extraction_result:
                return extraction_result
            
            # Create documents for indexing
            documents = []
            for doc_data in extraction_result.get("documents", []):
                doc = Document(
                    content=doc_data["content"],
                    meta={
                        **doc_data["metadata"],
                        "source": document_path,
                        "format": document_format.value,
                        "indexed_at": datetime.now(timezone.utc).isoformat()
                    }
                )
                documents.append(doc)
            
            # Run indexing pipeline
            result = self.pipeline.run({"embedder": {"documents": documents}})
            
            indexed_docs = result.get("writer", {}).get("documents_written", 0)
            
            return {
                "indexed": True,
                "document_path": document_path,
                "documents_indexed": indexed_docs,
                "chunks": len(documents),
                "index_size": len(self.document_store.filter_documents())
            }
            
        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            return {"error": str(e), "indexed": False}
    
    async def _search_documents(self, query: str) -> Dict[str, Any]:
        """Search indexed documents"""
        
        if not HAYSTACK_AVAILABLE or not query:
            return {
                "query": query,
                "results": [
                    {"content": f"Mock search result 1 for: {query}", "score": 0.9},
                    {"content": f"Mock search result 2 for: {query}", "score": 0.8}
                ],
                "total_results": 2
            }
        
        try:
            # Run search pipeline
            result = self.pipeline.run({"text_embedder": {"text": query}})
            
            # Extract search results
            documents = result.get("retriever", {}).get("documents", [])
            
            search_results = []
            for doc in documents:
                search_results.append({
                    "content": doc.content,
                    "metadata": doc.meta,
                    "score": doc.score if hasattr(doc, 'score') else 1.0
                })
            
            return {
                "query": query,
                "results": search_results,
                "total_results": len(search_results),
                "search_time": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return {"error": str(e), "query": query, "results": []}
    
    async def _update_processing_metrics(self, processing_time: float, success: bool) -> None:
        """Update processing metrics"""
        
        # Update document count
        self.processing_metrics["documents_processed"] += 1
        
        # Update total and average processing time
        self.processing_metrics["total_processing_time"] += processing_time
        doc_count = self.processing_metrics["documents_processed"]
        self.processing_metrics["average_processing_time"] = (
            self.processing_metrics["total_processing_time"] / doc_count
        )
        
        # Update success rate
        if success:
            current_success_rate = self.processing_metrics["success_rate"]
            successful_docs = current_success_rate * (doc_count - 1) + 1
            self.processing_metrics["success_rate"] = successful_docs / doc_count
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor information and metrics"""
        
        return {
            "processor_id": self.processor_id,
            "processor_type": self.processor_type.value,
            "session_id": self.session_id,
            "processing_metrics": self.processing_metrics.copy(),
            "document_store_size": len(self.document_store.filter_documents()) if self.document_store else 0,
            "haystack_available": HAYSTACK_AVAILABLE,
            "llamaindex_available": LLAMAINDEX_AVAILABLE
        }

class ForgeLlamaIndexProcessor:
    """Enhanced LlamaIndex processor with Forge 1 capabilities"""
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        performance_monitor: PerformanceMonitor,
        security_manager: SecurityManager,
        quality_assurance: QualityAssuranceSystem,
        **kwargs
    ):
        """Initialize enhanced LlamaIndex processor"""
        
        self.memory_manager = memory_manager
        self.performance_monitor = performance_monitor
        self.security_manager = security_manager
        self.quality_assurance = quality_assurance
        
        # Forge 1 enhancements
        self.processor_id = f"llama_proc_{uuid.uuid4().hex[:8]}"
        self.session_id = kwargs.get("session_id", str(uuid.uuid4()))
        
        # LlamaIndex components
        if LLAMAINDEX_AVAILABLE:
            self.node_parser = SimpleNodeParser()
            self.index = None
            self.query_engine = None
        else:
            self.node_parser = None
            self.index = None
            self.query_engine = None
        
        # Processing metrics
        self.processing_metrics = {
            "documents_indexed": 0,
            "queries_processed": 0,
            "average_query_time": 0.0,
            "index_size": 0,
            "quality_score": 0.0
        }
        
        logger.info(f"Created enhanced LlamaIndex processor {self.processor_id}")
    
    async def create_index_from_directory(self, directory_path: str) -> Dict[str, Any]:
        """Create LlamaIndex from directory"""
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Security validation
            await self.security_manager.validate_directory_access(
                directory_path=directory_path,
                processor_id=self.processor_id
            )
            
            if not LLAMAINDEX_AVAILABLE:
                # Mock index creation
                return {
                    "index_created": True,
                    "documents_processed": 10,
                    "index_id": f"mock_index_{uuid.uuid4().hex[:8]}"
                }
            
            # Load documents from directory
            reader = SimpleDirectoryReader(directory_path)
            documents = reader.load_data()
            
            # Create index
            self.index = VectorStoreIndex.from_documents(documents)
            self.query_engine = self.index.as_query_engine()
            
            # Update metrics
            self.processing_metrics["documents_indexed"] = len(documents)
            self.processing_metrics["index_size"] = len(documents)
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Track performance
            await self.performance_monitor.track_index_creation(
                processor_id=self.processor_id,
                processing_time=processing_time,
                documents_count=len(documents),
                success=True
            )
            
            return {
                "index_created": True,
                "documents_processed": len(documents),
                "processing_time": processing_time,
                "index_id": self.processor_id
            }
            
        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            await self.performance_monitor.track_index_creation(
                processor_id=self.processor_id,
                processing_time=processing_time,
                success=False,
                error=str(e)
            )
            
            return {
                "index_created": False,
                "error": str(e),
                "processing_time": processing_time
            }
    
    async def query_index(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query the LlamaIndex"""
        
        start_time = datetime.now(timezone.utc)
        query_id = f"query_{uuid.uuid4().hex[:8]}"
        
        try:
            if not self.query_engine:
                return {
                    "query_id": query_id,
                    "error": "No index available for querying",
                    "status": "failed"
                }
            
            # Security validation
            await self.security_manager.validate_query(
                query=query,
                processor_id=self.processor_id,
                context=context or {}
            )
            
            if not LLAMAINDEX_AVAILABLE:
                # Mock query response
                return {
                    "query_id": query_id,
                    "query": query,
                    "response": f"Mock LlamaIndex response to: {query}",
                    "sources": ["mock_doc_1.txt", "mock_doc_2.txt"]
                }
            
            # Execute query
            response = self.query_engine.query(query)
            
            # Store query and response in memory
            await self.memory_manager.store_memory(
                content=f"Query: {query}\nResponse: {str(response)}",
                memory_type="document_query",
                metadata={
                    "query_id": query_id,
                    "processor_id": self.processor_id,
                    "session_id": self.session_id,
                    "query": query
                }
            )
            
            query_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Update metrics
            self.processing_metrics["queries_processed"] += 1
            current_avg = self.processing_metrics["average_query_time"]
            query_count = self.processing_metrics["queries_processed"]
            self.processing_metrics["average_query_time"] = (
                (current_avg * (query_count - 1) + query_time) / query_count
            )
            
            # Track performance
            await self.performance_monitor.track_query_execution(
                processor_id=self.processor_id,
                query_id=query_id,
                query_time=query_time,
                success=True
            )
            
            return {
                "query_id": query_id,
                "query": query,
                "response": str(response),
                "query_time": query_time,
                "sources": getattr(response, 'source_nodes', []),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            query_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            await self.performance_monitor.track_query_execution(
                processor_id=self.processor_id,
                query_id=query_id,
                query_time=query_time,
                success=False,
                error=str(e)
            )
            
            return {
                "query_id": query_id,
                "query": query,
                "error": str(e),
                "query_time": query_time,
                "status": "failed"
            }
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor information and metrics"""
        
        return {
            "processor_id": self.processor_id,
            "session_id": self.session_id,
            "processing_metrics": self.processing_metrics.copy(),
            "has_index": self.index is not None,
            "has_query_engine": self.query_engine is not None,
            "llamaindex_available": LLAMAINDEX_AVAILABLE
        }

class HaystackLlamaIndexAdapter:
    """Comprehensive Haystack/LlamaIndex adapter for Forge 1"""
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        model_router: ModelRouter,
        performance_monitor: PerformanceMonitor,
        quality_assurance: QualityAssuranceSystem,
        security_manager: SecurityManager
    ):
        """Initialize Haystack/LlamaIndex adapter with Forge 1 systems"""
        
        self.memory_manager = memory_manager
        self.model_router = model_router
        self.performance_monitor = performance_monitor
        self.quality_assurance = quality_assurance
        self.security_manager = security_manager
        
        # Document processing state
        self.active_processors = {}
        self.document_stores = {}
        
        # Performance metrics
        self.integration_metrics = {
            "processors_created": 0,
            "documents_processed": 0,
            "queries_executed": 0,
            "average_processing_time": 0.0,
            "success_rate": 0.0,
            "quality_score": 0.0
        }
        
        logger.info("Haystack/LlamaIndex adapter initialized with Forge 1 enterprise enhancements")
    
    async def create_document_processor(
        self,
        processor_type: DocumentProcessingType,
        processor_config: Dict[str, Any] = None
    ) -> ForgeDocumentProcessor:
        """Create enhanced document processor"""
        
        config = processor_config or {}
        
        processor = ForgeDocumentProcessor(
            processor_type=processor_type,
            memory_manager=self.memory_manager,
            performance_monitor=self.performance_monitor,
            security_manager=self.security_manager,
            quality_assurance=self.quality_assurance,
            **config
        )
        
        # Store processor
        self.active_processors[processor.processor_id] = processor
        
        # Update metrics
        self.integration_metrics["processors_created"] += 1
        
        logger.info(f"Created document processor {processor.processor_id} ({processor_type.value})")
        
        return processor
    
    async def create_llamaindex_processor(
        self,
        processor_config: Dict[str, Any] = None
    ) -> ForgeLlamaIndexProcessor:
        """Create enhanced LlamaIndex processor"""
        
        config = processor_config or {}
        
        processor = ForgeLlamaIndexProcessor(
            memory_manager=self.memory_manager,
            performance_monitor=self.performance_monitor,
            security_manager=self.security_manager,
            quality_assurance=self.quality_assurance,
            **config
        )
        
        # Store processor
        self.active_processors[processor.processor_id] = processor
        
        # Update metrics
        self.integration_metrics["processors_created"] += 1
        
        logger.info(f"Created LlamaIndex processor {processor.processor_id}")
        
        return processor
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration metrics"""
        
        return {
            "integration_metrics": self.integration_metrics.copy(),
            "active_processors": len(self.active_processors),
            "haystack_available": HAYSTACK_AVAILABLE,
            "llamaindex_available": LLAMAINDEX_AVAILABLE,
            "capabilities": [
                "document_processing",
                "enterprise_security",
                "quality_assurance",
                "performance_monitoring",
                "advanced_search",
                "content_extraction"
            ]
        }
    
    def get_active_processors(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active processors"""
        
        return {
            processor_id: processor.get_processor_info()
            for processor_id, processor in self.active_processors.items()
        }
    
    async def cleanup_processor(self, processor_id: str) -> bool:
        """Clean up processor resources"""
        
        if processor_id in self.active_processors:
            processor = self.active_processors[processor_id]
            
            # Cleanup processor resources if needed
            if hasattr(processor, 'document_store') and processor.document_store:
                # Clear document store
                processor.document_store.delete_documents()
            
            # Remove processor
            del self.active_processors[processor_id]
            
            logger.info(f"Cleaned up document processor {processor_id}")
            return True
        
        return False