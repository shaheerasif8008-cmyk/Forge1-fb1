"""
Weaviate Memory Store Adapter

Enhanced memory store adapter that integrates Weaviate vector database
with MemGPT compression and implements the Forge1 MemoryAdapter interface.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union

from forge1.integrations.base_adapter import BaseAdapter, HealthCheckResult, AdapterStatus, ExecutionContext, TenantContext
from forge1.integrations.vector.weaviate_client import WeaviateAdapter, VectorData, VectorQuery, VectorResult
from forge1.core.memory.memgpt_summarizer import MemGPTSummarizer, CompressionStrategy, TTLPolicy, CompressionResult
from forge1.core.memory_models import MemoryContext, MemoryQuery, MemorySearchResult, MemorySearchResponse, MemoryType, SecurityLevel
from forge1.core.model_router import ModelRouter
from forge1.core.tenancy import get_current_tenant
from forge1.core.dlp import redact_payload

logger = logging.getLogger(__name__)

class WeaviateMemoryAdapter(BaseAdapter):
    """Enhanced memory adapter integrating Weaviate and MemGPT compression"""
    
    def __init__(self, weaviate_adapter: Optional[WeaviateAdapter] = None, 
                 memgpt_summarizer: Optional[MemGPTSummarizer] = None):
        super().__init__("weaviate_memory", {})
        
        self.weaviate = weaviate_adapter or WeaviateAdapter()
        self.model_router = ModelRouter()
        self.memgpt_summarizer = memgpt_summarizer or MemGPTSummarizer(self.model_router)
        
        # Memory management settings
        self.compression_threshold_tokens = 8000
        self.auto_compression_enabled = True
        self.ttl_policy = TTLPolicy(
            max_age_days=30,
            compression_threshold_tokens=8000,
            archive_after_days=90,
            delete_after_days=365
        )
        
        # Performance tracking
        self._operation_stats = {
            "store_operations": 0,
            "retrieve_operations": 0,
            "search_operations": 0,
            "compression_operations": 0,
            "total_memories_stored": 0,
            "total_memories_compressed": 0
        }
    
    async def initialize(self) -> bool:
        """Initialize the memory adapter and underlying services"""
        try:
            # Initialize Weaviate adapter
            if not await self.weaviate.initialize():
                logger.error("Failed to initialize Weaviate adapter")
                return False
            
            # Initialize model router (if needed)
            # The model router is typically initialized globally
            
            logger.info("Weaviate memory adapter initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate memory adapter: {e}")
            return False
    
    async def health_check(self) -> HealthCheckResult:
        """Perform comprehensive health check"""
        start_time = time.time()
        
        try:
            # Check Weaviate health
            weaviate_health = await self.weaviate.health_check()
            
            # Check model router health
            model_router_health = await self.model_router.health_check()
            
            # Determine overall status
            if (weaviate_health.status == AdapterStatus.HEALTHY and 
                model_router_health.get("status") == "healthy"):
                status = AdapterStatus.HEALTHY
                message = "Memory adapter healthy"
            elif (weaviate_health.status in [AdapterStatus.HEALTHY, AdapterStatus.DEGRADED] and
                  model_router_health.get("status") in ["healthy", "degraded"]):
                status = AdapterStatus.DEGRADED
                message = "Memory adapter degraded"
            else:
                status = AdapterStatus.UNHEALTHY
                message = "Memory adapter unhealthy"
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                status=status,
                message=message,
                details={
                    "weaviate_status": weaviate_health.status.value,
                    "model_router_status": model_router_health.get("status", "unknown"),
                    "operation_stats": self._operation_stats.copy(),
                    "compression_enabled": self.auto_compression_enabled,
                    "compression_threshold": self.compression_threshold_tokens
                },
                timestamp=time.time(),
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=AdapterStatus.UNHEALTHY,
                message=f"Memory adapter health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
                response_time_ms=response_time
            )
    
    async def cleanup(self) -> bool:
        """Clean up memory adapter resources"""
        try:
            await self.weaviate.cleanup()
            logger.info("Weaviate memory adapter cleaned up successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup memory adapter: {e}")
            return False
    
    async def store_memory(self, memory: MemoryContext, context: Optional[ExecutionContext] = None) -> str:
        """Store memory with automatic compression and vector indexing"""
        
        if not await self.ensure_initialized():
            raise RuntimeError("Memory adapter not initialized")
        
        if not context:
            context = self.create_execution_context(str(uuid.uuid4()))
        
        try:
            start_time = time.time()
            
            # Ensure tenant context matches
            if memory.owner_id != context.tenant_context.user_id:
                context.tenant_context.user_id = memory.owner_id
            
            # Apply DLP redaction
            safe_content, violations = redact_payload(memory.content)
            memory.content = safe_content
            
            if memory.summary:
                safe_summary, summary_violations = redact_payload(memory.summary)
                memory.summary = safe_summary
                violations.extend(summary_violations)
            
            # Add DLP violation tags
            if violations:
                violation_types = list(set(v.get("type", "unknown") for v in violations))
                memory.tags.extend([f"dlp:{vtype}" for vtype in violation_types])
            
            # Check if compression is needed
            content_text = self._extract_text_from_memory(memory)
            estimated_tokens = len(content_text) // 4  # Rough estimation
            
            compressed_memory = None
            if (self.auto_compression_enabled and 
                estimated_tokens > self.compression_threshold_tokens):
                
                logger.info(f"Memory {memory.id} exceeds token threshold, applying compression")
                
                try:
                    compressed_memory = await self.memgpt_summarizer.compress_memory_context(
                        memory, CompressionStrategy.ROLLING_SUMMARY
                    )
                    
                    # Update memory content with compressed version
                    memory.content = {"compressed_content": compressed_memory.compressed_content}
                    memory.summary = f"Compressed summary (ratio: {compressed_memory.compression_ratio:.2f})"
                    memory.tags.append(f"compressed:{compressed_memory.compression_strategy.value}")
                    
                    self._operation_stats["compression_operations"] += 1
                    self._operation_stats["total_memories_compressed"] += 1
                    
                except Exception as compression_error:
                    logger.warning(f"Failed to compress memory {memory.id}: {compression_error}")
                    # Continue with uncompressed memory
            
            # Generate embedding for the memory
            embedding_text = self._prepare_embedding_text(memory)
            embedding = await self._generate_embedding(embedding_text)
            
            # Prepare vector data
            vector_data = VectorData(
                id=memory.id,
                vector=embedding,
                properties={
                    "content": json.dumps(memory.content) if isinstance(memory.content, dict) else str(memory.content),
                    "summary": memory.summary or "",
                    "memory_type": memory.memory_type.value,
                    "employee_id": memory.employee_id,
                    "session_id": memory.session_id or "",
                    "security_level": memory.security_level.value,
                    "owner_id": memory.owner_id,
                    "shared_with": json.dumps(memory.shared_with),
                    "source": memory.source or "",
                    "tags": json.dumps(memory.tags),
                    "keywords": json.dumps(memory.keywords),
                    "created_at": memory.created_at.timestamp(),
                    "updated_at": memory.updated_at.timestamp() if memory.updated_at else memory.created_at.timestamp(),
                    "expires_at": memory.expires_at.timestamp() if memory.expires_at else None,
                    "parent_memory_id": memory.parent_memory_id or "",
                    "related_memory_ids": json.dumps(memory.related_memory_ids),
                    "access_count": memory.access_count,
                    "last_accessed": memory.last_accessed.timestamp() if memory.last_accessed else None,
                    "dlp_violations": len(violations),
                    "compressed": compressed_memory is not None,
                    "compression_ratio": compressed_memory.compression_ratio if compressed_memory else 1.0,
                    "fidelity_score": compressed_memory.fidelity_score if compressed_memory else 1.0
                },
                class_name="Memory",
                tenant_id=context.tenant_context.tenant_id,
                metadata={
                    "compression_applied": compressed_memory is not None,
                    "dlp_violations": len(violations)
                }
            )
            
            # Store in Weaviate
            stored_id = await self.weaviate.store_vector(vector_data, context)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self._operation_stats["store_operations"] += 1
            self._operation_stats["total_memories_stored"] += 1
            
            # Log operation
            self.log_operation(
                "store_memory",
                context,
                duration_ms=execution_time,
                success=True
            )
            
            logger.info(f"Stored memory {memory.id} with {len(violations)} DLP violations, "
                       f"compressed: {compressed_memory is not None}")
            
            return stored_id
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            self.log_operation(
                "store_memory",
                context,
                duration_ms=execution_time,
                success=False,
                error=str(e)
            )
            
            logger.error(f"Failed to store memory {memory.id}: {e}")
            raise
    
    async def retrieve_memory(self, memory_id: str, context: Optional[ExecutionContext] = None) -> Optional[MemoryContext]:
        """Retrieve a specific memory by ID"""
        
        if not await self.ensure_initialized():
            raise RuntimeError("Memory adapter not initialized")
        
        if not context:
            context = self.create_execution_context(str(uuid.uuid4()))
        
        try:
            start_time = time.time()
            
            # Search for the specific memory
            query = VectorQuery(
                class_name="Memory",
                where_filter={"path": ["id"], "operator": "Equal", "valueString": memory_id},
                limit=1,
                tenant_id=context.tenant_context.tenant_id,
                additional_properties=["*"]
            )
            
            results = await self.weaviate.search_vectors(query, context)
            
            if not results:
                return None
            
            # Convert result to MemoryContext
            result = results[0]
            memory = self._vector_result_to_memory_context(result)
            
            # Check access permissions
            if not self._check_memory_access(memory, context.tenant_context):
                logger.warning(f"Access denied to memory {memory_id} for user {context.tenant_context.user_id}")
                return None
            
            # Update access tracking
            memory.access_count += 1
            memory.last_accessed = datetime.now(timezone.utc)
            
            # TODO: Update access count in Weaviate (would require update operation)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self._operation_stats["retrieve_operations"] += 1
            
            # Log operation
            self.log_operation(
                "retrieve_memory",
                context,
                duration_ms=execution_time,
                success=True
            )
            
            logger.info(f"Retrieved memory {memory_id}")
            return memory
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            self.log_operation(
                "retrieve_memory",
                context,
                duration_ms=execution_time,
                success=False,
                error=str(e)
            )
            
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            return None
    
    async def search_memories(self, query: MemoryQuery, context: Optional[ExecutionContext] = None) -> MemorySearchResponse:
        """Search memories using semantic similarity and filters"""
        
        if not await self.ensure_initialized():
            raise RuntimeError("Memory adapter not initialized")
        
        if not context:
            context = self.create_execution_context(str(uuid.uuid4()))
        
        start_time = time.time()
        query_id = str(uuid.uuid4())
        
        try:
            # Generate query embedding if text provided
            query_embedding = None
            if query.query_text:
                query_embedding = await self._generate_embedding(query.query_text)
            
            # Build where filter
            where_filter = {}
            
            # Memory type filter
            if query.memory_types:
                memory_type_values = [mt.value for mt in query.memory_types]
                if len(memory_type_values) == 1:
                    where_filter["memory_type"] = {"equal": memory_type_values[0]}
                else:
                    where_filter["memory_type"] = {"valueStringArray": memory_type_values}
            
            # Employee filter
            if query.employee_ids:
                if len(query.employee_ids) == 1:
                    where_filter["employee_id"] = {"equal": query.employee_ids[0]}
                else:
                    where_filter["employee_id"] = {"valueStringArray": query.employee_ids}
            
            # Session filter
            if query.session_ids:
                if len(query.session_ids) == 1:
                    where_filter["session_id"] = {"equal": query.session_ids[0]}
                else:
                    where_filter["session_id"] = {"valueStringArray": query.session_ids}
            
            # Security level filter
            if query.security_levels:
                security_values = [sl.value for sl in query.security_levels]
                if len(security_values) == 1:
                    where_filter["security_level"] = {"equal": security_values[0]}
                else:
                    where_filter["security_level"] = {"valueStringArray": security_values}
            
            # Age filter
            if query.max_age_days:
                cutoff_timestamp = (datetime.now(timezone.utc) - timedelta(days=query.max_age_days)).timestamp()
                where_filter["created_at"] = {"greaterThanEqual": cutoff_timestamp}
            
            # Create Weaviate query
            weaviate_query = VectorQuery(
                vector=query_embedding,
                class_name="Memory",
                where_filter=where_filter,
                limit=query.limit,
                offset=query.offset,
                certainty=0.7,  # Default certainty threshold
                tenant_id=context.tenant_context.tenant_id,
                additional_properties=["*"]
            )
            
            # Execute search
            vector_results = await self.weaviate.search_vectors(weaviate_query, context)
            
            # Convert results to MemorySearchResult
            search_results = []
            for i, vector_result in enumerate(vector_results):
                memory = self._vector_result_to_memory_context(vector_result)
                
                # Check access permissions
                if not self._check_memory_access(memory, context.tenant_context):
                    continue
                
                search_result = MemorySearchResult(
                    memory=memory,
                    similarity_score=vector_result.certainty,
                    rank=query.offset + i + 1,
                    explanation=f"Semantic similarity: {vector_result.certainty:.3f}"
                )
                search_results.append(search_result)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self._operation_stats["search_operations"] += 1
            
            # Log operation
            self.log_operation(
                "search_memories",
                context,
                duration_ms=execution_time,
                success=True
            )
            
            logger.info(f"Memory search returned {len(search_results)} results")
            
            return MemorySearchResponse(
                results=search_results,
                total_count=len(search_results),  # Weaviate doesn't provide total count easily
                query_time_ms=execution_time,
                query_id=query_id
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            self.log_operation(
                "search_memories",
                context,
                duration_ms=execution_time,
                success=False,
                error=str(e)
            )
            
            logger.error(f"Memory search failed: {e}")
            
            return MemorySearchResponse(
                results=[],
                total_count=0,
                query_time_ms=execution_time,
                query_id=query_id
            )
    
    async def delete_memory(self, memory_id: str, context: Optional[ExecutionContext] = None) -> bool:
        """Delete a memory"""
        
        if not await self.ensure_initialized():
            raise RuntimeError("Memory adapter not initialized")
        
        if not context:
            context = self.create_execution_context(str(uuid.uuid4()))
        
        try:
            # First retrieve the memory to check permissions
            memory = await self.retrieve_memory(memory_id, context)
            if not memory:
                return False
            
            # Check delete permissions
            if memory.owner_id != context.tenant_context.user_id:
                logger.warning(f"Delete access denied to memory {memory_id} for user {context.tenant_context.user_id}")
                return False
            
            # Delete from Weaviate
            success = await self.weaviate.delete_vector(
                memory_id, 
                "Memory", 
                context.tenant_context.tenant_id, 
                context
            )
            
            if success:
                logger.info(f"Deleted memory {memory_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    async def compress_old_memories(self, tenant_id: str, context: Optional[ExecutionContext] = None) -> CompressionResult:
        """Compress old memories for a tenant based on TTL policy"""
        
        if not context:
            context = self.create_execution_context(str(uuid.uuid4()))
        
        try:
            # Use MemGPT summarizer for compression
            result = await self.memgpt_summarizer.compress_old_memories(tenant_id, self.ttl_policy)
            
            # Update statistics
            self._operation_stats["compression_operations"] += 1
            self._operation_stats["total_memories_compressed"] += result.compressed_count
            
            logger.info(f"Compressed {result.compressed_count} memories for tenant {tenant_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to compress old memories for tenant {tenant_id}: {e}")
            raise
    
    def _extract_text_from_memory(self, memory: MemoryContext) -> str:
        """Extract searchable text from memory"""
        text_parts = []
        
        if memory.summary:
            text_parts.append(memory.summary)
        
        if isinstance(memory.content, dict):
            for key, value in memory.content.items():
                if isinstance(value, str):
                    text_parts.append(value)
                else:
                    text_parts.append(str(value))
        elif isinstance(memory.content, str):
            text_parts.append(memory.content)
        else:
            text_parts.append(str(memory.content))
        
        return " ".join(text_parts)
    
    def _prepare_embedding_text(self, memory: MemoryContext) -> str:
        """Prepare text for embedding generation"""
        text_parts = []
        
        # Add summary if available
        if memory.summary:
            text_parts.append(f"Summary: {memory.summary}")
        
        # Add content
        content_text = self._extract_text_from_memory(memory)
        text_parts.append(content_text)
        
        # Add keywords
        if memory.keywords:
            text_parts.append(f"Keywords: {', '.join(memory.keywords)}")
        
        return "\n".join(text_parts)
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using model router"""
        try:
            client = await self.model_router.get_optimal_client("embedding_generation")
            
            # Mock embedding generation (in production would use actual embedding service)
            # This would integrate with the actual embedding service
            import hashlib
            import numpy as np
            
            # Create deterministic but varied embedding based on text
            text_hash = hashlib.md5(text.encode()).hexdigest()
            np.random.seed(int(text_hash[:8], 16))
            embedding = np.random.normal(0, 1, 1536).tolist()
            
            # Normalize to unit vector
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (np.array(embedding) / norm).tolist()
            
            await self.model_router.release_client(client)
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536
    
    def _vector_result_to_memory_context(self, result: VectorResult) -> MemoryContext:
        """Convert VectorResult to MemoryContext"""
        props = result.properties
        
        # Parse JSON fields
        content = props.get("content", "")
        try:
            content = json.loads(content) if content else {}
        except json.JSONDecodeError:
            pass
        
        shared_with = props.get("shared_with", "[]")
        try:
            shared_with = json.loads(shared_with) if shared_with else []
        except json.JSONDecodeError:
            shared_with = []
        
        tags = props.get("tags", "[]")
        try:
            tags = json.loads(tags) if tags else []
        except json.JSONDecodeError:
            tags = []
        
        keywords = props.get("keywords", "[]")
        try:
            keywords = json.loads(keywords) if keywords else []
        except json.JSONDecodeError:
            keywords = []
        
        related_memory_ids = props.get("related_memory_ids", "[]")
        try:
            related_memory_ids = json.loads(related_memory_ids) if related_memory_ids else []
        except json.JSONDecodeError:
            related_memory_ids = []
        
        # Convert timestamps
        created_at = datetime.fromtimestamp(props.get("created_at", time.time()), tz=timezone.utc)
        updated_at = None
        if props.get("updated_at"):
            updated_at = datetime.fromtimestamp(props["updated_at"], tz=timezone.utc)
        
        expires_at = None
        if props.get("expires_at"):
            expires_at = datetime.fromtimestamp(props["expires_at"], tz=timezone.utc)
        
        last_accessed = None
        if props.get("last_accessed"):
            last_accessed = datetime.fromtimestamp(props["last_accessed"], tz=timezone.utc)
        
        return MemoryContext(
            id=result.id,
            employee_id=props.get("employee_id", ""),
            session_id=props.get("session_id") or None,
            memory_type=MemoryType(props.get("memory_type", "conversation")),
            content=content,
            summary=props.get("summary"),
            keywords=keywords,
            embeddings=result.vector,
            embedding_model="text-embedding-ada-002",  # Default model
            access_count=props.get("access_count", 0),
            last_accessed=last_accessed,
            security_level=SecurityLevel(props.get("security_level", "standard")),
            owner_id=props.get("owner_id", ""),
            shared_with=shared_with,
            source=props.get("source"),
            tags=tags,
            created_at=created_at,
            updated_at=updated_at,
            expires_at=expires_at,
            parent_memory_id=props.get("parent_memory_id") or None,
            related_memory_ids=related_memory_ids
        )
    
    def _check_memory_access(self, memory: MemoryContext, tenant_context: TenantContext) -> bool:
        """Check if user has access to memory"""
        # Owner always has access
        if memory.owner_id == tenant_context.user_id:
            return True
        
        # Public memories are accessible
        if memory.security_level == SecurityLevel.PUBLIC:
            return True
        
        # Check if user is in shared_with list
        if tenant_context.user_id in memory.shared_with:
            return True
        
        # Check tenant isolation
        if not memory.tags or f"tenant:{tenant_context.tenant_id}" not in memory.tags:
            return False
        
        return False
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get operation statistics"""
        return self._operation_stats.copy()
    
    def reset_statistics(self):
        """Reset operation statistics"""
        self._operation_stats = {
            "store_operations": 0,
            "retrieve_operations": 0,
            "search_operations": 0,
            "compression_operations": 0,
            "total_memories_stored": 0,
            "total_memories_compressed": 0
        }

# Global Weaviate memory adapter instance
weaviate_memory_adapter = WeaviateMemoryAdapter()