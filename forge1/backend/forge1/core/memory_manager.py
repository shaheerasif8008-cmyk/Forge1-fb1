# forge1/backend/forge1/core/memory_manager.py
"""
Forge 1 Memory Manager

Implements semantic memory management with:
- Embedding generation and storage
- Semantic search and retrieval
- Contextual relevance scoring
- Memory lifecycle management
"""

import asyncio
import json
import os
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import openai
from openai import AsyncOpenAI
import asyncpg

from forge1.core.database_config import DatabaseManager, get_database_manager
from forge1.core.tenancy import get_current_tenant, tenant_prefix
from forge1.core.memory_models import (
    MemoryContext, MemoryQuery, MemorySearchResult, MemorySearchResponse,
    MemoryType, SecurityLevel, RelevanceScore, MemoryStats,
    EmbeddingRequest, EmbeddingResponse
)
from forge1.core.dlp import redact_payload
try:
    from forge1.dlp.presidio_adapter import HAS_PRESIDIO, redact_payload_presidio
except Exception:
    HAS_PRESIDIO = False
    def redact_payload_presidio(payload, language="en"):
        return payload, []

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating and managing embeddings"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-ada-002"):
        self.client = AsyncOpenAI(api_key=api_key) if api_key else None
        self.model = model
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = 3600  # 1 hour
        
        if not self.client:
            logger.warning("OpenAI API key not provided, using mock embeddings")
    
    async def generate_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embedding for text"""
        start_time = time.time()
        
        # Check cache first
        cache_key = request.cache_key or f"{request.model}:{hash(request.text)}"
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if time.time() - cached_result['timestamp'] < self.cache_ttl:
                logger.debug(f"Using cached embedding for: {request.text[:50]}...")
                return EmbeddingResponse(
                    embedding=cached_result['embedding'],
                    model=request.model,
                    token_count=cached_result['token_count'],
                    processing_time_ms=(time.time() - start_time) * 1000,
                    cached=True
                )
        
        try:
            if self.client:
                # Use real OpenAI API
                response = await self.client.embeddings.create(
                    input=request.text,
                    model=request.model
                )
                
                embedding = response.data[0].embedding
                token_count = response.usage.total_tokens
                
            else:
                # Use mock embedding for development
                embedding = self._generate_mock_embedding(request.text)
                token_count = len(request.text.split())
            
            processing_time = (time.time() - start_time) * 1000
            
            # Cache the result
            self.cache[cache_key] = {
                'embedding': embedding,
                'token_count': token_count,
                'timestamp': time.time()
            }
            
            logger.debug(f"Generated embedding for text: {request.text[:50]}... (tokens: {token_count})")
            
            return EmbeddingResponse(
                embedding=embedding,
                model=request.model,
                token_count=token_count,
                processing_time_ms=processing_time,
                cached=False
            )
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Fallback to mock embedding
            return EmbeddingResponse(
                embedding=self._generate_mock_embedding(request.text),
                model=request.model,
                token_count=len(request.text.split()),
                processing_time_ms=(time.time() - start_time) * 1000,
                cached=False
            )
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate mock embedding for development/testing"""
        # Create deterministic but varied embedding based on text
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.normal(0, 1, 1536).tolist()
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        return (np.array(embedding) / norm).tolist()
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

class MemoryManager:
    """Advanced memory management with semantic search capabilities"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager
        self.embedding_service = EmbeddingService()
        self._initialized = False
    
    async def initialize(self):
        """Initialize the memory manager"""
        if self._initialized:
            return
        
        if not self.db_manager:
            self.db_manager = await get_database_manager()
        
        self._initialized = True
        logger.info("Memory manager initialized")
    
    async def store_memory(self, memory: MemoryContext) -> str:
        """Store a memory context with embeddings"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Ensure tenant tag present (A.2)
            tenant_tag = f"tenant:{get_current_tenant()}"
            if tenant_tag not in memory.tags:
                memory.tags.append(tenant_tag)
            # DLP redaction of content and summary (B.1) with provider selection
            dlp_provider = os.getenv("DLP_PROVIDER", "builtin").lower()
            if dlp_provider == "presidio" and HAS_PRESIDIO:
                redacted_content, violations = redact_payload_presidio(memory.content)
            else:
                redacted_content, violations = redact_payload(memory.content)
            memory.content = redacted_content
            if memory.summary:
                if dlp_provider == "presidio" and HAS_PRESIDIO:
                    memory.summary, v2 = redact_payload_presidio(memory.summary)
                else:
                    memory.summary, v2 = redact_payload(memory.summary)
                violations.extend(v2)

            # Tag memory with DLP violations summary
            if violations:
                violation_types = sorted({v["type"] for v in violations})
                memory.tags.extend([f"dlp:{t}" for t in violation_types if f"dlp:{t}" not in memory.tags])

            # Generate embeddings for the memory content (post-redaction)
            content_text = self._extract_text_from_content(memory.content)
            if memory.summary:
                content_text = f"{memory.summary}\n{content_text}"
            
            embedding_request = EmbeddingRequest(
                text=content_text,
                user_id=memory.owner_id
            )
            
            embedding_response = await self.embedding_service.generate_embedding(embedding_request)
            memory.embeddings = embedding_response.embedding
            memory.embedding_model = embedding_response.model
            
            # Store in PostgreSQL
            async with self.db_manager.postgres.acquire() as conn:
                memory_id = await conn.fetchval("""
                    INSERT INTO forge1_memory.memory_contexts (
                        id, employee_id, session_id, memory_type, content, summary, keywords,
                        embeddings, embedding_model, security_level, owner_id, shared_with,
                        source, tags, created_at, updated_at, expires_at,
                        parent_memory_id, related_memory_ids
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19
                    ) RETURNING id
                """, 
                    memory.id, memory.employee_id, memory.session_id, memory.memory_type.value,
                    json.dumps(memory.content), memory.summary, memory.keywords,
                    memory.embeddings, memory.embedding_model, memory.security_level.value,
                    memory.owner_id, memory.shared_with, memory.source, memory.tags,
                    memory.created_at, memory.updated_at, memory.expires_at,
                    memory.parent_memory_id, memory.related_memory_ids
                )
            
            # Store in vector database for fast similarity search
            await self._store_in_vector_db(memory)
            
            # Cache in Redis for fast access
            await self._cache_memory(memory)
            
            logger.info(f"Stored memory {memory_id} for employee {memory.employee_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise
    
    async def retrieve_memory(self, memory_id: str, user_id: str) -> Optional[MemoryContext]:
        """Retrieve a specific memory by ID"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Try cache first
            cached_memory = await self._get_cached_memory(memory_id)
            if cached_memory and cached_memory.can_access(user_id):
                cached_memory.update_access()
                await self._update_access_tracking(memory_id)
                return cached_memory
            
            # Retrieve from database
            async with self.db_manager.postgres.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM forge1_memory.memory_contexts 
                    WHERE id = $1
                """, memory_id)
                
                if not row:
                    return None
                
                memory = self._row_to_memory_context(row)
                
                # Check access permissions
                if not memory.can_access(user_id):
                    logger.warning(f"Access denied to memory {memory_id} for user {user_id}")
                    return None
                
                # Update access tracking
                memory.update_access()
                await self._update_access_tracking(memory_id)
                
                # Cache for future access
                await self._cache_memory(memory)
                
                return memory
                
        except Exception as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            return None
    
    async def search_memories(self, query: MemoryQuery, user_id: str) -> MemorySearchResponse:
        """Search memories using semantic similarity and filters"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        query_id = str(uuid.uuid4())
        
        try:
            # Enforce tenant filter (A.2)
            tenant_tag = f"tenant:{get_current_tenant()}"
            # Generate query embedding if text provided
            query_embedding = query.query_embedding
            if query.query_text and not query_embedding:
                embedding_request = EmbeddingRequest(
                    text=query.query_text,
                    user_id=user_id
                )
                embedding_response = await self.embedding_service.generate_embedding(embedding_request)
                query_embedding = embedding_response.embedding
            
            # Build SQL query with filters
            sql_conditions = ["1=1"]  # Base condition
            sql_params = []
            param_count = 0
            
            # Security filter - only accessible memories
            param_count += 1
            sql_conditions.append(f"(owner_id = ${param_count} OR security_level = 'public' OR ${param_count} = ANY(shared_with))")
            sql_params.append(user_id)
            
            # Memory type filter
            if query.memory_types:
                param_count += 1
                sql_conditions.append(f"memory_type = ANY(${param_count})")
                sql_params.append([mt.value for mt in query.memory_types])
            
            # Employee filter
            if query.employee_ids:
                param_count += 1
                sql_conditions.append(f"employee_id = ANY(${param_count})")
                sql_params.append(query.employee_ids)
            
            # Session filter
            if query.session_ids:
                param_count += 1
                sql_conditions.append(f"session_id = ANY(${param_count})")
                sql_params.append(query.session_ids)
            
            # Keywords filter
            if query.keywords:
                param_count += 1
                sql_conditions.append(f"keywords && ${param_count}")
                sql_params.append(query.keywords)
            
            # Tags filter
            if query.tags:
                param_count += 1
                sql_conditions.append(f"tags && ${param_count}")
                sql_params.append(query.tags)
            # Tenant tag enforced
            param_count += 1
            sql_conditions.append(f"tags && ${param_count}")
            sql_params.append([tenant_tag])
            
            # Age filter
            if query.max_age_days:
                param_count += 1
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=query.max_age_days)
                sql_conditions.append(f"created_at >= ${param_count}")
                sql_params.append(cutoff_date)
            
            # Security level filter
            if query.security_levels:
                param_count += 1
                sql_conditions.append(f"security_level = ANY(${param_count})")
                sql_params.append([sl.value for sl in query.security_levels])
            
            # Relevance score filter
            if query.min_relevance_score > 0:
                param_count += 1
                sql_conditions.append(f"overall_relevance_score >= ${param_count}")
                sql_params.append(query.min_relevance_score)
            
            # Build final query
            where_clause = " AND ".join(sql_conditions)
            
            # Determine ordering
            if query_embedding and query.sort_by == "relevance_score":
                # Use vector similarity for ordering
                order_clause = "ORDER BY (embeddings <=> $1::vector) ASC"
                sql_params.insert(0, query_embedding)
                # Adjust parameter numbers
                for i in range(len(sql_conditions)):
                    if "$" in sql_conditions[i]:
                        sql_conditions[i] = sql_conditions[i].replace(f"${i+1}", f"${i+2}")
                where_clause = " AND ".join(sql_conditions)
            else:
                # Use standard ordering
                order_direction = "DESC" if query.sort_order == "desc" else "ASC"
                order_clause = f"ORDER BY {query.sort_by} {order_direction}"
            
            # Execute search query
            async with self.db_manager.postgres.acquire() as conn:
                # Get total count
                count_query = f"""
                    SELECT COUNT(*) FROM forge1_memory.memory_contexts 
                    WHERE {where_clause}
                """
                total_count = await conn.fetchval(count_query, *sql_params)
                
                # Get results with pagination
                search_query = f"""
                    SELECT * FROM forge1_memory.memory_contexts 
                    WHERE {where_clause}
                    {order_clause}
                    LIMIT ${len(sql_params) + 1} OFFSET ${len(sql_params) + 2}
                """
                sql_params.extend([query.limit, query.offset])
                
                rows = await conn.fetch(search_query, *sql_params)
            
            # Convert to search results
            results = []
            for i, row in enumerate(rows):
                memory = self._row_to_memory_context(row)
                
                # Calculate similarity score
                similarity_score = 0.0
                if query_embedding and memory.embeddings:
                    similarity_score = self.embedding_service.cosine_similarity(
                        query_embedding, memory.embeddings
                    )
                
                # Calculate relevance score
                relevance_score = await self._calculate_relevance_score(
                    memory, query.query_text, similarity_score
                )
                memory.relevance_score = relevance_score
                
                result = MemorySearchResult(
                    memory=memory,
                    similarity_score=similarity_score,
                    rank=query.offset + i + 1,
                    explanation=self._generate_search_explanation(memory, query, similarity_score)
                )
                results.append(result)
            
            query_time_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Memory search completed: {len(results)}/{total_count} results in {query_time_ms:.2f}ms")
            
            return MemorySearchResponse(
                results=results,
                total_count=total_count,
                query_time_ms=query_time_ms,
                query_id=query_id
            )
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return MemorySearchResponse(
                results=[],
                total_count=0,
                query_time_ms=(time.time() - start_time) * 1000,
                query_id=query_id
            )
    
    async def update_memory(self, memory_id: str, updates: Dict[str, Any], user_id: str) -> bool:
        """Update an existing memory"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Retrieve current memory to check permissions
            current_memory = await self.retrieve_memory(memory_id, user_id)
            if not current_memory:
                return False
            
            # Check write permissions
            if current_memory.owner_id != user_id:
                logger.warning(f"Write access denied to memory {memory_id} for user {user_id}")
                return False
            
            # Prepare update fields
            update_fields = []
            update_params = []
            param_count = 0
            
            # Handle content updates (regenerate embeddings if needed)
            if 'content' in updates or 'summary' in updates:
                new_content = updates.get('content', current_memory.content)
                new_summary = updates.get('summary', current_memory.summary)
                
                # Regenerate embeddings
                content_text = self._extract_text_from_content(new_content)
                if new_summary:
                    content_text = f"{new_summary}\n{content_text}"
                
                embedding_request = EmbeddingRequest(text=content_text, user_id=user_id)
                embedding_response = await self.embedding_service.generate_embedding(embedding_request)
                
                updates['embeddings'] = embedding_response.embedding
                updates['embedding_model'] = embedding_response.model
            
            # Build update query
            for field, value in updates.items():
                if field in ['content', 'summary', 'keywords', 'embeddings', 'embedding_model', 
                           'security_level', 'shared_with', 'source', 'tags', 'expires_at']:
                    param_count += 1
                    update_fields.append(f"{field} = ${param_count}")
                    
                    if field == 'content':
                        update_params.append(json.dumps(value))
                    elif field == 'security_level':
                        update_params.append(value.value if hasattr(value, 'value') else value)
                    else:
                        update_params.append(value)
            
            if not update_fields:
                return True  # No valid updates
            
            # Add updated_at
            param_count += 1
            update_fields.append(f"updated_at = ${param_count}")
            update_params.append(datetime.now(timezone.utc))
            
            # Add memory_id for WHERE clause
            param_count += 1
            update_params.append(memory_id)
            
            # Execute update
            async with self.db_manager.postgres.acquire() as conn:
                update_query = f"""
                    UPDATE forge1_memory.memory_contexts 
                    SET {', '.join(update_fields)}
                    WHERE id = ${param_count}
                """
                
                result = await conn.execute(update_query, *update_params)
                
                if result == "UPDATE 1":
                    # Update vector database if embeddings changed
                    if 'embeddings' in updates:
                        updated_memory = await self.retrieve_memory(memory_id, user_id)
                        if updated_memory:
                            await self._store_in_vector_db(updated_memory)
                    
                    # Invalidate cache
                    await self._invalidate_cache(memory_id)
                    
                    logger.info(f"Updated memory {memory_id}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False
    
    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Delete a memory"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Check permissions
            memory = await self.retrieve_memory(memory_id, user_id)
            if not memory or memory.owner_id != user_id:
                return False
            
            async with self.db_manager.postgres.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM forge1_memory.memory_contexts 
                    WHERE id = $1 AND owner_id = $2
                """, memory_id, user_id)
                
                if result == "DELETE 1":
                    # Remove from vector database
                    await self._remove_from_vector_db(memory_id)
                    
                    # Remove from cache
                    await self._invalidate_cache(memory_id)
                    
                    logger.info(f"Deleted memory {memory_id}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    async def get_memory_stats(self, employee_id: Optional[str] = None, user_id: Optional[str] = None) -> MemoryStats:
        """Get memory statistics"""
        if not self._initialized:
            await self.initialize()
        
        try:
            conditions = []
            params = []
            
            if employee_id:
                conditions.append("employee_id = $1")
                params.append(employee_id)
            
            if user_id:
                param_num = len(params) + 1
                conditions.append(f"owner_id = ${param_num}")
                params.append(user_id)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            async with self.db_manager.postgres.acquire() as conn:
                # Basic stats
                stats_row = await conn.fetchrow(f"""
                    SELECT 
                        COUNT(*) as total_memories,
                        AVG(overall_relevance_score) as avg_relevance,
                        MIN(created_at) as oldest_date,
                        MAX(created_at) as newest_date
                    FROM forge1_memory.memory_contexts 
                    WHERE {where_clause}
                """, *params)
                
                # Memory type distribution
                type_rows = await conn.fetch(f"""
                    SELECT memory_type, COUNT(*) as count
                    FROM forge1_memory.memory_contexts 
                    WHERE {where_clause}
                    GROUP BY memory_type
                """, *params)
                
                # Security level distribution
                security_rows = await conn.fetch(f"""
                    SELECT security_level, COUNT(*) as count
                    FROM forge1_memory.memory_contexts 
                    WHERE {where_clause}
                    GROUP BY security_level
                """, *params)
                
                # Most accessed memory
                most_accessed_row = await conn.fetchrow(f"""
                    SELECT id FROM forge1_memory.memory_contexts 
                    WHERE {where_clause}
                    ORDER BY access_count DESC 
                    LIMIT 1
                """, *params)
            
            # Build stats object
            memories_by_type = {
                MemoryType(row['memory_type']): row['count'] 
                for row in type_rows
            }
            
            memories_by_security = {
                SecurityLevel(row['security_level']): row['count'] 
                for row in security_rows
            }
            
            return MemoryStats(
                total_memories=stats_row['total_memories'] or 0,
                memories_by_type=memories_by_type,
                memories_by_security_level=memories_by_security,
                average_relevance_score=float(stats_row['avg_relevance'] or 0.0),
                most_accessed_memory_id=most_accessed_row['id'] if most_accessed_row else None,
                oldest_memory_date=stats_row['oldest_date'],
                newest_memory_date=stats_row['newest_date'],
                total_storage_mb=0.0  # Would calculate actual storage size
            )
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return MemoryStats()
    
    # Helper methods
    
    def _extract_text_from_content(self, content: Dict[str, Any]) -> str:
        """Extract searchable text from memory content"""
        if isinstance(content, str):
            return content
        
        text_parts = []
        
        def extract_text_recursive(obj):
            if isinstance(obj, str):
                text_parts.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_text_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_text_recursive(item)
        
        extract_text_recursive(content)
        return " ".join(text_parts)
    
    def _row_to_memory_context(self, row) -> MemoryContext:
        """Convert database row to MemoryContext object"""
        return MemoryContext(
            id=str(row['id']),
            employee_id=str(row['employee_id']),
            session_id=str(row['session_id']) if row['session_id'] else None,
            memory_type=MemoryType(row['memory_type']),
            content=row['content'],
            summary=row['summary'],
            keywords=row['keywords'] or [],
            embeddings=row['embeddings'],
            embedding_model=row['embedding_model'],
            access_count=row['access_count'],
            last_accessed=row['last_accessed'],
            security_level=SecurityLevel(row['security_level']),
            owner_id=row['owner_id'],
            shared_with=row['shared_with'] or [],
            source=row['source'],
            tags=row['tags'] or [],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            expires_at=row['expires_at'],
            parent_memory_id=str(row['parent_memory_id']) if row['parent_memory_id'] else None,
            related_memory_ids=[str(id) for id in (row['related_memory_ids'] or [])]
        )
    
    async def _calculate_relevance_score(self, memory: MemoryContext, query_text: Optional[str], similarity_score: float) -> RelevanceScore:
        """Calculate comprehensive relevance score"""
        # Temporal relevance (newer is more relevant)
        age_days = (datetime.now(timezone.utc) - memory.created_at).days
        temporal_relevance = max(0.0, 1.0 - (age_days / 365.0))  # Decay over a year
        
        # Context relevance (based on access patterns)
        context_relevance = min(1.0, memory.access_count / 10.0)  # Normalize to 0-1
        
        # Usage frequency (recent access is more relevant)
        hours_since_access = (datetime.now(timezone.utc) - memory.last_accessed).total_seconds() / 3600
        usage_frequency = max(0.0, 1.0 - (hours_since_access / (24 * 7)))  # Decay over a week
        
        return RelevanceScore(
            semantic_similarity=similarity_score,
            temporal_relevance=temporal_relevance,
            context_relevance=context_relevance,
            usage_frequency=usage_frequency
        )
    
    def _generate_search_explanation(self, memory: MemoryContext, query: MemoryQuery, similarity_score: float) -> str:
        """Generate explanation for why this memory was returned"""
        explanations = []
        
        if similarity_score > 0.8:
            explanations.append("High semantic similarity")
        elif similarity_score > 0.6:
            explanations.append("Moderate semantic similarity")
        
        if query.keywords and any(kw in memory.keywords for kw in query.keywords):
            explanations.append("Matching keywords")
        
        if query.tags and any(tag in memory.tags for tag in query.tags):
            explanations.append("Matching tags")
        
        if memory.access_count > 5:
            explanations.append("Frequently accessed")
        
        return "; ".join(explanations) if explanations else "General match"
    
    async def _store_in_vector_db(self, memory: MemoryContext):
        """Store memory in vector database for fast similarity search"""
        try:
            if not memory.embeddings:
                return
            
            vector_data = {
                "id": memory.id,
                "values": memory.embeddings,
                "metadata": {
                    "employee_id": memory.employee_id,
                    "memory_type": memory.memory_type.value,
                    "owner_id": memory.owner_id,
                    "security_level": memory.security_level.value,
                    "created_at": memory.created_at.isoformat(),
                    "tenant": get_current_tenant()
                }
            }
            
            if hasattr(self.db_manager.vector_db, 'upsert'):
                # Pinecone
                self.db_manager.vector_db.upsert(vectors=[vector_data], namespace=get_current_tenant())
            
            logger.debug(f"Stored memory {memory.id} in vector database")
            
        except Exception as e:
            logger.error(f"Failed to store memory in vector DB: {e}")
    
    async def _remove_from_vector_db(self, memory_id: str):
        """Remove memory from vector database"""
        try:
            if hasattr(self.db_manager.vector_db, 'delete'):
                # Pinecone
                self.db_manager.vector_db.delete(ids=[memory_id])
            
            logger.debug(f"Removed memory {memory_id} from vector database")
            
        except Exception as e:
            logger.error(f"Failed to remove memory from vector DB: {e}")
    
    async def _cache_memory(self, memory: MemoryContext):
        """Cache memory in Redis for fast access"""
        try:
            cache_key = tenant_prefix(f"memory:{memory.id}")
            cache_data = memory.json()
            
            await self.db_manager.redis.setex(
                cache_key, 
                3600,  # 1 hour TTL
                cache_data
            )
            
            logger.debug(f"Cached memory {memory.id}")
            
        except Exception as e:
            logger.error(f"Failed to cache memory: {e}")
    
    async def _get_cached_memory(self, memory_id: str) -> Optional[MemoryContext]:
        """Get memory from cache"""
        try:
            cache_key = tenant_prefix(f"memory:{memory_id}")
            cache_data = await self.db_manager.redis.get(cache_key)
            
            if cache_data:
                return MemoryContext.parse_raw(cache_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached memory: {e}")
            return None
    
    async def _invalidate_cache(self, memory_id: str):
        """Invalidate cached memory"""
        try:
            cache_key = tenant_prefix(f"memory:{memory_id}")
            await self.db_manager.redis.delete(cache_key)
            
            logger.debug(f"Invalidated cache for memory {memory_id}")
            
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")
    
    async def _update_access_tracking(self, memory_id: str):
        """Update access tracking in database"""
        try:
            async with self.db_manager.postgres.acquire() as conn:
                await conn.execute("""
                    UPDATE forge1_memory.memory_contexts 
                    SET access_count = access_count + 1, 
                        last_accessed = NOW(),
                        updated_at = NOW()
                    WHERE id = $1
                """, memory_id)
                
        except Exception as e:
            logger.error(f"Failed to update access tracking: {e}")

# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None

async def get_memory_manager() -> MemoryManager:
    """Get or create the global memory manager instance"""
    global _memory_manager
    
    if _memory_manager is None:
        _memory_manager = MemoryManager()
        await _memory_manager.initialize()
    
    return _memory_manager

# Export main classes
__all__ = [
    "MemoryManager",
    "EmbeddingService", 
    "get_memory_manager"
]
