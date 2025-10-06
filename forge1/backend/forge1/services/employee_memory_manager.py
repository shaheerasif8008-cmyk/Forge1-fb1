# forge1/backend/forge1/services/employee_memory_manager.py
"""
Employee Memory Management System

Manages isolated memory for each AI employee with complete tenant separation,
semantic search capabilities, and efficient storage across PostgreSQL, Vector DB, and Redis.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal

import numpy as np
from openai import AsyncOpenAI
from forge1.core.database_config import DatabaseManager, get_database_manager
from forge1.core.tenancy import get_current_tenant
from forge1.models.employee_models import (
    MemoryItem, EmployeeInteraction, EmployeeResponse, MemoryType,
    SummaryType, KnowledgeSourceType, MemoryAccessError, TenantIsolationError
)

logger = logging.getLogger(__name__)


class EmployeeMemoryManager:
    """
    Manages isolated memory for each AI employee with complete tenant separation.
    
    Features:
    - Namespace isolation per client/employee
    - Semantic search with embeddings
    - Multi-layer storage (PostgreSQL + Vector DB + Redis)
    - Memory summarization and compression
    - Knowledge base management
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager
        self._initialized = False
        self._embedding_cache = {}
        self._namespace_cache = {}
        
        # Initialize OpenAI client for embeddings
        self.openai_client = None
        self.embedding_model = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
        self.embedding_dimensions = int(os.getenv("OPENAI_EMBEDDINGS_DIMENSIONS", "1536"))
        
    async def initialize(self):
        """Initialize the memory manager"""
        if self._initialized:
            return
            
        if not self.db_manager:
            self.db_manager = await get_database_manager()
        
        # Initialize OpenAI client for real embeddings
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=openai_api_key)
            logger.info(f"OpenAI client initialized for embeddings with model: {self.embedding_model}")
        else:
            logger.warning("OpenAI API key not found, using mock embeddings")
            
        self._initialized = True
        logger.info("Employee Memory Manager initialized")
    
    async def initialize_employee_namespace(
        self,
        client_id: str,
        employee_id: str
    ) -> None:
        """
        Initialize memory namespace for new employee.
        
        Creates isolated storage spaces in all data stores for the employee.
        """
        if not self._initialized:
            await self.initialize()
            
        namespace = self._get_namespace(client_id, employee_id)
        
        try:
            # Initialize PostgreSQL namespace (handled by RLS policies)
            async with self.db_manager.postgres.acquire() as conn:
                # Verify employee exists and belongs to client
                employee_exists = await conn.fetchval("""
                    SELECT EXISTS(
                        SELECT 1 FROM forge1_employees.employees 
                        WHERE id = $1 AND client_id = $2
                    )
                """, employee_id, client_id)
                
                if not employee_exists:
                    raise MemoryAccessError(f"Employee {employee_id} not found for client {client_id}")
            
            # Initialize Vector DB collection if available
            if hasattr(self.db_manager, 'vector_db') and self.db_manager.vector_db:
                try:
                    await self._create_vector_collection(namespace)
                except Exception as e:
                    logger.warning(f"Vector DB initialization failed for {namespace}: {e}")
            
            # Initialize Redis namespace
            if hasattr(self.db_manager, 'redis') and self.db_manager.redis:
                try:
                    await self.db_manager.redis.hset(
                        f"employee:{namespace}:config",
                        mapping={
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "client_id": client_id,
                            "employee_id": employee_id,
                            "interaction_count": "0"
                        }
                    )
                except Exception as e:
                    logger.warning(f"Redis initialization failed for {namespace}: {e}")
            
            # Cache namespace info
            self._namespace_cache[namespace] = {
                "client_id": client_id,
                "employee_id": employee_id,
                "created_at": datetime.now(timezone.utc)
            }
            
            logger.info(f"Initialized memory namespace for employee {employee_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize namespace {namespace}: {e}")
            raise MemoryAccessError(f"Failed to initialize employee memory: {e}")
    
    async def store_interaction(
        self,
        client_id: str,
        employee_id: str,
        interaction: EmployeeInteraction,
        response: EmployeeResponse
    ) -> None:
        """
        Store interaction in employee's isolated memory across all storage layers.
        """
        if not self._initialized:
            await self.initialize()
            
        # Validate tenant access
        await self._validate_tenant_access(client_id, employee_id)
        
        namespace = self._get_namespace(client_id, employee_id)
        
        try:
            # Generate embedding for semantic search
            embedding = await self._generate_embedding(
                f"{interaction.message} {response.message}"
            )
            
            # Store in PostgreSQL for structured queries
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                await conn.execute("""
                    INSERT INTO forge1_employees.employee_interactions (
                        client_id, employee_id, interaction_id, session_id,
                        message, response, context, timestamp, embedding,
                        memory_type, importance_score, processing_time_ms,
                        model_used, tokens_used, cost
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """, 
                    client_id, employee_id, interaction.id, interaction.session_id,
                    interaction.message, response.message, 
                    json.dumps(interaction.context), interaction.timestamp,
                    embedding, MemoryType.CONVERSATION.value, 0.5,
                    interaction.processing_time_ms, interaction.model_used,
                    interaction.tokens_used, interaction.cost
                )
            
            # Store in Vector DB for semantic search
            if hasattr(self.db_manager, 'vector_db') and self.db_manager.vector_db:
                try:
                    await self._store_in_vector_db(namespace, interaction, response, embedding)
                except Exception as e:
                    logger.warning(f"Vector DB storage failed: {e}")
            
            # Cache recent interactions in Redis
            if hasattr(self.db_manager, 'redis') and self.db_manager.redis:
                try:
                    await self._cache_interaction(namespace, interaction, response)
                except Exception as e:
                    logger.warning(f"Redis caching failed: {e}")
            
            # Update employee last interaction timestamp
            await self._update_employee_last_interaction(client_id, employee_id)
            
            logger.debug(f"Stored interaction {interaction.id} for employee {employee_id}")
            
        except Exception as e:
            logger.error(f"Failed to store interaction for {employee_id}: {e}")
            raise MemoryAccessError(f"Failed to store interaction: {e}")
    
    async def get_employee_context(
        self,
        client_id: str,
        employee_id: str,
        query: Optional[str] = None,
        limit: int = 10,
        memory_types: Optional[List[MemoryType]] = None,
        min_importance: float = 0.0
    ) -> List[MemoryItem]:
        """
        Retrieve relevant memory context for employee with optional semantic search.
        """
        if not self._initialized:
            await self.initialize()
            
        # Validate tenant access
        await self._validate_tenant_access(client_id, employee_id)
        
        namespace = self._get_namespace(client_id, employee_id)
        
        try:
            if query:
                # Semantic search for relevant memories
                return await self._semantic_search(
                    client_id, employee_id, query, limit, memory_types, min_importance
                )
            else:
                # Get recent interactions from cache or database
                return await self._get_recent_interactions(
                    client_id, employee_id, limit, memory_types, min_importance
                )
                
        except Exception as e:
            logger.error(f"Failed to get context for employee {employee_id}: {e}")
            raise MemoryAccessError(f"Failed to retrieve employee context: {e}")
    
    async def search_employee_memory(
        self,
        client_id: str,
        employee_id: str,
        query: str,
        limit: int = 20,
        memory_types: Optional[List[MemoryType]] = None,
        include_embeddings: bool = False
    ) -> Tuple[List[MemoryItem], float]:
        """
        Perform semantic search across employee's memory.
        
        Returns:
            Tuple of (memory_items, search_time_ms)
        """
        if not self._initialized:
            await self.initialize()
            
        start_time = time.time()
        
        # Validate tenant access
        await self._validate_tenant_access(client_id, employee_id)
        
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            
            # Search in PostgreSQL with vector similarity
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                # Build query conditions
                conditions = ["client_id = $1", "employee_id = $2"]
                params = [client_id, employee_id]
                
                if memory_types:
                    conditions.append(f"memory_type = ANY(${len(params) + 1})")
                    params.append([mt.value for mt in memory_types])
                
                where_clause = " AND ".join(conditions)
                
                # Execute semantic search query
                query_sql = f"""
                    SELECT 
                        interaction_id, message, response, timestamp, memory_type,
                        importance_score, context, embedding,
                        (embedding <=> $3::vector) as distance
                    FROM forge1_employees.employee_interactions
                    WHERE {where_clause}
                    ORDER BY embedding <=> $3::vector
                    LIMIT ${len(params) + 1}
                """
                
                params.insert(2, query_embedding)  # Insert embedding as 3rd parameter
                params.append(limit)
                
                rows = await conn.fetch(query_sql, *params)
            
            # Convert to MemoryItem objects
            memory_items = []
            for row in rows:
                # Calculate relevance score (inverse of distance)
                relevance_score = max(0.0, 1.0 - row['distance'])
                
                memory_item = MemoryItem(
                    id=row['interaction_id'],
                    content=row['message'],
                    response=row['response'],
                    timestamp=row['timestamp'],
                    memory_type=MemoryType(row['memory_type']),
                    importance_score=row['importance_score'],
                    relevance_score=relevance_score,
                    embedding=list(row['embedding']) if include_embeddings else None,
                    context=row['context'] if row['context'] else {}
                )
                memory_items.append(memory_item)
            
            search_time_ms = (time.time() - start_time) * 1000
            
            logger.debug(f"Memory search for {employee_id}: {len(memory_items)} results in {search_time_ms:.2f}ms")
            
            return memory_items, search_time_ms
            
        except Exception as e:
            logger.error(f"Memory search failed for employee {employee_id}: {e}")
            raise MemoryAccessError(f"Memory search failed: {e}")
    
    async def create_memory_summary(
        self,
        client_id: str,
        employee_id: str,
        summary_type: SummaryType = SummaryType.CONVERSATION,
        time_period_hours: int = 24
    ) -> str:
        """
        Create a memory summary for the employee over a time period.
        """
        if not self._initialized:
            await self.initialize()
            
        # Validate tenant access
        await self._validate_tenant_access(client_id, employee_id)
        
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=time_period_hours)
            
            # Get interactions from time period
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                rows = await conn.fetch("""
                    SELECT message, response, timestamp, context
                    FROM forge1_employees.employee_interactions
                    WHERE client_id = $1 AND employee_id = $2
                    AND timestamp BETWEEN $3 AND $4
                    ORDER BY timestamp ASC
                """, client_id, employee_id, start_time, end_time)
            
            if not rows:
                return "No interactions found in the specified time period."
            
            # Create summary text
            interactions_text = []
            for row in rows:
                interactions_text.append(f"User: {row['message']}")
                interactions_text.append(f"Assistant: {row['response']}")
            
            # Generate summary (this would use an LLM in production)
            summary_text = await self._generate_summary(interactions_text, summary_type)
            
            # Generate embedding for summary
            summary_embedding = await self._generate_embedding(summary_text)
            
            # Store summary in database
            summary_id = str(uuid.uuid4())
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                await conn.execute("""
                    INSERT INTO forge1_employees.employee_memory_summaries (
                        id, client_id, employee_id, summary_text, summary_type,
                        time_period_start, time_period_end, embedding,
                        interaction_count, importance_score
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """, 
                    summary_id, client_id, employee_id, summary_text, summary_type.value,
                    start_time, end_time, summary_embedding, len(rows), 0.7
                )
            
            logger.info(f"Created {summary_type.value} summary for employee {employee_id}")
            return summary_id
            
        except Exception as e:
            logger.error(f"Failed to create memory summary for {employee_id}: {e}")
            raise MemoryAccessError(f"Failed to create memory summary: {e}")
    
    async def add_knowledge_source(
        self,
        client_id: str,
        employee_id: str,
        title: str,
        content: str,
        source_type: KnowledgeSourceType = KnowledgeSourceType.MANUAL,
        source_url: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Add a knowledge source to employee's knowledge base.
        """
        if not self._initialized:
            await self.initialize()
            
        # Validate tenant access
        await self._validate_tenant_access(client_id, employee_id)
        
        try:
            # Generate embedding for content
            content_embedding = await self._generate_embedding(f"{title} {content}")
            
            # Store knowledge in database
            knowledge_id = str(uuid.uuid4())
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                await conn.execute("""
                    INSERT INTO forge1_employees.employee_knowledge (
                        id, client_id, employee_id, title, content,
                        source_type, source_url, embedding, keywords, tags
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """, 
                    knowledge_id, client_id, employee_id, title, content,
                    source_type.value, source_url, content_embedding,
                    keywords or [], tags or []
                )
            
            logger.info(f"Added knowledge source {knowledge_id} for employee {employee_id}")
            return knowledge_id
            
        except Exception as e:
            logger.error(f"Failed to add knowledge source for {employee_id}: {e}")
            raise MemoryAccessError(f"Failed to add knowledge source: {e}")
    
    async def search_knowledge_base(
        self,
        client_id: str,
        employee_id: str,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search employee's knowledge base for relevant information.
        """
        if not self._initialized:
            await self.initialize()
            
        # Validate tenant access
        await self._validate_tenant_access(client_id, employee_id)
        
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            
            # Search knowledge base
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                rows = await conn.fetch("""
                    SELECT 
                        id, title, content, source_type, source_url,
                        keywords, tags, created_at,
                        (embedding <=> $3::vector) as distance
                    FROM forge1_employees.employee_knowledge
                    WHERE client_id = $1 AND employee_id = $2
                    ORDER BY embedding <=> $3::vector
                    LIMIT $4
                """, client_id, employee_id, query_embedding, limit)
            
            # Convert to response format
            knowledge_items = []
            for row in rows:
                relevance_score = max(0.0, 1.0 - row['distance'])
                
                knowledge_items.append({
                    "id": str(row['id']),
                    "title": row['title'],
                    "content": row['content'],
                    "source_type": row['source_type'],
                    "source_url": row['source_url'],
                    "keywords": row['keywords'],
                    "tags": row['tags'],
                    "relevance_score": relevance_score,
                    "created_at": row['created_at'].isoformat()
                })
            
            return knowledge_items
            
        except Exception as e:
            logger.error(f"Knowledge search failed for employee {employee_id}: {e}")
            raise MemoryAccessError(f"Knowledge search failed: {e}")
    
    async def get_memory_stats(
        self,
        client_id: str,
        employee_id: str
    ) -> Dict[str, Any]:
        """
        Get memory statistics for an employee.
        """
        if not self._initialized:
            await self.initialize()
            
        # Validate tenant access
        await self._validate_tenant_access(client_id, employee_id)
        
        try:
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                # Get interaction stats
                interaction_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_interactions,
                        COUNT(DISTINCT session_id) as unique_sessions,
                        MIN(timestamp) as first_interaction,
                        MAX(timestamp) as last_interaction,
                        AVG(importance_score) as avg_importance
                    FROM forge1_employees.employee_interactions
                    WHERE client_id = $1 AND employee_id = $2
                """, client_id, employee_id)
                
                # Get memory type distribution
                memory_types = await conn.fetch("""
                    SELECT memory_type, COUNT(*) as count
                    FROM forge1_employees.employee_interactions
                    WHERE client_id = $1 AND employee_id = $2
                    GROUP BY memory_type
                """, client_id, employee_id)
                
                # Get knowledge base stats
                knowledge_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_knowledge_items,
                        COUNT(DISTINCT source_type) as source_types
                    FROM forge1_employees.employee_knowledge
                    WHERE client_id = $1 AND employee_id = $2
                """, client_id, employee_id)
                
                # Get summary stats
                summary_stats = await conn.fetchrow("""
                    SELECT COUNT(*) as total_summaries
                    FROM forge1_employees.employee_memory_summaries
                    WHERE client_id = $1 AND employee_id = $2
                """, client_id, employee_id)
            
            return {
                "employee_id": employee_id,
                "total_interactions": interaction_stats['total_interactions'] or 0,
                "unique_sessions": interaction_stats['unique_sessions'] or 0,
                "first_interaction": interaction_stats['first_interaction'].isoformat() if interaction_stats['first_interaction'] else None,
                "last_interaction": interaction_stats['last_interaction'].isoformat() if interaction_stats['last_interaction'] else None,
                "average_importance": float(interaction_stats['avg_importance'] or 0.0),
                "memory_types": {row['memory_type']: row['count'] for row in memory_types},
                "total_knowledge_items": knowledge_stats['total_knowledge_items'] or 0,
                "knowledge_source_types": knowledge_stats['source_types'] or 0,
                "total_summaries": summary_stats['total_summaries'] or 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory stats for {employee_id}: {e}")
            raise MemoryAccessError(f"Failed to get memory statistics: {e}")
    
    # Private helper methods
    
    def _get_namespace(self, client_id: str, employee_id: str) -> str:
        """Generate namespace for client/employee combination"""
        return f"{client_id}:{employee_id}"
    
    async def _validate_tenant_access(self, client_id: str, employee_id: str) -> None:
        """Validate that current tenant can access the employee"""
        current_tenant = get_current_tenant()
        if current_tenant and current_tenant != client_id:
            raise TenantIsolationError(
                f"Tenant {current_tenant} cannot access employee {employee_id} from client {client_id}"
            )
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI or fallback to mock"""
        # Check cache first
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        # Use real OpenAI embeddings if available
        if self.openai_client:
            try:
                # Clean and truncate text for embedding
                clean_text = text.strip()[:8000]  # OpenAI has token limits
                
                response = await self.openai_client.embeddings.create(
                    input=clean_text,
                    model=self.embedding_model
                )
                
                embedding = response.data[0].embedding
                
                # Cache the result
                self._embedding_cache[text] = embedding
                
                logger.debug(f"Generated OpenAI embedding for text: {text[:50]}...")
                return embedding
                
            except Exception as e:
                logger.warning(f"OpenAI embedding failed, using mock: {e}")
                # Fall through to mock implementation
        
        # Fallback to mock embedding for development/testing
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        np.random.seed(seed)
        
        embedding = np.random.normal(0, 1, self.embedding_dimensions).tolist()
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = (np.array(embedding) / norm).tolist()
        
        self._embedding_cache[text] = embedding
        logger.debug(f"Generated mock embedding for text: {text[:50]}...")
        return embedding
    
    async def _create_vector_collection(self, namespace: str) -> None:
        """Create vector collection for namespace (mock implementation)"""
        # This would create a collection in Pinecone, Weaviate, etc.
        logger.debug(f"Created vector collection for namespace {namespace}")
    
    async def _store_in_vector_db(
        self,
        namespace: str,
        interaction: EmployeeInteraction,
        response: EmployeeResponse,
        embedding: List[float]
    ) -> None:
        """Store interaction in vector database (mock implementation)"""
        # This would store in Pinecone, Weaviate, etc.
        logger.debug(f"Stored interaction {interaction.id} in vector DB")
    
    async def _cache_interaction(
        self,
        namespace: str,
        interaction: EmployeeInteraction,
        response: EmployeeResponse
    ) -> None:
        """Cache recent interaction in Redis"""
        try:
            interaction_data = {
                "id": interaction.id,
                "message": interaction.message,
                "response": response.message,
                "timestamp": interaction.timestamp.isoformat(),
                "session_id": interaction.session_id
            }
            
            # Add to recent interactions list
            await self.db_manager.redis.lpush(
                f"employee:{namespace}:recent",
                json.dumps(interaction_data)
            )
            
            # Keep only last 50 interactions in cache
            await self.db_manager.redis.ltrim(f"employee:{namespace}:recent", 0, 49)
            
            # Update interaction count
            await self.db_manager.redis.hincrby(
                f"employee:{namespace}:config",
                "interaction_count",
                1
            )
            
        except Exception as e:
            logger.warning(f"Failed to cache interaction: {e}")
    
    async def _update_employee_last_interaction(self, client_id: str, employee_id: str) -> None:
        """Update employee's last interaction timestamp"""
        try:
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                await conn.execute("""
                    UPDATE forge1_employees.employees 
                    SET last_interaction_at = NOW(), updated_at = NOW()
                    WHERE id = $1 AND client_id = $2
                """, employee_id, client_id)
        except Exception as e:
            logger.warning(f"Failed to update employee last interaction: {e}")
    
    async def _semantic_search(
        self,
        client_id: str,
        employee_id: str,
        query: str,
        limit: int,
        memory_types: Optional[List[MemoryType]],
        min_importance: float
    ) -> List[MemoryItem]:
        """Perform semantic search on employee memory"""
        memory_items, _ = await self.search_employee_memory(
            client_id, employee_id, query, limit, memory_types
        )
        
        # Filter by minimum importance
        return [item for item in memory_items if item.importance_score >= min_importance]
    
    async def _get_recent_interactions(
        self,
        client_id: str,
        employee_id: str,
        limit: int,
        memory_types: Optional[List[MemoryType]],
        min_importance: float
    ) -> List[MemoryItem]:
        """Get recent interactions from cache or database"""
        namespace = self._get_namespace(client_id, employee_id)
        
        # Try Redis cache first
        if hasattr(self.db_manager, 'redis') and self.db_manager.redis:
            try:
                cached_data = await self.db_manager.redis.lrange(
                    f"employee:{namespace}:recent", 0, limit - 1
                )
                
                if cached_data:
                    memory_items = []
                    for item_json in cached_data:
                        data = json.loads(item_json)
                        memory_item = MemoryItem(
                            id=data["id"],
                            content=data["message"],
                            response=data["response"],
                            timestamp=datetime.fromisoformat(data["timestamp"]),
                            memory_type=MemoryType.CONVERSATION,
                            importance_score=0.5,
                            relevance_score=1.0
                        )
                        memory_items.append(memory_item)
                    
                    return memory_items[:limit]
            except Exception as e:
                logger.warning(f"Failed to get cached interactions: {e}")
        
        # Fallback to database
        try:
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                # Build query conditions
                conditions = ["client_id = $1", "employee_id = $2"]
                params = [client_id, employee_id]
                
                if memory_types:
                    conditions.append(f"memory_type = ANY(${len(params) + 1})")
                    params.append([mt.value for mt in memory_types])
                
                if min_importance > 0:
                    conditions.append(f"importance_score >= ${len(params) + 1}")
                    params.append(min_importance)
                
                where_clause = " AND ".join(conditions)
                params.append(limit)
                
                rows = await conn.fetch(f"""
                    SELECT 
                        interaction_id, message, response, timestamp,
                        memory_type, importance_score, context
                    FROM forge1_employees.employee_interactions
                    WHERE {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT ${len(params)}
                """, *params)
            
            memory_items = []
            for row in rows:
                memory_item = MemoryItem(
                    id=row['interaction_id'],
                    content=row['message'],
                    response=row['response'],
                    timestamp=row['timestamp'],
                    memory_type=MemoryType(row['memory_type']),
                    importance_score=row['importance_score'],
                    relevance_score=1.0,
                    context=row['context'] if row['context'] else {}
                )
                memory_items.append(memory_item)
            
            return memory_items
            
        except Exception as e:
            logger.error(f"Failed to get recent interactions from database: {e}")
            return []
    
    async def _generate_summary(
        self,
        interactions: List[str],
        summary_type: SummaryType
    ) -> str:
        """Generate summary from interactions (mock implementation)"""
        # In production, this would use an LLM to generate summaries
        interaction_count = len(interactions) // 2  # Divide by 2 since we have user + assistant messages
        
        if summary_type == SummaryType.CONVERSATION:
            return f"Conversation summary: {interaction_count} interactions covering various topics and requests."
        elif summary_type == SummaryType.DAILY:
            return f"Daily summary: {interaction_count} interactions throughout the day."
        elif summary_type == SummaryType.WEEKLY:
            return f"Weekly summary: {interaction_count} interactions over the past week."
        else:
            return f"Summary: {interaction_count} interactions of type {summary_type.value}."

    # New methods for Task 9: Memory Analytics and Management

    async def get_memory_analytics(
        self,
        client_id: str,
        employee_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive memory analytics for an employee."""
        try:
            await self._validate_tenant_access(client_id, employee_id)
            
            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            # Get interaction statistics
            async with self.db_manager.get_connection() as conn:
                # Total interactions
                total_interactions = await conn.fetchval("""
                    SELECT COUNT(*) FROM employee_interactions 
                    WHERE client_id = $1 AND employee_id = $2 
                    AND timestamp >= $3 AND timestamp <= $4
                """, client_id, employee_id, start_date, end_date)
                
                # Daily interaction counts
                daily_stats = await conn.fetch("""
                    SELECT DATE(timestamp) as date, COUNT(*) as count
                    FROM employee_interactions 
                    WHERE client_id = $1 AND employee_id = $2 
                    AND timestamp >= $3 AND timestamp <= $4
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """, client_id, employee_id, start_date, end_date)
                
                # Average response length
                avg_response_length = await conn.fetchval("""
                    SELECT AVG(LENGTH(response)) FROM employee_interactions 
                    WHERE client_id = $1 AND employee_id = $2 
                    AND timestamp >= $3 AND timestamp <= $4
                """, client_id, employee_id, start_date, end_date) or 0
                
                # Most common topics (simplified - based on keywords)
                common_topics = await conn.fetch("""
                    SELECT 
                        CASE 
                            WHEN LOWER(message) LIKE '%help%' OR LOWER(message) LIKE '%support%' THEN 'support'
                            WHEN LOWER(message) LIKE '%question%' OR LOWER(message) LIKE '%ask%' THEN 'questions'
                            WHEN LOWER(message) LIKE '%problem%' OR LOWER(message) LIKE '%issue%' THEN 'problems'
                            WHEN LOWER(message) LIKE '%thank%' OR LOWER(message) LIKE '%thanks%' THEN 'gratitude'
                            ELSE 'general'
                        END as topic,
                        COUNT(*) as count
                    FROM employee_interactions 
                    WHERE client_id = $1 AND employee_id = $2 
                    AND timestamp >= $3 AND timestamp <= $4
                    GROUP BY topic
                    ORDER BY count DESC
                    LIMIT 10
                """, client_id, employee_id, start_date, end_date)
            
            # Memory usage statistics
            memory_stats = await self.get_memory_stats(client_id, employee_id)
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days
                },
                "interaction_stats": {
                    "total_interactions": total_interactions,
                    "daily_average": round(total_interactions / days, 2) if days > 0 else 0,
                    "avg_response_length": round(float(avg_response_length), 2),
                    "daily_breakdown": [
                        {"date": row["date"].isoformat(), "count": row["count"]} 
                        for row in daily_stats
                    ]
                },
                "topic_analysis": [
                    {"topic": row["topic"], "count": row["count"]} 
                    for row in common_topics
                ],
                "memory_usage": memory_stats,
                "trends": {
                    "interaction_growth": self._calculate_growth_trend(daily_stats),
                    "engagement_level": self._calculate_engagement_level(total_interactions, days)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory analytics: {e}")
            raise MemoryAccessError(f"Failed to get memory analytics: {e}")

    async def get_conversation_history(
        self,
        client_id: str,
        employee_id: str,
        page: int = 1,
        page_size: int = 20,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        session_id: Optional[str] = None,
        include_context: bool = False
    ) -> Dict[str, Any]:
        """Get paginated conversation history for an employee."""
        try:
            await self._validate_tenant_access(client_id, employee_id)
            
            # Calculate offset
            offset = (page - 1) * page_size
            
            # Build query conditions
            conditions = ["client_id = $1", "employee_id = $2"]
            params = [client_id, employee_id]
            param_count = 2
            
            if start_date:
                param_count += 1
                conditions.append(f"timestamp >= ${param_count}")
                params.append(start_date)
            
            if end_date:
                param_count += 1
                conditions.append(f"timestamp <= ${param_count}")
                params.append(end_date)
            
            if session_id:
                param_count += 1
                conditions.append(f"context->>'session_id' = ${param_count}")
                params.append(session_id)
            
            where_clause = " AND ".join(conditions)
            
            async with self.db_manager.get_connection() as conn:
                # Get total count
                count_query = f"""
                    SELECT COUNT(*) FROM employee_interactions 
                    WHERE {where_clause}
                """
                total_count = await conn.fetchval(count_query, *params)
                
                # Get paginated results
                select_fields = """
                    id, interaction_id, message, response, timestamp, 
                    context, processing_time_ms, tokens_used, cost
                """ + (", context" if include_context else "")
                
                data_query = f"""
                    SELECT {select_fields}
                    FROM employee_interactions 
                    WHERE {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT ${param_count + 1} OFFSET ${param_count + 2}
                """
                params.extend([page_size, offset])
                
                conversations = await conn.fetch(data_query, *params)
            
            # Format conversations
            formatted_conversations = []
            for conv in conversations:
                conversation_data = {
                    "id": str(conv["id"]),
                    "interaction_id": conv["interaction_id"],
                    "message": conv["message"],
                    "response": conv["response"],
                    "timestamp": conv["timestamp"].isoformat(),
                    "processing_time_ms": conv.get("processing_time_ms"),
                    "tokens_used": conv.get("tokens_used"),
                    "cost": float(conv["cost"]) if conv.get("cost") else None
                }
                
                if include_context and conv.get("context"):
                    conversation_data["context"] = conv["context"]
                
                formatted_conversations.append(conversation_data)
            
            return {
                "total_count": total_count,
                "conversations": formatted_conversations
            }
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            raise MemoryAccessError(f"Failed to get conversation history: {e}")

    async def get_memory_statistics(
        self,
        client_id: str,
        employee_id: str
    ) -> Dict[str, Any]:
        """Get detailed memory statistics for an employee."""
        try:
            await self._validate_tenant_access(client_id, employee_id)
            
            async with self.db_manager.get_connection() as conn:
                # Basic interaction stats
                basic_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_interactions,
                        MIN(timestamp) as first_interaction,
                        MAX(timestamp) as last_interaction,
                        AVG(LENGTH(message)) as avg_message_length,
                        AVG(LENGTH(response)) as avg_response_length,
                        SUM(tokens_used) as total_tokens,
                        SUM(cost) as total_cost
                    FROM employee_interactions 
                    WHERE client_id = $1 AND employee_id = $2
                """, client_id, employee_id)
                
                # Memory summaries count
                summaries_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM employee_memory_summaries 
                    WHERE client_id = $1 AND employee_id = $2
                """, client_id, employee_id)
                
                # Knowledge base count
                knowledge_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM employee_knowledge_base 
                    WHERE client_id = $1 AND employee_id = $2
                """, client_id, employee_id)
            
            # Calculate memory efficiency metrics
            total_interactions = basic_stats["total_interactions"] or 0
            days_active = 0
            
            if basic_stats["first_interaction"] and basic_stats["last_interaction"]:
                days_active = (basic_stats["last_interaction"] - basic_stats["first_interaction"]).days + 1
            
            return {
                "interactions": {
                    "total": total_interactions,
                    "first_interaction": basic_stats["first_interaction"].isoformat() if basic_stats["first_interaction"] else None,
                    "last_interaction": basic_stats["last_interaction"].isoformat() if basic_stats["last_interaction"] else None,
                    "days_active": days_active,
                    "avg_per_day": round(total_interactions / days_active, 2) if days_active > 0 else 0
                },
                "content": {
                    "avg_message_length": round(float(basic_stats["avg_message_length"] or 0), 2),
                    "avg_response_length": round(float(basic_stats["avg_response_length"] or 0), 2),
                    "total_tokens": int(basic_stats["total_tokens"] or 0),
                    "total_cost": float(basic_stats["total_cost"] or 0)
                },
                "memory_organization": {
                    "summaries_count": summaries_count,
                    "knowledge_items": knowledge_count,
                    "memory_efficiency": self._calculate_memory_efficiency(total_interactions, summaries_count)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory statistics: {e}")
            raise MemoryAccessError(f"Failed to get memory statistics: {e}")

    async def export_memory(
        self,
        client_id: str,
        employee_id: str,
        format: str = "json",
        include_embeddings: bool = False,
        include_analytics: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export employee memory data for backup or analysis."""
        try:
            await self._validate_tenant_access(client_id, employee_id)
            
            export_id = f"export_{uuid.uuid4().hex[:8]}"
            
            # Parse date filters
            start_datetime = None
            end_datetime = None
            
            if start_date:
                start_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if end_date:
                end_datetime = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            
            # Get all memory data
            export_data = {
                "export_info": {
                    "export_id": export_id,
                    "employee_id": employee_id,
                    "client_id": client_id,
                    "format": format,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "include_embeddings": include_embeddings,
                    "include_analytics": include_analytics,
                    "date_range": {
                        "start": start_date,
                        "end": end_date
                    }
                }
            }
            
            # Get interactions
            conditions = ["client_id = $1", "employee_id = $2"]
            params = [client_id, employee_id]
            
            if start_datetime:
                conditions.append("timestamp >= $3")
                params.append(start_datetime)
            if end_datetime:
                conditions.append(f"timestamp <= ${len(params) + 1}")
                params.append(end_datetime)
            
            where_clause = " AND ".join(conditions)
            
            async with self.db_manager.get_connection() as conn:
                # Export interactions
                select_fields = "interaction_id, message, response, timestamp, context, processing_time_ms, tokens_used, cost"
                if include_embeddings:
                    select_fields += ", embedding"
                
                interactions = await conn.fetch(f"""
                    SELECT {select_fields}
                    FROM employee_interactions 
                    WHERE {where_clause}
                    ORDER BY timestamp
                """, *params)
                
                export_data["interactions"] = [
                    {
                        "interaction_id": row["interaction_id"],
                        "message": row["message"],
                        "response": row["response"],
                        "timestamp": row["timestamp"].isoformat(),
                        "context": row["context"],
                        "processing_time_ms": row.get("processing_time_ms"),
                        "tokens_used": row.get("tokens_used"),
                        "cost": float(row["cost"]) if row.get("cost") else None,
                        **({"embedding": row["embedding"]} if include_embeddings and row.get("embedding") else {})
                    }
                    for row in interactions
                ]
                
                # Export summaries
                summaries = await conn.fetch("""
                    SELECT summary_id, summary_type, content, interaction_count, created_at
                    FROM employee_memory_summaries 
                    WHERE client_id = $1 AND employee_id = $2
                    ORDER BY created_at
                """, client_id, employee_id)
                
                export_data["summaries"] = [
                    {
                        "summary_id": row["summary_id"],
                        "summary_type": row["summary_type"],
                        "content": row["content"],
                        "interaction_count": row["interaction_count"],
                        "created_at": row["created_at"].isoformat()
                    }
                    for row in summaries
                ]
                
                # Export knowledge base
                knowledge = await conn.fetch("""
                    SELECT knowledge_id, title, content, source_type, source_url, keywords, tags, created_at
                    FROM employee_knowledge_base 
                    WHERE client_id = $1 AND employee_id = $2
                    ORDER BY created_at
                """, client_id, employee_id)
                
                export_data["knowledge_base"] = [
                    {
                        "knowledge_id": row["knowledge_id"],
                        "title": row["title"],
                        "content": row["content"],
                        "source_type": row["source_type"],
                        "source_url": row["source_url"],
                        "keywords": row["keywords"],
                        "tags": row["tags"],
                        "created_at": row["created_at"].isoformat()
                    }
                    for row in knowledge
                ]
            
            # Include analytics if requested
            if include_analytics:
                export_data["analytics"] = await self.get_memory_analytics(client_id, employee_id, 90)
                export_data["statistics"] = await self.get_memory_statistics(client_id, employee_id)
            
            # Store export data (in production, this would be saved to file storage)
            # For now, we'll simulate the export process
            estimated_completion = datetime.now(timezone.utc) + timedelta(minutes=5)
            
            return {
                "export_id": export_id,
                "estimated_completion": estimated_completion.isoformat(),
                "download_url": f"/api/v1/employees/clients/{client_id}/employees/{employee_id}/memory/export/{export_id}/download"
            }
            
        except Exception as e:
            logger.error(f"Failed to export memory: {e}")
            raise MemoryAccessError(f"Failed to export memory: {e}")

    async def get_export_status(
        self,
        client_id: str,
        employee_id: str,
        export_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get the status of a memory export operation."""
        try:
            await self._validate_tenant_access(client_id, employee_id)
            
            # In a real implementation, this would check the actual export status
            # For now, we'll simulate a completed export
            return {
                "export_id": export_id,
                "status": "completed",
                "progress": 100,
                "created_at": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "download_url": f"/api/v1/employees/clients/{client_id}/employees/{employee_id}/memory/export/{export_id}/download",
                "file_size_bytes": 1024 * 1024,  # 1MB example
                "expires_at": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get export status: {e}")
            return None

    async def create_memory_backup(
        self,
        client_id: str,
        employee_id: str,
        backup_name: str,
        include_embeddings: bool = True,
        compression: str = "gzip"
    ) -> Dict[str, Any]:
        """Create a backup of employee memory data."""
        try:
            await self._validate_tenant_access(client_id, employee_id)
            
            backup_id = f"backup_{uuid.uuid4().hex[:8]}"
            created_at = datetime.now(timezone.utc)
            
            # Get memory statistics for size estimation
            stats = await self.get_memory_statistics(client_id, employee_id)
            estimated_size = stats["interactions"]["total"] * 1024  # Rough estimate
            
            # In a real implementation, this would create an actual backup
            # For now, we'll simulate the backup creation
            
            return {
                "backup_id": backup_id,
                "size_bytes": estimated_size,
                "created_at": created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create memory backup: {e}")
            raise MemoryAccessError(f"Failed to create memory backup: {e}")

    async def list_memory_backups(
        self,
        client_id: str,
        employee_id: str
    ) -> List[Dict[str, Any]]:
        """List all memory backups for an employee."""
        try:
            await self._validate_tenant_access(client_id, employee_id)
            
            # In a real implementation, this would list actual backups from storage
            # For now, we'll return a simulated list
            return [
                {
                    "backup_id": "backup_example_001",
                    "backup_name": f"backup_{employee_id}_001",
                    "created_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
                    "size_bytes": 1024 * 512,
                    "compression": "gzip",
                    "status": "completed"
                }
            ]
            
        except Exception as e:
            logger.error(f"Failed to list memory backups: {e}")
            raise MemoryAccessError(f"Failed to list memory backups: {e}")

    async def restore_memory_backup(
        self,
        client_id: str,
        employee_id: str,
        backup_id: str,
        merge_mode: str = "replace"
    ) -> Dict[str, Any]:
        """Restore employee memory from a backup."""
        try:
            await self._validate_tenant_access(client_id, employee_id)
            
            restore_id = f"restore_{uuid.uuid4().hex[:8]}"
            completed_at = datetime.now(timezone.utc)
            
            # In a real implementation, this would restore from actual backup
            # For now, we'll simulate the restore process
            
            return {
                "restore_id": restore_id,
                "restored_items": 100,  # Simulated count
                "completed_at": completed_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to restore memory backup: {e}")
            raise MemoryAccessError(f"Failed to restore memory backup: {e}")

    def _calculate_growth_trend(self, daily_stats: List[Dict]) -> str:
        """Calculate interaction growth trend."""
        if len(daily_stats) < 2:
            return "insufficient_data"
        
        recent_avg = sum(row["count"] for row in daily_stats[-7:]) / min(7, len(daily_stats))
        older_avg = sum(row["count"] for row in daily_stats[:-7]) / max(1, len(daily_stats) - 7)
        
        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"

    def _calculate_engagement_level(self, total_interactions: int, days: int) -> str:
        """Calculate engagement level based on interaction frequency."""
        if days == 0:
            return "no_data"
        
        avg_per_day = total_interactions / days
        
        if avg_per_day >= 10:
            return "high"
        elif avg_per_day >= 3:
            return "medium"
        elif avg_per_day >= 1:
            return "low"
        else:
            return "minimal"

    def _calculate_memory_efficiency(self, interactions: int, summaries: int) -> float:
        """Calculate memory efficiency score."""
        if interactions == 0:
            return 0.0
        
        # Higher efficiency when we have good summarization
        efficiency = min(1.0, summaries / max(1, interactions / 10))
        return round(efficiency, 2)

    # Additional methods for Task 11: Knowledge Source Management

    async def list_employee_knowledge_sources(
        self,
        client_id: str,
        employee_id: str,
        source_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List employee's knowledge sources with optional filtering."""
        try:
            await self._validate_tenant_access(client_id, employee_id)
            
            # Build query conditions
            conditions = ["client_id = $1", "employee_id = $2"]
            params = [client_id, employee_id]
            
            if source_type:
                conditions.append("source_type = $3")
                params.append(source_type)
                limit_param = "$4"
            else:
                limit_param = "$3"
            
            where_clause = " AND ".join(conditions)
            
            async with self.db_manager.get_connection() as conn:
                knowledge_sources = await conn.fetch(f"""
                    SELECT 
                        knowledge_id, title, content, source_type, source_url,
                        keywords, tags, created_at, updated_at
                    FROM employee_knowledge_base 
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT {limit_param}
                """, *params, limit)
            
            return [
                {
                    "knowledge_id": row["knowledge_id"],
                    "title": row["title"],
                    "content": row["content"][:200] + "..." if len(row["content"]) > 200 else row["content"],
                    "source_type": row["source_type"],
                    "source_url": row["source_url"],
                    "keywords": row["keywords"],
                    "tags": row["tags"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None
                }
                for row in knowledge_sources
            ]
            
        except Exception as e:
            logger.error(f"Failed to list knowledge sources for employee {employee_id}: {e}")
            raise MemoryAccessError(f"Failed to list knowledge sources: {e}")

    async def remove_knowledge_source(
        self,
        client_id: str,
        employee_id: str,
        knowledge_id: str
    ) -> bool:
        """Remove a knowledge source from employee's knowledge base."""
        try:
            await self._validate_tenant_access(client_id, employee_id)
            
            async with self.db_manager.get_connection() as conn:
                # Remove knowledge source
                result = await conn.execute("""
                    DELETE FROM employee_knowledge_base 
                    WHERE client_id = $1 AND employee_id = $2 AND knowledge_id = $3
                """, client_id, employee_id, knowledge_id)
                
                # Check if any rows were affected
                return result == "DELETE 1"
            
        except Exception as e:
            logger.error(f"Failed to remove knowledge source {knowledge_id}: {e}")
            return False

    async def clear_employee_knowledge_base(
        self,
        client_id: str,
        employee_id: str
    ) -> bool:
        """Clear all knowledge sources for an employee."""
        try:
            await self._validate_tenant_access(client_id, employee_id)
            
            async with self.db_manager.get_connection() as conn:
                await conn.execute("""
                    DELETE FROM employee_knowledge_base 
                    WHERE client_id = $1 AND employee_id = $2
                """, client_id, employee_id)
            
            logger.info(f"Cleared knowledge base for employee {employee_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear knowledge base for employee {employee_id}: {e}")
            return False
