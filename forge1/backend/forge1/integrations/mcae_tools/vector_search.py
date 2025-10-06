"""
Tenant-Aware Vector Search

Vector search tool with tenant isolation that ensures employees
can only search within their tenant's vector space.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union
import uuid
from datetime import datetime, timezone

from forge1.core.tenancy import get_current_tenant, set_current_tenant
from forge1.core.memory_manager import MemoryManager
from forge1.core.memory_models import MemoryQuery, MemoryType

logger = logging.getLogger(__name__)


class VectorSearchError(Exception):
    """Exception raised when vector search fails"""
    pass


class TenantAccessError(Exception):
    """Exception raised when tenant access is denied"""
    pass


class TenantAwareVectorSearch:
    """
    Vector search tool with tenant isolation.
    
    Provides semantic search capabilities while ensuring strict tenant
    boundaries and employee-specific access controls.
    """
    
    def __init__(self, tenant_id: str, employee_id: str, memory_manager: MemoryManager):
        """
        Initialize the tenant-aware vector search.
        
        Args:
            tenant_id: Tenant ID for isolation
            employee_id: Employee ID for access control
            memory_manager: Forge1's memory manager for vector operations
        """
        self.tenant_id = tenant_id
        self.employee_id = employee_id
        self.memory_manager = memory_manager
        
        # Search statistics
        self.stats = {
            "searches_performed": 0,
            "total_results_returned": 0,
            "average_search_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Simple cache for recent searches
        self._search_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
    async def search(self, query: str, limit: int = 10, filters: Optional[Dict] = None, 
                    search_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search within tenant's vector space only.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            filters: Additional filters to apply
            search_types: Types of content to search (documents, conversations, etc.)
            
        Returns:
            List of search results with relevance scores
            
        Raises:
            TenantAccessError: If tenant access is denied
            VectorSearchError: If search fails
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Set tenant context
            set_current_tenant(self.tenant_id)
            
            # Validate tenant access
            current_tenant = get_current_tenant()
            if current_tenant != self.tenant_id:
                raise TenantAccessError(f"Tenant access denied: {current_tenant} != {self.tenant_id}")
            
            # Check cache first
            cache_key = self._generate_cache_key(query, limit, filters, search_types)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.stats["cache_hits"] += 1
                return cached_result
            
            self.stats["cache_misses"] += 1
            
            # Build memory query with tenant isolation
            memory_query = self._build_memory_query(query, limit, filters, search_types)
            
            # Perform search through Forge1's memory manager
            search_response = await self.memory_manager.search_memories(memory_query, self.employee_id)
            
            # Process and format results
            results = await self._process_search_results(search_response, query)
            
            # Cache the results
            self._cache_result(cache_key, results)
            
            # Update statistics
            search_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._update_stats(len(results), search_time)
            
            logger.info(f"Vector search completed: {len(results)} results for tenant {self.tenant_id}")
            return results
            
        except TenantAccessError:
            logger.warning(f"Tenant access denied for search by employee {self.employee_id}")
            raise
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise VectorSearchError(f"Search failed: {e}")
    
    async def semantic_search(self, query: str, limit: int = 10, 
                            similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform semantic search with similarity threshold.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of semantically similar results
        """
        try:
            # Perform regular search
            results = await self.search(query, limit * 2)  # Get more results to filter
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result.get('similarity_score', 0.0) >= similarity_threshold
            ]
            
            # Limit results
            return filtered_results[:limit]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise VectorSearchError(f"Semantic search failed: {e}")
    
    async def search_by_category(self, query: str, category: str, 
                               limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search within a specific category.
        
        Args:
            query: Search query text
            category: Category to search within (documents, conversations, tasks, etc.)
            limit: Maximum number of results
            
        Returns:
            List of category-specific results
        """
        try:
            # Map category to memory types
            category_mapping = {
                "documents": [MemoryType.DOCUMENT],
                "conversations": [MemoryType.CONVERSATION],
                "tasks": [MemoryType.TASK],
                "workflows": [MemoryType.WORKFLOW],
                "feedback": [MemoryType.FEEDBACK],
                "all": None  # Search all types
            }
            
            memory_types = category_mapping.get(category.lower())
            if memory_types is None and category.lower() != "all":
                raise VectorSearchError(f"Unknown category: {category}")
            
            # Build filters for category
            filters = {"memory_types": memory_types} if memory_types else {}
            
            return await self.search(query, limit, filters)
            
        except Exception as e:
            logger.error(f"Category search failed: {e}")
            raise VectorSearchError(f"Category search failed: {e}")
    
    async def find_similar_content(self, content_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find content similar to a specific item.
        
        Args:
            content_id: ID of the content to find similar items for
            limit: Maximum number of similar items to return
            
        Returns:
            List of similar content items
        """
        try:
            # Set tenant context
            set_current_tenant(self.tenant_id)
            
            # Retrieve the reference content
            reference_memory = await self.memory_manager.retrieve_memory(content_id, self.employee_id)
            if not reference_memory:
                raise VectorSearchError(f"Reference content not found: {content_id}")
            
            # Use the reference content's embeddings for similarity search
            if not reference_memory.embeddings:
                raise VectorSearchError(f"Reference content has no embeddings: {content_id}")
            
            # Build query using embeddings
            memory_query = MemoryQuery(
                query_embedding=reference_memory.embeddings,
                limit=limit + 1,  # +1 to exclude the reference item itself
                tags=[f"tenant:{self.tenant_id}"],
                min_relevance_score=0.5
            )
            
            # Perform search
            search_response = await self.memory_manager.search_memories(memory_query, self.employee_id)
            
            # Process results and exclude the reference item
            results = []
            for result in search_response.results:
                if result.memory.id != content_id:  # Exclude reference item
                    processed_result = await self._process_single_result(result, "similarity search")
                    results.append(processed_result)
                    
                    if len(results) >= limit:
                        break
            
            logger.info(f"Found {len(results)} similar items for content {content_id}")
            return results
            
        except Exception as e:
            logger.error(f"Similar content search failed: {e}")
            raise VectorSearchError(f"Similar content search failed: {e}")
    
    def _build_memory_query(self, query: str, limit: int, filters: Optional[Dict], 
                          search_types: Optional[List[str]]) -> MemoryQuery:
        """Build memory query with tenant isolation"""
        
        # Base tags for tenant isolation
        tags = [f"tenant:{self.tenant_id}"]
        
        # Add employee-specific tag if searching personal content
        if filters and filters.get("include_personal", True):
            tags.append(f"employee:{self.employee_id}")
        
        # Add additional filter tags
        if filters:
            for key, value in filters.items():
                if key not in ["include_personal", "memory_types"] and value:
                    if isinstance(value, list):
                        tags.extend([f"{key}:{v}" for v in value])
                    else:
                        tags.append(f"{key}:{value}")
        
        # Map search types to memory types
        memory_types = None
        if search_types:
            type_mapping = {
                "documents": MemoryType.DOCUMENT,
                "conversations": MemoryType.CONVERSATION,
                "tasks": MemoryType.TASK,
                "workflows": MemoryType.WORKFLOW,
                "feedback": MemoryType.FEEDBACK,
                "general": MemoryType.GENERAL
            }
            
            memory_types = []
            for search_type in search_types:
                if search_type.lower() in type_mapping:
                    memory_types.append(type_mapping[search_type.lower()])
        
        # Override with filter memory types if specified
        if filters and "memory_types" in filters:
            memory_types = filters["memory_types"]
        
        return MemoryQuery(
            query_text=query,
            limit=limit,
            tags=tags,
            memory_types=memory_types,
            sort_by="relevance_score",
            sort_order="desc",
            min_relevance_score=0.3  # Minimum relevance threshold
        )
    
    async def _process_search_results(self, search_response, query: str) -> List[Dict[str, Any]]:
        """Process and format search results"""
        results = []
        
        for result in search_response.results:
            processed_result = await self._process_single_result(result, query)
            results.append(processed_result)
        
        return results
    
    async def _process_single_result(self, result, query: str) -> Dict[str, Any]:
        """Process a single search result"""
        memory = result.memory
        
        # Extract content preview
        content_preview = ""
        if isinstance(memory.content, dict):
            # Extract text from structured content
            if "text" in memory.content:
                content_preview = str(memory.content["text"])[:200]
            else:
                content_preview = str(memory.content)[:200]
        else:
            content_preview = str(memory.content)[:200]
        
        return {
            "id": memory.id,
            "title": memory.summary or f"{memory.memory_type.value} content",
            "content_preview": content_preview,
            "similarity_score": result.similarity_score,
            "relevance_score": memory.relevance_score.overall if hasattr(memory, 'relevance_score') else 0.0,
            "memory_type": memory.memory_type.value,
            "source": memory.source,
            "created_at": memory.created_at.isoformat(),
            "tags": [tag for tag in memory.tags if not tag.startswith("tenant:")],  # Hide tenant tags
            "keywords": memory.keywords,
            "tenant_id": self.tenant_id,
            "employee_id": self.employee_id,
            "search_query": query,
            "rank": result.rank if hasattr(result, 'rank') else 0
        }
    
    def _generate_cache_key(self, query: str, limit: int, filters: Optional[Dict], 
                          search_types: Optional[List[str]]) -> str:
        """Generate cache key for search parameters"""
        import hashlib
        
        cache_data = {
            "tenant_id": self.tenant_id,
            "employee_id": self.employee_id,
            "query": query,
            "limit": limit,
            "filters": filters or {},
            "search_types": search_types or []
        }
        
        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[Dict]]:
        """Get cached search result if valid"""
        if cache_key in self._search_cache:
            cached_data = self._search_cache[cache_key]
            age = (datetime.now(timezone.utc) - cached_data["timestamp"]).total_seconds()
            
            if age < self._cache_ttl:
                return cached_data["results"]
            else:
                # Remove expired cache entry
                del self._search_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, results: List[Dict]):
        """Cache search results"""
        self._search_cache[cache_key] = {
            "results": results,
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Limit cache size
        if len(self._search_cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                self._search_cache.keys(),
                key=lambda k: self._search_cache[k]["timestamp"]
            )[:20]
            
            for key in oldest_keys:
                del self._search_cache[key]
    
    def _update_stats(self, result_count: int, search_time: float):
        """Update search statistics"""
        self.stats["searches_performed"] += 1
        self.stats["total_results_returned"] += result_count
        
        # Update average search time
        current_avg = self.stats["average_search_time"]
        search_count = self.stats["searches_performed"]
        self.stats["average_search_time"] = (
            (current_avg * (search_count - 1) + search_time) / search_count
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            **self.stats,
            "tenant_id": self.tenant_id,
            "employee_id": self.employee_id,
            "cache_size": len(self._search_cache),
            "cache_hit_rate": (
                self.stats["cache_hits"] / 
                max(self.stats["cache_hits"] + self.stats["cache_misses"], 1)
            )
        }
    
    def clear_cache(self):
        """Clear search cache"""
        self._search_cache.clear()
        logger.info(f"Cleared vector search cache for tenant {self.tenant_id}")
    
    def reset_stats(self):
        """Reset search statistics"""
        self.stats = {
            "searches_performed": 0,
            "total_results_returned": 0,
            "average_search_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(f"Reset vector search stats for tenant {self.tenant_id}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test search functionality with a simple query
            test_results = await self.search("test", limit=1)
            
            return {
                "status": "healthy",
                "tenant_id": self.tenant_id,
                "employee_id": self.employee_id,
                "test_search_results": len(test_results),
                "stats": self.get_stats()
            }
        except Exception as e:
            logger.error(f"Vector search health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "tenant_id": self.tenant_id,
                "employee_id": self.employee_id
            }