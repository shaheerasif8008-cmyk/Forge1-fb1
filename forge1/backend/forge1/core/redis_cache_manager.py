# forge1/backend/forge1/core/redis_cache_manager.py
"""
Redis Cache Manager for Employee Lifecycle System

Provides high-performance caching for frequently accessed employees and data
with intelligent cache invalidation and performance monitoring.

Requirements: 5.1, 5.2, 5.3
"""

import asyncio
import json
import logging
import os
import pickle
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import asdict

import redis.asyncio as redis
from redis.asyncio import ConnectionPool

from forge1.models.employee_models import Employee, EmployeeStatus

logger = logging.getLogger(__name__)


class RedisCacheManager:
    """
    High-performance Redis cache manager for employee data.
    
    Features:
    - Employee object caching with TTL
    - Memory context caching
    - Performance metrics tracking
    - Intelligent cache invalidation
    - Connection pooling
    - Async operations
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        max_connections: int = 20,
        default_ttl: int = 3600  # 1 hour
    ):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.max_connections = max_connections
        self.default_ttl = default_ttl
        
        # Connection pool for better performance
        self.pool = None
        self.redis_client = None
        self._initialized = False
        
        # Cache key prefixes
        self.prefixes = {
            "employee": "emp:",
            "memory": "mem:",
            "stats": "stats:",
            "config": "cfg:",
            "session": "sess:",
            "metrics": "metrics:"
        }
        
        # Performance metrics
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_sets": 0,
            "cache_deletes": 0,
            "total_operations": 0,
            "avg_response_time": 0.0,
            "connection_errors": 0
        }
    
    async def initialize(self):
        """Initialize Redis connection pool and client"""
        if self._initialized:
            return
        
        try:
            # Create connection pool
            self.pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            # Create Redis client
            self.redis_client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.redis_client.ping()
            
            self._initialized = True
            logger.info(f"Redis cache manager initialized with {self.max_connections} max connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache manager: {e}")
            self.metrics["connection_errors"] += 1
            # Continue without Redis caching
            self._initialized = False
    
    async def close(self):
        """Close Redis connections"""
        if self.redis_client:
            await self.redis_client.close()
        if self.pool:
            await self.pool.disconnect()
        
        self._initialized = False
        logger.info("Redis cache manager closed")
    
    # Employee Caching Methods
    
    async def cache_employee(
        self,
        client_id: str,
        employee_id: str,
        employee: Employee,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache employee object with optional TTL"""
        if not self._initialized:
            return False
        
        start_time = time.time()
        
        try:
            cache_key = self._get_employee_key(client_id, employee_id)
            
            # Serialize employee object
            employee_data = {
                "id": employee.id,
                "client_id": employee.client_id,
                "name": employee.name,
                "role": employee.role,
                "status": employee.status.value,
                "created_at": employee.created_at.isoformat(),
                "updated_at": employee.updated_at.isoformat(),
                "last_interaction_at": employee.last_interaction_at.isoformat() if employee.last_interaction_at else None,
                "personality": {
                    "communication_style": employee.personality.communication_style.value,
                    "formality_level": employee.personality.formality_level.value,
                    "expertise_level": employee.personality.expertise_level.value,
                    "response_length": employee.personality.response_length.value,
                    "creativity_level": employee.personality.creativity_level,
                    "empathy_level": employee.personality.empathy_level,
                    "custom_traits": employee.personality.custom_traits
                },
                "model_preferences": {
                    "primary_model": employee.model_preferences.primary_model,
                    "fallback_models": employee.model_preferences.fallback_models,
                    "temperature": employee.model_preferences.temperature,
                    "max_tokens": employee.model_preferences.max_tokens,
                    "specialized_models": employee.model_preferences.specialized_models
                },
                "tool_access": employee.tool_access,
                "knowledge_sources": employee.knowledge_sources,
                "cached_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Cache with TTL
            cache_ttl = ttl or self.default_ttl
            await self.redis_client.setex(
                cache_key,
                cache_ttl,
                json.dumps(employee_data)
            )
            
            # Update metrics
            self.metrics["cache_sets"] += 1
            self._update_response_time(time.time() - start_time)
            
            logger.debug(f"Cached employee {employee_id} for {cache_ttl}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache employee {employee_id}: {e}")
            self.metrics["connection_errors"] += 1
            return False
    
    async def get_cached_employee(
        self,
        client_id: str,
        employee_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached employee data"""
        if not self._initialized:
            return None
        
        start_time = time.time()
        
        try:
            cache_key = self._get_employee_key(client_id, employee_id)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                self.metrics["cache_hits"] += 1
                employee_data = json.loads(cached_data)
                
                # Check if cache is still fresh (additional validation)
                cached_at = datetime.fromisoformat(employee_data["cached_at"])
                if datetime.now(timezone.utc) - cached_at > timedelta(seconds=self.default_ttl):
                    # Cache expired, remove it
                    await self.invalidate_employee_cache(client_id, employee_id)
                    self.metrics["cache_misses"] += 1
                    return None
                
                self._update_response_time(time.time() - start_time)
                logger.debug(f"Cache hit for employee {employee_id}")
                return employee_data
            else:
                self.metrics["cache_misses"] += 1
                logger.debug(f"Cache miss for employee {employee_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get cached employee {employee_id}: {e}")
            self.metrics["connection_errors"] += 1
            self.metrics["cache_misses"] += 1
            return None
    
    async def invalidate_employee_cache(
        self,
        client_id: str,
        employee_id: str
    ) -> bool:
        """Invalidate cached employee data"""
        if not self._initialized:
            return False
        
        try:
            cache_key = self._get_employee_key(client_id, employee_id)
            result = await self.redis_client.delete(cache_key)
            
            self.metrics["cache_deletes"] += 1
            logger.debug(f"Invalidated cache for employee {employee_id}")
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to invalidate cache for employee {employee_id}: {e}")
            self.metrics["connection_errors"] += 1
            return False
    
    # Memory Context Caching
    
    async def cache_memory_context(
        self,
        client_id: str,
        employee_id: str,
        memory_context: List[Dict[str, Any]],
        ttl: int = 300  # 5 minutes for memory context
    ) -> bool:
        """Cache employee memory context"""
        if not self._initialized:
            return False
        
        try:
            cache_key = self._get_memory_key(client_id, employee_id)
            
            memory_data = {
                "context": memory_context,
                "cached_at": datetime.now(timezone.utc).isoformat()
            }
            
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(memory_data)
            )
            
            self.metrics["cache_sets"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache memory context for {employee_id}: {e}")
            return False
    
    async def get_cached_memory_context(
        self,
        client_id: str,
        employee_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached memory context"""
        if not self._initialized:
            return None
        
        try:
            cache_key = self._get_memory_key(client_id, employee_id)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                memory_data = json.loads(cached_data)
                self.metrics["cache_hits"] += 1
                return memory_data["context"]
            else:
                self.metrics["cache_misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Failed to get cached memory context for {employee_id}: {e}")
            self.metrics["cache_misses"] += 1
            return None
    
    # Performance Metrics Caching
    
    async def cache_performance_metrics(
        self,
        metric_key: str,
        metrics_data: Dict[str, Any],
        ttl: int = 60  # 1 minute for metrics
    ) -> bool:
        """Cache performance metrics"""
        if not self._initialized:
            return False
        
        try:
            cache_key = f"{self.prefixes['metrics']}{metric_key}"
            
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(metrics_data)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache metrics {metric_key}: {e}")
            return False
    
    async def get_cached_metrics(
        self,
        metric_key: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached performance metrics"""
        if not self._initialized:
            return None
        
        try:
            cache_key = f"{self.prefixes['metrics']}{metric_key}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached metrics {metric_key}: {e}")
            return None
    
    # Session Caching
    
    async def cache_session_data(
        self,
        session_id: str,
        session_data: Dict[str, Any],
        ttl: int = 1800  # 30 minutes
    ) -> bool:
        """Cache session data for employee interactions"""
        if not self._initialized:
            return False
        
        try:
            cache_key = f"{self.prefixes['session']}{session_id}"
            
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(session_data)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache session {session_id}: {e}")
            return False
    
    async def get_cached_session(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached session data"""
        if not self._initialized:
            return None
        
        try:
            cache_key = f"{self.prefixes['session']}{session_id}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached session {session_id}: {e}")
            return None
    
    # Bulk Operations
    
    async def invalidate_client_cache(
        self,
        client_id: str
    ) -> int:
        """Invalidate all cached data for a client"""
        if not self._initialized:
            return 0
        
        try:
            # Find all keys for this client
            patterns = [
                f"{self.prefixes['employee']}{client_id}:*",
                f"{self.prefixes['memory']}{client_id}:*",
                f"{self.prefixes['stats']}{client_id}:*"
            ]
            
            deleted_count = 0
            for pattern in patterns:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    deleted_count += await self.redis_client.delete(*keys)
            
            self.metrics["cache_deletes"] += deleted_count
            logger.info(f"Invalidated {deleted_count} cache entries for client {client_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to invalidate client cache for {client_id}: {e}")
            return 0
    
    async def warm_cache(
        self,
        employees: List[Employee]
    ) -> int:
        """Warm cache with frequently accessed employees"""
        if not self._initialized:
            return 0
        
        cached_count = 0
        
        for employee in employees:
            success = await self.cache_employee(
                employee.client_id,
                employee.id,
                employee,
                ttl=self.default_ttl * 2  # Longer TTL for warm cache
            )
            if success:
                cached_count += 1
        
        logger.info(f"Warmed cache with {cached_count} employees")
        return cached_count
    
    # Health and Monitoring
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Redis health and return status"""
        if not self._initialized:
            return {"status": "disconnected", "error": "Not initialized"}
        
        try:
            start_time = time.time()
            
            # Test basic operations
            test_key = "health_check_test"
            await self.redis_client.set(test_key, "test", ex=10)
            result = await self.redis_client.get(test_key)
            await self.redis_client.delete(test_key)
            
            response_time = time.time() - start_time
            
            # Get Redis info
            info = await self.redis_client.info()
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "cache_metrics": self.get_cache_metrics()
            }
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "cache_metrics": self.get_cache_metrics()
            }
    
    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics"""
        total_ops = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        hit_rate = (self.metrics["cache_hits"] / max(total_ops, 1)) * 100
        
        return {
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"],
            "cache_sets": self.metrics["cache_sets"],
            "cache_deletes": self.metrics["cache_deletes"],
            "hit_rate_percent": round(hit_rate, 2),
            "total_operations": self.metrics["total_operations"],
            "avg_response_time_ms": round(self.metrics["avg_response_time"] * 1000, 2),
            "connection_errors": self.metrics["connection_errors"]
        }
    
    def reset_metrics(self):
        """Reset cache metrics"""
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_sets": 0,
            "cache_deletes": 0,
            "total_operations": 0,
            "avg_response_time": 0.0,
            "connection_errors": 0
        }
    
    # Private helper methods
    
    def _get_employee_key(self, client_id: str, employee_id: str) -> str:
        """Generate cache key for employee"""
        return f"{self.prefixes['employee']}{client_id}:{employee_id}"
    
    def _get_memory_key(self, client_id: str, employee_id: str) -> str:
        """Generate cache key for memory context"""
        return f"{self.prefixes['memory']}{client_id}:{employee_id}"
    
    def _update_response_time(self, response_time: float):
        """Update average response time metric"""
        self.metrics["total_operations"] += 1
        current_avg = self.metrics["avg_response_time"]
        total_ops = self.metrics["total_operations"]
        
        # Calculate running average
        self.metrics["avg_response_time"] = (
            (current_avg * (total_ops - 1) + response_time) / total_ops
        )


# Global cache manager instance
_cache_manager = None


async def get_cache_manager() -> RedisCacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = RedisCacheManager()
        await _cache_manager.initialize()
    
    return _cache_manager


async def close_cache_manager():
    """Close global cache manager"""
    global _cache_manager
    
    if _cache_manager:
        await _cache_manager.close()
        _cache_manager = None