"""
MCAE Performance Optimization and Monitoring

Performance monitoring, optimization, and resource management for MCAE integration
with caching, connection pooling, and resource utilization tracking.
"""

import asyncio
import time
import psutil
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone, timedelta
from collections import deque
from dataclasses import dataclass, field

from forge1.integrations.mcae_logging import mcae_logger, log_performance


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    name: str
    value: float
    timestamp: datetime
    tenant_id: Optional[str] = None
    employee_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceUsage:
    """System resource usage snapshot"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    timestamp: datetime


class PerformanceCache:
    """High-performance cache for MCAE operations"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self.cache:
            # Check TTL
            if time.time() - self.access_times[key] < self.ttl_seconds:
                self.access_times[key] = time.time()
                self.hit_count += 1
                return self.cache[key]
            else:
                # Expired
                del self.cache[key]
                del self.access_times[key]
        
        self.miss_count += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache"""
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.access_times.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds
        }


class ConnectionPool:
    """Connection pool for MCAE operations"""
    
    def __init__(self, max_connections: int = 50, timeout_seconds: int = 30):
        self.max_connections = max_connections
        self.timeout_seconds = timeout_seconds
        self.active_connections = {}
        self.available_connections = deque()
        self.connection_count = 0
        self.stats = {
            "connections_created": 0,
            "connections_reused": 0,
            "connections_timeout": 0,
            "peak_connections": 0
        }
    
    async def get_connection(self, connection_type: str) -> Any:
        """Get connection from pool"""
        # Try to reuse existing connection
        if self.available_connections:
            connection = self.available_connections.popleft()
            self.active_connections[id(connection)] = {
                "connection": connection,
                "type": connection_type,
                "acquired_at": time.time()
            }
            self.stats["connections_reused"] += 1
            return connection
        
        # Create new connection if under limit
        if self.connection_count < self.max_connections:
            connection = await self._create_connection(connection_type)
            self.active_connections[id(connection)] = {
                "connection": connection,
                "type": connection_type,
                "acquired_at": time.time()
            }
            self.connection_count += 1
            self.stats["connections_created"] += 1
            
            # Update peak connections
            if self.connection_count > self.stats["peak_connections"]:
                self.stats["peak_connections"] = self.connection_count
            
            return connection
        
        # Wait for available connection
        raise RuntimeError("Connection pool exhausted")
    
    async def release_connection(self, connection: Any) -> None:
        """Release connection back to pool"""
        connection_id = id(connection)
        
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            
            # Check if connection is still valid
            if await self._is_connection_valid(connection):
                self.available_connections.append(connection)
            else:
                self.connection_count -= 1
    
    async def _create_connection(self, connection_type: str) -> Any:
        """Create new connection (mock implementation)"""
        # In real implementation, this would create actual connections
        # (database, HTTP, MCAE agent connections, etc.)
        return f"mock_connection_{connection_type}_{time.time()}"
    
    async def _is_connection_valid(self, connection: Any) -> bool:
        """Check if connection is still valid"""
        # Mock validation - in real implementation, this would ping the connection
        return True
    
    async def cleanup_expired_connections(self) -> int:
        """Clean up expired connections"""
        current_time = time.time()
        expired_connections = []
        
        for conn_id, conn_info in self.active_connections.items():
            if current_time - conn_info["acquired_at"] > self.timeout_seconds:
                expired_connections.append(conn_id)
        
        for conn_id in expired_connections:
            del self.active_connections[conn_id]
            self.connection_count -= 1
            self.stats["connections_timeout"] += 1
        
        return len(expired_connections)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            **self.stats,
            "active_connections": len(self.active_connections),
            "available_connections": len(self.available_connections),
            "total_connections": self.connection_count,
            "max_connections": self.max_connections,
            "utilization": (self.connection_count / self.max_connections) * 100
        }


class MCAEPerformanceMonitor:
    """
    Performance monitoring and optimization for MCAE integration.
    
    Tracks metrics, manages caching, monitors resource usage,
    and provides optimization recommendations.
    """
    
    def __init__(self):
        # Performance metrics storage
        self.metrics = deque(maxlen=10000)  # Keep last 10k metrics
        self.resource_usage = deque(maxlen=1000)  # Keep last 1k resource snapshots
        
        # Performance components
        self.cache = PerformanceCache()
        self.connection_pool = ConnectionPool()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_interval = 30  # seconds
        self.monitoring_task = None
        
        # Performance thresholds
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "workflow_execution_time": 30.0,  # seconds
            "cache_hit_rate": 70.0,  # percent
            "connection_utilization": 90.0  # percent
        }
        
        # Optimization recommendations
        self.recommendations = []
        
        # Statistics
        self.stats = {
            "monitoring_started_at": None,
            "total_metrics_collected": 0,
            "performance_alerts": 0,
            "optimization_recommendations": 0
        }
    
    async def start_monitoring(self, interval_seconds: int = 30) -> None:
        """Start performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_interval = interval_seconds
        self.monitoring_active = True
        self.stats["monitoring_started_at"] = datetime.now(timezone.utc)
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        mcae_logger.log_performance("monitoring_started", 1.0, interval=interval_seconds)
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        mcae_logger.log_performance("monitoring_stopped", 1.0)
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect resource usage
                await self._collect_resource_usage()
                
                # Clean up expired connections
                await self.connection_pool.cleanup_expired_connections()
                
                # Generate recommendations
                await self._generate_recommendations()
                
                # Sleep until next interval
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                mcae_logger.log(
                    mcae_logger.LogLevel.ERROR,
                    mcae_logger.LogCategory.PERFORMANCE,
                    f"Monitoring loop error: {e}"
                )
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_resource_usage(self) -> None:
        """Collect system resource usage"""
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            
            # Get disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
            
            # Get network I/O
            network_io = psutil.net_io_counters()
            network_sent_mb = network_io.bytes_sent / (1024 * 1024) if network_io else 0
            network_recv_mb = network_io.bytes_recv / (1024 * 1024) if network_io else 0
            
            # Create resource usage snapshot
            usage = ResourceUsage(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                timestamp=datetime.now(timezone.utc)
            )
            
            self.resource_usage.append(usage)
            self.stats["total_metrics_collected"] += 1
            
            # Check thresholds
            await self._check_performance_thresholds(usage)
            
        except Exception as e:
            mcae_logger.log(
                mcae_logger.LogLevel.ERROR,
                mcae_logger.LogCategory.PERFORMANCE,
                f"Resource collection error: {e}"
            )
    
    async def _check_performance_thresholds(self, usage: ResourceUsage) -> None:
        """Check performance thresholds and generate alerts"""
        alerts = []
        
        if usage.cpu_percent > self.thresholds["cpu_percent"]:
            alerts.append(f"High CPU usage: {usage.cpu_percent:.1f}%")
        
        if usage.memory_percent > self.thresholds["memory_percent"]:
            alerts.append(f"High memory usage: {usage.memory_percent:.1f}%")
        
        # Check cache performance
        cache_stats = self.cache.get_stats()
        if cache_stats["hit_rate"] < self.thresholds["cache_hit_rate"]:
            alerts.append(f"Low cache hit rate: {cache_stats['hit_rate']:.1f}%")
        
        # Check connection pool utilization
        pool_stats = self.connection_pool.get_stats()
        if pool_stats["utilization"] > self.thresholds["connection_utilization"]:
            alerts.append(f"High connection pool utilization: {pool_stats['utilization']:.1f}%")
        
        # Log alerts
        for alert in alerts:
            mcae_logger.log(
                mcae_logger.LogLevel.WARNING,
                mcae_logger.LogCategory.PERFORMANCE,
                f"Performance alert: {alert}"
            )
            self.stats["performance_alerts"] += 1
    
    async def _generate_recommendations(self) -> None:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Analyze recent resource usage
        if len(self.resource_usage) >= 10:
            recent_usage = list(self.resource_usage)[-10:]
            avg_cpu = sum(u.cpu_percent for u in recent_usage) / len(recent_usage)
            avg_memory = sum(u.memory_percent for u in recent_usage) / len(recent_usage)
            
            if avg_cpu > 70:
                recommendations.append({
                    "type": "cpu_optimization",
                    "message": "Consider optimizing CPU-intensive operations or scaling horizontally",
                    "priority": "high" if avg_cpu > 85 else "medium"
                })
            
            if avg_memory > 75:
                recommendations.append({
                    "type": "memory_optimization",
                    "message": "Consider increasing cache TTL or implementing memory pooling",
                    "priority": "high" if avg_memory > 90 else "medium"
                })
        
        # Analyze cache performance
        cache_stats = self.cache.get_stats()
        if cache_stats["hit_rate"] < 60:
            recommendations.append({
                "type": "cache_optimization",
                "message": "Consider increasing cache size or adjusting TTL settings",
                "priority": "medium"
            })
        
        # Analyze connection pool
        pool_stats = self.connection_pool.get_stats()
        if pool_stats["utilization"] > 80:
            recommendations.append({
                "type": "connection_optimization",
                "message": "Consider increasing connection pool size",
                "priority": "medium"
            })
        
        # Add new recommendations
        for rec in recommendations:
            if rec not in self.recommendations:
                self.recommendations.append({
                    **rec,
                    "timestamp": datetime.now(timezone.utc),
                    "id": len(self.recommendations)
                })
                self.stats["optimization_recommendations"] += 1
        
        # Limit recommendations list
        if len(self.recommendations) > 100:
            self.recommendations = self.recommendations[-50:]
    
    def record_metric(
        self,
        name: str,
        value: float,
        tenant_id: Optional[str] = None,
        employee_id: Optional[str] = None,
        **metadata
    ) -> None:
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            tenant_id=tenant_id,
            employee_id=employee_id,
            metadata=metadata
        )
        
        self.metrics.append(metric)
        log_performance(name, value, tenant_id=tenant_id, employee_id=employee_id, **metadata)
    
    def get_metrics(
        self,
        name: Optional[str] = None,
        tenant_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[PerformanceMetric]:
        """Get performance metrics with filtering"""
        filtered_metrics = list(self.metrics)
        
        if name:
            filtered_metrics = [m for m in filtered_metrics if m.name == name]
        
        if tenant_id:
            filtered_metrics = [m for m in filtered_metrics if m.tenant_id == tenant_id]
        
        if since:
            filtered_metrics = [m for m in filtered_metrics if m.timestamp >= since]
        
        return filtered_metrics[-limit:]
    
    def get_resource_usage(self, since: Optional[datetime] = None, limit: int = 100) -> List[ResourceUsage]:
        """Get resource usage data"""
        usage_data = list(self.resource_usage)
        
        if since:
            usage_data = [u for u in usage_data if u.timestamp >= since]
        
        return usage_data[-limit:]
    
    def get_performance_summary(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary"""
        # Get recent metrics
        recent_metrics = self.get_metrics(tenant_id=tenant_id, limit=1000)
        recent_usage = self.get_resource_usage(limit=100)
        
        # Calculate averages
        avg_cpu = sum(u.cpu_percent for u in recent_usage) / len(recent_usage) if recent_usage else 0
        avg_memory = sum(u.memory_percent for u in recent_usage) / len(recent_usage) if recent_usage else 0
        
        # Workflow execution times
        workflow_metrics = [m for m in recent_metrics if m.name.endswith("_execution_time")]
        avg_workflow_time = sum(m.value for m in workflow_metrics) / len(workflow_metrics) if workflow_metrics else 0
        
        return {
            "resource_usage": {
                "avg_cpu_percent": avg_cpu,
                "avg_memory_percent": avg_memory,
                "current_cpu": recent_usage[-1].cpu_percent if recent_usage else 0,
                "current_memory": recent_usage[-1].memory_percent if recent_usage else 0
            },
            "workflow_performance": {
                "avg_execution_time": avg_workflow_time,
                "total_workflows": len(workflow_metrics),
                "workflows_over_threshold": len([m for m in workflow_metrics if m.value > self.thresholds["workflow_execution_time"]])
            },
            "cache_performance": self.cache.get_stats(),
            "connection_pool": self.connection_pool.get_stats(),
            "recommendations": len([r for r in self.recommendations if r.get("priority") == "high"]),
            "monitoring_stats": self.stats
        }
    
    def get_recommendations(self, priority: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get optimization recommendations"""
        recommendations = self.recommendations
        
        if priority:
            recommendations = [r for r in recommendations if r.get("priority") == priority]
        
        return sorted(recommendations, key=lambda r: r["timestamp"], reverse=True)
    
    async def optimize_cache(self) -> Dict[str, Any]:
        """Perform cache optimization"""
        old_stats = self.cache.get_stats()
        
        # Clear expired entries
        self.cache.clear()
        
        # Adjust cache size based on usage patterns
        if old_stats["hit_rate"] < 50:
            self.cache.max_size = min(self.cache.max_size * 2, 50000)
        elif old_stats["hit_rate"] > 90:
            self.cache.max_size = max(self.cache.max_size // 2, 1000)
        
        new_stats = self.cache.get_stats()
        
        return {
            "optimization": "cache_tuning",
            "old_size": old_stats["max_size"],
            "new_size": new_stats["max_size"],
            "old_hit_rate": old_stats["hit_rate"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for performance monitor"""
        try:
            recent_usage = self.get_resource_usage(limit=1)
            current_usage = recent_usage[0] if recent_usage else None
            
            status = "healthy"
            if current_usage:
                if current_usage.cpu_percent > 90 or current_usage.memory_percent > 95:
                    status = "critical"
                elif current_usage.cpu_percent > 80 or current_usage.memory_percent > 85:
                    status = "degraded"
            
            return {
                "status": status,
                "monitoring_active": self.monitoring_active,
                "current_resource_usage": {
                    "cpu_percent": current_usage.cpu_percent if current_usage else 0,
                    "memory_percent": current_usage.memory_percent if current_usage else 0
                },
                "cache_stats": self.cache.get_stats(),
                "connection_pool_stats": self.connection_pool.get_stats(),
                "recent_alerts": self.stats["performance_alerts"],
                "recommendations_count": len(self.recommendations)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global performance monitor instance
performance_monitor = MCAEPerformanceMonitor()


# Convenience functions
async def start_monitoring(interval_seconds: int = 30):
    """Start performance monitoring"""
    await performance_monitor.start_monitoring(interval_seconds)


async def stop_monitoring():
    """Stop performance monitoring"""
    await performance_monitor.stop_monitoring()


def record_performance_metric(name: str, value: float, **kwargs):
    """Record a performance metric"""
    performance_monitor.record_metric(name, value, **kwargs)


def get_performance_summary(tenant_id: Optional[str] = None) -> Dict[str, Any]:
    """Get performance summary"""
    return performance_monitor.get_performance_summary(tenant_id)


def get_cache() -> PerformanceCache:
    """Get performance cache instance"""
    return performance_monitor.cache


def get_connection_pool() -> ConnectionPool:
    """Get connection pool instance"""
    return performance_monitor.connection_pool


# Performance decorators
def monitor_performance(metric_name: str):
    """Decorator to monitor function performance"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                record_performance_metric(f"{metric_name}_success", duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                record_performance_metric(f"{metric_name}_error", duration, error=str(e))
                raise
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                record_performance_metric(f"{metric_name}_success", duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                record_performance_metric(f"{metric_name}_error", duration, error=str(e))
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def cache_result(cache_key_func: Callable = None, ttl_seconds: int = 3600):
    """Decorator to cache function results"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # Try cache first
            cached_result = get_cache().get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            get_cache().set(cache_key, result)
            
            return result
        
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # Try cache first
            cached_result = get_cache().get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            get_cache().set(cache_key, result)
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator