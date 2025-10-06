# forge1/backend/forge1/core/performance_monitor.py
"""
Performance Monitor for Employee Lifecycle System

Comprehensive performance monitoring and metrics collection for
employee interactions, database operations, and system health.

Requirements: 5.4, 5.5
"""

import asyncio
import logging
import time
import psutil
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = "ms"


@dataclass
class SystemMetrics:
    """System-level performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    timestamp: datetime


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Features:
    - Real-time metrics collection
    - Performance trend analysis
    - Alerting for performance issues
    - Resource usage monitoring
    - Database performance tracking
    - Employee interaction metrics
    """
    
    def __init__(
        self,
        max_metrics_history: int = 10000,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        self.max_metrics_history = max_metrics_history
        self.alert_thresholds = alert_thresholds or {
            "response_time_ms": 5000,  # 5 seconds
            "cpu_percent": 80,
            "memory_percent": 85,
            "error_rate_percent": 5,
            "database_connection_time_ms": 1000
        }
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=max_metrics_history)
        self.current_metrics: Dict[str, PerformanceMetric] = {}
        
        # Performance counters
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        
        # System monitoring
        self.system_metrics_history: deque = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        # Performance baselines
        self.baselines = {
            "employee_load_time_ms": 100,
            "interaction_processing_time_ms": 2000,
            "memory_query_time_ms": 50,
            "database_query_time_ms": 100
        }
    
    async def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous system monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )
        
        logger.info(f"Performance monitoring started with {interval_seconds}s interval")
    
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        unit: str = "ms"
    ):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            tags=tags or {},
            unit=unit
        )
        
        # Store in history
        self.metrics_history.append(metric)
        
        # Update current metrics
        self.current_metrics[name] = metric
        
        # Update timers for trend analysis
        self.timers[name].append(value)
        if len(self.timers[name]) > 100:  # Keep last 100 values
            self.timers[name].pop(0)
        
        # Check for alerts
        self._check_alert_thresholds(metric)
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        self.counters[name] += value
        
        # Also record as a metric for trend analysis
        self.record_metric(f"{name}_count", self.counters[name], tags, "count")
    
    def time_operation(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        return OperationTimer(self, operation_name, tags)
    
    async def record_employee_interaction(
        self,
        client_id: str,
        employee_id: str,
        processing_time_ms: float,
        tokens_used: int,
        model_used: str,
        success: bool
    ):
        """Record employee interaction performance metrics"""
        tags = {
            "client_id": client_id,
            "employee_id": employee_id,
            "model": model_used,
            "status": "success" if success else "error"
        }
        
        # Record timing
        self.record_metric("employee_interaction_time", processing_time_ms, tags)
        
        # Record token usage
        self.record_metric("tokens_used", tokens_used, tags, "tokens")
        
        # Update counters
        self.increment_counter("employee_interactions_total", 1, tags)
        
        if not success:
            self.increment_counter("employee_interaction_errors", 1, tags)
    
    async def record_database_operation(
        self,
        operation_type: str,
        table_name: str,
        execution_time_ms: float,
        success: bool
    ):
        """Record database operation performance"""
        tags = {
            "operation": operation_type,
            "table": table_name,
            "status": "success" if success else "error"
        }
        
        self.record_metric("database_operation_time", execution_time_ms, tags)
        self.increment_counter("database_operations_total", 1, tags)
        
        if not success:
            self.increment_counter("database_operation_errors", 1, tags)
    
    async def record_cache_operation(
        self,
        operation_type: str,
        cache_key: str,
        hit: bool,
        response_time_ms: float
    ):
        """Record cache operation performance"""
        tags = {
            "operation": operation_type,
            "result": "hit" if hit else "miss"
        }
        
        self.record_metric("cache_operation_time", response_time_ms, tags)
        self.increment_counter(f"cache_{operation_type}_total", 1, tags)
        
        if hit:
            self.increment_counter("cache_hits", 1, tags)
        else:
            self.increment_counter("cache_misses", 1, tags)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics summary"""
        now = datetime.now(timezone.utc)
        
        # Calculate rates and averages
        metrics_summary = {}
        
        for name, values in self.timers.items():
            if values:
                metrics_summary[name] = {
                    "current": values[-1],
                    "average": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "p95": self._percentile(values, 95),
                    "p99": self._percentile(values, 99)
                }
        
        # Add counter values
        for name, count in self.counters.items():
            metrics_summary[f"{name}_total"] = count
        
        # Calculate error rates
        total_interactions = self.counters.get("employee_interactions_total", 0)
        interaction_errors = self.counters.get("employee_interaction_errors", 0)
        
        if total_interactions > 0:
            error_rate = (interaction_errors / total_interactions) * 100
            metrics_summary["error_rate_percent"] = round(error_rate, 2)
        
        # Calculate cache hit rate
        cache_hits = self.counters.get("cache_hits", 0)
        cache_misses = self.counters.get("cache_misses", 0)
        total_cache_ops = cache_hits + cache_misses
        
        if total_cache_ops > 0:
            hit_rate = (cache_hits / total_cache_ops) * 100
            metrics_summary["cache_hit_rate_percent"] = round(hit_rate, 2)
        
        return {
            "timestamp": now.isoformat(),
            "metrics": metrics_summary,
            "system": self._get_current_system_metrics(),
            "alerts": self._get_active_alerts()
        }
    
    def get_performance_trends(
        self,
        metric_name: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get performance trends for a specific metric"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Filter metrics by time and name
        relevant_metrics = [
            m for m in self.metrics_history
            if m.name == metric_name and m.timestamp >= cutoff_time
        ]
        
        if not relevant_metrics:
            return {"error": f"No data found for metric {metric_name}"}
        
        values = [m.value for m in relevant_metrics]
        timestamps = [m.timestamp for m in relevant_metrics]
        
        # Calculate trend statistics
        trend_data = {
            "metric_name": metric_name,
            "time_range_hours": hours,
            "data_points": len(values),
            "statistics": {
                "average": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                "p95": self._percentile(values, 95),
                "p99": self._percentile(values, 99)
            },
            "trend": self._calculate_trend(values),
            "first_timestamp": timestamps[0].isoformat(),
            "last_timestamp": timestamps[-1].isoformat()
        }
        
        # Add baseline comparison if available
        if metric_name in self.baselines:
            baseline = self.baselines[metric_name]
            current_avg = trend_data["statistics"]["average"]
            deviation_percent = ((current_avg - baseline) / baseline) * 100
            
            trend_data["baseline_comparison"] = {
                "baseline_value": baseline,
                "current_average": current_avg,
                "deviation_percent": round(deviation_percent, 2),
                "performance": "better" if current_avg < baseline else "worse"
            }
        
        return trend_data
    
    def get_system_health_score(self) -> Dict[str, Any]:
        """Calculate overall system health score"""
        scores = {}
        
        # Performance score (based on response times)
        interaction_times = self.timers.get("employee_interaction_time", [])
        if interaction_times:
            avg_time = statistics.mean(interaction_times)
            baseline = self.baselines["interaction_processing_time_ms"]
            performance_score = max(0, 100 - ((avg_time - baseline) / baseline) * 100)
            scores["performance"] = min(100, max(0, performance_score))
        else:
            scores["performance"] = 100
        
        # Reliability score (based on error rates)
        total_ops = self.counters.get("employee_interactions_total", 0)
        errors = self.counters.get("employee_interaction_errors", 0)
        
        if total_ops > 0:
            error_rate = (errors / total_ops) * 100
            reliability_score = max(0, 100 - (error_rate * 10))  # 10% error = 0 score
            scores["reliability"] = reliability_score
        else:
            scores["reliability"] = 100
        
        # Resource utilization score
        if self.system_metrics_history:
            latest_system = self.system_metrics_history[-1]
            cpu_score = max(0, 100 - latest_system.cpu_percent)
            memory_score = max(0, 100 - latest_system.memory_percent)
            scores["resources"] = (cpu_score + memory_score) / 2
        else:
            scores["resources"] = 100
        
        # Overall health score
        overall_score = statistics.mean(scores.values())
        
        return {
            "overall_score": round(overall_score, 1),
            "component_scores": {k: round(v, 1) for k, v in scores.items()},
            "health_status": self._get_health_status(overall_score),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def add_alert_callback(self, callback: Callable[[PerformanceMetric], None]):
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)
    
    def set_baseline(self, metric_name: str, baseline_value: float):
        """Set performance baseline for a metric"""
        self.baselines[metric_name] = baseline_value
        logger.info(f"Set baseline for {metric_name}: {baseline_value}")
    
    # Private methods
    
    async def _monitoring_loop(self, interval_seconds: int):
        """Continuous system monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # Record system metrics as performance metrics
                self.record_metric("cpu_percent", system_metrics.cpu_percent, unit="%")
                self.record_metric("memory_percent", system_metrics.memory_percent, unit="%")
                self.record_metric("memory_used_mb", system_metrics.memory_used_mb, unit="MB")
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv
        }
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_usage_percent=disk.percent,
            network_io=network_io,
            timestamp=datetime.now(timezone.utc)
        )
    
    def _get_current_system_metrics(self) -> Optional[Dict[str, Any]]:
        """Get latest system metrics"""
        if not self.system_metrics_history:
            return None
        
        latest = self.system_metrics_history[-1]
        return {
            "cpu_percent": latest.cpu_percent,
            "memory_percent": latest.memory_percent,
            "memory_used_mb": round(latest.memory_used_mb, 2),
            "memory_available_mb": round(latest.memory_available_mb, 2),
            "disk_usage_percent": latest.disk_usage_percent,
            "timestamp": latest.timestamp.isoformat()
        }
    
    def _check_alert_thresholds(self, metric: PerformanceMetric):
        """Check if metric exceeds alert thresholds"""
        threshold_key = metric.name
        if threshold_key in self.alert_thresholds:
            threshold = self.alert_thresholds[threshold_key]
            
            if metric.value > threshold:
                # Trigger alert
                for callback in self.alert_callbacks:
                    try:
                        callback(metric)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {e}")
                
                logger.warning(
                    f"Performance alert: {metric.name} = {metric.value} {metric.unit} "
                    f"(threshold: {threshold} {metric.unit})"
                )
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active performance alerts"""
        alerts = []
        
        for name, metric in self.current_metrics.items():
            if name in self.alert_thresholds:
                threshold = self.alert_thresholds[name]
                if metric.value > threshold:
                    alerts.append({
                        "metric": name,
                        "current_value": metric.value,
                        "threshold": threshold,
                        "severity": self._get_alert_severity(metric.value, threshold),
                        "timestamp": metric.timestamp.isoformat()
                    })
        
        return alerts
    
    def _get_alert_severity(self, value: float, threshold: float) -> str:
        """Determine alert severity based on threshold deviation"""
        deviation = (value - threshold) / threshold
        
        if deviation > 0.5:  # 50% over threshold
            return "critical"
        elif deviation > 0.2:  # 20% over threshold
            return "high"
        else:
            return "medium"
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for values"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple trend calculation using first and last quartiles
        quarter_size = len(values) // 4
        if quarter_size == 0:
            return "insufficient_data"
        
        first_quarter_avg = statistics.mean(values[:quarter_size])
        last_quarter_avg = statistics.mean(values[-quarter_size:])
        
        change_percent = ((last_quarter_avg - first_quarter_avg) / first_quarter_avg) * 100
        
        if change_percent > 10:
            return "increasing"
        elif change_percent < -10:
            return "decreasing"
        else:
            return "stable"
    
    def _get_health_status(self, score: float) -> str:
        """Get health status based on score"""
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "fair"
        elif score >= 40:
            return "poor"
        else:
            return "critical"


class OperationTimer:
    """Context manager for timing operations"""
    
    def __init__(
        self,
        monitor: PerformanceMonitor,
        operation_name: str,
        tags: Optional[Dict[str, str]] = None
    ):
        self.monitor = monitor
        self.operation_name = operation_name
        self.tags = tags or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            
            # Add success/error tag based on exception
            tags = self.tags.copy()
            tags["status"] = "error" if exc_type else "success"
            
            self.monitor.record_metric(
                self.operation_name,
                duration_ms,
                tags
            )


# Global performance monitor instance
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    
    return _performance_monitor