"""
Custom Business Metrics Collection

Provides comprehensive business metrics collection for Forge1 with
tenant-aware labeling and integration with OpenTelemetry metrics.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram, UpDownCounter

from forge1.integrations.observability.otel_init import otel_integration
from forge1.core.tenancy import get_current_tenant

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"

@dataclass
class MetricDefinition:
    """Metric definition"""
    name: str
    description: str
    unit: str
    metric_type: MetricType
    labels: List[str]

class ForgeMetrics:
    """Business metrics collection for Forge1"""
    
    def __init__(self, meter_provider: Optional[metrics.MeterProvider] = None):
        self.meter = otel_integration.get_meter() if otel_integration else None
        
        # Business metrics
        self._task_counter = None
        self._task_duration_histogram = None
        self._model_usage_counter = None
        self._model_cost_histogram = None
        self._policy_decision_counter = None
        self._memory_operations_counter = None
        self._vector_operations_counter = None
        self._queue_depth_gauge = None
        self._active_sessions_gauge = None
        self._error_counter = None
        
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize all business metrics"""
        if not self.meter:
            logger.warning("No meter available, metrics will not be collected")
            return
        
        # Task execution metrics
        self._task_counter = self.meter.create_counter(
            name="forge1_tasks_total",
            description="Total number of tasks executed",
            unit="1"
        )
        
        self._task_duration_histogram = self.meter.create_histogram(
            name="forge1_task_duration_seconds",
            description="Task execution duration",
            unit="s"
        )
        
        # Model usage metrics
        self._model_usage_counter = self.meter.create_counter(
            name="forge1_model_requests_total",
            description="Total number of model requests",
            unit="1"
        )
        
        self._model_cost_histogram = self.meter.create_histogram(
            name="forge1_model_cost_dollars",
            description="Model usage cost",
            unit="USD"
        )
        
        # Policy decision metrics
        self._policy_decision_counter = self.meter.create_counter(
            name="forge1_policy_decisions_total",
            description="Total number of policy decisions",
            unit="1"
        )
        
        # Memory operation metrics
        self._memory_operations_counter = self.meter.create_counter(
            name="forge1_memory_operations_total",
            description="Total number of memory operations",
            unit="1"
        )
        
        # Vector operation metrics
        self._vector_operations_counter = self.meter.create_counter(
            name="forge1_vector_operations_total",
            description="Total number of vector operations",
            unit="1"
        )
        
        # Queue depth gauge
        self._queue_depth_gauge = self.meter.create_up_down_counter(
            name="forge1_queue_depth",
            description="Current queue depth",
            unit="1"
        )
        
        # Active sessions gauge
        self._active_sessions_gauge = self.meter.create_up_down_counter(
            name="forge1_active_sessions",
            description="Number of active sessions",
            unit="1"
        )
        
        # Error counter
        self._error_counter = self.meter.create_counter(
            name="forge1_errors_total",
            description="Total number of errors",
            unit="1"
        )
        
        logger.info("Business metrics initialized")
    
    def record_task_execution(self, task_type: str, duration: float, 
                            tenant_id: Optional[str] = None, 
                            success: bool = True, **labels):
        """Record task execution metrics"""
        if not self._task_counter or not self._task_duration_histogram:
            return
        
        # Get tenant ID from context if not provided
        if not tenant_id:
            tenant_id = get_current_tenant() or "unknown"
        
        # Common labels
        common_labels = {
            "task_type": task_type,
            "tenant_id": tenant_id,
            "success": str(success).lower()
        }
        common_labels.update(labels)
        
        # Record task count
        self._task_counter.add(1, common_labels)
        
        # Record task duration
        self._task_duration_histogram.record(duration, common_labels)
        
        logger.debug(f"Recorded task execution: {task_type}, duration: {duration}s, success: {success}")
    
    def record_model_usage(self, model: str, tokens: int, cost: float, 
                          tenant_id: Optional[str] = None, **labels):
        """Record model usage metrics"""
        if not self._model_usage_counter or not self._model_cost_histogram:
            return
        
        # Get tenant ID from context if not provided
        if not tenant_id:
            tenant_id = get_current_tenant() or "unknown"
        
        # Common labels
        common_labels = {
            "model": model,
            "tenant_id": tenant_id
        }
        common_labels.update(labels)
        
        # Record model request
        self._model_usage_counter.add(1, common_labels)
        
        # Record cost
        if cost > 0:
            self._model_cost_histogram.record(cost, common_labels)
        
        logger.debug(f"Recorded model usage: {model}, tokens: {tokens}, cost: ${cost}")
    
    def record_policy_decision(self, decision: str, resource: str, 
                             tenant_id: Optional[str] = None, **labels):
        """Record policy decision metrics"""
        if not self._policy_decision_counter:
            return
        
        # Get tenant ID from context if not provided
        if not tenant_id:
            tenant_id = get_current_tenant() or "unknown"
        
        # Common labels
        common_labels = {
            "decision": decision,
            "resource": resource,
            "tenant_id": tenant_id
        }
        common_labels.update(labels)
        
        # Record policy decision
        self._policy_decision_counter.add(1, common_labels)
        
        logger.debug(f"Recorded policy decision: {decision} for {resource}")    
    
def record_memory_operation(self, operation: str, duration: float, 
                               tenant_id: Optional[str] = None, 
                               success: bool = True, **labels):
        """Record memory operation metrics"""
        if not self._memory_operations_counter:
            return
        
        # Get tenant ID from context if not provided
        if not tenant_id:
            tenant_id = get_current_tenant() or "unknown"
        
        # Common labels
        common_labels = {
            "operation": operation,
            "tenant_id": tenant_id,
            "success": str(success).lower()
        }
        common_labels.update(labels)
        
        # Record memory operation
        self._memory_operations_counter.add(1, common_labels)
        
        logger.debug(f"Recorded memory operation: {operation}, duration: {duration}s, success: {success}")
    
    def record_vector_operation(self, operation: str, duration: float, 
                              tenant_id: Optional[str] = None, 
                              success: bool = True, **labels):
        """Record vector operation metrics"""
        if not self._vector_operations_counter:
            return
        
        # Get tenant ID from context if not provided
        if not tenant_id:
            tenant_id = get_current_tenant() or "unknown"
        
        # Common labels
        common_labels = {
            "operation": operation,
            "tenant_id": tenant_id,
            "success": str(success).lower()
        }
        common_labels.update(labels)
        
        # Record vector operation
        self._vector_operations_counter.add(1, common_labels)
        
        logger.debug(f"Recorded vector operation: {operation}, duration: {duration}s, success: {success}")
    
    def update_queue_depth(self, queue_name: str, depth: int, 
                          tenant_id: Optional[str] = None):
        """Update queue depth gauge"""
        if not self._queue_depth_gauge:
            return
        
        # Get tenant ID from context if not provided
        if not tenant_id:
            tenant_id = get_current_tenant() or "unknown"
        
        labels = {
            "queue_name": queue_name,
            "tenant_id": tenant_id
        }
        
        # Update queue depth
        self._queue_depth_gauge.add(depth, labels)
        
        logger.debug(f"Updated queue depth: {queue_name} = {depth}")
    
    def update_active_sessions(self, session_count: int, 
                             tenant_id: Optional[str] = None):
        """Update active sessions gauge"""
        if not self._active_sessions_gauge:
            return
        
        # Get tenant ID from context if not provided
        if not tenant_id:
            tenant_id = get_current_tenant() or "unknown"
        
        labels = {
            "tenant_id": tenant_id
        }
        
        # Update active sessions
        self._active_sessions_gauge.add(session_count, labels)
        
        logger.debug(f"Updated active sessions: {session_count}")
    
    def record_error(self, error_type: str, component: str, 
                    tenant_id: Optional[str] = None, **labels):
        """Record error metrics"""
        if not self._error_counter:
            return
        
        # Get tenant ID from context if not provided
        if not tenant_id:
            tenant_id = get_current_tenant() or "unknown"
        
        # Common labels
        common_labels = {
            "error_type": error_type,
            "component": component,
            "tenant_id": tenant_id
        }
        common_labels.update(labels)
        
        # Record error
        self._error_counter.add(1, common_labels)
        
        logger.debug(f"Recorded error: {error_type} in {component}")
    
    def record_custom_metric(self, metric_name: str, value: float, 
                           labels: Dict[str, str], 
                           metric_type: MetricType = MetricType.COUNTER):
        """Record custom metric"""
        if not self.meter:
            return
        
        # Add tenant ID to labels if not present
        if "tenant_id" not in labels:
            tenant_id = get_current_tenant()
            if tenant_id:
                labels["tenant_id"] = tenant_id
        
        # Create or get metric based on type
        if metric_type == MetricType.COUNTER:
            metric = self.meter.create_counter(
                name=f"forge1_custom_{metric_name}",
                description=f"Custom metric: {metric_name}",
                unit="1"
            )
            metric.add(value, labels)
        elif metric_type == MetricType.HISTOGRAM:
            metric = self.meter.create_histogram(
                name=f"forge1_custom_{metric_name}",
                description=f"Custom metric: {metric_name}",
                unit="1"
            )
            metric.record(value, labels)
        elif metric_type == MetricType.GAUGE:
            metric = self.meter.create_up_down_counter(
                name=f"forge1_custom_{metric_name}",
                description=f"Custom metric: {metric_name}",
                unit="1"
            )
            metric.add(value, labels)
        
        logger.debug(f"Recorded custom metric: {metric_name} = {value}")
    
    def get_metric_definitions(self) -> List[MetricDefinition]:
        """Get all metric definitions"""
        return [
            MetricDefinition(
                name="forge1_tasks_total",
                description="Total number of tasks executed",
                unit="1",
                metric_type=MetricType.COUNTER,
                labels=["task_type", "tenant_id", "success"]
            ),
            MetricDefinition(
                name="forge1_task_duration_seconds",
                description="Task execution duration",
                unit="s",
                metric_type=MetricType.HISTOGRAM,
                labels=["task_type", "tenant_id", "success"]
            ),
            MetricDefinition(
                name="forge1_model_requests_total",
                description="Total number of model requests",
                unit="1",
                metric_type=MetricType.COUNTER,
                labels=["model", "tenant_id"]
            ),
            MetricDefinition(
                name="forge1_model_cost_dollars",
                description="Model usage cost",
                unit="USD",
                metric_type=MetricType.HISTOGRAM,
                labels=["model", "tenant_id"]
            ),
            MetricDefinition(
                name="forge1_policy_decisions_total",
                description="Total number of policy decisions",
                unit="1",
                metric_type=MetricType.COUNTER,
                labels=["decision", "resource", "tenant_id"]
            ),
            MetricDefinition(
                name="forge1_memory_operations_total",
                description="Total number of memory operations",
                unit="1",
                metric_type=MetricType.COUNTER,
                labels=["operation", "tenant_id", "success"]
            ),
            MetricDefinition(
                name="forge1_vector_operations_total",
                description="Total number of vector operations",
                unit="1",
                metric_type=MetricType.COUNTER,
                labels=["operation", "tenant_id", "success"]
            ),
            MetricDefinition(
                name="forge1_queue_depth",
                description="Current queue depth",
                unit="1",
                metric_type=MetricType.GAUGE,
                labels=["queue_name", "tenant_id"]
            ),
            MetricDefinition(
                name="forge1_active_sessions",
                description="Number of active sessions",
                unit="1",
                metric_type=MetricType.GAUGE,
                labels=["tenant_id"]
            ),
            MetricDefinition(
                name="forge1_errors_total",
                description="Total number of errors",
                unit="1",
                metric_type=MetricType.COUNTER,
                labels=["error_type", "component", "tenant_id"]
            )
        ]

# Global metrics instance
forge_metrics = ForgeMetrics()