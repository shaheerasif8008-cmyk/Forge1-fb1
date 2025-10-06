"""
OpenTelemetry Integration for LlamaIndex and workflows-py

Provides comprehensive tracing, metrics, and observability for all workflow
and tool operations with proper tenant/employee context tagging.
"""

import logging
import time
from typing import Dict, Any, Optional
from contextlib import contextmanager

from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

class Forge1Tracer:
    """Forge1-specific OpenTelemetry tracer with context awareness"""
    
    def __init__(self, service_name: str = "forge1-workflows"):
        self.service_name = service_name
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
        
        # Initialize metrics
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize custom metrics"""
        self.workflow_duration = self.meter.create_histogram(
            name="workflow_duration_seconds",
            description="Duration of workflow execution",
            unit="s"
        )
        
        self.tool_duration = self.meter.create_histogram(
            name="tool_duration_seconds", 
            description="Duration of tool execution",
            unit="s"
        )
        
        self.workflow_counter = self.meter.create_counter(
            name="workflows_total",
            description="Total number of workflows executed"
        )
        
        self.tool_counter = self.meter.create_counter(
            name="tools_total",
            description="Total number of tools executed"
        )
    
    @contextmanager
    def trace_workflow(self, workflow_id: str, tenant_id: str, employee_id: str, **kwargs):
        """Trace workflow execution"""
        with self.tracer.start_as_current_span(
            f"workflow.{workflow_id}",
            attributes={
                "workflow.id": workflow_id,
                "tenant.id": tenant_id,
                "employee.id": employee_id,
                "service.name": self.service_name,
                **kwargs
            }
        ) as span:
            start_time = time.time()
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            finally:
                duration = time.time() - start_time
                self.workflow_duration.record(duration, {
                    "tenant_id": tenant_id,
                    "employee_id": employee_id,
                    "workflow_id": workflow_id
                })
                self.workflow_counter.add(1, {
                    "tenant_id": tenant_id,
                    "employee_id": employee_id,
                    "status": "success" if span.status.status_code == StatusCode.OK else "error"
                })
    
    @contextmanager
    def trace_tool(self, tool_name: str, tenant_id: str, employee_id: str, **kwargs):
        """Trace tool execution"""
        with self.tracer.start_as_current_span(
            f"tool.{tool_name}",
            attributes={
                "tool.name": tool_name,
                "tenant.id": tenant_id,
                "employee.id": employee_id,
                "service.name": self.service_name,
                **kwargs
            }
        ) as span:
            start_time = time.time()
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            finally:
                duration = time.time() - start_time
                self.tool_duration.record(duration, {
                    "tenant_id": tenant_id,
                    "employee_id": employee_id,
                    "tool_name": tool_name
                })
                self.tool_counter.add(1, {
                    "tenant_id": tenant_id,
                    "employee_id": employee_id,
                    "tool_name": tool_name,
                    "status": "success" if span.status.status_code == StatusCode.OK else "error"
                })

# Global tracer instance
forge1_tracer = Forge1Tracer()
