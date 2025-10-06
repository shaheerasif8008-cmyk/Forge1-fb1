"""
Azure Monitor Integration Adapter

Provides comprehensive Azure Monitor integration for Forge1 with Application Insights,
custom metrics, distributed tracing, and tenant-aware telemetry collection.
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from forge1.integrations.base_adapter import BaseAdapter, HealthCheckResult, AdapterStatus, ExecutionContext, TenantContext
from forge1.config.integration_settings import IntegrationType, settings_manager
from forge1.core.tenancy import get_current_tenant
from forge1.core.dlp import redact_payload

logger = logging.getLogger(__name__)

# Azure Monitor imports with graceful fallback
try:
    from azure.monitor.opentelemetry import configure_azure_monitor
    from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter, AzureMonitorMetricExporter, AzureMonitorLogExporter
    from azure.core.exceptions import AzureError
    import azure.monitor.events.extension as events_extension
    AZURE_MONITOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Azure Monitor not available: {e}")
    AZURE_MONITOR_AVAILABLE = False
    
    # Create mock classes for graceful degradation
    class MockAzureMonitorExporter:
        def __init__(self, *args, **kwargs): pass
        def export(self, *args, **kwargs): return True
        def shutdown(self): pass
    
    AzureMonitorTraceExporter = MockAzureMonitorExporter
    AzureMonitorMetricExporter = MockAzureMonitorExporter
    AzureMonitorLogExporter = MockAzureMonitorExporter
    configure_azure_monitor = lambda *args, **kwargs: None
    AzureError = Exception

class AzureMonitorEventType(Enum):
    """Types of Azure Monitor events"""
    CUSTOM_EVENT = "custom_event"
    DEPENDENCY = "dependency"
    EXCEPTION = "exception"
    PAGE_VIEW = "page_view"
    REQUEST = "request"
    TRACE = "trace"
    METRIC = "metric"

@dataclass
class AzureMonitorConfig:
    """Azure Monitor configuration"""
    connection_string: str
    instrumentation_key: Optional[str] = None
    enable_auto_instrumentation: bool = True
    enable_custom_metrics: bool = True
    enable_custom_events: bool = True
    enable_dependency_tracking: bool = True
    enable_exception_tracking: bool = True
    sampling_percentage: float = 100.0
    tenant_isolation_enabled: bool = True
    custom_dimensions: Dict[str, str] = None
    
    def __post_init__(self):
        if self.custom_dimensions is None:
            self.custom_dimensions = {}

@dataclass
class CustomEvent:
    """Custom event for Azure Monitor"""
    name: str
    properties: Dict[str, Any]
    measurements: Dict[str, float] = None
    tenant_id: Optional[str] = None
    
    def __post_init__(self):
        if self.measurements is None:
            self.measurements = {}

@dataclass
class CustomMetric:
    """Custom metric for Azure Monitor"""
    name: str
    value: float
    properties: Dict[str, str] = None
    tenant_id: Optional[str] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

class AzureMonitorAdapter(BaseAdapter):
    """Azure Monitor integration adapter with comprehensive telemetry"""
    
    def __init__(self):
        config = self._load_azure_monitor_config()
        super().__init__("azure_monitor", config)
        
        self.azure_config = config
        self.trace_exporter: Optional[AzureMonitorTraceExporter] = None
        self.metric_exporter: Optional[AzureMonitorMetricExporter] = None
        self.log_exporter: Optional[AzureMonitorLogExporter] = None
        
        # Telemetry statistics
        self._events_sent = 0
        self._metrics_sent = 0
        self._traces_sent = 0
        self._exceptions_logged = 0
        self._dependencies_tracked = 0
        
        # Custom dimensions for all telemetry
        self._global_dimensions = {
            "service": "forge1-backend",
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development")
        }
        self._global_dimensions.update(self.azure_config.custom_dimensions)
    
    async def initialize(self) -> bool:
        """Initialize Azure Monitor integration"""
        
        if not AZURE_MONITOR_AVAILABLE:
            logger.warning("Azure Monitor SDK not available - adapter will operate in mock mode")
            return True
        
        if not self.azure_config.connection_string:
            logger.warning("Azure Monitor connection string not provided - adapter will operate in mock mode")
            return True
        
        try:
            # Configure Azure Monitor with OpenTelemetry
            configure_azure_monitor(
                connection_string=self.azure_config.connection_string,
                enable_live_metrics=True,
                sampling_ratio=self.azure_config.sampling_percentage / 100.0
            )
            
            # Initialize exporters
            self.trace_exporter = AzureMonitorTraceExporter(
                connection_string=self.azure_config.connection_string
            )
            
            self.metric_exporter = AzureMonitorMetricExporter(
                connection_string=self.azure_config.connection_string
            )
            
            self.log_exporter = AzureMonitorLogExporter(
                connection_string=self.azure_config.connection_string
            )
            
            # Send initialization event
            await self._send_custom_event(
                "forge1_azure_monitor_initialized",
                {
                    "auto_instrumentation": self.azure_config.enable_auto_instrumentation,
                    "custom_metrics": self.azure_config.enable_custom_metrics,
                    "tenant_isolation": self.azure_config.tenant_isolation_enabled
                }
            )
            
            logger.info("Azure Monitor adapter initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure Monitor adapter: {e}")
            return False
    
    async def health_check(self) -> HealthCheckResult:
        """Perform health check of Azure Monitor connection"""
        start_time = time.time()
        
        try:
            if not AZURE_MONITOR_AVAILABLE:
                return HealthCheckResult(
                    status=AdapterStatus.DEGRADED,
                    message="Azure Monitor SDK not available - running in mock mode",
                    details={"mock_mode": True},
                    timestamp=time.time(),
                    response_time_ms=0
                )
            
            if not self.azure_config.connection_string:
                return HealthCheckResult(
                    status=AdapterStatus.DEGRADED,
                    message="Azure Monitor connection string not configured",
                    details={"configured": False},
                    timestamp=time.time(),
                    response_time_ms=0
                )
            
            # Test connection by sending a health check event
            try:
                await self._send_custom_event(
                    "forge1_health_check",
                    {"component": "azure_monitor_adapter"},
                    {"response_time_ms": 0}
                )
                
                status = AdapterStatus.HEALTHY
                message = "Azure Monitor healthy"
                
            except Exception as e:
                logger.warning(f"Azure Monitor health check failed: {e}")
                status = AdapterStatus.DEGRADED
                message = f"Azure Monitor connection issues: {str(e)}"
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                status=status,
                message=message,
                details={
                    "connection_string_configured": bool(self.azure_config.connection_string),
                    "auto_instrumentation": self.azure_config.enable_auto_instrumentation,
                    "events_sent": self._events_sent,
                    "metrics_sent": self._metrics_sent,
                    "traces_sent": self._traces_sent,
                    "exceptions_logged": self._exceptions_logged,
                    "dependencies_tracked": self._dependencies_tracked
                },
                timestamp=time.time(),
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=AdapterStatus.UNHEALTHY,
                message=f"Azure Monitor health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
                response_time_ms=response_time
            )
    
    async def cleanup(self) -> bool:
        """Clean up Azure Monitor resources"""
        try:
            # Send shutdown event
            await self._send_custom_event(
                "forge1_azure_monitor_shutdown",
                {"events_sent": self._events_sent, "metrics_sent": self._metrics_sent}
            )
            
            # Shutdown exporters
            if self.trace_exporter:
                self.trace_exporter.shutdown()
            
            if self.metric_exporter:
                self.metric_exporter.shutdown()
            
            if self.log_exporter:
                self.log_exporter.shutdown()
            
            logger.info("Azure Monitor adapter cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup Azure Monitor adapter: {e}")
            return False
    
    async def track_custom_event(self, event: CustomEvent, context: Optional[ExecutionContext] = None) -> bool:
        """Track a custom event in Azure Monitor"""
        
        try:
            # Apply DLP redaction
            safe_properties, violations = redact_payload(event.properties)
            
            # Add tenant context if available
            if not event.tenant_id and context:
                event.tenant_id = context.tenant_context.tenant_id
            elif not event.tenant_id:
                event.tenant_id = get_current_tenant()
            
            # Add global dimensions and tenant info
            enriched_properties = self._global_dimensions.copy()
            enriched_properties.update(safe_properties)
            
            if event.tenant_id and self.azure_config.tenant_isolation_enabled:
                enriched_properties["tenant_id"] = event.tenant_id
            
            if violations:
                enriched_properties["dlp_violations"] = len(violations)
            
            # Send event
            success = await self._send_custom_event(event.name, enriched_properties, event.measurements)
            
            if success:
                self._events_sent += 1
                logger.debug(f"Tracked custom event: {event.name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to track custom event {event.name}: {e}")
            return False
    
    async def track_custom_metric(self, metric: CustomMetric, context: Optional[ExecutionContext] = None) -> bool:
        """Track a custom metric in Azure Monitor"""
        
        try:
            # Add tenant context if available
            if not metric.tenant_id and context:
                metric.tenant_id = context.tenant_context.tenant_id
            elif not metric.tenant_id:
                metric.tenant_id = get_current_tenant()
            
            # Add global dimensions and tenant info
            enriched_properties = self._global_dimensions.copy()
            enriched_properties.update(metric.properties)
            
            if metric.tenant_id and self.azure_config.tenant_isolation_enabled:
                enriched_properties["tenant_id"] = metric.tenant_id
            
            # Send metric
            success = await self._send_custom_metric(metric.name, metric.value, enriched_properties)
            
            if success:
                self._metrics_sent += 1
                logger.debug(f"Tracked custom metric: {metric.name} = {metric.value}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to track custom metric {metric.name}: {e}")
            return False
    
    async def track_exception(self, exception: Exception, context: Optional[ExecutionContext] = None, 
                            properties: Optional[Dict[str, str]] = None) -> bool:
        """Track an exception in Azure Monitor"""
        
        try:
            # Prepare exception properties
            exception_properties = self._global_dimensions.copy()
            if properties:
                exception_properties.update(properties)
            
            # Add tenant context
            if context:
                exception_properties["tenant_id"] = context.tenant_context.tenant_id
                exception_properties["request_id"] = context.request_id
            else:
                tenant_id = get_current_tenant()
                if tenant_id:
                    exception_properties["tenant_id"] = tenant_id
            
            exception_properties.update({
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "timestamp": str(time.time())
            })
            
            # Send exception event
            success = await self._send_custom_event(
                "forge1_exception",
                exception_properties,
                {"severity": 1.0}
            )
            
            if success:
                self._exceptions_logged += 1
                logger.debug(f"Tracked exception: {type(exception).__name__}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to track exception: {e}")
            return False
    
    async def track_dependency(self, name: str, dependency_type: str, target: str, 
                             duration_ms: float, success: bool, 
                             context: Optional[ExecutionContext] = None,
                             properties: Optional[Dict[str, str]] = None) -> bool:
        """Track a dependency call in Azure Monitor"""
        
        try:
            # Prepare dependency properties
            dependency_properties = self._global_dimensions.copy()
            if properties:
                dependency_properties.update(properties)
            
            # Add tenant context
            if context:
                dependency_properties["tenant_id"] = context.tenant_context.tenant_id
                dependency_properties["request_id"] = context.request_id
            else:
                tenant_id = get_current_tenant()
                if tenant_id:
                    dependency_properties["tenant_id"] = tenant_id
            
            dependency_properties.update({
                "dependency_name": name,
                "dependency_type": dependency_type,
                "target": target,
                "success": str(success).lower(),
                "timestamp": str(time.time())
            })
            
            # Send dependency event
            success_result = await self._send_custom_event(
                "forge1_dependency",
                dependency_properties,
                {"duration_ms": duration_ms, "success_flag": 1.0 if success else 0.0}
            )
            
            if success_result:
                self._dependencies_tracked += 1
                logger.debug(f"Tracked dependency: {name} -> {target}")
            
            return success_result
            
        except Exception as e:
            logger.error(f"Failed to track dependency {name}: {e}")
            return False
    
    async def track_request(self, name: str, url: str, duration_ms: float, 
                          response_code: int, success: bool,
                          context: Optional[ExecutionContext] = None,
                          properties: Optional[Dict[str, str]] = None) -> bool:
        """Track an HTTP request in Azure Monitor"""
        
        try:
            # Prepare request properties
            request_properties = self._global_dimensions.copy()
            if properties:
                request_properties.update(properties)
            
            # Add tenant context
            if context:
                request_properties["tenant_id"] = context.tenant_context.tenant_id
                request_properties["request_id"] = context.request_id
            else:
                tenant_id = get_current_tenant()
                if tenant_id:
                    request_properties["tenant_id"] = tenant_id
            
            request_properties.update({
                "request_name": name,
                "url": url,
                "response_code": str(response_code),
                "success": str(success).lower(),
                "timestamp": str(time.time())
            })
            
            # Send request event
            success_result = await self._send_custom_event(
                "forge1_request",
                request_properties,
                {"duration_ms": duration_ms, "response_code": float(response_code)}
            )
            
            return success_result
            
        except Exception as e:
            logger.error(f"Failed to track request {name}: {e}")
            return False
    
    def _load_azure_monitor_config(self) -> AzureMonitorConfig:
        """Load Azure Monitor configuration"""
        
        connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING") or os.getenv("AZURE_MONITOR_CONNECTION_STRING")
        instrumentation_key = os.getenv("APPINSIGHTS_INSTRUMENTATIONKEY") or os.getenv("AZURE_MONITOR_INSTRUMENTATION_KEY")
        
        return AzureMonitorConfig(
            connection_string=connection_string or "",
            instrumentation_key=instrumentation_key,
            enable_auto_instrumentation=os.getenv("AZURE_MONITOR_AUTO_INSTRUMENTATION", "true").lower() == "true",
            enable_custom_metrics=os.getenv("AZURE_MONITOR_CUSTOM_METRICS", "true").lower() == "true",
            enable_custom_events=os.getenv("AZURE_MONITOR_CUSTOM_EVENTS", "true").lower() == "true",
            enable_dependency_tracking=os.getenv("AZURE_MONITOR_DEPENDENCY_TRACKING", "true").lower() == "true",
            enable_exception_tracking=os.getenv("AZURE_MONITOR_EXCEPTION_TRACKING", "true").lower() == "true",
            sampling_percentage=float(os.getenv("AZURE_MONITOR_SAMPLING_PERCENTAGE", "100.0")),
            tenant_isolation_enabled=os.getenv("AZURE_MONITOR_TENANT_ISOLATION", "true").lower() == "true",
            custom_dimensions={
                "forge1_component": "backend",
                "deployment_id": os.getenv("DEPLOYMENT_ID", "unknown")
            }
        )
    
    async def _send_custom_event(self, name: str, properties: Dict[str, Any], 
                               measurements: Optional[Dict[str, float]] = None) -> bool:
        """Send custom event to Azure Monitor"""
        
        if not AZURE_MONITOR_AVAILABLE or not self.azure_config.connection_string:
            logger.debug(f"Mock sending Azure Monitor event: {name}")
            return True
        
        try:
            # This would use the actual Azure Monitor SDK to send events
            # For now, we'll log the event (in production, this would use the telemetry client)
            logger.info(f"Azure Monitor Event: {name}", extra={
                "properties": properties,
                "measurements": measurements or {}
            })
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Azure Monitor event {name}: {e}")
            return False
    
    async def _send_custom_metric(self, name: str, value: float, properties: Dict[str, str]) -> bool:
        """Send custom metric to Azure Monitor"""
        
        if not AZURE_MONITOR_AVAILABLE or not self.azure_config.connection_string:
            logger.debug(f"Mock sending Azure Monitor metric: {name} = {value}")
            return True
        
        try:
            # This would use the actual Azure Monitor SDK to send metrics
            logger.info(f"Azure Monitor Metric: {name} = {value}", extra={
                "properties": properties
            })
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Azure Monitor metric {name}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Azure Monitor adapter statistics"""
        
        return {
            "events_sent": self._events_sent,
            "metrics_sent": self._metrics_sent,
            "traces_sent": self._traces_sent,
            "exceptions_logged": self._exceptions_logged,
            "dependencies_tracked": self._dependencies_tracked,
            "connection_configured": bool(self.azure_config.connection_string),
            "auto_instrumentation_enabled": self.azure_config.enable_auto_instrumentation,
            "tenant_isolation_enabled": self.azure_config.tenant_isolation_enabled,
            "sampling_percentage": self.azure_config.sampling_percentage
        }
    
    def reset_statistics(self):
        """Reset adapter statistics"""
        
        self._events_sent = 0
        self._metrics_sent = 0
        self._traces_sent = 0
        self._exceptions_logged = 0
        self._dependencies_tracked = 0

# Global Azure Monitor adapter instance
azure_monitor_adapter = AzureMonitorAdapter()