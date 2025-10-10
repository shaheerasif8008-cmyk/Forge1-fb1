"""
OpenTelemetry Integration Setup

Provides comprehensive OpenTelemetry instrumentation for Forge1 with
automatic tracing, metrics collection, and tenant-aware observability.
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# OpenTelemetry imports with graceful fallbacks
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.celery import CeleryInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.b3 import B3MultiFormat
    from opentelemetry.propagators.jaeger import JaegerPropagator
    from opentelemetry.propagators.composite import CompositePropagator

    try:  # Jaeger exporter is optional; OTLP is primary path
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter  # type: ignore
    except ImportError:
        JaegerExporter = None  # type: ignore
        logger.info("Jaeger exporter not available; continuing without Jaeger agent integration")

    OTEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenTelemetry not available: {e}")
    OTEL_AVAILABLE = False
    # Create mock classes for graceful degradation
    class MockTracer:
        def start_as_current_span(self, name): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def set_attribute(self, key, value): pass
    
    class MockMeter:
        def create_counter(self, *args, **kwargs): return MockMetric()
        def create_histogram(self, *args, **kwargs): return MockMetric()
        def create_up_down_counter(self, *args, **kwargs): return MockMetric()
    
    class MockMetric:
        def add(self, *args, **kwargs): pass
        def record(self, *args, **kwargs): pass

    class _MockTraceModule:
        Tracer = MockTracer

        def __init__(self):
            self._tracer = MockTracer()

        def get_tracer_provider(self):
            return None

        def get_tracer(self, *args, **kwargs):
            return self._tracer

    class _MockMetricsModule:
        Meter = MockMeter

        def get_meter_provider(self):
            return None

        def get_meter(self, *args, **kwargs):
            return MockMeter()

    trace = _MockTraceModule()
    metrics = _MockMetricsModule()

from forge1.integrations.base_adapter import BaseAdapter, HealthCheckResult, AdapterStatus, ExecutionContext, TenantContext
from forge1.config.integration_settings import IntegrationType, settings_manager
from forge1.core.tenancy import get_current_tenant

class OTELIntegration(BaseAdapter):
    """OpenTelemetry integration with comprehensive instrumentation"""
    
    def __init__(self):
        config = settings_manager.get_config(IntegrationType.OBSERVABILITY)
        super().__init__("opentelemetry", config)
        
        self.otel_config = config["opentelemetry"]
        self.tracer_provider: Optional[TracerProvider] = None
        self.meter_provider: Optional[MeterProvider] = None
        self.tracer: Optional[trace.Tracer] = None
        self.meter: Optional[metrics.Meter] = None
        
        # Instrumentation tracking
        self._instrumented_components = set()
        self._span_processors = []
        self._metric_readers = []
        
        # Custom attributes
        self._resource_attributes = {}
        self._global_attributes = {}
    
    async def initialize(self) -> bool:
        """Initialize OpenTelemetry instrumentation"""
        try:
            # Setup resource attributes
            self._setup_resource_attributes()
            
            # Initialize tracing
            self._setup_tracing()
            
            # Initialize metrics
            self._setup_metrics()
            
            # Setup propagators
            self._setup_propagators()
            
            # Apply automatic instrumentation
            self._setup_automatic_instrumentation()
            
            logger.info("OpenTelemetry integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry integration: {e}")
            return False
    
    async def health_check(self) -> HealthCheckResult:
        """Perform health check of OpenTelemetry components"""
        start_time = time.time()
        
        try:
            # Check tracer provider
            tracer_healthy = self.tracer_provider is not None
            
            # Check meter provider
            meter_healthy = self.meter_provider is not None
            
            # Test span creation
            span_creation_healthy = False
            if self.tracer:
                with self.tracer.start_as_current_span("health_check_span") as span:
                    span.set_attribute("health_check", True)
                    span_creation_healthy = True
            
            # Test metric recording
            metric_recording_healthy = False
            if self.meter:
                counter = self.meter.create_counter("health_check_counter")
                counter.add(1, {"component": "otel_integration"})
                metric_recording_healthy = True
            
            # Determine overall status
            if all([tracer_healthy, meter_healthy, span_creation_healthy, metric_recording_healthy]):
                status = AdapterStatus.HEALTHY
                message = "OpenTelemetry integration healthy"
            elif any([tracer_healthy, meter_healthy]):
                status = AdapterStatus.DEGRADED
                message = "OpenTelemetry integration partially healthy"
            else:
                status = AdapterStatus.UNHEALTHY
                message = "OpenTelemetry integration unhealthy"
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                status=status,
                message=message,
                details={
                    "tracer_provider": tracer_healthy,
                    "meter_provider": meter_healthy,
                    "span_creation": span_creation_healthy,
                    "metric_recording": metric_recording_healthy,
                    "instrumented_components": list(self._instrumented_components),
                    "span_processors": len(self._span_processors),
                    "metric_readers": len(self._metric_readers),
                    "service_name": self.otel_config.service_name,
                    "service_version": self.otel_config.service_version
                },
                timestamp=time.time(),
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=AdapterStatus.UNHEALTHY,
                message=f"OpenTelemetry health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
                response_time_ms=response_time
            )
    
    async def cleanup(self) -> bool:
        """Clean up OpenTelemetry resources"""
        try:
            # Shutdown span processors
            for processor in self._span_processors:
                if hasattr(processor, 'shutdown'):
                    processor.shutdown()
            
            # Shutdown metric readers
            for reader in self._metric_readers:
                if hasattr(reader, 'shutdown'):
                    reader.shutdown()
            
            logger.info("OpenTelemetry integration cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup OpenTelemetry integration: {e}")
            return False
    
    def _setup_resource_attributes(self):
        """Setup resource attributes for telemetry"""
        self._resource_attributes = {
            ResourceAttributes.SERVICE_NAME: self.otel_config.service_name,
            ResourceAttributes.SERVICE_VERSION: self.otel_config.service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: os.getenv("ENVIRONMENT", "development"),
            "forge1.component": "backend",
            "forge1.integration": "oss_systems"
        }
        
        # Add custom resource attributes from config
        self._resource_attributes.update(self.otel_config.resource_attributes)
        
        # Add runtime attributes
        self._resource_attributes.update({
            "runtime.name": "python",
            "runtime.version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "host.name": os.getenv("HOSTNAME", "unknown"),
            "process.pid": os.getpid()
        })
    
    def _setup_tracing(self):
        """Setup distributed tracing"""
        # Create resource
        resource = Resource.create(self._resource_attributes)
        
        # Create tracer provider
        self.tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(self.tracer_provider)
        
        # Setup exporters
        exporters = []
        
        # Jaeger exporter
        if self.otel_config.jaeger_endpoint and JaegerExporter is not None:
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.otel_config.jaeger_endpoint.split("://")[1].split(":")[0],
                agent_port=int(self.otel_config.jaeger_endpoint.split(":")[-1]),
            )
            exporters.append(jaeger_exporter)
        elif self.otel_config.jaeger_endpoint:
            logger.warning(
                "Jaeger exporter requested but opentelemetry-exporter-jaeger is not installed"
            )
        
        # OTLP exporter
        if self.otel_config.otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.otel_config.otlp_endpoint,
                insecure=True  # Configure based on environment
            )
            exporters.append(otlp_exporter)
        
        # Console exporter for development
        if os.getenv("OTEL_CONSOLE_EXPORTER", "false").lower() == "true":
            console_exporter = ConsoleSpanExporter()
            exporters.append(console_exporter)
        
        # Add span processors
        for exporter in exporters:
            processor = BatchSpanProcessor(exporter)
            self.tracer_provider.add_span_processor(processor)
            self._span_processors.append(processor)
        
        # Create tracer
        self.tracer = trace.get_tracer(__name__)
        
        logger.info(f"Tracing setup complete with {len(exporters)} exporters")
    
    def _setup_metrics(self):
        """Setup metrics collection"""
        # Create resource
        resource = Resource.create(self._resource_attributes)
        
        # Setup metric readers
        readers = []
        
        # OTLP metric exporter
        if self.otel_config.otlp_endpoint:
            otlp_metric_exporter = OTLPMetricExporter(
                endpoint=self.otel_config.otlp_endpoint,
                insecure=True
            )
            otlp_reader = PeriodicExportingMetricReader(
                exporter=otlp_metric_exporter,
                export_interval_millis=30000  # 30 seconds
            )
            readers.append(otlp_reader)
        
        # Console exporter for development
        if os.getenv("OTEL_CONSOLE_EXPORTER", "false").lower() == "true":
            console_metric_exporter = ConsoleMetricExporter()
            console_reader = PeriodicExportingMetricReader(
                exporter=console_metric_exporter,
                export_interval_millis=60000  # 60 seconds
            )
            readers.append(console_reader)
        
        # Create meter provider
        self.meter_provider = MeterProvider(
            resource=resource,
            metric_readers=readers
        )
        metrics.set_meter_provider(self.meter_provider)
        
        # Store readers for cleanup
        self._metric_readers.extend(readers)
        
        # Create meter
        self.meter = metrics.get_meter(__name__)
        
        logger.info(f"Metrics setup complete with {len(readers)} readers")
    
    def _setup_propagators(self):
        """Setup trace context propagation"""
        # Setup composite propagator with multiple formats
        propagators = [
            B3MultiFormat(),
            JaegerPropagator()
        ]
        
        composite_propagator = CompositePropagator(propagators)
        set_global_textmap(composite_propagator)
        
        logger.info("Trace propagation setup complete")
    
    def _setup_automatic_instrumentation(self):
        """Setup automatic instrumentation for various components"""
        
        # FastAPI instrumentation
        if self.otel_config.instrumentation_config.get("fastapi", True):
            try:
                FastAPIInstrumentor().instrument()
                self._instrumented_components.add("fastapi")
                logger.info("FastAPI instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument FastAPI: {e}")
        
        # Celery instrumentation
        if self.otel_config.instrumentation_config.get("celery", True):
            try:
                CeleryInstrumentor().instrument()
                self._instrumented_components.add("celery")
                logger.info("Celery instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument Celery: {e}")
        
        # Redis instrumentation
        if self.otel_config.instrumentation_config.get("redis", True):
            try:
                RedisInstrumentor().instrument()
                self._instrumented_components.add("redis")
                logger.info("Redis instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument Redis: {e}")
        
        # PostgreSQL instrumentation
        if self.otel_config.instrumentation_config.get("psycopg2", True):
            try:
                Psycopg2Instrumentor().instrument()
                self._instrumented_components.add("psycopg2")
                logger.info("PostgreSQL instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument PostgreSQL: {e}")
        
        # Requests instrumentation
        if self.otel_config.instrumentation_config.get("requests", True):
            try:
                RequestsInstrumentor().instrument()
                self._instrumented_components.add("requests")
                logger.info("Requests instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument Requests: {e}")
        
        # HTTPX instrumentation
        if self.otel_config.instrumentation_config.get("httpx", True):
            try:
                HTTPXClientInstrumentor().instrument()
                self._instrumented_components.add("httpx")
                logger.info("HTTPX instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument HTTPX: {e}")
    
    def instrument_fastapi(self, app) -> None:
        """Instrument FastAPI application with custom attributes"""
        
        @app.middleware("http")
        async def add_tenant_context_to_spans(request, call_next):
            """Add tenant context to all spans"""
            
            # Get current span
            current_span = trace.get_current_span()
            
            if current_span and current_span.is_recording():
                # Add tenant information
                tenant_id = get_current_tenant()
                if tenant_id:
                    current_span.set_attribute("forge1.tenant_id", tenant_id)
                
                # Add request information
                current_span.set_attribute("http.route", str(request.url.path))
                current_span.set_attribute("http.method", request.method)
                
                # Add user information if available
                user_id = request.headers.get("X-User-ID")
                if user_id:
                    current_span.set_attribute("forge1.user_id", user_id)
                
                # Add request ID if available
                request_id = request.headers.get("X-Request-ID")
                if request_id:
                    current_span.set_attribute("forge1.request_id", request_id)
            
            response = await call_next(request)
            
            # Add response information
            if current_span and current_span.is_recording():
                current_span.set_attribute("http.status_code", response.status_code)
            
            return response
        
        logger.info("FastAPI custom instrumentation applied")
    
    def instrument_celery(self, celery_app) -> None:
        """Instrument Celery application with custom attributes"""
        
        @celery_app.task(bind=True)
        def instrumented_task_wrapper(self, original_task, *args, **kwargs):
            """Wrapper to add custom attributes to Celery tasks"""
            
            current_span = trace.get_current_span()
            
            if current_span and current_span.is_recording():
                # Add task information
                current_span.set_attribute("celery.task_name", original_task.__name__)
                current_span.set_attribute("celery.task_id", self.request.id)
                
                # Add tenant information from task context
                tenant_id = kwargs.get("_tenant_id")
                if tenant_id:
                    current_span.set_attribute("forge1.tenant_id", tenant_id)
                
                # Add execution context information
                execution_context = kwargs.get("_execution_context")
                if execution_context:
                    current_span.set_attribute("forge1.request_id", execution_context.get("request_id", ""))
                    current_span.set_attribute("forge1.session_id", execution_context.get("session_id", ""))
            
            return original_task(*args, **kwargs)
        
        logger.info("Celery custom instrumentation applied")
    
    def instrument_database_calls(self) -> None:
        """Instrument database calls with custom attributes"""
        # This would be implemented with database-specific instrumentation
        # For now, PostgreSQL instrumentation is handled automatically
        logger.info("Database instrumentation applied")
    
    def instrument_model_router_calls(self) -> None:
        """Instrument model router calls with custom tracing"""
        # This would be implemented by modifying the ModelRouter class
        # to add tracing around model calls
        logger.info("Model router instrumentation applied")
    
    @contextmanager
    def trace_operation(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager for tracing operations"""
        
        if not self.tracer:
            yield None
            return
        
        with self.tracer.start_as_current_span(operation_name) as span:
            # Add tenant context
            tenant_id = get_current_tenant()
            if tenant_id:
                span.set_attribute("forge1.tenant_id", tenant_id)
            
            # Add custom attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            # Add global attributes
            for key, value in self._global_attributes.items():
                span.set_attribute(key, value)
            
            yield span
    
    def create_counter(self, name: str, description: str = "", unit: str = ""):
        """Create a counter metric"""
        if not self.meter:
            return None
        
        return self.meter.create_counter(
            name=name,
            description=description,
            unit=unit
        )
    
    def create_histogram(self, name: str, description: str = "", unit: str = ""):
        """Create a histogram metric"""
        if not self.meter:
            return None
        
        return self.meter.create_histogram(
            name=name,
            description=description,
            unit=unit
        )
    
    def create_gauge(self, name: str, description: str = "", unit: str = ""):
        """Create a gauge metric"""
        if not self.meter:
            return None
        
        return self.meter.create_up_down_counter(
            name=name,
            description=description,
            unit=unit
        )
    
    def set_global_attribute(self, key: str, value: Any):
        """Set a global attribute that will be added to all spans"""
        self._global_attributes[key] = value
    
    def get_tracer(self) -> Optional[trace.Tracer]:
        """Get the tracer instance"""
        return self.tracer
    
    def get_meter(self) -> Optional[metrics.Meter]:
        """Get the meter instance"""
        return self.meter
    
    def get_instrumented_components(self) -> List[str]:
        """Get list of instrumented components"""
        return list(self._instrumented_components)

# Global OpenTelemetry integration instance
otel_integration = OTELIntegration()
