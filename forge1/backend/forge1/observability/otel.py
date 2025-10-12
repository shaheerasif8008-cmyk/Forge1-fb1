"""Lightweight OpenTelemetry bootstrap for Forge1 services."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from opentelemetry import metrics, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    MetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

try:  # optional instrumentations that may not be present in all environments
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
except ImportError:  # pragma: no cover - dependency optional for non-api runtimes
    FastAPIInstrumentor = None  # type: ignore

try:
    from opentelemetry.instrumentation.celery import CeleryInstrumentor
except ImportError:  # pragma: no cover
    CeleryInstrumentor = None  # type: ignore

try:
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
except ImportError:  # pragma: no cover
    RequestsInstrumentor = None  # type: ignore

try:
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
except ImportError:  # pragma: no cover
    HTTPXClientInstrumentor = None  # type: ignore

try:
    from opentelemetry.instrumentation.redis import RedisInstrumentor
except ImportError:  # pragma: no cover
    RedisInstrumentor = None  # type: ignore

try:
    from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
except ImportError:  # pragma: no cover
    Psycopg2Instrumentor = None  # type: ignore

from forge1.core.logging_config import init_logger

logger = init_logger("forge1.observability.otel")


@dataclass
class ObservabilityConfig:
    """Configuration controlling OpenTelemetry bootstrap."""

    service_name: str = field(
        default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "forge1-backend"),
    )
    service_version: str = field(
        default_factory=lambda: os.getenv("FORGE1_VERSION", "0.1.0"),
    )
    environment: str = field(
        default_factory=lambda: os.getenv("FORGE1_ENVIRONMENT", "development"),
    )
    otlp_endpoint: str = field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
    )
    otlp_headers: str = field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_HEADERS", ""),
    )
    otlp_insecure: bool = field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() in {"1", "true", "yes"},
    )
    metrics_enabled: bool = field(
        default_factory=lambda: os.getenv("FORGE1_OTEL_ENABLE_METRICS", "true").lower() in {"1", "true", "yes"},
    )
    metrics_export_interval_seconds: float = field(
        default_factory=lambda: float(os.getenv("FORGE1_OTEL_METRICS_INTERVAL", "30")),
    )
    enable_console_exporter: bool = field(
        default_factory=lambda: os.getenv("FORGE1_OTEL_CONSOLE_EXPORTER", "false").lower() in {"1", "true", "yes"},
    )

    def resource(self) -> Resource:
        return Resource.create(
            {
                "service.name": self.service_name,
                "service.version": self.service_version,
                "service.environment": self.environment,
            }
        )

    def headers_dict(self) -> Dict[str, str]:
        if not self.otlp_headers:
            return {}
        headers: Dict[str, str] = {}
        for pair in self.otlp_headers.split(","):
            if not pair:
                continue
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            headers[key.strip()] = value.strip()
        return headers


@dataclass
class ObservabilityState:
    """Runtime handles for the configured observability stack."""

    config: ObservabilityConfig
    tracer_provider: TracerProvider
    span_processor: BatchSpanProcessor
    meter_provider: Optional[MeterProvider]
    metric_reader: Optional[PeriodicExportingMetricReader]
    instrumented: Dict[str, bool]
    app: Optional[Any] = None


_CURRENT_STATE: Optional[ObservabilityState] = None


def get_current_observability_state() -> Optional[ObservabilityState]:
    return _CURRENT_STATE


def _build_span_exporter(config: ObservabilityConfig, span_exporter: Optional[SpanExporter]) -> SpanExporter:
    if span_exporter is not None:
        return span_exporter
    return OTLPSpanExporter(
        endpoint=config.otlp_endpoint,
        headers=config.headers_dict(),
        insecure=config.otlp_insecure,
    )


def _build_metric_exporter(config: ObservabilityConfig, metric_exporter: Optional[MetricExporter]) -> Optional[MetricExporter]:
    if not config.metrics_enabled:
        return None
    if metric_exporter is not None:
        return metric_exporter
    return OTLPMetricExporter(
        endpoint=config.otlp_endpoint,
        headers=config.headers_dict(),
        insecure=config.otlp_insecure,
    )


def _instrument_requests() -> bool:
    if RequestsInstrumentor is None:  # pragma: no cover - optional dependency
        return False
    RequestsInstrumentor().instrument()
    return True


def _instrument_httpx() -> bool:
    if HTTPXClientInstrumentor is None:  # pragma: no cover - optional dependency
        return False
    HTTPXClientInstrumentor().instrument()
    return True


def _instrument_redis() -> bool:
    if RedisInstrumentor is None:  # pragma: no cover - optional dependency
        return False
    RedisInstrumentor().instrument()
    return True


def _instrument_psycopg() -> bool:
    if Psycopg2Instrumentor is None:  # pragma: no cover - optional dependency
        return False
    Psycopg2Instrumentor().instrument()
    return True


def _instrument_celery() -> bool:
    if CeleryInstrumentor is None:  # pragma: no cover - optional dependency
        return False
    CeleryInstrumentor().instrument()
    return True


def setup_observability(
    *,
    app: Optional[Any] = None,
    celery_app: Optional[Any] = None,
    config: Optional[ObservabilityConfig] = None,
    span_exporter: Optional[SpanExporter] = None,
    metric_exporter: Optional[MetricExporter] = None,
) -> ObservabilityState:
    """Configure OpenTelemetry exporters and auto-instrumentation.

    The function is idempotent: repeated calls return the existing state unless
    :func:`shutdown_observability` is invoked between calls.
    """

    global _CURRENT_STATE

    if _CURRENT_STATE is not None:
        return _CURRENT_STATE

    config = config or ObservabilityConfig()
    resource = config.resource()

    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    exporter = _build_span_exporter(config, span_exporter)
    span_processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(span_processor)

    if config.enable_console_exporter:
        tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    meter_provider: Optional[MeterProvider] = None
    metric_reader: Optional[PeriodicExportingMetricReader] = None
    exporter_metrics = _build_metric_exporter(config, metric_exporter)
    if exporter_metrics is not None:
        metric_reader = PeriodicExportingMetricReader(
            exporter_metrics,
            export_interval_millis=int(config.metrics_export_interval_seconds * 1000),
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
    else:
        meter_provider = MeterProvider(resource=resource)
        metrics.set_meter_provider(meter_provider)

    instrumented: Dict[str, bool] = {}

    if app is not None and FastAPIInstrumentor is not None:
        FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)
        instrumented["fastapi"] = True
    elif FastAPIInstrumentor is not None:
        FastAPIInstrumentor().instrument()
        instrumented["fastapi"] = True
    else:
        instrumented["fastapi"] = False

    instrumented["requests"] = _instrument_requests()
    instrumented["httpx"] = _instrument_httpx()
    instrumented["redis"] = _instrument_redis()
    instrumented["psycopg"] = _instrument_psycopg()
    instrumented["celery"] = False
    if celery_app is not None:
        instrumented["celery"] = _instrument_celery()
    else:
        # We still instrument Celery globally when available because tasks may
        # be initialized lazily in other modules.
        instrumented["celery"] = _instrument_celery()

    logger.info(
        "OpenTelemetry configured",
        extra={
            "service": config.service_name,
            "environment": config.environment,
            "instrumented": instrumented,
        },
    )

    state = ObservabilityState(
        config=config,
        tracer_provider=tracer_provider,
        span_processor=span_processor,
        meter_provider=meter_provider,
        metric_reader=metric_reader,
        instrumented=instrumented,
        app=app,
    )
    _CURRENT_STATE = state
    return state


def shutdown_observability(state: Optional[ObservabilityState] = None) -> None:
    """Flush exporters and uninstrument libraries."""

    global _CURRENT_STATE

    state = state or _CURRENT_STATE
    if state is None:
        return

    try:
        state.span_processor.shutdown()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Span processor shutdown failed", extra={"error": str(exc)})

    if state.metric_reader is not None:
        try:
            state.metric_reader.shutdown()
        except Exception as exc:  # pragma: no cover
            logger.warning("Metric reader shutdown failed", extra={"error": str(exc)})

    if state.app is not None and FastAPIInstrumentor is not None:
        try:
            FastAPIInstrumentor().uninstrument_app(state.app)
        except Exception:  # pragma: no cover - fastapi instrumentation may not support
            pass
    elif FastAPIInstrumentor is not None:
        try:
            FastAPIInstrumentor().uninstrument()
        except Exception:  # pragma: no cover
            pass

    if state.instrumented.get("requests") and RequestsInstrumentor is not None:
        RequestsInstrumentor().uninstrument()
    if state.instrumented.get("httpx") and HTTPXClientInstrumentor is not None:
        HTTPXClientInstrumentor().uninstrument()
    if state.instrumented.get("redis") and RedisInstrumentor is not None:
        RedisInstrumentor().uninstrument()
    if state.instrumented.get("psycopg") and Psycopg2Instrumentor is not None:
        Psycopg2Instrumentor().uninstrument()
    if state.instrumented.get("celery") and CeleryInstrumentor is not None:
        try:
            CeleryInstrumentor().uninstrument()
        except TypeError:
            # Older versions require explicit app; ignore for noop instrumentation
            pass

    _CURRENT_STATE = None

