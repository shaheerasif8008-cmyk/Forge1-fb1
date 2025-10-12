from __future__ import annotations

import os

try:
    from fastapi import FastAPI  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    FastAPI = None  # type: ignore
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from forge1.observability import ObservabilityConfig, setup_observability, shutdown_observability


def test_setup_observability_exports_spans(monkeypatch):
    app = FastAPI() if FastAPI is not None else None
    exporter = InMemorySpanExporter()
    config = ObservabilityConfig(metrics_enabled=False)

    shutdown_observability()
    state = setup_observability(app=app, span_exporter=exporter, config=config)

    tracer = state.tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("observability-test"):
        pass

    state.span_processor.force_flush()
    spans = exporter.get_finished_spans()
    assert spans
    assert spans[0].name == "observability-test"

    shutdown_observability(state)


def test_observability_config_headers(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "authorization=token,foo=bar")
    config = ObservabilityConfig(metrics_enabled=False)
    headers = config.headers_dict()
    assert headers == {"authorization": "token", "foo": "bar"}
    shutdown_observability()


def test_observability_idempotent(monkeypatch):
    os.environ.pop("OTEL_EXPORTER_OTLP_HEADERS", None)
    config = ObservabilityConfig(metrics_enabled=False)
    state_a = setup_observability(config=config)
    state_b = setup_observability(config=config)
    assert state_a is state_b
    shutdown_observability(state_a)
