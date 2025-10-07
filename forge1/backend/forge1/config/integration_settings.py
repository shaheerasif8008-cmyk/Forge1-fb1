"""Lightweight integration settings manager for Forge1.

The original project references a more fully featured configuration system that
is not included in this workspace. This module provides minimal defaults so the
backend can start in a local/testing environment.
"""

from __future__ import annotations

from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict


class IntegrationType(Enum):
    POLICY = "policy"
    VECTOR = "vector"
    METERING = "metering"
    OBSERVABILITY = "observability"


class _SettingsManager:
    def __init__(self) -> None:
        self._configs: Dict[IntegrationType, Any] = {
            IntegrationType.POLICY: SimpleNamespace(
                server_url="http://localhost:8181",
                timeout_seconds=5,
                verify_ssl=False,
                auth_token=None,
                policy_bundle_url=None,
                policy_bundle_path=None,
                retry_attempts=2,
            ),
            IntegrationType.VECTOR: SimpleNamespace(
                scheme="http",
                host="localhost",
                port=8080,
                auth_client_secret=None,
                timeout_config=(5, 5),
                additional_headers={},
                startup_period=5,
            ),
            IntegrationType.METERING: SimpleNamespace(
                api_endpoint="http://localhost:8889",
                api_key=None,
                timeout_seconds=5,
                kafka_brokers=None,
                kafka_topic="forge1-usage-events",
                retry_attempts=3,
                batch_size=50,
                flush_interval_seconds=5,
                enable_kafka=False,
            ),
            IntegrationType.OBSERVABILITY: {
                "opentelemetry": SimpleNamespace(
                    service_name="forge1-backend",
                    service_version="1.0.0",
                    jaeger_endpoint=None,
                    otlp_endpoint=None,
                    instrumentation_config={
                        "fastapi": True,
                        "celery": True,
                        "redis": True,
                        "psycopg2": True,
                        "requests": True,
                        "httpx": True,
                    },
                ),
                "azure_monitor": SimpleNamespace(connection_string=None),
            },
        }

    def get_config(self, integration_type: IntegrationType) -> Any:
        return self._configs[integration_type]

    def update_config(self, integration_type: IntegrationType, config: Any) -> None:
        self._configs[integration_type] = config


settings_manager = _SettingsManager()
