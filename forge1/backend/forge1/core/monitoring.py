"""Stub monitoring utilities for Forge1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from forge1.core.logging_config import init_logger

logger = init_logger("forge1.core.monitoring")


@dataclass
class MetricsCollector:
    namespace: str = "default"

    def increment(self, metric: str, value: int = 1, tags: Dict[str, Any] | None = None) -> None:
        raise NotImplementedError("stub")

    def record_metric(self, metric: str, value: float, tags: Dict[str, Any] | None = None) -> None:
        raise NotImplementedError("stub")


__all__ = ["MetricsCollector"]
