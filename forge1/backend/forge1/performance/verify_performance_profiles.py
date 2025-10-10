"""Performance profile verification utilities."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List

from forge1.core.logging_config import init_logger


logger = init_logger("forge1.performance.verification")

DEFAULT_THRESHOLDS = {"mean": 0.25, "p95": 0.6, "max": 1.2}


@dataclass
class PerformanceCheck:
    name: str
    passed: bool
    details: str


@dataclass
class PerformanceReport:
    checks: List[PerformanceCheck]

    @property
    def all_passed(self) -> bool:
        return all(check.passed for check in self.checks)


def _calculate_percentile(latencies: Iterable[float], percentile: float) -> float:
    values = sorted(latencies)
    if not values:
        return 0.0
    index = int(round((percentile / 100.0) * (len(values) - 1)))
    return values[index]


def verify_latency_profile(latencies: Iterable[float], thresholds: Dict[str, float]) -> PerformanceCheck:
    """Validate latency distribution against thresholds."""
    latencies = list(latencies)
    if not latencies:
        return PerformanceCheck("latency_profile", False, "No latency samples provided")

    metrics = {
        "mean": statistics.mean(latencies),
        "p95": _calculate_percentile(latencies, 95),
        "max": max(latencies),
    }
    violations = {k: v for k, v in metrics.items() if v > thresholds.get(k, float("inf"))}
    if violations:
        return PerformanceCheck(
            "latency_profile",
            False,
            f"Threshold breaches detected: {violations}",
        )
    return PerformanceCheck("latency_profile", True, f"Latency metrics within thresholds: {metrics}")


def verify_error_rate(error_events: int, total_requests: int, max_error_ratio: float = 0.02) -> PerformanceCheck:
    if total_requests <= 0:
        return PerformanceCheck("error_rate", False, "Total requests must be > 0")
    ratio = error_events / total_requests
    if ratio > max_error_ratio:
        return PerformanceCheck(
            "error_rate",
            False,
            f"Error ratio {ratio:.2%} exceeds limit of {max_error_ratio:.2%}",
        )
    return PerformanceCheck("error_rate", True, f"Error ratio {ratio:.2%} within limit {max_error_ratio:.2%}")


def run_performance_profile_suite(
    latencies: Iterable[float],
    error_events: int,
    total_requests: int,
    thresholds: Dict[str, float] | None = None,
) -> PerformanceReport:
    """Execute performance validation across latency and error metrics."""

    logger.info("Running performance profile verification")

    thresholds = thresholds or DEFAULT_THRESHOLDS
    checks = [
        verify_latency_profile(latencies, thresholds),
        verify_error_rate(error_events, total_requests),
    ]
    return PerformanceReport(checks=checks)


__all__ = [
    "PerformanceCheck",
    "PerformanceReport",
    "run_performance_profile_suite",
    "verify_latency_profile",
    "verify_error_rate",
]
