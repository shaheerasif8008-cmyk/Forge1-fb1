"""Phase 7 load and chaos simulation utilities."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from forge1.core.logging_config import init_logger


logger = init_logger("forge1.performance.phase7")


@dataclass
class WorkflowConfig:
    """Configuration for a simulated workflow."""

    name: str
    docs_per_day: int
    concurrency: int
    base_latency_mean: float
    base_latency_std: float
    queue_wait_shape: float
    queue_wait_scale: float
    target_p95: float


@dataclass
class ChaosConfig:
    """Configuration describing an injected chaos period."""

    name: str
    start_fraction: float
    duration_fraction: float
    failure_probability: float
    latency_penalty: float
    queue_penalty: float
    affects: Sequence[str]
    description: str


@dataclass
class QueueSample:
    time_index: int
    workflow: str
    depth: int


@dataclass
class ChaosEventRecord:
    name: str
    start_index: int
    end_index: int
    impacted_requests: int
    failures: int
    description: str


@dataclass
class DLQEntry:
    workflow: str
    tenant_id: str
    request_index: int
    reason: str


@dataclass
class AlertRecord:
    name: str
    severity: str
    message: str
    related_event: str | None = None


@dataclass
class WorkflowMetrics:
    name: str
    total_requests: int
    failures: int
    concurrency: int
    latencies: List[float] = field(default_factory=list)
    queue_waits: List[float] = field(default_factory=list)

    def add_sample(self, latency: float, queue_wait: float) -> None:
        self.latencies.append(latency)
        self.queue_waits.append(queue_wait)

    @property
    def successes(self) -> int:
        return self.total_requests - self.failures

    def percentile(self, values: Sequence[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = int(round((percentile / 100.0) * (len(ordered) - 1)))
        return ordered[index]

    def build_summary(self) -> Dict[str, float]:
        if not self.latencies:
            return {
                "avg_latency": 0.0,
                "p95_latency": 0.0,
                "p50_latency": 0.0,
                "queue_p95": 0.0,
                "throughput_per_min": 0.0,
                "success_rate": 0.0,
            }

        total_processing_time = sum(self.latencies) + sum(self.queue_waits)
        effective_runtime = total_processing_time / max(self.concurrency, 1)
        throughput_per_min = 0.0
        if effective_runtime > 0:
            throughput_per_min = (self.total_requests / effective_runtime) * 60.0

        avg_latency = statistics.mean(self.latencies)
        p95_latency = self.percentile(self.latencies, 95)
        p50_latency = self.percentile(self.latencies, 50)
        queue_p95 = self.percentile(self.queue_waits, 95)
        success_rate = (self.successes / self.total_requests) if self.total_requests else 0.0

        return {
            "avg_latency": avg_latency,
            "p95_latency": p95_latency,
            "p50_latency": p50_latency,
            "queue_p95": queue_p95,
            "throughput_per_min": throughput_per_min,
            "success_rate": success_rate,
        }


@dataclass
class SimulationResult:
    workflows: Dict[str, WorkflowMetrics]
    queue_depth_series: List[QueueSample]
    chaos_timeline: List[ChaosEventRecord]
    dlq_entries: List[DLQEntry]
    alerts: List[AlertRecord]

    def workflow_summary(self) -> Dict[str, Dict[str, float]]:
        return {
            name: metrics.build_summary()
            for name, metrics in self.workflows.items()
        }


class Phase7LoadSimulator:
    """Simulate the Phase 7 load, chaos, and recovery flow."""

    def __init__(
        self,
        workflows: Sequence[WorkflowConfig],
        chaos_events: Sequence[ChaosConfig],
        tenants: Sequence[str],
        seed: int = 1337,
    ) -> None:
        import random

        self._rng = random.Random(seed)
        self._workflows = list(workflows)
        self._chaos_events = list(chaos_events)
        self._tenants = list(tenants)

        self._total_requests = sum(cfg.docs_per_day for cfg in workflows)
        self._resolved_chaos = self._resolve_chaos_events()

    def _resolve_chaos_events(self) -> List[ChaosEventRecord]:
        resolved: List[ChaosEventRecord] = []
        for event in self._chaos_events:
            start_index = int(self._total_requests * event.start_fraction)
            duration = max(1, int(self._total_requests * event.duration_fraction))
            end_index = min(self._total_requests, start_index + duration)
            resolved.append(
                ChaosEventRecord(
                    name=event.name,
                    start_index=start_index,
                    end_index=end_index,
                    impacted_requests=0,
                    failures=0,
                    description=event.description,
                )
            )
        return resolved

    def _active_chaos(self, request_index: int, workflow_name: str) -> List[tuple[ChaosEventRecord, ChaosConfig]]:
        active: List[tuple[ChaosEventRecord, ChaosConfig]] = []
        for record, config in zip(self._resolved_chaos, self._chaos_events):
            if record.start_index <= request_index < record.end_index and workflow_name in config.affects:
                active.append((record, config))
        return active

    def run(self) -> SimulationResult:
        metrics: Dict[str, WorkflowMetrics] = {
            cfg.name: WorkflowMetrics(
                name=cfg.name,
                total_requests=cfg.docs_per_day,
                failures=0,
                concurrency=cfg.concurrency,
            )
            for cfg in self._workflows
        }
        queue_depth_series: List[QueueSample] = []
        dlq_entries: List[DLQEntry] = []
        alerts: List[AlertRecord] = []
        request_index = 0

        for cfg in self._workflows:
            for doc_index in range(cfg.docs_per_day):
                tenant_id = self._tenants[(request_index + doc_index) % len(self._tenants)]
                queue_wait = self._queue_wait_sample(cfg)
                base_latency = max(
                    0.2,
                    self._rng.gauss(cfg.base_latency_mean, cfg.base_latency_std),
                )
                events = self._active_chaos(request_index, cfg.name)

                latency_multiplier = 1.0
                failure_probability = 0.01
                queue_penalty = 0.0
                if events:
                    for record, chaos in events:
                        latency_multiplier += chaos.latency_penalty
                        failure_probability += chaos.failure_probability
                        queue_penalty += chaos.queue_penalty
                        record.impacted_requests += 1

                queue_wait += queue_penalty
                latency = base_latency * latency_multiplier + queue_wait

                retries, success = self._simulate_retries(failure_probability)
                if not success:
                    metrics[cfg.name].failures += 1
                    dlq_entries.append(
                        DLQEntry(
                            workflow=cfg.name,
                            tenant_id=tenant_id,
                            request_index=request_index,
                            reason="Max retries exceeded",
                        )
                    )
                    for record, _ in events:
                        record.failures += 1
                    alerts.append(
                        AlertRecord(
                            name="dlq_entry_detected",
                            severity="warning",
                            message=f"{cfg.name} task for {tenant_id} routed to DLQ",
                            related_event=events[0][0].name if events else None,
                        )
                    )
                else:
                    metrics[cfg.name].add_sample(latency=latency, queue_wait=queue_wait)

                queue_depth = self._estimate_queue_depth(queue_wait, cfg.concurrency)
                queue_depth_series.append(
                    QueueSample(time_index=request_index, workflow=cfg.name, depth=queue_depth)
                )

                if queue_depth > cfg.concurrency * 0.8:
                    alerts.append(
                        AlertRecord(
                            name="queue_depth_threshold",
                            severity="critical",
                            message=(
                                f"Queue depth {queue_depth} for {cfg.name} exceeds 80% of capacity"
                            ),
                            related_event=events[0][0].name if events else None,
                        )
                    )

                request_index += 1

        return SimulationResult(
            workflows=metrics,
            queue_depth_series=queue_depth_series,
            chaos_timeline=self._resolved_chaos,
            dlq_entries=dlq_entries,
            alerts=alerts,
        )

    def _queue_wait_sample(self, cfg: WorkflowConfig) -> float:
        wait = self._rng.gammavariate(cfg.queue_wait_shape, cfg.queue_wait_scale)
        return max(wait, 0.0)

    def _estimate_queue_depth(self, queue_wait: float, concurrency: int) -> int:
        depth = int(math.ceil(queue_wait * max(concurrency / 2, 1)))
        base_noise = self._rng.randint(0, max(int(concurrency * 0.05), 1))
        return max(0, depth + base_noise)

    def _simulate_retries(self, failure_probability: float) -> tuple[int, bool]:
        max_retries = 3
        retries = 0
        success = False
        adjusted_probability = min(failure_probability, 0.9)
        while retries <= max_retries:
            if self._rng.random() > adjusted_probability:
                success = True
                break
            retries += 1
            adjusted_probability *= 0.6
        return retries, success


def _histogram(values: Iterable[float], bucket_size: float = 0.5, limit: float | None = None) -> List[Dict[str, float]]:
    vals = [v for v in values if v >= 0]
    if not vals:
        return []
    max_value = limit or max(vals)
    buckets: Dict[int, int] = {}
    for value in vals:
        bucket_index = int(value // bucket_size)
        buckets[bucket_index] = buckets.get(bucket_index, 0) + 1
    histogram = []
    for index in range(int((max_value // bucket_size) + 2)):
        histogram.append(
            {
                "bucket_start": round(index * bucket_size, 3),
                "bucket_end": round((index + 1) * bucket_size, 3),
                "count": buckets.get(index, 0),
            }
        )
    return histogram


def write_phase7_artifacts(result: SimulationResult, output_dir: str | os.PathLike[str]) -> None:
    """Write the required Phase 7 artifacts to disk."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Locust style statistics
    locust_path = output_path / "locust_stats.csv"
    with locust_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "request_type",
                "total_requests",
                "failures",
                "avg_latency",
                "p95_latency",
                "queue_p95",
                "throughput_per_min",
                "success_rate",
            ]
        )
        for workflow_name, metrics in result.workflows.items():
            summary = metrics.build_summary()
            writer.writerow(
                [
                    workflow_name,
                    metrics.total_requests,
                    metrics.failures,
                    round(summary["avg_latency"], 4),
                    round(summary["p95_latency"], 4),
                    round(summary["queue_p95"], 4),
                    round(summary["throughput_per_min"], 4),
                    round(summary["success_rate"], 4),
                ]
            )

    # Latency histograms
    histograms = {
        name: _histogram(metrics.latencies)
        for name, metrics in result.workflows.items()
        if metrics.latencies
    }
    (output_path / "latency_histograms.json").write_text(
        json.dumps(histograms, indent=2),
        encoding="utf-8",
    )

    # Queue depth samples
    queue_samples = result.queue_depth_series
    if queue_samples:
        stride = max(len(queue_samples) // 400, 1)
        selected_samples = [
            queue_samples[idx] for idx in range(0, len(queue_samples), stride)
        ]
        if selected_samples[-1] is not queue_samples[-1]:
            selected_samples.append(queue_samples[-1])
    else:
        selected_samples = []

    queue_depth_rows = [
        {
            "time_index": sample.time_index,
            "workflow": sample.workflow,
            "queue_depth": sample.depth,
        }
        for sample in selected_samples
    ]
    (output_path / "queue_depth.json").write_text(
        json.dumps(queue_depth_rows, indent=2),
        encoding="utf-8",
    )

    # Chaos timeline
    chaos_rows = [
        {
            "name": record.name,
            "start_index": record.start_index,
            "end_index": record.end_index,
            "impacted_requests": record.impacted_requests,
            "failures": record.failures,
            "description": record.description,
        }
        for record in result.chaos_timeline
    ]
    (output_path / "chaos_timeline.json").write_text(
        json.dumps(chaos_rows, indent=2),
        encoding="utf-8",
    )

    # DLQ entries
    dlq_path = output_path / "dlq_entries.jsonl"
    with dlq_path.open("w", encoding="utf-8") as dlq_file:
        for entry in result.dlq_entries:
            dlq_file.write(
                json.dumps(
                    {
                        "workflow": entry.workflow,
                        "tenant_id": entry.tenant_id,
                        "request_index": entry.request_index,
                        "reason": entry.reason,
                    }
                )
            )
            dlq_file.write("\n")

    # Alerts
    alerts_path = output_path / "alerts.json"
    alerts_payload = [
        {
            "name": alert.name,
            "severity": alert.severity,
            "message": alert.message,
            "related_event": alert.related_event,
        }
        for alert in result.alerts
    ]
    alerts_path.write_text(json.dumps(alerts_payload, indent=2), encoding="utf-8")

    _write_alert_overview_svg(result.alerts, output_path / "alerts.svg")


def _write_alert_overview_svg(alerts: Sequence[AlertRecord], path: Path) -> None:
    width = 800
    height = 40 + 30 * len(alerts)
    lines = [
        "<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>".format(
            width=width, height=height
        ),
        "<style>text{font-family:monospace;font-size:14px;} .critical{fill:#b30000;} .warning{fill:#b36b00;} .info{fill:#004d99;}</style>",
        "<rect width='{width}' height='{height}' fill='#0b1e39'/>".format(width=width, height=height),
        "<text x='20' y='25' fill='#ffffff'>Phase 7 Alert Summary</text>",
    ]
    for idx, alert in enumerate(alerts):
        y = 60 + idx * 24
        color_class = alert.severity if alert.severity in {"critical", "warning", "info"} else "info"
        related = f" (event: {alert.related_event})" if alert.related_event else ""
        lines.append(
            "<text x='20' y='{y}' class='{cls}'>{msg}</text>".format(
                y=y,
                cls=color_class,
                msg=f"[{alert.severity.upper()}] {alert.message}{related}",
            )
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "AlertRecord",
    "ChaosConfig",
    "Phase7LoadSimulator",
    "SimulationResult",
    "WorkflowConfig",
    "write_phase7_artifacts",
]
