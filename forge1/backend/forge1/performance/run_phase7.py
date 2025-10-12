"""Command line entrypoint to execute Phase 7 load and chaos simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from forge1.core.logging_config import init_logger
from forge1.performance.phase7_simulation import (
    ChaosConfig,
    Phase7LoadSimulator,
    SimulationResult,
    WorkflowConfig,
    write_phase7_artifacts,
)
from forge1.performance.verify_performance_profiles import run_performance_profile_suite


logger = init_logger("forge1.performance.phase7.runner")

DEFAULT_TENANTS: List[str] = [
    "tenant-legal",
    "tenant-finance",
    "tenant-health",
    "tenant-marketing",
    "tenant-operations",
]


def _default_workflow_configs() -> List[WorkflowConfig]:
    return [
        WorkflowConfig(
            name="law_nda",
            docs_per_day=6200,
            concurrency=140,
            base_latency_mean=2.1,
            base_latency_std=0.35,
            queue_wait_shape=1.5,
            queue_wait_scale=0.2,
            target_p95=6.0,
        ),
        WorkflowConfig(
            name="finance_pnl",
            docs_per_day=3800,
            concurrency=120,
            base_latency_mean=1.4,
            base_latency_std=0.25,
            queue_wait_shape=1.2,
            queue_wait_scale=0.18,
            target_p95=4.0,
        ),
    ]

def _default_chaos_configs() -> List[ChaosConfig]:
    return [
        ChaosConfig(
            name="slack_502_burst",
            start_fraction=0.18,
            duration_fraction=0.06,
            failure_probability=0.22,
            latency_penalty=0.65,
            queue_penalty=0.35,
            affects=("law_nda",),
            description="Simulated Slack API 502 burst handled via retries and DLQ fallback.",
        ),
        ChaosConfig(
            name="drive_429_spike",
            start_fraction=0.42,
            duration_fraction=0.05,
            failure_probability=0.18,
            latency_penalty=0.5,
            queue_penalty=0.25,
            affects=("law_nda", "finance_pnl"),
            description="Drive connector rate limiting requiring backoff and jitter.",
        ),
        ChaosConfig(
            name="vector_latency_injection",
            start_fraction=0.55,
            duration_fraction=0.08,
            failure_probability=0.06,
            latency_penalty=0.75,
            queue_penalty=0.1,
            affects=("law_nda", "finance_pnl"),
            description="Vector store latency injection validating circuit breakers.",
        ),
        ChaosConfig(
            name="model_failover",
            start_fraction=0.78,
            duration_fraction=0.04,
            failure_probability=0.12,
            latency_penalty=0.45,
            queue_penalty=0.2,
            affects=("finance_pnl",),
            description="Primary model provider failure forcing failover sequence.",
        ),
    ]

def build_default_simulator(seed: int = 1337) -> tuple[Phase7LoadSimulator, List[WorkflowConfig]]:
    workflows = _default_workflow_configs()
    chaos_events = _default_chaos_configs()
    simulator = Phase7LoadSimulator(
        workflows=workflows,
        chaos_events=chaos_events,
        tenants=DEFAULT_TENANTS,
        seed=seed,
    )
    return simulator, workflows


def _gate_results(result: SimulationResult, output_dir: Path, workflows: List[WorkflowConfig]) -> None:
    failures: List[str] = []
    summary = {}

    for workflow_name, metrics in result.workflows.items():
        stats = metrics.build_summary()
        summary[workflow_name] = stats
        target_p95 = next(cfg.target_p95 for cfg in workflows if cfg.name == workflow_name)
        if stats["p95_latency"] > target_p95:
            failures.append(
                f"{workflow_name} p95 latency {stats['p95_latency']:.2f}s exceeds {target_p95:.2f}s"
            )
        if stats["queue_p95"] > 1.0:
            failures.append(
                f"{workflow_name} queue wait p95 {stats['queue_p95']:.2f}s exceeds 1.0s"
            )

    if not result.dlq_entries:
        failures.append("No DLQ entries captured during chaos scenarios")

    report = run_performance_profile_suite(
        latencies=[lat for m in result.workflows.values() for lat in m.latencies],
        error_events=sum(m.failures for m in result.workflows.values()),
        total_requests=sum(m.total_requests for m in result.workflows.values()),
        thresholds={"p95": 6.0, "mean": 3.0, "max": 8.0},
    )
    summary["profile_checks"] = {
        check.name: {"passed": check.passed, "details": check.details}
        for check in report.checks
    }

    summary_path = output_dir / "phase7_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if failures:
        failure_path = output_dir / "phase7_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise SystemExit("; ".join(failures))

    logger.info("Phase 7 gates satisfied", extra={"summary_file": str(summary_path)})


def run(output_dir: str, seed: int = 1337) -> SimulationResult:
    simulator, workflows = build_default_simulator(seed=seed)
    result = simulator.run()
    output_path = Path(output_dir)
    write_phase7_artifacts(result, output_path)
    _gate_results(result, output_path, workflows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 7 load and chaos simulations")
    parser.add_argument(
        "--output",
        default="artifacts/phase7",
        help="Directory to write artifacts into.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed to make the simulation reproducible.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run(str(output_dir), seed=args.seed)

    logger.info(
        "Simulation complete",
        extra={
            "total_requests": sum(m.total_requests for m in result.workflows.values()),
            "dlq_entries": len(result.dlq_entries),
            "alerts": len(result.alerts),
        },
    )


if __name__ == "__main__":
    main()
