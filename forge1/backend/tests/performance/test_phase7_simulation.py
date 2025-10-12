"""Tests for the Phase 7 load and chaos simulation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge1.performance.phase7_simulation import Phase7LoadSimulator, WorkflowConfig, ChaosConfig
from forge1.performance.run_phase7 import run as run_phase7


@pytest.fixture()
def small_simulator() -> Phase7LoadSimulator:
    workflows = [
        WorkflowConfig(
            name="test_flow",
            docs_per_day=200,
            concurrency=40,
            base_latency_mean=1.2,
            base_latency_std=0.1,
            queue_wait_shape=1.1,
            queue_wait_scale=0.1,
            target_p95=3.5,
        )
    ]
    chaos = [
        ChaosConfig(
            name="test_event",
            start_fraction=0.2,
            duration_fraction=0.6,
            failure_probability=0.95,
            latency_penalty=0.6,
            queue_penalty=0.3,
            affects=("test_flow",),
            description="Test chaos event",
        )
    ]
    tenants = ["tenant-a", "tenant-b", "tenant-c"]
    return Phase7LoadSimulator(workflows=workflows, chaos_events=chaos, tenants=tenants, seed=7)


def test_simulation_produces_dlq_and_alerts(small_simulator: Phase7LoadSimulator) -> None:
    result = small_simulator.run()
    assert result.dlq_entries, "expected DLQ entries when chaos injected"
    assert any(alert.severity == "warning" for alert in result.alerts)
    assert all(sample.depth >= 0 for sample in result.queue_depth_series)


def test_phase7_run_generates_artifacts(tmp_path: Path) -> None:
    result = run_phase7(output_dir=str(tmp_path))

    summary_path = tmp_path / "phase7_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "law_nda" in summary and "finance_pnl" in summary
    assert summary["law_nda"]["p95_latency"] <= 6.0
    assert summary["finance_pnl"]["p95_latency"] <= 4.0
    assert summary["law_nda"]["queue_p95"] <= 1.0
    assert summary["finance_pnl"]["queue_p95"] <= 1.0
    assert result.dlq_entries, "chaos simulations should push some tasks into the DLQ"

    required_artifacts = [
        "locust_stats.csv",
        "latency_histograms.json",
        "queue_depth.json",
        "chaos_timeline.json",
        "dlq_entries.jsonl",
        "alerts.json",
        "alerts.svg",
    ]
    for artifact in required_artifacts:
        assert (tmp_path / artifact).exists(), f"missing artifact {artifact}"

    chaos_timeline = json.loads((tmp_path / "chaos_timeline.json").read_text(encoding="utf-8"))
    assert any(event["impacted_requests"] > 0 for event in chaos_timeline)
    assert any(event["failures"] >= 0 for event in chaos_timeline)
