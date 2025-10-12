"""Tests for the Phase 6 flow executors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from forge1.workflows.phase6_flows import (
        execute_phase6_flows,
        run_finance_pnl_flow,
        run_law_nda_flow,
    )
except ModuleNotFoundError:  # pragma: no cover - namespace guard for optional deps
    pytest.skip("Phase 6 flows not available", allow_module_level=True)


def _assert_flow_artifacts(flow_dir: Path, expected_files: list[str]) -> None:
    for name in expected_files:
        target = flow_dir / name
        assert target.exists(), f"Expected artifact missing: {target}"
        assert target.stat().st_size > 0, f"Artifact {target} is empty"


def test_run_law_nda_flow_creates_artifacts(tmp_path: Path) -> None:
    result = run_law_nda_flow(tmp_path)

    assert result.flow_name == "law_nda"
    assert all(step.status == "success" for step in result.steps)

    _assert_flow_artifacts(
        tmp_path,
        ["flow.json", "summary.txt", "usage_events.json"],
    )

    usage_payload = json.loads((tmp_path / "usage_events.json").read_text(encoding="utf-8"))
    assert usage_payload, "Expected usage events for NDA flow"


def test_run_finance_pnl_flow_generates_report(tmp_path: Path) -> None:
    result = run_finance_pnl_flow(tmp_path)

    assert result.flow_name == "finance_pnl"
    assert result.steps[-1].name == "email_send"

    _assert_flow_artifacts(
        tmp_path,
        ["flow.json", "report.html", "usage_events.json"],
    )

    report_html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Monthly P&" in report_html


def test_execute_phase6_flows_writes_shared_summary(tmp_path: Path) -> None:
    results = execute_phase6_flows(tmp_path)

    assert set(results) == {"law_nda", "finance_pnl"}

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Phase 6 Flow Execution" in summary

    usage_events = json.loads((tmp_path / "usage_events.json").read_text(encoding="utf-8"))
    assert usage_events, "Expected combined usage events"
