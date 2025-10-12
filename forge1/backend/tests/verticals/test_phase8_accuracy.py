from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge1.verticals.phase8_accuracy import (
    DomainEvaluation,
    run_finance_accuracy_evaluation,
    run_legal_accuracy_evaluation,
    run_phase8_evaluations,
)


@pytest.fixture()
def artifact_dir(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


def test_legal_evaluation_passes_targets():
    evaluation = run_legal_accuracy_evaluation()
    assert isinstance(evaluation, DomainEvaluation)
    metrics = {metric.name: metric for metric in evaluation.metrics}
    assert metrics["precision_at_5"].passed is True
    assert metrics["precision_at_5"].value >= 0.8
    assert metrics["hallucination_rate"].passed is True
    assert metrics["hallucination_rate"].value <= 0.02


def test_finance_evaluation_variance_within_threshold():
    evaluation = run_finance_accuracy_evaluation()
    metrics = {metric.name: metric for metric in evaluation.metrics}
    for metric in metrics.values():
        assert metric.passed is True
        assert metric.value <= metric.target


def test_phase8_runner_writes_artifacts(artifact_dir: Path):
    results = run_phase8_evaluations(artifact_dir)
    assert set(results.keys()) == {"legal", "finance"}

    legal_file = artifact_dir / "legal_evaluation.json"
    finance_file = artifact_dir / "finance_evaluation.json"
    summary_file = artifact_dir / "phase8_summary.json"
    methodology_file = artifact_dir / "methodology.md"

    for file in [legal_file, finance_file, summary_file, methodology_file]:
        assert file.exists(), f"Expected artifact missing: {file}"

    summary = json.loads(summary_file.read_text())
    assert summary["overall_passed"] is True
    assert summary["legal"]["passed"] is True
    assert summary["finance"]["passed"] is True
