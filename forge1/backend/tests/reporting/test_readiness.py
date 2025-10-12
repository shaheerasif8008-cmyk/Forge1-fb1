from pathlib import Path

from forge1.reporting import build_readiness_report


def test_build_readiness_report_passes_all_phases():
    repo_root = Path(__file__).resolve().parents[4]
    artifacts_dir = repo_root / "artifacts"

    report = build_readiness_report(artifacts_dir)

    assert report.recommendation == "GO"
    assert report.readiness_score >= 0.9
    assert len(report.phases) == 9
    assert all(phase.status == "pass" for phase in report.phases)
