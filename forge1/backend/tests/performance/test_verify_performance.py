from forge1.performance.verify_performance_profiles import (
    PerformanceReport,
    run_performance_profile_suite,
    verify_error_rate,
    verify_latency_profile,
)


def test_latency_profile_detects_threshold_breach():
    latencies = [0.1, 0.2, 1.5]
    result = verify_latency_profile(latencies, {"mean": 0.3, "p95": 0.5, "max": 1.0})
    assert not result.passed
    assert "Threshold breaches" in result.details


def test_error_rate_check():
    result = verify_error_rate(1, 100)
    assert result.passed
    result = verify_error_rate(10, 100, max_error_ratio=0.05)
    assert not result.passed


def test_performance_suite_aggregates_checks():
    report = run_performance_profile_suite(
        latencies=[0.1, 0.2, 0.3],
        error_events=0,
        total_requests=50,
        thresholds={"mean": 0.4, "p95": 0.5, "max": 1.0},
    )
    assert isinstance(report, PerformanceReport)
    assert report.all_passed
