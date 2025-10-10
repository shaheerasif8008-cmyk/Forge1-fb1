from forge1.compliance.verify_security_compliance import (
    ComplianceReport,
    run_security_compliance_suite,
    verify_dlp_redaction,
    verify_policy_presence,
    verify_policy_rules,
)


def test_policy_presence_detects_missing():
    result = verify_policy_presence({"tool_access", "document_access"})
    assert not result.passed
    assert "routing_constraints" in result.details


def test_policy_rules_validate_structure():
    good = {"tool_access": {"version": "1.0", "rules": []}}
    assert verify_policy_rules(good).passed

    bad = {"tool_access": []}
    res = verify_policy_rules(bad)
    assert not res.passed
    assert "tool_access" in res.details


def test_compliance_suite_passes_when_all_checks_good():
    report = run_security_compliance_suite(
        {"tool_access", "document_access", "routing_constraints"},
        {
            "tool_access": {"version": "1.0", "rules": []},
            "document_access": {"version": "1.0", "rules": []},
            "routing_constraints": {"version": "1.0", "rules": []},
        },
        {"message": "[REDACTED]", "ssn": "***-**-****"},
    )
    assert isinstance(report, ComplianceReport)
    assert report.all_passed


def test_dlp_redaction_detects_pii():
    res = verify_dlp_redaction({"ssn": "123-45-6789"})
    assert not res.passed
    assert "123-45-6789" in res.details
