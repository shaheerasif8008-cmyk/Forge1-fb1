"""Security and compliance verification utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from forge1.core.logging_config import init_logger


logger = init_logger("forge1.compliance.verification")

REQUIRED_POLICIES = {"tool_access", "document_access", "routing_constraints"}
PII_PATTERN = re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b|\b\d{16}\b")


@dataclass
class ComplianceCheck:
    name: str
    passed: bool
    details: str


@dataclass
class ComplianceReport:
    checks: List[ComplianceCheck]

    @property
    def all_passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> Dict[str, Any]:
        return {"all_passed": self.all_passed, "checks": [check.__dict__ for check in self.checks]}


def verify_policy_presence(policies: Iterable[str]) -> ComplianceCheck:
    """Ensure mandatory policy bundles are present."""
    missing = REQUIRED_POLICIES - set(policies)
    if missing:
        return ComplianceCheck(
            name="policy_presence",
            passed=False,
            details=f"Missing policies: {', '.join(sorted(missing))}",
        )
    return ComplianceCheck(name="policy_presence", passed=True, details="All required policies available")


def verify_policy_rules(policy_definitions: Dict[str, Dict[str, Any]]) -> ComplianceCheck:
    """Perform lightweight validation of policy structure."""
    malformed = []
    for policy_name, definition in policy_definitions.items():
        if not isinstance(definition, dict):
            malformed.append(policy_name)
            continue
        if "version" not in definition or "rules" not in definition:
            malformed.append(policy_name)
    if malformed:
        return ComplianceCheck(
            name="policy_structure",
            passed=False,
            details=f"Malformed policy definitions: {', '.join(sorted(malformed))}",
        )
    return ComplianceCheck(name="policy_structure", passed=True, details="Policy definitions well-formed")


def verify_dlp_redaction(sample_payload: Dict[str, Any]) -> ComplianceCheck:
    """Ensure payload is redacted before storage/logging."""
    matches = PII_PATTERN.findall(str(sample_payload))
    if matches:
        return ComplianceCheck(
            name="dlp_redaction",
            passed=False,
            details=f"Detected unredacted PII values: {matches}",
        )
    return ComplianceCheck(name="dlp_redaction", passed=True, details="No unredacted PII detected")


def run_security_compliance_suite(
    available_policies: Iterable[str],
    policy_definitions: Dict[str, Dict[str, Any]],
    sanitized_payload: Dict[str, Any],
) -> ComplianceReport:
    """Execute bundled compliance checks."""

    logger.info("Running security & compliance verification")

    checks = [
        verify_policy_presence(available_policies),
        verify_policy_rules(policy_definitions),
        verify_dlp_redaction(sanitized_payload),
    ]
    return ComplianceReport(checks=checks)


__all__ = [
    "ComplianceCheck",
    "ComplianceReport",
    "run_security_compliance_suite",
    "verify_policy_presence",
    "verify_policy_rules",
    "verify_dlp_redaction",
]
