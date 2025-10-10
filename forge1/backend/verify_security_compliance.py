#!/usr/bin/env python3
"""Run Forge1 security and compliance verification checks."""

from __future__ import annotations

import json
from pathlib import Path

from forge1.compliance.verify_security_compliance import run_security_compliance_suite


DEFAULT_POLICIES = {
    "tool_access": {"version": "1.0", "rules": []},
    "document_access": {"version": "1.0", "rules": []},
    "routing_constraints": {"version": "1.0", "rules": []},
}

SANITIZED_PAYLOAD = {"message": "[REDACTED]", "ssn": "***-**-****"}


def main() -> int:
    report = run_security_compliance_suite(
        available_policies=DEFAULT_POLICIES.keys(),
        policy_definitions=DEFAULT_POLICIES,
        sanitized_payload=SANITIZED_PAYLOAD,
    )
    output_path = Path("artifacts/security/security_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"Security compliance checks completed. Report saved to {output_path}")
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
