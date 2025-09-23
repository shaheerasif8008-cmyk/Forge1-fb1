# forge1/backend/forge1/tests/test_compliance_security.py
"""Security and compliance tests (Task 10.2)"""

import pytest
from forge1.core.compliance_engine import ComplianceEngine
from forge1.core.security_manager import SecurityManager, Permission


@pytest.mark.asyncio
async def test_compliance_pii_detection():
    engine = ComplianceEngine()
    text = "Customer SSN is 123-45-6789 and email test@example.com"
    result = await engine.analyze_content(text, framework="GDPR")
    assert result["compliant"] is False
    assert any(v["type"] == "pii_detected" for v in result["violations"]) is True


@pytest.mark.asyncio
async def test_security_permission_checks():
    sec = SecurityManager()
    # Simulate a user id and build context (method handles caching)
    user_id = "alice@manager.com"
    ok = await sec.check_permission(user_id, Permission.COMPLIANCE_VIEW)
    # Managers have COMPLIANCE_VIEW per mapping
    assert ok is True

