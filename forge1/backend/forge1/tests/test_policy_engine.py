import pytest

from forge1.core.policy_engine import PolicyEngine


def test_policy_engine_basic_abac():
    pe = PolicyEngine()

    user = {"roles": ["user"], "tenant_id": "t1", "security_level": "standard"}
    # Tenant mismatch denied
    dec = pe.evaluate(user, resource="memory", action="read", attributes={"tenant_id": "t2"})
    assert dec.allow is False and dec.reason == "tenant_mismatch"

    # Destructive denied for non-admin
    dec = pe.evaluate(user, resource="memory", action="delete", attributes={"tenant_id": "t1"})
    assert dec.allow is False and "insufficient_role" in dec.reason

    # Security level insufficient
    dec = pe.evaluate(user, resource="memory", action="read", attributes={"tenant_id": "t1", "required_security_level": "confidential"})
    assert dec.allow is False and dec.reason == "security_level_insufficient"

    # Allow case
    user2 = {"roles": ["manager"], "tenant_id": "t1", "security_level": "enterprise"}
    dec = pe.evaluate(user2, resource="memory", action="delete", attributes={"tenant_id": "t1"})
    assert dec.allow is True

