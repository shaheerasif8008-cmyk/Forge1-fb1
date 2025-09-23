"""
Simple ABAC Policy Engine (Phase A.3)

Evaluates allow/deny decisions based on user/resource/action attributes.
This is a lightweight placeholder to be swapped with OPA/Rego later.
"""

from typing import Dict, Any, Tuple


class PolicyDecision:
    def __init__(self, allow: bool, reason: str = ""):
        self.allow = allow
        self.reason = reason


class PolicyEngine:
    def __init__(self, rules: Dict[str, Any] = None):
        # rules can be extended to load from DB or files
        self.rules = rules or {}

    def evaluate(self, user: Dict[str, Any], resource: str, action: str, attributes: Dict[str, Any]) -> PolicyDecision:
        # Basic examples:
        # - Block destructive actions unless role is admin/manager
        # - Restrict access by tenant match
        # - Enforce security level constraints

        roles = set((user.get("roles") or []))
        tenant = user.get("tenant_id")
        resource_tenant = attributes.get("tenant_id", tenant)

        if tenant and resource_tenant and (tenant != resource_tenant):
            return PolicyDecision(False, "tenant_mismatch")

        destructive = {"delete", "truncate", "purge"}
        if action.lower() in destructive and roles.isdisjoint({"admin", "manager"}):
            return PolicyDecision(False, "insufficient_role_for_destructive_action")

        required_level = attributes.get("required_security_level")
        user_level = user.get("security_level")
        if required_level and user_level and user_level not in {required_level, "enterprise"}:
            return PolicyDecision(False, "security_level_insufficient")

        # Default allow
        return PolicyDecision(True, "allow")


__all__ = ["PolicyEngine", "PolicyDecision"]

