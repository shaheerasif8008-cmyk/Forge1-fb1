"""Minimal OPA/Rego policy evaluation utilities used by Forge1."""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

from forge1.core.dlp import redact_payload
from forge1.core.logging_config import init_logger

LOGGER = init_logger("forge1.policy.opa")


class PolicyDecision(str, Enum):
    """Possible decisions emitted by the policy engine."""

    ALLOW = "allow"
    DENY = "deny"


@dataclass(slots=True)
class PolicyInput:
    """Structured payload sent to the policy engine."""

    subject: Dict[str, Any]
    resource: Dict[str, Any]
    action: str
    environment: Dict[str, Any]
    tenant_id: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "input": {
                "subject": self.subject,
                "resource": self.resource,
                "action": self.action,
                "environment": self.environment,
                "tenant_id": self.tenant_id,
            }
        }


@dataclass(slots=True)
class PolicyResult:
    decision: PolicyDecision
    reason: str
    policy: str
    metadata: Dict[str, Any]
    violations: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "policy": self.policy,
            "metadata": self.metadata,
            "violations": self.violations,
        }


class DecisionLogger:
    """Persist policy decisions to a JSONL log for auditability."""

    def __init__(self, path: Optional[pathlib.Path] = None) -> None:
        target = path or pathlib.Path(os.environ.get("FORGE1_POLICY_LOG", "artifacts/policy/decision_log.jsonl"))
        target.parent.mkdir(parents=True, exist_ok=True)
        self._path = target
        self._lock = asyncio.Lock()

    async def write(self, result: PolicyResult, context: Dict[str, Any]) -> None:
        payload = {
            "decision": result.to_dict(),
            "context": context,
        }
        async with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, sort_keys=True) + "\n")


class LocalPolicyEvaluator:
    """Fallback evaluator used in tests when an OPA server is unavailable."""

    def evaluate(self, policy: str, document: Dict[str, Any]) -> PolicyResult:
        if policy.endswith("tool_access"):
            return self._tool_access(document)
        if policy.endswith("doc_access"):
            return self._doc_access(document)
        if policy.endswith("routing_constraints"):
            return self._routing(document)
        raise ValueError(f"Unsupported policy: {policy}")

    def _tool_access(self, document: Dict[str, Any]) -> PolicyResult:
        subject = document["input"]["subject"]
        resource = document["input"]["resource"]
        tenant_id = document["input"]["tenant_id"]
        action = document["input"]["action"]

        if tenant_id != resource.get("tenant_id"):
            return PolicyResult(PolicyDecision.DENY, "tenant_mismatch", "forge1.policies.tool_access", {}, [])

        sensitivity = resource.get("sensitivity", "standard")
        if sensitivity == "high" and subject.get("role") not in {"admin", "security"}:
            return PolicyResult(PolicyDecision.DENY, "insufficient_role", "forge1.policies.tool_access", {}, [])

        if action.lower() != "execute":
            return PolicyResult(PolicyDecision.DENY, "unsupported_action", "forge1.policies.tool_access", {}, [])

        return PolicyResult(PolicyDecision.ALLOW, "allow", "forge1.policies.tool_access", {}, [])

    def _doc_access(self, document: Dict[str, Any]) -> PolicyResult:
        subject = document["input"]["subject"]
        resource = document["input"]["resource"]
        tenant_id = document["input"]["tenant_id"]

        if tenant_id != resource.get("tenant_id"):
            return PolicyResult(PolicyDecision.DENY, "tenant_mismatch", "forge1.policies.doc_access", {}, [])

        classification = resource.get("classification", "public")
        if classification in {"restricted", "confidential"} and subject.get("role") not in {"admin", "manager"}:
            return PolicyResult(PolicyDecision.DENY, "role_restricted", "forge1.policies.doc_access", {}, [])

        return PolicyResult(PolicyDecision.ALLOW, "allow", "forge1.policies.doc_access", {}, [])

    def _routing(self, document: Dict[str, Any]) -> PolicyResult:
        subject = document["input"]["subject"]
        environment = document["input"].get("environment", {})
        target_model = environment.get("target_model")
        if not target_model:
            return PolicyResult(PolicyDecision.DENY, "missing_model", "forge1.policies.routing_constraints", {}, [])

        role = subject.get("role")
        allowed_models = {"admin": {"gpt-4", "gpt-3.5"}, "manager": {"gpt-3.5"}, "user": {"gpt-3.5"}}
        permitted = allowed_models.get(role, {"gpt-3.5"})
        if target_model not in permitted:
            return PolicyResult(PolicyDecision.DENY, "model_not_permitted", "forge1.policies.routing_constraints", {}, [])

        return PolicyResult(PolicyDecision.ALLOW, "allow", "forge1.policies.routing_constraints", {}, [])


class OPAClient:
    """Evaluate Rego policies either via an OPA server or the local evaluator."""

    def __init__(self, *, decision_logger: Optional[DecisionLogger] = None) -> None:
        self._server_url = os.environ.get("OPA_SERVER_URL")
        self._client: Optional[httpx.AsyncClient] = None
        self._local = LocalPolicyEvaluator()
        self._decision_logger = decision_logger or DecisionLogger()

    async def _ensure_client(self) -> Optional[httpx.AsyncClient]:
        if not self._server_url:
            return None
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._server_url.rstrip("/"))
        return self._client

    async def evaluate(self, policy: str, policy_input: PolicyInput) -> PolicyResult:
        payload = policy_input.as_dict()
        violations: List[Dict[str, Any]] = []

        # Apply redaction to the resource payload prior to evaluation.
        safe_resource, violations = redact_payload(policy_input.resource)
        payload["input"]["resource"] = safe_resource

        if client := await self._ensure_client():
            response = await client.post(f"/v1/data/{policy}", json=payload)
            response.raise_for_status()
            result = response.json().get("result", {})
            allow = bool(result.get("allow"))
            decision = PolicyDecision.ALLOW if allow else PolicyDecision.DENY
            reason = result.get("reason", "allow" if allow else "deny")
        else:
            local_result = self._local.evaluate(policy, payload)
            decision = local_result.decision
            reason = local_result.reason

        policy_result = PolicyResult(
            decision=decision,
            reason=reason,
            policy=policy,
            metadata={"subject": policy_input.subject, "environment": policy_input.environment},
            violations=violations,
        )

        LOGGER.info(
            "policy decision rendered",
            extra={
                "policy": policy,
                "decision": policy_result.decision.value,
                "tenant_id": policy_input.tenant_id,
                "reason": policy_result.reason,
            },
        )

        await self._decision_logger.write(
            policy_result,
            {
                "tenant_id": policy_input.tenant_id,
                "action": policy_input.action,
                "resource": safe_resource,
                "policy": policy,
            },
        )

        return policy_result

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


# Backwards compatibility alias used by existing imports
OPAAdapter = OPAClient

# Module-level adapter for legacy call sites that expect an instantiated client.
opa_adapter = OPAAdapter()

__all__ = [
    "DecisionLogger",
    "opa_adapter",
    "OPAAdapter",
    "OPAClient",
    "PolicyDecision",
    "PolicyInput",
    "PolicyResult",
]
