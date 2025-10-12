"""FastAPI middleware that enforces OPA policy decisions."""

from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, Optional

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from forge1.core.logging_config import init_logger
from forge1.policy.opa_client import OPAAdapter, PolicyDecision, PolicyInput

LOGGER = init_logger("forge1.middleware.policy")


class PolicyEnforcementMiddleware(BaseHTTPMiddleware):
    """Evaluate requests against OPA policies using request metadata."""

    def __init__(
        self,
        app,
        *,
        opa_adapter: Optional[OPAAdapter] = None,
        bypass_paths: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(app)
        self._opa = opa_adapter or OPAAdapter()
        self._bypass = set(bypass_paths or {"/", "/health", "/metrics", "/auth/token"})

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        if request.url.path in self._bypass:
            return await call_next(request)

        policy_ref = self._resolve_policy(request)
        if not policy_ref:
            return await call_next(request)

        start = time.monotonic()
        auth = getattr(request.state, "auth", None)
        if not auth:
            raise HTTPException(status_code=401, detail="Authentication required for policy enforcement")

        resource = self._resource_from_request(request, auth.tenant_id)
        action = request.headers.get(
            "X-Policy-Action",
            "execute" if request.method.upper() in {"POST", "PUT", "PATCH"} else request.method.lower(),
        )

        policy_input = PolicyInput(
            subject={
                "user_id": auth.user_id,
                "role": getattr(auth.role, "value", getattr(auth, "role", "user")),
                "tenant_id": auth.tenant_id,
            },
            resource=resource,
            action=action,
            environment={"path": request.url.path, "method": request.method},
            tenant_id=auth.tenant_id,
        )

        result = await self._opa.evaluate(policy_ref, policy_input)
        duration_ms = (time.monotonic() - start) * 1000

        LOGGER.debug(
            "policy evaluation completed",
            extra={
                "policy": policy_ref,
                "decision": result.decision.value,
                "duration_ms": duration_ms,
                "path": request.url.path,
            },
        )

        if result.decision == PolicyDecision.DENY:
            return JSONResponse(
                status_code=403,
                content={
                    "detail": result.reason,
                    "policy": policy_ref,
                },
            )

        return await call_next(request)

    def _resolve_policy(self, request: Request) -> Optional[str]:
        resource_header = request.headers.get("X-Policy-Resource")
        if resource_header == "tool":
            return "forge1.policies.tool_access"
        if resource_header == "document":
            return "forge1.policies.doc_access"
        if resource_header == "routing":
            return "forge1.policies.routing_constraints"
        return None

    def _resource_from_request(self, request: Request, default_tenant: str) -> Dict[str, str]:
        tenant_id = request.headers.get("X-Resource-Tenant", default_tenant)
        sensitivity = request.headers.get("X-Resource-Sensitivity", "standard")
        name = request.headers.get("X-Tool-Name", request.url.path)
        classification = request.headers.get("X-Document-Class", "public")
        target_model = request.headers.get("X-Target-Model")

        resource: Dict[str, str] = {
            "tenant_id": tenant_id,
            "sensitivity": sensitivity,
            "name": name,
            "classification": classification,
        }
        if target_model:
            resource["target_model"] = target_model
        return resource


policy_enforcement_middleware = PolicyEnforcementMiddleware

__all__ = ["PolicyEnforcementMiddleware", "policy_enforcement_middleware"]
