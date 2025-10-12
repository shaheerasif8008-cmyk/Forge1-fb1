"""Tenant-aware authentication middleware."""

from __future__ import annotations

from typing import Iterable, Optional

from fastapi import HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from forge1.auth.jwt import TokenPayload, TokenVerificationError, UserRole, verify_token
from forge1.core.logging_config import init_logger
from forge1.core.security import SecretManager
from forge1.core.tenancy import TenantContext, clear_tenant_context, set_tenant_context

logger = init_logger("forge1.middleware.tenant")


class TenantContextMiddleware(BaseHTTPMiddleware):
    """Populate ``request.state.auth`` and the global tenant context."""

    def __init__(
        self,
        app,
        *,
        secret_manager: SecretManager,
        jwt_secret_name: str = "FORGE1_JWT_SECRET",
        allow_anonymous_paths: Iterable[str] | None = None,
    ) -> None:
        super().__init__(app)
        self.secret_manager = secret_manager
        self.jwt_secret_name = jwt_secret_name
        self.allow_anonymous_paths = set(allow_anonymous_paths or {"/health", "/metrics", "/"})

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in self.allow_anonymous_paths:
            return await call_next(request)

        try:
            payload = await self._authenticate(request)
            if payload:
                request.state.auth = payload
                tenant_context = TenantContext(
                    tenant_id=payload.tenant_id,
                    user_id=payload.user_id,
                    role=payload.role.value,
                )
                set_tenant_context(tenant_context)
            else:
                request.state.auth = None
        except HTTPException:
            clear_tenant_context()
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            clear_tenant_context()
            logger.error("Tenant authentication failure", extra={"error": str(exc)})
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Authentication error") from exc

        try:
            response = await call_next(request)
        finally:
            clear_tenant_context()

        return response

    async def _authenticate(self, request: Request) -> Optional[TokenPayload]:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ", 1)[1]
            secret = self._get_jwt_secret()
            try:
                return verify_token(token, secret)
            except TokenVerificationError as exc:
                logger.warning("Invalid JWT", extra={"error": str(exc)})
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

        tenant_id = request.headers.get("X-Tenant-ID")
        user_id = request.headers.get("X-User-ID")
        role = request.headers.get("X-Role", UserRole.USER.value)

        if tenant_id and user_id:
            logger.debug("Using header-based authentication", extra={"tenant_id": tenant_id, "user_id": user_id})
            return TokenPayload({
                "tenant_id": tenant_id,
                "user_id": user_id,
                "role": role,
            })

        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authentication headers")

    def _get_jwt_secret(self) -> str:
        secret = self.secret_manager.get_secret(self.jwt_secret_name)
        if not secret:
            raise RuntimeError(f"JWT secret '{self.jwt_secret_name}' is not configured")
        return secret


__all__ = ["TenantContextMiddleware"]
