"""JWT utilities and FastAPI dependencies for Forge1."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Optional

import jwt
from fastapi import HTTPException, Request, status

from forge1.core.logging_config import init_logger

logger = init_logger("forge1.auth.jwt")


class UserRole(str, Enum):
    """Supported user roles for Phase 1 foundations."""

    ADMIN = "admin"
    USER = "user"
    SERVICE = "service"


class TokenPayload(Dict[str, Any]):
    """Runtime representation of validated JWT claims."""

    @property
    def tenant_id(self) -> str:
        return self["tenant_id"]

    @property
    def user_id(self) -> str:
        return self["user_id"]

    @property
    def role(self) -> UserRole:
        return UserRole(self["role"])


class TokenVerificationError(Exception):
    """Raised when a JWT cannot be validated."""


def issue_token(
    *,
    secret: str,
    user_id: str,
    tenant_id: str,
    role: UserRole,
    expires_in: timedelta | None = None,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """Issue a signed JWT for the supplied identity."""

    expires = datetime.now(timezone.utc) + (expires_in or timedelta(minutes=30))
    payload: Dict[str, Any] = {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "role": role.value,
        "exp": expires,
        "iat": datetime.now(timezone.utc),
    }
    if extra_claims:
        payload.update(extra_claims)

    token = jwt.encode(payload, secret, algorithm="HS256")
    logger.debug("Issued JWT", extra={"tenant_id": tenant_id, "user_id": user_id, "role": role.value})
    return token


def verify_token(token: str, secret: str) -> TokenPayload:
    """Validate ``token`` using ``secret``.

    Returns the decoded payload or raises :class:`TokenVerificationError` when the
    token cannot be decoded or does not contain mandatory claims.
    """

    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:  # pragma: no cover - library provides varied errors
        raise TokenVerificationError(str(exc)) from exc

    for required in ("user_id", "tenant_id", "role"):
        if required not in payload:
            raise TokenVerificationError(f"Missing claim: {required}")

    return TokenPayload(payload)


def require_role(*roles: UserRole) -> Callable[[Request], TokenPayload]:
    """FastAPI dependency enforcing that the active user has one of ``roles``."""

    allowed_roles: Iterable[UserRole]
    if not roles:
        allowed_roles = (UserRole.USER, UserRole.ADMIN, UserRole.SERVICE)
    else:
        allowed_roles = roles

    async def dependency(request: Request) -> TokenPayload:
        payload: Optional[TokenPayload] = getattr(request.state, "auth", None)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )

        if payload.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role",
            )
        return payload

    return dependency


__all__ = ["UserRole", "TokenPayload", "TokenVerificationError", "issue_token", "verify_token", "require_role"]
