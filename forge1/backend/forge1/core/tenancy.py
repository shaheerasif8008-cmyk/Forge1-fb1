"""Minimal tenant context utilities used across the backend."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class TenantContext:
    """Container for per-request multi-tenant metadata."""

    tenant_id: str
    user_id: str
    role: str


_tenant_ctx: ContextVar[Optional[TenantContext]] = ContextVar("tenant_ctx", default=None)


def set_tenant_context(context: Optional[TenantContext]) -> None:
    """Bind a tenant context to the current async execution flow."""

    _tenant_ctx.set(context)


def clear_tenant_context() -> None:
    """Remove the tenant context for the current execution flow."""

    _tenant_ctx.set(None)


def get_tenant_context() -> Optional[TenantContext]:
    """Return the active :class:`TenantContext`, if any."""

    return _tenant_ctx.get()


def get_current_tenant() -> Optional[str]:
    """Convenience accessor returning the tenant identifier."""

    context = get_tenant_context()
    return context.tenant_id if context else None


def tenant_prefix(explicit_tenant: Optional[str] = None) -> str:
    """Return a stable prefix suitable for Redis or vector namespaces."""

    tenant_id = explicit_tenant or get_current_tenant() or "default"
    return f"tenant:{tenant_id}"


def set_current_tenant(tenant_id: Optional[str], user_id: Optional[str] = None, role: Optional[str] = None) -> None:
    """Compatibility helper that mirrors the legacy API used across the codebase."""

    if tenant_id is None:
        clear_tenant_context()
        return

    context = TenantContext(
        tenant_id=tenant_id,
        user_id=user_id or "system",
        role=role or "system",
    )
    set_tenant_context(context)


def get_current_user() -> Optional[str]:
    context = get_tenant_context()
    return context.user_id if context else None


__all__ = [
    "TenantContext",
    "set_tenant_context",
    "set_current_tenant",
    "clear_tenant_context",
    "get_tenant_context",
    "get_current_tenant",
    "get_current_user",
    "tenant_prefix",
]
