import contextvars
from typing import Optional

_tenant_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("forge1_tenant", default="default_tenant")


def set_current_tenant(tenant_id: Optional[str]) -> None:
    _tenant_ctx.set(tenant_id or "default_tenant")


def get_current_tenant() -> str:
    return _tenant_ctx.get()


def tenant_prefix(prefix: str) -> str:
    return f"t:{get_current_tenant()}:{prefix}"


__all__ = [
    "set_current_tenant",
    "get_current_tenant",
    "tenant_prefix",
]

