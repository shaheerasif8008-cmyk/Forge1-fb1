"""
Forge 1 Multi-Tenancy Support
Enterprise multi-tenant architecture with data isolation
"""

import threading
from typing import Optional

# Thread-local storage for tenant context
_tenant_context = threading.local()

def set_current_tenant(tenant_id: Optional[str]):
    """Set the current tenant ID for the request context"""
    _tenant_context.tenant_id = tenant_id

def get_current_tenant() -> Optional[str]:
    """Get the current tenant ID from the request context"""
    return getattr(_tenant_context, 'tenant_id', None)

def clear_tenant_context():
    """Clear the tenant context"""
    if hasattr(_tenant_context, 'tenant_id'):
        delattr(_tenant_context, 'tenant_id')


def tenant_prefix(tenant_id: Optional[str]) -> str:
    """Return a normalized tenant prefix for cache namespaces."""
    tenant = tenant_id or get_current_tenant() or "default"
    return f"tenant:{tenant}"
