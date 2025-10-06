"""
Workflow Context Injection System

Ensures that tenant_id and employee_id context flows through all MCAE operations
and validates context to prevent cross-tenant access violations.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from contextvars import ContextVar
from functools import wraps

from forge1.core.tenancy import get_current_tenant, set_current_tenant
from forge1.integrations.mcae_adapter import TenantIsolationViolationError

logger = logging.getLogger(__name__)

# Context variables for workflow execution
workflow_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar('workflow_context', default=None)
tenant_context: ContextVar[Optional[str]] = ContextVar('tenant_context', default=None)
employee_context: ContextVar[Optional[str]] = ContextVar('employee_context', default=None)


class WorkflowContextInjector:
    """
    Manages workflow context injection and validation throughout MCAE operations.
    
    Ensures that all MCAE agents and tools receive proper tenant and employee context
    and validates that operations don't cross tenant boundaries.
    """
    
    def __init__(self):
        self.active_contexts = {}
        self.context_validators = []
        self.context_interceptors = []
        
        # Statistics
        self.stats = {
            "contexts_created": 0,
            "contexts_validated": 0,
            "validation_failures": 0,
            "cross_tenant_attempts": 0
        }
    
    def create_workflow_context(
        self,
        tenant_id: str,
        employee_id: str,
        session_id: str,
        user_id: str,
        additional_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Create a complete workflow context with all necessary information.
        
        Args:
            tenant_id: Tenant ID for isolation
            employee_id: Employee ID for personalization
            session_id: Session ID for workflow tracking
            user_id: User ID for audit trails
            additional_context: Additional context data
            
        Returns:
            Complete workflow context dictionary
        """
        context = {
            "tenant_id": tenant_id,
            "employee_id": employee_id,
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "context_id": f"ctx_{tenant_id}_{employee_id}_{session_id}",
            "validation_token": self._generate_validation_token(tenant_id, employee_id)
        }
        
        if additional_context:
            context.update(additional_context)
        
        # Store active context
        self.active_contexts[context["context_id"]] = context
        self.stats["contexts_created"] += 1
        
        logger.debug(f"Created workflow context: {context['context_id']}")
        return context
    
    def inject_context(self, context: Dict[str, Any]) -> None:
        """
        Inject context into the current execution environment.
        
        Args:
            context: Workflow context to inject
        """
        # Set context variables
        workflow_context.set(context)
        tenant_context.set(context["tenant_id"])
        employee_context.set(context["employee_id"])
        
        # Set Forge1 tenant context
        set_current_tenant(context["tenant_id"])
        
        logger.debug(f"Injected context for tenant {context['tenant_id']}, employee {context['employee_id']}")
    
    def get_current_context(self) -> Optional[Dict[str, Any]]:
        """Get the current workflow context"""
        return workflow_context.get()
    
    def get_current_tenant_id(self) -> Optional[str]:
        """Get the current tenant ID from context"""
        return tenant_context.get()
    
    def get_current_employee_id(self) -> Optional[str]:
        """Get the current employee ID from context"""
        return employee_context.get()
    
    def validate_context(self, required_tenant_id: str, required_employee_id: str) -> bool:
        """
        Validate that current context matches required tenant and employee.
        
        Args:
            required_tenant_id: Expected tenant ID
            required_employee_id: Expected employee ID
            
        Returns:
            True if context is valid, False otherwise
            
        Raises:
            TenantIsolationViolationError: If tenant isolation is violated
        """
        current_context = self.get_current_context()
        current_tenant = self.get_current_tenant_id()
        current_employee = self.get_current_employee_id()
        
        self.stats["contexts_validated"] += 1
        
        # Check if context exists
        if not current_context or not current_tenant or not current_employee:
            self.stats["validation_failures"] += 1
            logger.error("No workflow context found during validation")
            return False
        
        # Validate tenant isolation
        if current_tenant != required_tenant_id:
            self.stats["cross_tenant_attempts"] += 1
            self.stats["validation_failures"] += 1
            
            error_msg = f"Tenant isolation violation: current={current_tenant}, required={required_tenant_id}"
            logger.critical(error_msg)
            
            raise TenantIsolationViolationError(error_msg)
        
        # Validate employee context
        if current_employee != required_employee_id:
            self.stats["validation_failures"] += 1
            logger.warning(f"Employee context mismatch: current={current_employee}, required={required_employee_id}")
            return False
        
        # Validate context integrity
        validation_token = current_context.get("validation_token")
        expected_token = self._generate_validation_token(required_tenant_id, required_employee_id)
        
        if validation_token != expected_token:
            self.stats["validation_failures"] += 1
            logger.error("Context validation token mismatch")
            return False
        
        logger.debug(f"Context validation successful for tenant {required_tenant_id}, employee {required_employee_id}")
        return True
    
    def clear_context(self) -> None:
        """Clear the current workflow context"""
        current_context = self.get_current_context()
        
        if current_context:
            context_id = current_context.get("context_id")
            if context_id and context_id in self.active_contexts:
                del self.active_contexts[context_id]
        
        # Clear context variables
        workflow_context.set(None)
        tenant_context.set(None)
        employee_context.set(None)
        
        logger.debug("Cleared workflow context")
    
    def with_context(self, context: Dict[str, Any]):
        """
        Context manager for executing code with specific workflow context.
        
        Args:
            context: Workflow context to use
        """
        return WorkflowContextManager(self, context)
    
    def context_required(self, tenant_id: str, employee_id: str):
        """
        Decorator to ensure functions are called with proper context.
        
        Args:
            tenant_id: Required tenant ID
            employee_id: Required employee ID
        """
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.validate_context(tenant_id, employee_id):
                    raise TenantIsolationViolationError(f"Invalid context for {func.__name__}")
                return await func(*args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.validate_context(tenant_id, employee_id):
                    raise TenantIsolationViolationError(f"Invalid context for {func.__name__}")
                return func(*args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    def add_context_validator(self, validator: Callable[[Dict[str, Any]], bool]) -> None:
        """Add a custom context validator"""
        self.context_validators.append(validator)
        logger.info(f"Added context validator: {validator.__name__}")
    
    def add_context_interceptor(self, interceptor: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """Add a context interceptor that can modify context"""
        self.context_interceptors.append(interceptor)
        logger.info(f"Added context interceptor: {interceptor.__name__}")
    
    def _generate_validation_token(self, tenant_id: str, employee_id: str) -> str:
        """Generate a validation token for context integrity"""
        import hashlib
        
        token_data = f"{tenant_id}:{employee_id}:{datetime.now().strftime('%Y%m%d')}"
        return hashlib.sha256(token_data.encode()).hexdigest()[:16]
    
    def _run_validators(self, context: Dict[str, Any]) -> bool:
        """Run all registered validators on the context"""
        for validator in self.context_validators:
            try:
                if not validator(context):
                    logger.warning(f"Context validation failed: {validator.__name__}")
                    return False
            except Exception as e:
                logger.error(f"Context validator {validator.__name__} failed: {e}")
                return False
        
        return True
    
    def _run_interceptors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run all registered interceptors on the context"""
        modified_context = context.copy()
        
        for interceptor in self.context_interceptors:
            try:
                modified_context = interceptor(modified_context)
            except Exception as e:
                logger.error(f"Context interceptor {interceptor.__name__} failed: {e}")
        
        return modified_context
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context injection statistics"""
        return {
            **self.stats,
            "active_contexts": len(self.active_contexts),
            "validators_registered": len(self.context_validators),
            "interceptors_registered": len(self.context_interceptors),
            "validation_success_rate": (
                (self.stats["contexts_validated"] - self.stats["validation_failures"]) /
                max(self.stats["contexts_validated"], 1)
            ) * 100
        }
    
    def cleanup_expired_contexts(self, max_age_hours: int = 24) -> int:
        """Clean up expired contexts"""
        from datetime import timedelta
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        expired_contexts = []
        
        for context_id, context in self.active_contexts.items():
            created_at = datetime.fromisoformat(context["created_at"].replace("Z", "+00:00"))
            if created_at < cutoff_time:
                expired_contexts.append(context_id)
        
        for context_id in expired_contexts:
            del self.active_contexts[context_id]
        
        logger.info(f"Cleaned up {len(expired_contexts)} expired contexts")
        return len(expired_contexts)


class WorkflowContextManager:
    """Context manager for workflow context injection"""
    
    def __init__(self, injector: WorkflowContextInjector, context: Dict[str, Any]):
        self.injector = injector
        self.context = context
        self.previous_context = None
    
    def __enter__(self):
        # Save previous context
        self.previous_context = self.injector.get_current_context()
        
        # Inject new context
        self.injector.inject_context(self.context)
        
        return self.context
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        if self.previous_context:
            self.injector.inject_context(self.previous_context)
        else:
            self.injector.clear_context()


# Global context injector instance
context_injector = WorkflowContextInjector()


# Convenience functions
def create_context(tenant_id: str, employee_id: str, session_id: str, user_id: str, **kwargs) -> Dict[str, Any]:
    """Create workflow context"""
    return context_injector.create_workflow_context(tenant_id, employee_id, session_id, user_id, kwargs)


def inject_context(context: Dict[str, Any]) -> None:
    """Inject workflow context"""
    context_injector.inject_context(context)


def get_context() -> Optional[Dict[str, Any]]:
    """Get current workflow context"""
    return context_injector.get_current_context()


def validate_context(tenant_id: str, employee_id: str) -> bool:
    """Validate current context"""
    return context_injector.validate_context(tenant_id, employee_id)


def clear_context() -> None:
    """Clear current context"""
    context_injector.clear_context()


def with_context(context: Dict[str, Any]):
    """Context manager for workflow context"""
    return context_injector.with_context(context)


def context_required(tenant_id: str, employee_id: str):
    """Decorator for context validation"""
    return context_injector.context_required(tenant_id, employee_id)


# Context validation decorators
def require_tenant_context(func):
    """Decorator to ensure tenant context is present"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        current_tenant = context_injector.get_current_tenant_id()
        if not current_tenant:
            raise TenantIsolationViolationError("No tenant context available")
        return await func(*args, **kwargs)
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        current_tenant = context_injector.get_current_tenant_id()
        if not current_tenant:
            raise TenantIsolationViolationError("No tenant context available")
        return func(*args, **kwargs)
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def require_employee_context(func):
    """Decorator to ensure employee context is present"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        current_employee = context_injector.get_current_employee_id()
        if not current_employee:
            raise TenantIsolationViolationError("No employee context available")
        return await func(*args, **kwargs)
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        current_employee = context_injector.get_current_employee_id()
        if not current_employee:
            raise TenantIsolationViolationError("No employee context available")
        return func(*args, **kwargs)
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


# Context propagation utilities
def propagate_context_to_mcae_agents(agents: Dict[str, Any]) -> None:
    """Propagate current context to MCAE agents"""
    current_context = get_context()
    if not current_context:
        logger.warning("No context to propagate to MCAE agents")
        return
    
    for agent_name, agent in agents.items():
        try:
            if hasattr(agent, 'set_context'):
                agent.set_context(current_context)
            elif hasattr(agent, 'context'):
                agent.context = current_context
            
            logger.debug(f"Propagated context to agent: {agent_name}")
        except Exception as e:
            logger.error(f"Failed to propagate context to agent {agent_name}: {e}")


def ensure_context_in_mcae_tools(tools: List[Any]) -> None:
    """Ensure MCAE tools have proper context"""
    current_context = get_context()
    if not current_context:
        logger.warning("No context to propagate to MCAE tools")
        return
    
    for tool in tools:
        try:
            if hasattr(tool, 'set_tenant_context'):
                tool.set_tenant_context(
                    current_context["tenant_id"],
                    current_context["employee_id"]
                )
            
            logger.debug(f"Set context for tool: {tool.__class__.__name__}")
        except Exception as e:
            logger.error(f"Failed to set context for tool {tool.__class__.__name__}: {e}")


# Health check function
async def health_check() -> Dict[str, Any]:
    """Health check for context injection system"""
    try:
        stats = context_injector.get_stats()
        
        return {
            "status": "healthy",
            "stats": stats,
            "current_context_exists": get_context() is not None,
            "current_tenant": context_injector.get_current_tenant_id(),
            "current_employee": context_injector.get_current_employee_id()
        }
    except Exception as e:
        logger.error(f"Context injection health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }