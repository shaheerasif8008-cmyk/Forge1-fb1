"""
Tool Permission Enforcement System

Validates tool access based on employee configuration and enforces
tenant-aware permissions for all MCAE tool executions.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timezone
from enum import Enum

from forge1.core.tenancy import get_current_tenant
from forge1.integrations.workflow_context_injector import get_context, require_tenant_context, require_employee_context

logger = logging.getLogger(__name__)


class PermissionLevel(str, Enum):
    """Tool permission levels"""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class ToolCategory(str, Enum):
    """Tool categories for permission grouping"""
    DOCUMENT = "document"
    COMMUNICATION = "communication"
    SEARCH = "search"
    STORAGE = "storage"
    ANALYSIS = "analysis"
    INTEGRATION = "integration"
    SYSTEM = "system"


class PermissionDeniedError(Exception):
    """Raised when tool access is denied"""
    pass


class ToolPermissionEnforcer:
    """
    Enforces tool permissions for MCAE agents based on employee configuration
    and tenant-specific access rules.
    """
    
    def __init__(self):
        # Tool registry with metadata
        self.tool_registry = {}
        
        # Permission rules
        self.permission_rules = {}
        self.tenant_restrictions = {}
        self.employee_permissions = {}
        
        # Audit logging
        self.access_log = []
        self.max_log_entries = 10000
        
        # Statistics
        self.stats = {
            "permission_checks": 0,
            "access_granted": 0,
            "access_denied": 0,
            "tools_registered": 0,
            "audit_entries": 0
        }
        
        # Initialize default tools
        self._register_default_tools()
    
    def register_tool(
        self,
        tool_name: str,
        category: ToolCategory,
        required_permission: PermissionLevel,
        tenant_restricted: bool = False,
        description: str = "",
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Register a tool with permission requirements.
        
        Args:
            tool_name: Name of the tool
            category: Tool category
            required_permission: Minimum permission level required
            tenant_restricted: Whether tool access is tenant-restricted
            description: Tool description
            metadata: Additional tool metadata
        """
        self.tool_registry[tool_name] = {
            "category": category,
            "required_permission": required_permission,
            "tenant_restricted": tenant_restricted,
            "description": description,
            "metadata": metadata or {},
            "registered_at": datetime.now(timezone.utc).isoformat()
        }
        
        self.stats["tools_registered"] += 1
        logger.info(f"Registered tool: {tool_name} (category: {category.value}, permission: {required_permission.value})")
    
    def set_employee_permissions(
        self,
        tenant_id: str,
        employee_id: str,
        tool_permissions: Dict[str, PermissionLevel]
    ) -> None:
        """
        Set tool permissions for a specific employee.
        
        Args:
            tenant_id: Tenant ID
            employee_id: Employee ID
            tool_permissions: Dictionary of tool_name -> permission_level
        """
        key = f"{tenant_id}:{employee_id}"
        self.employee_permissions[key] = tool_permissions
        
        logger.info(f"Set permissions for employee {employee_id} in tenant {tenant_id}: {len(tool_permissions)} tools")
    
    def set_tenant_restrictions(
        self,
        tenant_id: str,
        restricted_tools: Set[str],
        allowed_categories: Optional[Set[ToolCategory]] = None
    ) -> None:
        """
        Set tenant-level tool restrictions.
        
        Args:
            tenant_id: Tenant ID
            restricted_tools: Set of tool names that are restricted
            allowed_categories: Set of allowed tool categories (if None, all are allowed)
        """
        self.tenant_restrictions[tenant_id] = {
            "restricted_tools": restricted_tools,
            "allowed_categories": allowed_categories,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Set restrictions for tenant {tenant_id}: {len(restricted_tools)} restricted tools")
    
    @require_tenant_context
    @require_employee_context
    async def check_tool_permission(
        self,
        tool_name: str,
        operation: str = "execute",
        additional_context: Optional[Dict] = None
    ) -> bool:
        """
        Check if current employee has permission to use a tool.
        
        Args:
            tool_name: Name of the tool to check
            operation: Operation being performed (execute, read, write)
            additional_context: Additional context for permission check
            
        Returns:
            True if permission is granted, False otherwise
            
        Raises:
            PermissionDeniedError: If access is explicitly denied
        """
        self.stats["permission_checks"] += 1
        
        # Get current context
        context = get_context()
        if not context:
            self._log_access_attempt(tool_name, operation, False, "No context available")
            raise PermissionDeniedError("No workflow context available")
        
        tenant_id = context["tenant_id"]
        employee_id = context["employee_id"]
        
        try:
            # Check if tool is registered
            if tool_name not in self.tool_registry:
                self._log_access_attempt(tool_name, operation, False, "Tool not registered")
                self.stats["access_denied"] += 1
                return False
            
            tool_info = self.tool_registry[tool_name]
            
            # Check tenant restrictions
            if not self._check_tenant_restrictions(tenant_id, tool_name, tool_info):
                self._log_access_attempt(tool_name, operation, False, "Tenant restriction")
                self.stats["access_denied"] += 1
                raise PermissionDeniedError(f"Tool {tool_name} is restricted for tenant {tenant_id}")
            
            # Check employee permissions
            if not self._check_employee_permissions(tenant_id, employee_id, tool_name, tool_info, operation):
                self._log_access_attempt(tool_name, operation, False, "Insufficient employee permissions")
                self.stats["access_denied"] += 1
                raise PermissionDeniedError(f"Employee {employee_id} lacks permission for tool {tool_name}")
            
            # Check operation-specific permissions
            if not self._check_operation_permissions(tool_name, operation, tool_info):
                self._log_access_attempt(tool_name, operation, False, "Operation not permitted")
                self.stats["access_denied"] += 1
                raise PermissionDeniedError(f"Operation {operation} not permitted for tool {tool_name}")
            
            # All checks passed
            self._log_access_attempt(tool_name, operation, True, "Permission granted")
            self.stats["access_granted"] += 1
            
            logger.debug(f"Tool permission granted: {tool_name} for employee {employee_id}")
            return True
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            self._log_access_attempt(tool_name, operation, False, f"Error: {e}")
            self.stats["access_denied"] += 1
            logger.error(f"Error checking tool permission: {e}")
            return False
    
    def get_employee_allowed_tools(self, tenant_id: str, employee_id: str) -> List[Dict[str, Any]]:
        """
        Get list of tools that an employee is allowed to use.
        
        Args:
            tenant_id: Tenant ID
            employee_id: Employee ID
            
        Returns:
            List of allowed tools with their metadata
        """
        allowed_tools = []
        
        for tool_name, tool_info in self.tool_registry.items():
            try:
                # Check tenant restrictions
                if not self._check_tenant_restrictions(tenant_id, tool_name, tool_info):
                    continue
                
                # Check employee permissions
                if not self._check_employee_permissions(tenant_id, employee_id, tool_name, tool_info, "execute"):
                    continue
                
                # Tool is allowed
                allowed_tools.append({
                    "name": tool_name,
                    "category": tool_info["category"].value,
                    "description": tool_info["description"],
                    "required_permission": tool_info["required_permission"].value,
                    "tenant_restricted": tool_info["tenant_restricted"]
                })
                
            except Exception as e:
                logger.error(f"Error checking tool {tool_name} for employee {employee_id}: {e}")
        
        return allowed_tools
    
    def validate_tool_access_list(self, tenant_id: str, employee_id: str, tool_list: List[str]) -> Dict[str, bool]:
        """
        Validate access to a list of tools for an employee.
        
        Args:
            tenant_id: Tenant ID
            employee_id: Employee ID
            tool_list: List of tool names to validate
            
        Returns:
            Dictionary mapping tool names to access status
        """
        access_status = {}
        
        for tool_name in tool_list:
            try:
                if tool_name not in self.tool_registry:
                    access_status[tool_name] = False
                    continue
                
                tool_info = self.tool_registry[tool_name]
                
                # Check all permissions
                tenant_ok = self._check_tenant_restrictions(tenant_id, tool_name, tool_info)
                employee_ok = self._check_employee_permissions(tenant_id, employee_id, tool_name, tool_info, "execute")
                
                access_status[tool_name] = tenant_ok and employee_ok
                
            except Exception as e:
                logger.error(f"Error validating tool {tool_name}: {e}")
                access_status[tool_name] = False
        
        return access_status
    
    def _check_tenant_restrictions(self, tenant_id: str, tool_name: str, tool_info: Dict) -> bool:
        """Check tenant-level restrictions"""
        
        # Check if tenant has specific restrictions
        if tenant_id in self.tenant_restrictions:
            restrictions = self.tenant_restrictions[tenant_id]
            
            # Check if tool is explicitly restricted
            if tool_name in restrictions["restricted_tools"]:
                return False
            
            # Check if tool category is allowed
            allowed_categories = restrictions.get("allowed_categories")
            if allowed_categories is not None:
                tool_category = tool_info["category"]
                if tool_category not in allowed_categories:
                    return False
        
        return True
    
    def _check_employee_permissions(
        self,
        tenant_id: str,
        employee_id: str,
        tool_name: str,
        tool_info: Dict,
        operation: str
    ) -> bool:
        """Check employee-level permissions"""
        
        key = f"{tenant_id}:{employee_id}"
        
        # Check if employee has specific permissions set
        if key in self.employee_permissions:
            employee_perms = self.employee_permissions[key]
            
            # Check specific tool permission
            if tool_name in employee_perms:
                employee_level = employee_perms[tool_name]
                required_level = tool_info["required_permission"]
                
                return self._permission_level_sufficient(employee_level, required_level)
            
            # Check category-based permissions
            category_key = f"category:{tool_info['category'].value}"
            if category_key in employee_perms:
                employee_level = employee_perms[category_key]
                required_level = tool_info["required_permission"]
                
                return self._permission_level_sufficient(employee_level, required_level)
        
        # Default: deny access if no explicit permission
        return False
    
    def _check_operation_permissions(self, tool_name: str, operation: str, tool_info: Dict) -> bool:
        """Check operation-specific permissions"""
        
        # Map operations to required permission levels
        operation_requirements = {
            "read": PermissionLevel.READ,
            "execute": PermissionLevel.WRITE,
            "write": PermissionLevel.WRITE,
            "admin": PermissionLevel.ADMIN,
            "delete": PermissionLevel.ADMIN
        }
        
        required_level = operation_requirements.get(operation, PermissionLevel.WRITE)
        tool_required_level = tool_info["required_permission"]
        
        # Tool's required level must be sufficient for the operation
        return self._permission_level_sufficient(tool_required_level, required_level)
    
    def _permission_level_sufficient(self, granted_level: PermissionLevel, required_level: PermissionLevel) -> bool:
        """Check if granted permission level is sufficient for required level"""
        
        level_hierarchy = {
            PermissionLevel.NONE: 0,
            PermissionLevel.READ: 1,
            PermissionLevel.WRITE: 2,
            PermissionLevel.ADMIN: 3
        }
        
        granted_value = level_hierarchy.get(granted_level, 0)
        required_value = level_hierarchy.get(required_level, 0)
        
        return granted_value >= required_value
    
    def _log_access_attempt(self, tool_name: str, operation: str, granted: bool, reason: str) -> None:
        """Log tool access attempt for audit purposes"""
        
        context = get_context()
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_name": tool_name,
            "operation": operation,
            "granted": granted,
            "reason": reason,
            "tenant_id": context.get("tenant_id") if context else None,
            "employee_id": context.get("employee_id") if context else None,
            "session_id": context.get("session_id") if context else None
        }
        
        self.access_log.append(log_entry)
        self.stats["audit_entries"] += 1
        
        # Limit log size
        if len(self.access_log) > self.max_log_entries:
            self.access_log = self.access_log[-self.max_log_entries:]
        
        # Log security-relevant events
        if not granted:
            logger.warning(f"Tool access denied: {tool_name} for {context.get('employee_id') if context else 'unknown'} - {reason}")
    
    def _register_default_tools(self) -> None:
        """Register default tools with their permission requirements"""
        
        # Document tools
        self.register_tool(
            "document_parser",
            ToolCategory.DOCUMENT,
            PermissionLevel.READ,
            tenant_restricted=True,
            description="Parse and extract content from documents"
        )
        
        # Search tools
        self.register_tool(
            "vector_search",
            ToolCategory.SEARCH,
            PermissionLevel.READ,
            tenant_restricted=True,
            description="Semantic search within tenant data"
        )
        
        # Communication tools
        self.register_tool(
            "slack_poster",
            ToolCategory.COMMUNICATION,
            PermissionLevel.WRITE,
            tenant_restricted=True,
            description="Post messages to Slack channels"
        )
        
        # Storage tools
        self.register_tool(
            "drive_fetcher",
            ToolCategory.STORAGE,
            PermissionLevel.READ,
            tenant_restricted=True,
            description="Fetch files from Google Drive"
        )
        
        # Analysis tools
        self.register_tool(
            "legal_research",
            ToolCategory.ANALYSIS,
            PermissionLevel.READ,
            tenant_restricted=True,
            description="Legal research and case law analysis"
        )
        
        # Integration tools
        self.register_tool(
            "api_client",
            ToolCategory.INTEGRATION,
            PermissionLevel.WRITE,
            tenant_restricted=True,
            description="Generic API client for external integrations"
        )
        
        # System tools (admin only)
        self.register_tool(
            "system_monitor",
            ToolCategory.SYSTEM,
            PermissionLevel.ADMIN,
            tenant_restricted=False,
            description="System monitoring and diagnostics"
        )
    
    def get_access_log(self, tenant_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get access log entries, optionally filtered by tenant"""
        
        if tenant_id:
            filtered_log = [
                entry for entry in self.access_log
                if entry.get("tenant_id") == tenant_id
            ]
        else:
            filtered_log = self.access_log
        
        return filtered_log[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get permission enforcement statistics"""
        
        return {
            **self.stats,
            "registered_tools": len(self.tool_registry),
            "tenant_restrictions": len(self.tenant_restrictions),
            "employee_permissions": len(self.employee_permissions),
            "access_success_rate": (
                self.stats["access_granted"] / 
                max(self.stats["permission_checks"], 1)
            ) * 100
        }
    
    def clear_access_log(self) -> None:
        """Clear the access log"""
        self.access_log.clear()
        self.stats["audit_entries"] = 0
        logger.info("Cleared tool access log")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for tool permission enforcer"""
        
        try:
            stats = self.get_stats()
            
            return {
                "status": "healthy",
                "stats": stats,
                "tools_registered": len(self.tool_registry),
                "recent_access_attempts": len(self.access_log[-10:])
            }
        except Exception as e:
            logger.error(f"Tool permission enforcer health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global permission enforcer instance
permission_enforcer = ToolPermissionEnforcer()


# Convenience functions
def check_tool_permission(tool_name: str, operation: str = "execute") -> bool:
    """Check tool permission for current context"""
    return asyncio.run(permission_enforcer.check_tool_permission(tool_name, operation))


def register_tool(tool_name: str, category: ToolCategory, required_permission: PermissionLevel, **kwargs) -> None:
    """Register a tool with permission requirements"""
    permission_enforcer.register_tool(tool_name, category, required_permission, **kwargs)


def set_employee_permissions(tenant_id: str, employee_id: str, tool_permissions: Dict[str, PermissionLevel]) -> None:
    """Set tool permissions for an employee"""
    permission_enforcer.set_employee_permissions(tenant_id, employee_id, tool_permissions)


def get_allowed_tools(tenant_id: str, employee_id: str) -> List[Dict[str, Any]]:
    """Get allowed tools for an employee"""
    return permission_enforcer.get_employee_allowed_tools(tenant_id, employee_id)


# Decorator for tool functions
def require_tool_permission(tool_name: str, operation: str = "execute"):
    """Decorator to enforce tool permissions"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            if not await permission_enforcer.check_tool_permission(tool_name, operation):
                raise PermissionDeniedError(f"Access denied to tool {tool_name}")
            return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            if not check_tool_permission(tool_name, operation):
                raise PermissionDeniedError(f"Access denied to tool {tool_name}")
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator