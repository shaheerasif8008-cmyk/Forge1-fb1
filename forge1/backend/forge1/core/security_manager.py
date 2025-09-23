# forge1/backend/forge1/core/security_manager.py
"""
Security Manager for Forge 1

Enterprise-grade security with:
- Azure KeyVault integration for secrets management
- Role-based access control (RBAC) system
- Enhanced authentication and authorization
- Request validation and sanitization
- Audit logging and compliance tracking
"""

import logging
import os
import time
import hashlib
import jwt
from typing import Dict, Any, Optional, List, Set
from fastapi import Request, Response, HTTPException
from enum import Enum
from dataclasses import dataclass
from forge1.core.policy_engine import PolicyEngine

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    VIEWER = "viewer"
    SYSTEM = "system"

class Permission(Enum):
    """System permissions"""
    READ_PLANS = "read_plans"
    WRITE_PLANS = "write_plans"
    DELETE_PLANS = "delete_plans"
    MANAGE_USERS = "manage_users"
    VIEW_METRICS = "view_metrics"
    MANAGE_MODELS = "manage_models"
    SYSTEM_ADMIN = "system_admin"
    COMPLIANCE_VIEW = "compliance_view"

@dataclass
class AuditLogEntry:
    """Audit log entry structure"""
    timestamp: float
    user_id: str
    action: str
    resource: str
    ip_address: str
    user_agent: str
    success: bool
    details: Dict[str, Any]

@dataclass
class UserContext:
    """Enhanced user context with RBAC"""
    user_id: str
    roles: List[UserRole]
    permissions: Set[Permission]
    tenant_id: Optional[str] = None
    security_level: str = "standard"
    last_validated: float = 0.0

class SecurityManager:
    """Enterprise security manager with Azure KeyVault and RBAC"""
    
    def __init__(self):
        self.blocked_ips = set()
        self.rate_limits = {}
        self.audit_log: List[AuditLogEntry] = []
        self.user_sessions = {}
        self.keyvault_client = None
        self.role_permissions = self._initialize_role_permissions()
        self._initialize_keyvault()
        self.policy_engine = PolicyEngine()
    
    def _initialize_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """Initialize role-based permissions mapping"""
        return {
            UserRole.ADMIN: {
                Permission.READ_PLANS, Permission.WRITE_PLANS, Permission.DELETE_PLANS,
                Permission.MANAGE_USERS, Permission.VIEW_METRICS, Permission.MANAGE_MODELS,
                Permission.SYSTEM_ADMIN, Permission.COMPLIANCE_VIEW
            },
            UserRole.MANAGER: {
                Permission.READ_PLANS, Permission.WRITE_PLANS, Permission.DELETE_PLANS,
                Permission.VIEW_METRICS, Permission.COMPLIANCE_VIEW
            },
            UserRole.USER: {
                Permission.READ_PLANS, Permission.WRITE_PLANS, Permission.VIEW_METRICS
            },
            UserRole.VIEWER: {
                Permission.READ_PLANS, Permission.VIEW_METRICS
            },
            UserRole.SYSTEM: {
                Permission.READ_PLANS, Permission.WRITE_PLANS, Permission.SYSTEM_ADMIN
            }
        }
    
    def _initialize_keyvault(self):
        """Initialize Azure KeyVault client"""
        try:
            from azure.keyvault.secrets import SecretClient
            from azure.identity import DefaultAzureCredential
            
            keyvault_url = os.getenv("AZURE_KEYVAULT_URL")
            if keyvault_url:
                credential = DefaultAzureCredential()
                self.keyvault_client = SecretClient(vault_url=keyvault_url, credential=credential)
                logger.info("Azure KeyVault client initialized successfully")
            else:
                logger.warning("AZURE_KEYVAULT_URL not configured, using environment variables for secrets")
                
        except ImportError:
            logger.warning("Azure KeyVault libraries not available, using environment variables")
        except Exception as e:
            logger.error(f"Failed to initialize KeyVault client: {e}")
    
    async def get_secret(self, secret_name: str) -> Optional[str]:
        """Get secret from Azure KeyVault or environment"""
        try:
            if self.keyvault_client:
                secret = await self.keyvault_client.get_secret(secret_name)
                return secret.value
            else:
                # Fallback to environment variables
                return os.getenv(secret_name.upper().replace("-", "_"))
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            return None
    
    async def validate_request(self, request: Request, call_next) -> Response:
        """Enhanced request validation with RBAC and audit logging"""
        
        start_time = time.time()
        client_ip = request.client.host
        user_agent = request.headers.get("user-agent", "unknown")
        
        # IP blocking check
        if client_ip in self.blocked_ips:
            await self._log_security_event(
                user_id="unknown",
                action="blocked_ip_access",
                resource=str(request.url),
                ip_address=client_ip,
                user_agent=user_agent,
                success=False,
                details={"reason": "IP blocked"}
            )
            logger.warning(f"Blocked request from IP: {client_ip}")
            return Response(status_code=403, content="Access denied")
        
        # Rate limiting check
        if await self._check_rate_limit(client_ip):
            await self._log_security_event(
                user_id="unknown",
                action="rate_limit_exceeded",
                resource=str(request.url),
                ip_address=client_ip,
                user_agent=user_agent,
                success=False,
                details={"reason": "Rate limit exceeded"}
            )
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return Response(status_code=429, content="Rate limit exceeded")
        
        # Request validation
        try:
            await self._validate_request_headers(request)
            await self._validate_request_content(request)
        except HTTPException as e:
            await self._log_security_event(
                user_id="unknown",
                action="request_validation_failed",
                resource=str(request.url),
                ip_address=client_ip,
                user_agent=user_agent,
                success=False,
                details={"error": str(e.detail)}
            )
            raise e
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log successful request
            processing_time = time.time() - start_time
            await self._log_security_event(
                user_id=getattr(request.state, 'user_id', 'unknown'),
                action="request_processed",
                resource=str(request.url),
                ip_address=client_ip,
                user_agent=user_agent,
                success=True,
                details={
                    "method": request.method,
                    "status_code": response.status_code,
                    "processing_time": processing_time
                }
            )
            
            return response
            
        except Exception as e:
            # Log failed request
            await self._log_security_event(
                user_id=getattr(request.state, 'user_id', 'unknown'),
                action="request_failed",
                resource=str(request.url),
                ip_address=client_ip,
                user_agent=user_agent,
                success=False,
                details={"error": str(e)}
            )
            raise e
    
    async def get_enhanced_user_details(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Get enhanced user details with RBAC and security validation"""
        
        try:
            # Import Microsoft's auth utility
            import sys
            sys.path.append('../../../../Multi-Agent-Custom-Automation-Engine-Solution-Accelerator/src/backend')
            from auth.auth_utils import get_authenticated_user_details
            
            # Get base user details
            base_user_details = get_authenticated_user_details(request_headers=headers)
            user_id = base_user_details.get("user_principal_id")
            
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid authentication")
            
            # Get or create user context with RBAC
            user_context = await self._get_user_context(user_id, headers)
            
            # Enhanced user details with RBAC
            enhanced_details = {
                **base_user_details,
                "user_id": user_id,
                "roles": [role.value for role in user_context.roles],
                "permissions": [perm.value for perm in user_context.permissions],
                "security_level": user_context.security_level,
                "tenant_id": user_context.tenant_id,
                "last_validated": user_context.last_validated,
                "session_valid": await self._validate_user_session(user_id)
            }
            
            return enhanced_details
            
        except Exception as e:
            logger.error(f"Failed to get enhanced user details: {e}")
            raise HTTPException(status_code=401, detail="Authentication failed")
    
    async def _get_user_context(self, user_id: str, headers: Dict[str, str]) -> UserContext:
        """Get or create user context with RBAC"""
        
        # Check cache first
        if user_id in self.user_sessions:
            context = self.user_sessions[user_id]
            if time.time() - context.last_validated < 3600:  # 1 hour cache
                return context
        
        # Determine user roles (in production, this would query a user management system)
        roles = await self._determine_user_roles(user_id, headers)
        
        # Calculate permissions based on roles
        permissions = set()
        for role in roles:
            permissions.update(self.role_permissions.get(role, set()))
        
        # Create user context
        context = UserContext(
            user_id=user_id,
            roles=roles,
            permissions=permissions,
            tenant_id=await self._get_user_tenant(user_id),
            security_level="enterprise",
            last_validated=time.time()
        )
        
        # Cache context
        self.user_sessions[user_id] = context
        
        return context
    
    async def _determine_user_roles(self, user_id: str, headers: Dict[str, str]) -> List[UserRole]:
        """Determine user roles (placeholder implementation)"""
        
        # In production, this would query Azure AD, database, or other user management system
        # For now, return default roles based on user ID patterns
        
        if user_id.endswith("@admin.com"):
            return [UserRole.ADMIN]
        elif user_id.endswith("@manager.com"):
            return [UserRole.MANAGER]
        elif user_id.endswith("@system.com"):
            return [UserRole.SYSTEM]
        else:
            return [UserRole.USER]
    
    async def _get_user_tenant(self, user_id: str) -> Optional[str]:
        """Get user tenant ID for multi-tenancy"""
        
        # Extract tenant from user ID or query tenant service
        # For now, use domain as tenant
        if "@" in user_id:
            domain = user_id.split("@")[1]
            return domain.replace(".", "_")
        
        return "default_tenant"
    
    async def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission"""
        
        try:
            if user_id not in self.user_sessions:
                # Force context refresh
                await self._get_user_context(user_id, {})
            
            context = self.user_sessions.get(user_id)
            if not context:
                return False
            
            return permission in context.permissions
            
        except Exception as e:
            logger.error(f"Permission check failed for user {user_id}: {e}")
            return False
    
    async def require_permission(self, user_id: str, permission: Permission) -> None:
        """Require specific permission or raise HTTPException"""
        
        if not await self.check_permission(user_id, permission):
            await self._log_security_event(
                user_id=user_id,
                action="permission_denied",
                resource=permission.value,
                ip_address="unknown",
                user_agent="unknown",
                success=False,
                details={"required_permission": permission.value}
            )
            raise HTTPException(
                status_code=403, 
                detail=f"Insufficient permissions. Required: {permission.value}"
            )

    async def enforce_policy(self, user_details: Dict[str, Any], resource: str, action: str, attributes: Dict[str, Any]) -> None:
        """Policy enforcement using OPA if configured, else ABAC fallback."""
        opa_url = os.getenv("OPA_URL")
        decision_data = None
        if opa_url:
            try:
                from forge1.policy.opa_client import OPAClient
                client = OPAClient(opa_url, timeout_ms=int(os.getenv("OPA_TIMEOUT_MS", "2000")))
                policy_path = os.getenv("OPA_POLICY_PATH", "forge1/allow")
                input_data = {
                    "user": user_details,
                    "resource": resource,
                    "action": action,
                    "attributes": attributes,
                }
                decision_data = await client.evaluate(policy_path, input_data)
            except Exception as e:
                logger.warning(f"OPA evaluation failed, falling back to ABAC: {e}
")
        if decision_data is not None and isinstance(decision_data, dict) and "allow" in decision_data:
            allow = bool(decision_data.get("allow"))
            reason = str(decision_data.get("reason", "opa_decision"))
            if not allow:
                await self._log_security_event(
                    user_id=user_details.get("user_principal_id", "unknown"),
                    action="policy_denied",
                    resource=resource,
                    ip_address="unknown",
                    user_agent="opa",
                    success=False,
                    details={"reason": reason, "action": action, "attributes": attributes}
                )
                raise HTTPException(status_code=403, detail=f"Policy denied: {reason}")
            return
        # Fallback ABAC
        decision = self.policy_engine.evaluate(user_details, resource, action, attributes)
        if not decision.allow:
            await self._log_security_event(
                user_id=user_details.get("user_principal_id", "unknown"),
                action="policy_denied",
                resource=resource,
                ip_address="unknown",
                user_agent="policy",
                success=False,
                details={"reason": decision.reason, "action": action, "attributes": attributes}
            )
            raise HTTPException(status_code=403, detail=f"Policy denied: {decision.reason}")
    
    async def sanitize_error_message(self, error_message: str) -> str:
        """Sanitize error messages to prevent information leakage"""
        
        # Remove sensitive information patterns
        sensitive_patterns = [
            r'password[=:]\s*\S+',
            r'token[=:]\s*\S+',
            r'key[=:]\s*\S+',
            r'secret[=:]\s*\S+',
        ]
        
        sanitized = error_message
        for pattern in sensitive_patterns:
            import re
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    async def _validate_request_headers(self, request: Request) -> None:
        """Validate request headers for security"""
        
        # Check for required security headers
        required_headers = ["user-agent"]
        for header in required_headers:
            if header not in request.headers:
                logger.warning(f"Missing required header: {header}")
        
        # Validate content type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                logger.warning(f"Unexpected content type: {content_type}")
    
    async def _validate_request_content(self, request: Request) -> None:
        """Validate request content for security"""
        
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > 10 * 1024 * 1024:  # 10MB limit
                    raise HTTPException(status_code=413, detail="Request too large")
            except ValueError:
                logger.warning("Invalid content-length header")
    
    async def _validate_user_session(self, user_id: str) -> bool:
        """Validate user session"""
        
        # Check if user session exists and is valid
        if user_id in self.user_sessions:
            context = self.user_sessions[user_id]
            # Session valid for 8 hours
            return time.time() - context.last_validated < 28800
        
        return False
    
    async def _log_security_event(self, user_id: str, action: str, resource: str, 
                                 ip_address: str, user_agent: str, success: bool, 
                                 details: Dict[str, Any]) -> None:
        """Log security event for audit trail and persist (B.3)"""
        from forge1.core.database_config import get_database_manager
        from forge1.core.tenancy import get_current_tenant

        audit_entry = AuditLogEntry(
            timestamp=time.time(),
            user_id=user_id,
            action=action,
            resource=resource,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details
        )
        
        self.audit_log.append(audit_entry)
        
        # Keep only recent entries (last 10000)
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]
        
        # Log to system logger for external SIEM integration
        log_level = logging.INFO if success else logging.WARNING
        logger.log(log_level, f"Security Event: {action} by {user_id} on {resource} - {'Success' if success else 'Failed'}")
        # Persist to audit DB
        try:
            db = await get_database_manager()
            tenant = get_current_tenant()
            async with db.postgres.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO forge1_audit.audit_log(event_type, user_id, employee_id, event_data, ip_address, user_agent)
                    VALUES ($1, $2, NULL, $3, $4, $5)
                    """,
                    action,
                    user_id,
                    {"resource": resource, "success": success, "details": details, "tenant": tenant},
                    ip_address,
                    user_agent,
                )
        except Exception as e:
            logger.debug(f"Audit persistence failed: {e}")
    
    async def get_audit_log(self, user_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get audit log entries for a user"""
        
        # Check permission
        await self.require_permission(user_id, Permission.COMPLIANCE_VIEW)
        
        cutoff_time = time.time() - (hours * 3600)
        
        filtered_entries = [
            {
                "timestamp": entry.timestamp,
                "user_id": entry.user_id,
                "action": entry.action,
                "resource": entry.resource,
                "ip_address": entry.ip_address,
                "success": entry.success,
                "details": entry.details
            }
            for entry in self.audit_log
            if entry.timestamp >= cutoff_time
        ]
        
        return filtered_entries
    
    async def _check_rate_limit(self, client_ip: str) -> bool:
        """Enhanced rate limiting with different tiers"""
        
        current_time = time.time()
        
        if client_ip not in self.rate_limits:
            self.rate_limits[client_ip] = []
        
        # Clean old requests (older than 1 minute)
        self.rate_limits[client_ip] = [
            req_time for req_time in self.rate_limits[client_ip]
            if current_time - req_time < 60
        ]
        
        # Different limits based on IP patterns
        if client_ip.startswith("127.") or client_ip.startswith("::1"):
            # Localhost - higher limit
            limit = 1000
        elif client_ip.startswith("10.") or client_ip.startswith("192.168."):
            # Internal network - higher limit
            limit = 500
        else:
            # External - standard limit
            limit = 100
        
        # Check if limit exceeded
        if len(self.rate_limits[client_ip]) >= limit:
            return True
        
        # Add current request
        self.rate_limits[client_ip].append(current_time)
        return False
    
    async def block_ip(self, ip_address: str, reason: str) -> None:
        """Block an IP address"""
        
        self.blocked_ips.add(ip_address)
        logger.warning(f"Blocked IP {ip_address}: {reason}")
        
        # Log security event
        await self._log_security_event(
            user_id="system",
            action="ip_blocked",
            resource=ip_address,
            ip_address=ip_address,
            user_agent="system",
            success=True,
            details={"reason": reason}
        )
    
    async def unblock_ip(self, ip_address: str) -> None:
        """Unblock an IP address"""
        
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            logger.info(f"Unblocked IP {ip_address}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced security manager health check"""
        
        try:
            keyvault_status = "connected" if self.keyvault_client else "not_configured"
            
            return {
                "status": "healthy",
                "keyvault_status": keyvault_status,
                "blocked_ips_count": len(self.blocked_ips),
                "active_sessions": len(self.user_sessions),
                "audit_log_entries": len(self.audit_log),
                "rate_limit_entries": len(self.rate_limits),
                "supported_roles": [role.value for role in UserRole],
                "supported_permissions": [perm.value for perm in Permission]
            }
            
        except Exception as e:
            logger.error(f"Security manager health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
