# forge1/backend/forge1/middleware/security_middleware.py
"""
Security Middleware for Employee Lifecycle System

Implements comprehensive security features including:
- Tenant context extraction and validation
- Access control for client/employee operations
- Security headers and CORS configuration
- Request/response security validation

Requirements: 2.4, 8.1, 8.2, 8.3, 8.4
"""

import logging
import time
import uuid
from typing import Dict, Optional, Any, Callable
from datetime import datetime, timezone

from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from forge1.core.tenancy import set_current_tenant, get_current_tenant
from forge1.core.audit_logger import AuditLogger
from forge1.core.encryption_manager import EncryptionManager

logger = logging.getLogger(__name__)


class SecurityContext:
    """Security context for requests"""
    
    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        user_id: str,
        request_id: str,
        ip_address: str,
        user_agent: str,
        authenticated: bool = True
    ):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.user_id = user_id
        self.request_id = request_id
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.authenticated = authenticated
        self.created_at = datetime.now(timezone.utc)


class TenantIsolationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce tenant isolation and security policies.
    
    Features:
    - Tenant context extraction from headers/JWT
    - Access control validation
    - Security headers injection
    - Request/response audit logging
    - Rate limiting per tenant
    """
    
    def __init__(
        self,
        app,
        audit_logger: Optional[AuditLogger] = None,
        encryption_manager: Optional[EncryptionManager] = None
    ):
        super().__init__(app)
        self.audit_logger = audit_logger or AuditLogger()
        self.encryption_manager = encryption_manager or EncryptionManager()
        self.security_bearer = HTTPBearer(auto_error=False)
        
        # Security configuration
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        # Rate limiting (simple in-memory implementation)
        self.rate_limits = {}
        self.rate_limit_window = 3600  # 1 hour
        self.rate_limit_max_requests = 1000  # per tenant per hour
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through security middleware"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Extract security context
            security_context = await self._extract_security_context(request, request_id)
            
            # Set tenant context
            set_current_tenant(security_context.tenant_id)
            
            # Store security context in request state
            request.state.security_context = security_context
            
            # Validate access permissions
            await self._validate_access_permissions(request, security_context)
            
            # Check rate limits
            await self._check_rate_limits(security_context)
            
            # Log request start
            await self._log_request_start(request, security_context)
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response)
            
            # Log successful response
            processing_time = time.time() - start_time
            await self._log_request_success(request, response, security_context, processing_time)
            
            return response
            
        except HTTPException as e:
            # Log security violation
            processing_time = time.time() - start_time
            await self._log_security_violation(request, e, processing_time)
            
            # Create error response with security headers
            error_response = JSONResponse(
                status_code=e.status_code,
                content={"error": e.detail, "request_id": request_id}
            )
            self._add_security_headers(error_response)
            return error_response
            
        except Exception as e:
            # Log unexpected error
            logger.error(f"Security middleware error: {e}")
            processing_time = time.time() - start_time
            await self._log_system_error(request, e, processing_time)
            
            # Create error response
            error_response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "request_id": request_id}
            )
            self._add_security_headers(error_response)
            return error_response
    
    async def _extract_security_context(self, request: Request, request_id: str) -> SecurityContext:
        """Extract security context from request"""
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Try to extract from headers first (for API clients)
        tenant_id = request.headers.get("x-tenant-id")
        client_id = request.headers.get("x-client-id")
        user_id = request.headers.get("x-user-id")
        
        if tenant_id and client_id and user_id:
            return SecurityContext(
                tenant_id=tenant_id,
                client_id=client_id,
                user_id=user_id,
                request_id=request_id,
                ip_address=client_ip,
                user_agent=user_agent,
                authenticated=True
            )
        
        # Try to extract from Authorization header (JWT)
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            
            # For demo purposes, handle demo token
            if token == "demo_token":
                return SecurityContext(
                    tenant_id="demo_client_001",
                    client_id="demo_client_001",
                    user_id="demo_user_001",
                    request_id=request_id,
                    ip_address=client_ip,
                    user_agent=user_agent,
                    authenticated=True
                )
            
            # In production, decode JWT and extract claims
            try:
                # jwt_payload = jwt.decode(token, secret_key, algorithms=["HS256"])
                # return SecurityContext from JWT claims
                pass
            except Exception as e:
                logger.warning(f"Invalid JWT token: {e}")
                raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        # Check if this is a public endpoint
        if self._is_public_endpoint(request.url.path):
            return SecurityContext(
                tenant_id="public",
                client_id="public",
                user_id="anonymous",
                request_id=request_id,
                ip_address=client_ip,
                user_agent=user_agent,
                authenticated=False
            )
        
        # No valid authentication found
        raise HTTPException(status_code=401, detail="Authentication required")
    
    async def _validate_access_permissions(self, request: Request, context: SecurityContext):
        """Validate access permissions for the request"""
        
        # Skip validation for public endpoints
        if not context.authenticated:
            return
        
        # Extract resource identifiers from path
        path_parts = request.url.path.strip("/").split("/")
        
        # Check client access for employee operations
        if "clients" in path_parts:
            try:
                client_index = path_parts.index("clients")
                if client_index + 1 < len(path_parts):
                    requested_client_id = path_parts[client_index + 1]
                    
                    # Validate tenant can access this client
                    if requested_client_id != context.client_id and context.tenant_id != "admin":
                        await self.audit_logger.log_security_violation(
                            event_type="unauthorized_client_access",
                            context=context,
                            details={
                                "requested_client_id": requested_client_id,
                                "actual_client_id": context.client_id,
                                "path": request.url.path
                            }
                        )
                        raise HTTPException(
                            status_code=403, 
                            detail="Access denied: insufficient permissions for client"
                        )
            except ValueError:
                # "clients" not in path, skip client validation
                pass
        
        # Additional permission checks can be added here
        # - Role-based access control
        # - Resource-specific permissions
        # - Time-based access restrictions
    
    async def _check_rate_limits(self, context: SecurityContext):
        """Check rate limits for the tenant"""
        
        if not context.authenticated:
            return
        
        current_time = time.time()
        tenant_key = f"rate_limit:{context.tenant_id}"
        
        # Clean old entries
        if tenant_key in self.rate_limits:
            self.rate_limits[tenant_key] = [
                timestamp for timestamp in self.rate_limits[tenant_key]
                if current_time - timestamp < self.rate_limit_window
            ]
        else:
            self.rate_limits[tenant_key] = []
        
        # Check if rate limit exceeded
        if len(self.rate_limits[tenant_key]) >= self.rate_limit_max_requests:
            await self.audit_logger.log_security_violation(
                event_type="rate_limit_exceeded",
                context=context,
                details={
                    "requests_in_window": len(self.rate_limits[tenant_key]),
                    "max_requests": self.rate_limit_max_requests,
                    "window_seconds": self.rate_limit_window
                }
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Add current request to rate limit tracking
        self.rate_limits[tenant_key].append(current_time)
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        # Add request tracking header
        response.headers["X-Request-ID"] = getattr(response, "request_id", "unknown")
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        # Check for forwarded headers (when behind proxy/load balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection IP
        return getattr(request.client, "host", "unknown")
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public (doesn't require authentication)"""
        public_endpoints = [
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/metrics"
        ]
        
        return any(path.startswith(endpoint) for endpoint in public_endpoints)
    
    async def _log_request_start(self, request: Request, context: SecurityContext):
        """Log request start for audit trail"""
        await self.audit_logger.log_request(
            event_type="request_start",
            context=context,
            details={
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "content_type": request.headers.get("content-type"),
                "content_length": request.headers.get("content-length")
            }
        )
    
    async def _log_request_success(
        self,
        request: Request,
        response: Response,
        context: SecurityContext,
        processing_time: float
    ):
        """Log successful request completion"""
        await self.audit_logger.log_request(
            event_type="request_success",
            context=context,
            details={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "processing_time_ms": round(processing_time * 1000, 2),
                "response_size": len(response.body) if hasattr(response, "body") else 0
            }
        )
    
    async def _log_security_violation(
        self,
        request: Request,
        exception: HTTPException,
        processing_time: float
    ):
        """Log security violations"""
        context = getattr(request.state, "security_context", None)
        
        await self.audit_logger.log_security_violation(
            event_type="security_violation",
            context=context,
            details={
                "method": request.method,
                "path": request.url.path,
                "status_code": exception.status_code,
                "error_detail": exception.detail,
                "processing_time_ms": round(processing_time * 1000, 2),
                "ip_address": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent")
            }
        )
    
    async def _log_system_error(
        self,
        request: Request,
        exception: Exception,
        processing_time: float
    ):
        """Log system errors"""
        context = getattr(request.state, "security_context", None)
        
        await self.audit_logger.log_system_event(
            event_type="system_error",
            context=context,
            details={
                "method": request.method,
                "path": request.url.path,
                "error_type": type(exception).__name__,
                "error_message": str(exception),
                "processing_time_ms": round(processing_time * 1000, 2)
            }
        )


class EmployeeAccessControlMiddleware:
    """
    Middleware specifically for employee-related access control.
    
    Validates that users can only access employees they own or have permission to access.
    """
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
    
    async def validate_employee_access(
        self,
        client_id: str,
        employee_id: str,
        security_context: SecurityContext,
        operation: str = "read"
    ) -> bool:
        """
        Validate that the current user can access the specified employee.
        
        Args:
            client_id: Client ID from the request
            employee_id: Employee ID from the request
            security_context: Current security context
            operation: Type of operation (read, write, delete)
        
        Returns:
            True if access is allowed, raises HTTPException otherwise
        """
        
        # Admin users can access everything
        if security_context.tenant_id == "admin":
            return True
        
        # Users can only access their own client's employees
        if client_id != security_context.client_id:
            raise HTTPException(
                status_code=403,
                detail="Access denied: cannot access employees from other clients"
            )
        
        # Additional checks can be added here:
        # - Check if employee exists and belongs to client
        # - Check user role permissions for the operation
        # - Check if employee is active/accessible
        
        if self.db_manager:
            try:
                async with self.db_manager.get_connection() as conn:
                    # Verify employee exists and belongs to client
                    employee_exists = await conn.fetchval("""
                        SELECT EXISTS(
                            SELECT 1 FROM employees 
                            WHERE id = $1 AND client_id = $2 AND status != 'deleted'
                        )
                    """, employee_id, client_id)
                    
                    if not employee_exists:
                        raise HTTPException(
                            status_code=404,
                            detail="Employee not found or access denied"
                        )
            except Exception as e:
                logger.error(f"Error validating employee access: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to validate employee access"
                )
        
        return True


# Utility functions for security context

def get_security_context(request: Request) -> Optional[SecurityContext]:
    """Get security context from request state"""
    return getattr(request.state, "security_context", None)


def require_authentication(request: Request) -> SecurityContext:
    """Require authentication and return security context"""
    context = get_security_context(request)
    if not context or not context.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    return context


def require_client_access(request: Request, client_id: str) -> SecurityContext:
    """Require access to specific client"""
    context = require_authentication(request)
    
    if context.client_id != client_id and context.tenant_id != "admin":
        raise HTTPException(
            status_code=403,
            detail="Access denied: insufficient permissions for client"
        )
    
    return context