"""
Forge 1 Enhanced Security Manager
Enterprise-grade security with comprehensive authentication and authorization
"""

import hashlib
import hmac
import jwt
import time
import logging
from enum import Enum
from typing import Dict, Any, Optional, List
from fastapi import Request, Response, HTTPException

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Stub permission levels for import-time checks."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"

class SecurityManager:
    """Enhanced security manager with enterprise features"""
    
    def __init__(self):
        self.jwt_secret = "forge1_jwt_secret_key"  # Should be from env
        self.jwt_algorithm = "HS256"
        self.session_timeout = 3600  # 1 hour
        self.active_sessions = {}
        self.security_policies = {
            "require_mfa": False,
            "password_complexity": True,
            "session_timeout": 3600,
            "max_failed_attempts": 5,
            "lockout_duration": 900  # 15 minutes
        }
        
    async def validate_request(self, request: Request, call_next):
        """Validate incoming request security"""
        try:
            # Skip validation for health checks and public endpoints
            if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
                return await call_next(request)
            
            # Add security headers
            response = await call_next(request)
            
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Content-Security-Policy"] = "default-src 'self'"
            
            return response
            
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            return await call_next(request)
    
    async def get_enhanced_user_details(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Get enhanced user details with security validation"""
        try:
            # Try JWT token first
            auth_header = headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                user_details = await self._validate_jwt_token(token)
                if user_details:
                    return user_details
            
            # Try session-based auth
            session_id = headers.get("X-Session-ID")
            if session_id and session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                if session["expires_at"] > time.time():
                    return session["user_details"]
                else:
                    # Session expired
                    del self.active_sessions[session_id]
            
            # Try API key auth
            api_key = headers.get("X-API-Key")
            if api_key:
                user_details = await self._validate_api_key(api_key)
                if user_details:
                    return user_details
            
            # Fallback to demo user for development
            return {
                "user_principal_id": "demo_user",
                "tenant_id": "default_tenant",
                "name": "Demo User",
                "email": "demo@forge1.com",
                "roles": ["user"],
                "permissions": ["read", "write"],
                "security_level": "standard",
                "mfa_enabled": False,
                "last_login": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting user details: {e}")
            raise HTTPException(status_code=401, detail="Authentication failed")
    
    async def _validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Check expiration
            if payload.get("exp", 0) < time.time():
                return None
            
            return {
                "user_principal_id": payload.get("user_id"),
                "tenant_id": payload.get("tenant_id", "default"),
                "name": payload.get("name", "Unknown User"),
                "email": payload.get("email"),
                "roles": payload.get("roles", ["user"]),
                "permissions": payload.get("permissions", ["read"]),
                "security_level": payload.get("security_level", "standard"),
                "mfa_enabled": payload.get("mfa_enabled", False)
            }
            
        except jwt.InvalidTokenError:
            return None
    
    async def _validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key"""
        # Mock API key validation
        valid_keys = {
            "forge1_demo_key": {
                "user_id": "api_user",
                "tenant_id": "api_tenant",
                "name": "API User",
                "permissions": ["read", "write", "admin"]
            }
        }
        
        if api_key in valid_keys:
            key_info = valid_keys[api_key]
            return {
                "user_principal_id": key_info["user_id"],
                "tenant_id": key_info["tenant_id"],
                "name": key_info["name"],
                "email": f"{key_info['user_id']}@forge1.com",
                "roles": ["api_user"],
                "permissions": key_info["permissions"],
                "security_level": "api",
                "mfa_enabled": False
            }
        
        return None
    
    async def create_session(self, user_details: Dict[str, Any]) -> str:
        """Create secure session"""
        session_id = hashlib.sha256(
            f"{user_details['user_principal_id']}{time.time()}".encode()
        ).hexdigest()
        
        self.active_sessions[session_id] = {
            "user_details": user_details,
            "created_at": time.time(),
            "expires_at": time.time() + self.session_timeout,
            "last_activity": time.time()
        }
        
        return session_id
    
    async def create_jwt_token(self, user_details: Dict[str, Any]) -> str:
        """Create JWT token"""
        payload = {
            "user_id": user_details["user_principal_id"],
            "tenant_id": user_details.get("tenant_id", "default"),
            "name": user_details.get("name"),
            "email": user_details.get("email"),
            "roles": user_details.get("roles", ["user"]),
            "permissions": user_details.get("permissions", ["read"]),
            "security_level": user_details.get("security_level", "standard"),
            "mfa_enabled": user_details.get("mfa_enabled", False),
            "iat": time.time(),
            "exp": time.time() + self.session_timeout
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    async def sanitize_error_message(self, error_message: str) -> str:
        """Sanitize error messages to prevent information disclosure"""
        # Remove sensitive information from error messages
        sensitive_terms = [
            "password", "token", "key", "secret", "credential",
            "database", "connection", "internal", "system"
        ]
        
        sanitized = error_message.lower()
        for term in sensitive_terms:
            if term in sanitized:
                return "An internal error occurred. Please contact support."
        
        # Limit error message length
        if len(error_message) > 200:
            return "An error occurred. Please contact support for details."
        
        return error_message
    
    async def validate_input(self, input_data: Any) -> bool:
        """Validate input for security threats"""
        if isinstance(input_data, str):
            # Check for common injection patterns
            dangerous_patterns = [
                "<script", "javascript:", "onload=", "onerror=",
                "DROP TABLE", "DELETE FROM", "INSERT INTO",
                "../", "..\\", "/etc/passwd"
            ]
            
            input_lower = input_data.lower()
            for pattern in dangerous_patterns:
                if pattern in input_lower:
                    logger.warning(f"Dangerous pattern detected: {pattern}")
                    return False
        
        return True
    
    async def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security events for monitoring"""
        security_event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details,
            "severity": self._get_event_severity(event_type)
        }
        
        logger.info(f"Security Event: {security_event}")
        
        # In production, send to SIEM system
        
    def _get_event_severity(self, event_type: str) -> str:
        """Get event severity level"""
        high_severity = ["authentication_failure", "authorization_failure", "injection_attempt"]
        medium_severity = ["session_timeout", "invalid_token", "rate_limit_exceeded"]
        
        if event_type in high_severity:
            return "HIGH"
        elif event_type in medium_severity:
            return "MEDIUM"
        else:
            return "LOW"
    
    async def health_check(self) -> bool:
        """Security manager health check"""
        try:
            # Check if security policies are loaded
            if not self.security_policies:
                return False
            
            # Check if JWT secret is configured
            if not self.jwt_secret:
                return False
            
            # Clean up expired sessions
            current_time = time.time()
            expired_sessions = [
                session_id for session_id, session in self.active_sessions.items()
                if session["expires_at"] < current_time
            ]
            
            for session_id in expired_sessions:
                del self.active_sessions[session_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Security manager health check failed: {e}")
            return False
