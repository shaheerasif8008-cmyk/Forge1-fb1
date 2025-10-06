"""
Policy Enforcement Middleware

FastAPI middleware for enforcing OPA policies across all Forge1 operations
with comprehensive audit logging and tenant-aware policy evaluation.
"""

import json
import logging
import time
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse

from forge1.policy.opa_client import OPAAdapter, PolicyInput, PolicyDecision
from forge1.integrations.base_adapter import ExecutionContext, TenantContext
from forge1.core.tenancy import get_current_tenant
from forge1.core.dlp import redact_payload

logger = logging.getLogger(__name__)

class PolicyEnforcementMiddleware:
    """FastAPI middleware for OPA policy enforcement"""
    
    def __init__(self, opa_adapter: Optional[OPAAdapter] = None):
        self.opa = opa_adapter or OPAAdapter()
        
        # Policy enforcement statistics
        self._enforcement_stats = {
            "total_requests": 0,
            "policy_evaluations": 0,
            "policy_denials": 0,
            "policy_errors": 0,
            "bypass_count": 0
        }
        
        # Paths that bypass policy enforcement
        self._bypass_paths = {
            "/health",
            "/metrics", 
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/forge1/health/detailed"
        }
        
        # Policy mappings for different operations
        self._policy_mappings = {
            "tool_execution": "forge1.policies.tool_access",
            "document_access": "forge1.policies.doc_access", 
            "model_routing": "forge1.policies.routing_constraints",
            "api_access": "forge1.policies.api_access"
        }
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Main middleware function"""
        
        start_time = time.time()
        self._enforcement_stats["total_requests"] += 1
        
        # Check if path should bypass policy enforcement
        if self._should_bypass_policy(request):
            self._enforcement_stats["bypass_count"] += 1
            return await call_next(request)
        
        try:
            # Extract request context
            context = await self._extract_request_context(request)
            
            # Determine policy to evaluate
            policy_name = self._determine_policy(request)
            
            if policy_name:
                # Evaluate policy
                policy_result = await self._evaluate_request_policy(request, context, policy_name)
                
                # Enforce policy decision
                if policy_result.decision == PolicyDecision.DENY:
                    return await self._handle_policy_denial(request, policy_result, context)
                
                # Add policy result to request state for downstream use
                request.state.policy_result = policy_result
            
            # Continue with request processing
            response = await call_next(request)
            
            # Add policy headers to response
            if hasattr(request.state, 'policy_result'):
                response.headers["X-Policy-Decision"] = request.state.policy_result.decision.value
                response.headers["X-Policy-Name"] = request.state.policy_result.policy_name
            
            # Log successful request
            processing_time = (time.time() - start_time) * 1000
            await self._log_policy_enforcement(request, context, "allow", processing_time)
            
            return response
            
        except Exception as e:
            self._enforcement_stats["policy_errors"] += 1
            processing_time = (time.time() - start_time) * 1000
            
            logger.error(f"Policy enforcement error: {e}", extra={
                "path": request.url.path,
                "method": request.method,
                "processing_time_ms": processing_time,
                "error": str(e)
            })
            
            # Fail-safe: deny on error
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Policy enforcement error",
                    "detail": "Access denied due to policy evaluation failure",
                    "request_id": getattr(request.state, 'request_id', 'unknown')
                }
            )
    
    async def enforce_tool_access_policy(self, context: ExecutionContext, tool_name: str, 
                                       tool_params: Dict[str, Any]) -> bool:
        """Enforce policy for tool execution"""
        
        try:
            # Apply DLP redaction to tool parameters
            safe_params, violations = redact_payload(tool_params)
            
            # Create policy input
            policy_input = PolicyInput(
                subject={
                    "user_id": context.tenant_context.user_id,
                    "employee_id": context.tenant_context.employee_id,
                    "role": context.tenant_context.role,
                    "security_level": context.tenant_context.security_level
                },
                resource={
                    "type": "tool",
                    "name": tool_name,
                    "parameters": safe_params,
                    "dlp_violations": len(violations)
                },
                action="execute",
                environment={
                    "timestamp": time.time(),
                    "request_id": context.request_id,
                    "session_id": context.session_id
                },
                tenant_id=context.tenant_context.tenant_id
            )
            
            # Evaluate policy
            result = await self.opa.evaluate_policy(
                policy_input, 
                self._policy_mappings["tool_execution"],
                context
            )
            
            # Log policy decision
            await self._log_tool_policy_decision(tool_name, result, context)
            
            return result.decision == PolicyDecision.ALLOW
            
        except Exception as e:
            logger.error(f"Tool access policy evaluation failed for {tool_name}: {e}")
            return False  # Fail-safe: deny on error
    
    async def enforce_document_access_policy(self, context: ExecutionContext, doc_id: str, 
                                           doc_metadata: Dict[str, Any]) -> bool:
        """Enforce policy for document access"""
        
        try:
            # Apply DLP redaction to document metadata
            safe_metadata, violations = redact_payload(doc_metadata)
            
            # Create policy input
            policy_input = PolicyInput(
                subject={
                    "user_id": context.tenant_context.user_id,
                    "employee_id": context.tenant_context.employee_id,
                    "role": context.tenant_context.role,
                    "security_level": context.tenant_context.security_level
                },
                resource={
                    "type": "document",
                    "id": doc_id,
                    "metadata": safe_metadata,
                    "classification": doc_metadata.get("classification", "unclassified"),
                    "owner": doc_metadata.get("owner"),
                    "dlp_violations": len(violations)
                },
                action="read",
                environment={
                    "timestamp": time.time(),
                    "request_id": context.request_id,
                    "session_id": context.session_id
                },
                tenant_id=context.tenant_context.tenant_id
            )
            
            # Evaluate policy
            result = await self.opa.evaluate_policy(
                policy_input,
                self._policy_mappings["document_access"],
                context
            )
            
            # Log policy decision
            await self._log_document_policy_decision(doc_id, result, context)
            
            return result.decision == PolicyDecision.ALLOW
            
        except Exception as e:
            logger.error(f"Document access policy evaluation failed for {doc_id}: {e}")
            return False  # Fail-safe: deny on error
    
    async def enforce_model_routing_policy(self, context: ExecutionContext, model_name: str, 
                                         request_data: Dict[str, Any]) -> bool:
        """Enforce policy for model routing constraints"""
        
        try:
            # Apply DLP redaction to request data
            safe_data, violations = redact_payload(request_data)
            
            # Create policy input
            policy_input = PolicyInput(
                subject={
                    "user_id": context.tenant_context.user_id,
                    "employee_id": context.tenant_context.employee_id,
                    "role": context.tenant_context.role,
                    "security_level": context.tenant_context.security_level
                },
                resource={
                    "type": "model",
                    "name": model_name,
                    "request_data": safe_data,
                    "has_pii": len(violations) > 0,
                    "dlp_violations": len(violations)
                },
                action="route",
                environment={
                    "timestamp": time.time(),
                    "request_id": context.request_id,
                    "session_id": context.session_id
                },
                tenant_id=context.tenant_context.tenant_id
            )
            
            # Evaluate policy
            result = await self.opa.evaluate_policy(
                policy_input,
                self._policy_mappings["model_routing"],
                context
            )
            
            # Log policy decision
            await self._log_model_policy_decision(model_name, result, context)
            
            return result.decision == PolicyDecision.ALLOW
            
        except Exception as e:
            logger.error(f"Model routing policy evaluation failed for {model_name}: {e}")
            return False  # Fail-safe: deny on error
    
    def _should_bypass_policy(self, request: Request) -> bool:
        """Check if request should bypass policy enforcement"""
        
        path = request.url.path
        
        # Check bypass paths
        if path in self._bypass_paths:
            return True
        
        # Check path prefixes
        bypass_prefixes = ["/static/", "/assets/"]
        if any(path.startswith(prefix) for prefix in bypass_prefixes):
            return True
        
        return False
    
    async def _extract_request_context(self, request: Request) -> ExecutionContext:
        """Extract execution context from request"""
        
        # Get tenant ID
        tenant_id = get_current_tenant() or request.headers.get("X-Tenant-ID", "unknown")
        
        # Get user information
        user_id = request.headers.get("X-User-ID", "anonymous")
        employee_id = request.headers.get("X-Employee-ID")
        role = request.headers.get("X-User-Role", "user")
        security_level = request.headers.get("X-Security-Level", "standard")
        
        # Get request metadata
        request_id = request.headers.get("X-Request-ID", str(time.time()))
        session_id = request.headers.get("X-Session-ID")
        
        # Create tenant context
        tenant_context = TenantContext(
            tenant_id=tenant_id,
            employee_id=employee_id,
            user_id=user_id,
            role=role,
            security_level=security_level
        )
        
        # Create execution context
        return ExecutionContext(
            tenant_context=tenant_context,
            request_id=request_id,
            session_id=session_id,
            metadata={
                "path": request.url.path,
                "method": request.method,
                "user_agent": request.headers.get("User-Agent", "unknown")
            }
        )
    
    def _determine_policy(self, request: Request) -> Optional[str]:
        """Determine which policy to evaluate for the request"""
        
        path = request.url.path
        method = request.method
        
        # API access policies
        if path.startswith("/api/"):
            return self._policy_mappings["api_access"]
        
        # Tool execution policies
        if "tool" in path or "execute" in path:
            return self._policy_mappings["tool_execution"]
        
        # Document access policies
        if "document" in path or "file" in path:
            return self._policy_mappings["document_access"]
        
        # Model routing policies
        if "model" in path or "generate" in path:
            return self._policy_mappings["model_routing"]
        
        # Default to API access policy for authenticated endpoints
        return self._policy_mappings["api_access"]
    
    async def _evaluate_request_policy(self, request: Request, context: ExecutionContext, 
                                     policy_name: str):
        """Evaluate policy for the request"""
        
        # Extract request body if present
        request_body = {}
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    request_body = json.loads(body.decode())
            except Exception as e:
                logger.warning(f"Failed to parse request body: {e}")
        
        # Create policy input
        policy_input = PolicyInput(
            subject={
                "user_id": context.tenant_context.user_id,
                "employee_id": context.tenant_context.employee_id,
                "role": context.tenant_context.role,
                "security_level": context.tenant_context.security_level
            },
            resource={
                "type": "api_endpoint",
                "path": request.url.path,
                "method": request.method,
                "query_params": dict(request.query_params),
                "body": request_body
            },
            action=request.method.lower(),
            environment={
                "timestamp": time.time(),
                "request_id": context.request_id,
                "session_id": context.session_id,
                "user_agent": request.headers.get("User-Agent", "unknown"),
                "ip_address": request.client.host if request.client else "unknown"
            },
            tenant_id=context.tenant_context.tenant_id
        )
        
        # Evaluate policy
        self._enforcement_stats["policy_evaluations"] += 1
        result = await self.opa.evaluate_policy(policy_input, policy_name, context)
        
        if result.decision == PolicyDecision.DENY:
            self._enforcement_stats["policy_denials"] += 1
        
        return result
    
    async def _handle_policy_denial(self, request: Request, policy_result, context: ExecutionContext) -> Response:
        """Handle policy denial with proper response and logging"""
        
        # Log policy denial
        await self._log_policy_enforcement(request, context, "deny", policy_result.evaluation_time_ms, policy_result.reason)
        
        # Create denial response
        response_content = {
            "error": "Access Denied",
            "detail": "Request denied by security policy",
            "policy": policy_result.policy_name,
            "request_id": context.request_id,
            "timestamp": time.time()
        }
        
        # Add additional context if available (but redacted)
        if policy_result.metadata:
            safe_metadata, _ = redact_payload(policy_result.metadata)
            response_content["policy_metadata"] = safe_metadata
        
        return JSONResponse(
            status_code=403,
            content=response_content,
            headers={
                "X-Policy-Decision": policy_result.decision.value,
                "X-Policy-Name": policy_result.policy_name,
                "X-Request-ID": context.request_id
            }
        )
    
    async def _log_policy_enforcement(self, request: Request, context: ExecutionContext, 
                                    decision: str, processing_time_ms: float, reason: str = ""):
        """Log policy enforcement decision"""
        
        log_data = {
            "event_type": "policy_enforcement",
            "decision": decision,
            "path": request.url.path,
            "method": request.method,
            "tenant_id": context.tenant_context.tenant_id,
            "user_id": context.tenant_context.user_id,
            "employee_id": context.tenant_context.employee_id,
            "request_id": context.request_id,
            "processing_time_ms": processing_time_ms,
            "timestamp": time.time()
        }
        
        if reason:
            log_data["reason"] = reason
        
        if decision == "deny":
            logger.warning("Policy enforcement denial", extra=log_data)
        else:
            logger.info("Policy enforcement decision", extra=log_data)
    
    async def _log_tool_policy_decision(self, tool_name: str, policy_result, context: ExecutionContext):
        """Log tool access policy decision"""
        
        logger.info(f"Tool access policy: {tool_name} -> {policy_result.decision.value}", extra={
            "event_type": "tool_policy_decision",
            "tool_name": tool_name,
            "decision": policy_result.decision.value,
            "reason": policy_result.reason,
            "tenant_id": context.tenant_context.tenant_id,
            "user_id": context.tenant_context.user_id,
            "request_id": context.request_id,
            "evaluation_time_ms": policy_result.evaluation_time_ms
        })
    
    async def _log_document_policy_decision(self, doc_id: str, policy_result, context: ExecutionContext):
        """Log document access policy decision"""
        
        logger.info(f"Document access policy: {doc_id} -> {policy_result.decision.value}", extra={
            "event_type": "document_policy_decision",
            "document_id": doc_id,
            "decision": policy_result.decision.value,
            "reason": policy_result.reason,
            "tenant_id": context.tenant_context.tenant_id,
            "user_id": context.tenant_context.user_id,
            "request_id": context.request_id,
            "evaluation_time_ms": policy_result.evaluation_time_ms
        })
    
    async def _log_model_policy_decision(self, model_name: str, policy_result, context: ExecutionContext):
        """Log model routing policy decision"""
        
        logger.info(f"Model routing policy: {model_name} -> {policy_result.decision.value}", extra={
            "event_type": "model_policy_decision",
            "model_name": model_name,
            "decision": policy_result.decision.value,
            "reason": policy_result.reason,
            "tenant_id": context.tenant_context.tenant_id,
            "user_id": context.tenant_context.user_id,
            "request_id": context.request_id,
            "evaluation_time_ms": policy_result.evaluation_time_ms
        })
    
    def get_enforcement_statistics(self) -> Dict[str, Any]:
        """Get policy enforcement statistics"""
        return self._enforcement_stats.copy()
    
    def add_bypass_path(self, path: str):
        """Add a path to bypass policy enforcement"""
        self._bypass_paths.add(path)
    
    def remove_bypass_path(self, path: str):
        """Remove a path from bypass list"""
        self._bypass_paths.discard(path)
    
    def update_policy_mapping(self, operation: str, policy_name: str):
        """Update policy mapping for an operation"""
        self._policy_mappings[operation] = policy_name

# Global policy enforcement middleware instance
policy_enforcement_middleware = PolicyEnforcementMiddleware()