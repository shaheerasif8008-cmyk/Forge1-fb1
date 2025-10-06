"""
OPA (Open Policy Agent) Integration Adapter

Provides centralized policy evaluation and enforcement using OPA
with tenant-aware policy management and comprehensive audit logging.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum

import httpx

from forge1.integrations.base_adapter import CachedAdapter, HealthCheckResult, AdapterStatus, ExecutionContext, TenantContext
from forge1.config.integration_settings import IntegrationType, settings_manager
from forge1.core.tenancy import get_current_tenant
from forge1.core.dlp import redact_payload

logger = logging.getLogger(__name__)

class PolicyDecision(Enum):
    """Policy decision types"""
    ALLOW = "allow"
    DENY = "deny"
    UNKNOWN = "unknown"

@dataclass
class PolicyInput:
    """Input for policy evaluation"""
    subject: Dict[str, Any]  # user/employee context
    resource: Dict[str, Any]  # resource being accessed
    action: str  # action being performed
    environment: Dict[str, Any]  # request context
    tenant_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for OPA evaluation"""
        return {
            "input": {
                "subject": self.subject,
                "resource": self.resource,
                "action": self.action,
                "environment": self.environment,
                "tenant_id": self.tenant_id
            }
        }

@dataclass
class PolicyResult:
    """Result of policy evaluation"""
    decision: PolicyDecision
    reason: str
    policy_name: str
    evaluation_time_ms: float
    metadata: Dict[str, Any]
    violations: List[str] = None
    
    def __post_init__(self):
        if self.violations is None:
            self.violations = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "policy_name": self.policy_name,
            "evaluation_time_ms": self.evaluation_time_ms,
            "metadata": self.metadata,
            "violations": self.violations
        }

@dataclass
class PolicyBundle:
    """Policy bundle information"""
    name: str
    version: str
    policies: List[str]
    last_updated: float
    checksum: str

class OPAAdapter(CachedAdapter):
    """OPA integration adapter for policy evaluation"""
    
    def __init__(self):
        config = settings_manager.get_config(IntegrationType.POLICY)
        super().__init__("opa", config, cache_ttl=300)  # 5 minute cache
        
        self.opa_config = config
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # Policy management
        self._loaded_policies: Dict[str, PolicyBundle] = {}
        self._policy_cache_hits = 0
        self._policy_cache_misses = 0
        self._policy_evaluations = 0
        self._policy_denials = 0
        
        # Default policies
        self._default_policies = {
            "tool_access": "forge1/policies/tool_access.rego",
            "doc_access": "forge1/policies/doc_access.rego", 
            "routing_constraints": "forge1/policies/routing_constraints.rego"
        }
    
    async def initialize(self) -> bool:
        """Initialize OPA client and load policies"""
        try:
            # Initialize HTTP client
            headers = {"Content-Type": "application/json"}
            if self.opa_config.auth_token:
                headers["Authorization"] = f"Bearer {self.opa_config.auth_token}"
            
            self.http_client = httpx.AsyncClient(
                base_url=self.opa_config.server_url,
                headers=headers,
                timeout=httpx.Timeout(self.opa_config.timeout_seconds),
                verify=self.opa_config.verify_ssl
            )
            
            # Test connection
            await self._test_connection()
            
            # Load policy bundle if configured
            if self.opa_config.policy_bundle_url:
                await self.load_policy_bundle_from_url(self.opa_config.policy_bundle_url)
            elif self.opa_config.policy_bundle_path:
                await self.load_policy_bundle_from_path(self.opa_config.policy_bundle_path)
            
            logger.info("OPA adapter initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OPA adapter: {e}")
            return False
    
    async def health_check(self) -> HealthCheckResult:
        """Perform health check of OPA server"""
        start_time = time.time()
        
        try:
            if not self.http_client:
                return HealthCheckResult(
                    status=AdapterStatus.UNHEALTHY,
                    message="OPA client not initialized",
                    details={},
                    timestamp=time.time(),
                    response_time_ms=0
                )
            
            # Test basic connectivity
            response = await self.http_client.get("/health")
            
            if response.status_code == 200:
                # Test policy evaluation with a simple query
                test_input = PolicyInput(
                    subject={"user_id": "test", "role": "user"},
                    resource={"type": "test"},
                    action="read",
                    environment={"timestamp": time.time()},
                    tenant_id="test"
                )
                
                # Try to evaluate a simple policy (this may fail if no policies loaded)
                try:
                    await self.evaluate_policy(test_input, "test_policy")
                    policy_evaluation_healthy = True
                except Exception:
                    policy_evaluation_healthy = False
                
                status = AdapterStatus.HEALTHY if policy_evaluation_healthy else AdapterStatus.DEGRADED
                message = "OPA server healthy" if policy_evaluation_healthy else "OPA server healthy but policy evaluation failed"
            else:
                status = AdapterStatus.UNHEALTHY
                message = f"OPA server unhealthy: HTTP {response.status_code}"
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                status=status,
                message=message,
                details={
                    "server_url": self.opa_config.server_url,
                    "loaded_policies": len(self._loaded_policies),
                    "policy_evaluations": self._policy_evaluations,
                    "policy_denials": self._policy_denials,
                    "cache_hits": self._policy_cache_hits,
                    "cache_misses": self._policy_cache_misses,
                    "cache_hit_ratio": self._policy_cache_hits / max(1, self._policy_cache_hits + self._policy_cache_misses)
                },
                timestamp=time.time(),
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=AdapterStatus.UNHEALTHY,
                message=f"OPA health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
                response_time_ms=response_time
            )
    
    async def cleanup(self) -> bool:
        """Clean up OPA resources"""
        try:
            if self.http_client:
                await self.http_client.aclose()
            
            logger.info("OPA adapter cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup OPA adapter: {e}")
            return False
    
    async def evaluate_policy(self, policy_input: PolicyInput, policy_name: str, 
                            context: Optional[ExecutionContext] = None) -> PolicyResult:
        """Evaluate a policy against input data"""
        
        if not await self.ensure_initialized():
            raise RuntimeError("OPA adapter not initialized")
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(policy_input, policy_name)
            cached_result = self.get_from_cache(cache_key, policy_input.tenant_id)
            
            if cached_result:
                self._policy_cache_hits += 1
                logger.debug(f"Policy evaluation cache hit for {policy_name}")
                return cached_result
            
            self._policy_cache_misses += 1
            
            # Apply DLP redaction to policy input
            safe_input_dict = policy_input.to_dict()
            safe_input_dict, violations = redact_payload(safe_input_dict)
            
            # Build OPA query URL
            policy_path = f"/v1/data/{policy_name.replace('.', '/')}"
            
            # Make policy evaluation request
            response = await self._execute_with_retry(
                lambda: self.http_client.post(policy_path, json=safe_input_dict)
            )
            
            response.raise_for_status()
            result_data = response.json()
            
            # Parse OPA response
            decision = PolicyDecision.DENY  # Default to deny
            reason = "Policy evaluation failed"
            metadata = {}
            
            if "result" in result_data:
                opa_result = result_data["result"]
                
                if isinstance(opa_result, bool):
                    decision = PolicyDecision.ALLOW if opa_result else PolicyDecision.DENY
                    reason = f"Policy {policy_name} returned {decision.value}"
                elif isinstance(opa_result, dict):
                    # Handle structured policy response
                    decision = PolicyDecision.ALLOW if opa_result.get("allow", False) else PolicyDecision.DENY
                    reason = opa_result.get("reason", f"Policy {policy_name} returned {decision.value}")
                    metadata = opa_result.get("metadata", {})
            
            evaluation_time = (time.time() - start_time) * 1000
            
            # Create policy result
            policy_result = PolicyResult(
                decision=decision,
                reason=reason,
                policy_name=policy_name,
                evaluation_time_ms=evaluation_time,
                metadata=metadata,
                violations=violations if violations else []
            )
            
            # Cache the result
            tenant_context = TenantContext(tenant_id=policy_input.tenant_id)
            self.set_in_cache(cache_key, policy_result, tenant_context)
            
            # Update statistics
            self._policy_evaluations += 1
            if decision == PolicyDecision.DENY:
                self._policy_denials += 1
            
            # Log policy decision
            logger.info(f"Policy evaluation: {policy_name} -> {decision.value}", extra={
                "policy_name": policy_name,
                "decision": decision.value,
                "reason": reason,
                "tenant_id": policy_input.tenant_id,
                "evaluation_time_ms": evaluation_time
            })
            
            return policy_result
            
        except Exception as e:
            evaluation_time = (time.time() - start_time) * 1000
            
            logger.error(f"Policy evaluation failed for {policy_name}: {e}")
            
            # Return deny decision on error (fail-safe)
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=f"Policy evaluation error: {str(e)}",
                policy_name=policy_name,
                evaluation_time_ms=evaluation_time,
                metadata={"error": str(e)}
            )
    
    async def load_policy_bundle_from_url(self, bundle_url: str) -> bool:
        """Load policy bundle from URL"""
        
        try:
            response = await self.http_client.get(bundle_url)
            response.raise_for_status()
            
            bundle_data = response.json()
            return await self._process_policy_bundle(bundle_data)
            
        except Exception as e:
            logger.error(f"Failed to load policy bundle from URL {bundle_url}: {e}")
            return False
    
    async def load_policy_bundle_from_path(self, bundle_path: str) -> bool:
        """Load policy bundle from local path"""
        
        try:
            # This would load policies from local filesystem
            # For now, simulate loading default policies
            
            default_bundle = {
                "name": "forge1_policies",
                "version": "1.0.0",
                "policies": list(self._default_policies.keys()),
                "manifest": {
                    "revision": "1",
                    "roots": ["forge1"]
                }
            }
            
            return await self._process_policy_bundle(default_bundle)
            
        except Exception as e:
            logger.error(f"Failed to load policy bundle from path {bundle_path}: {e}")
            return False
    
    async def update_policy(self, policy_name: str, policy_content: str) -> bool:
        """Update a specific policy"""
        
        if not await self.ensure_initialized():
            raise RuntimeError("OPA adapter not initialized")
        
        try:
            # Build policy update URL
            policy_path = f"/v1/policies/{policy_name}"
            
            # Send policy update
            response = await self.http_client.put(
                policy_path,
                content=policy_content,
                headers={"Content-Type": "text/plain"}
            )
            
            response.raise_for_status()
            
            # Invalidate cache for this policy
            self.invalidate_cache()
            
            logger.info(f"Updated policy: {policy_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update policy {policy_name}: {e}")
            return False
    
    async def _test_connection(self):
        """Test connection to OPA server"""
        try:
            response = await self.http_client.get("/health")
            if response.status_code not in [200, 404]:  # 404 is acceptable if no health endpoint
                raise ConnectionError(f"OPA connection test failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"OPA connection test failed: {e}")
    
    async def _process_policy_bundle(self, bundle_data: Dict[str, Any]) -> bool:
        """Process and load policy bundle"""
        
        try:
            bundle_name = bundle_data.get("name", "unknown")
            bundle_version = bundle_data.get("version", "1.0.0")
            policies = bundle_data.get("policies", [])
            
            # Create policy bundle
            policy_bundle = PolicyBundle(
                name=bundle_name,
                version=bundle_version,
                policies=policies,
                last_updated=time.time(),
                checksum=str(hash(json.dumps(bundle_data, sort_keys=True)))
            )
            
            # Store bundle
            self._loaded_policies[bundle_name] = policy_bundle
            
            # Send bundle to OPA if manifest is provided
            if "manifest" in bundle_data:
                bundle_path = "/v1/bundles/forge1"
                response = await self.http_client.put(bundle_path, json=bundle_data)
                response.raise_for_status()
            
            logger.info(f"Loaded policy bundle: {bundle_name} v{bundle_version} with {len(policies)} policies")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process policy bundle: {e}")
            return False
    
    def _generate_cache_key(self, policy_input: PolicyInput, policy_name: str) -> str:
        """Generate cache key for policy evaluation"""
        
        # Create a deterministic key based on input
        key_data = {
            "policy": policy_name,
            "subject": policy_input.subject,
            "resource": policy_input.resource,
            "action": policy_input.action,
            "environment": {k: v for k, v in policy_input.environment.items() 
                          if k not in ["timestamp", "request_id"]}  # Exclude volatile fields
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return f"policy_eval:{hash(key_str)}"
    
    async def _execute_with_retry(self, operation) -> httpx.Response:
        """Execute HTTP operation with retry logic"""
        
        last_exception = None
        
        for attempt in range(self.opa_config.retry_attempts + 1):
            try:
                return await operation()
                
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exception = e
                
                if attempt == self.opa_config.retry_attempts:
                    break
                
                # Exponential backoff
                delay = 2 ** attempt
                logger.warning(f"OPA operation failed (attempt {attempt + 1}/{self.opa_config.retry_attempts + 1}), "
                             f"retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
                
            except httpx.HTTPStatusError as e:
                # Don't retry for HTTP errors
                logger.error(f"OPA operation failed with HTTP error: {e}")
                raise
        
        raise last_exception
    
    def get_loaded_policies(self) -> Dict[str, PolicyBundle]:
        """Get information about loaded policies"""
        return self._loaded_policies.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get adapter statistics"""
        return {
            "policy_evaluations": self._policy_evaluations,
            "policy_denials": self._policy_denials,
            "cache_hits": self._policy_cache_hits,
            "cache_misses": self._policy_cache_misses,
            "cache_hit_ratio": self._policy_cache_hits / max(1, self._policy_cache_hits + self._policy_cache_misses),
            "loaded_policies": len(self._loaded_policies)
        }

# Global OPA adapter instance
opa_adapter = OPAAdapter()