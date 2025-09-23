# forge1/backend/forge1/integrations/api_integration_layer.py

"""
Enterprise API Integration Layer for Forge 1

Comprehensive API integration system supporting:
- REST APIs with full HTTP method support
- GraphQL APIs with query, mutation, and subscription support
- SOAP APIs with WSDL parsing and envelope handling
- Rate limiting and throttling
- Retry logic with exponential backoff
- Circuit breaker pattern for fault tolerance
- Request/response transformation and validation
- Authentication and authorization integration
- Performance monitoring and analytics
"""

import asyncio
import logging
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import aiohttp
import time
from dataclasses import dataclass, field
import hashlib
import hmac
import base64
from urllib.parse import urljoin, urlparse
import ssl

from forge1.auth.authentication_manager import AuthenticationManager, AuthenticationType
from forge1.core.security_manager import SecurityManager
from forge1.core.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

class APIType(Enum):
    """Supported API types"""
    REST = "rest"
    GRAPHQL = "graphql"
    SOAP = "soap"
    WEBHOOK = "webhook"

class HTTPMethod(Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

class AuthenticationMethod(Enum):
    """API authentication methods"""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    CUSTOM_HEADER = "custom_header"
    HMAC_SIGNATURE = "hmac_signature"
    MUTUAL_TLS = "mutual_tls"

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_second: float = 10.0
    requests_per_minute: int = 600
    requests_per_hour: int = 36000
    burst_size: int = 20
    enable_adaptive: bool = True

@dataclass
class RetryConfig:
    """Retry configuration"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_status_codes: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout: float = 30.0

@dataclass
class APIEndpoint:
    """API endpoint configuration"""
    url: str
    method: HTTPMethod = HTTPMethod.GET
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, Any] = field(default_factory=dict)
    body_template: Optional[str] = None
    response_transform: Optional[str] = None
    timeout: float = 30.0
    
class RateLimiter:
    """Token bucket rate limiter with adaptive capabilities"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.burst_size
        self.last_update = time.time()
        self.request_times = []
        
        # Adaptive rate limiting
        self.adaptive_rate = config.requests_per_second
        self.success_count = 0
        self.error_count = 0
        
    async def acquire(self) -> bool:
        """Acquire a token for rate limiting"""
        
        current_time = time.time()
        
        # Update tokens based on time elapsed
        time_elapsed = current_time - self.last_update
        tokens_to_add = time_elapsed * self.adaptive_rate
        self.tokens = min(self.config.burst_size, self.tokens + tokens_to_add)
        self.last_update = current_time
        
        # Check if we have tokens available
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            self._record_request(current_time)
            return True
        
        return False
    
    def _record_request(self, timestamp: float) -> None:
        """Record request timestamp for analytics"""
        
        self.request_times.append(timestamp)
        
        # Keep only recent requests (last hour)
        cutoff_time = timestamp - 3600
        self.request_times = [t for t in self.request_times if t > cutoff_time]
    
    def update_adaptive_rate(self, success: bool) -> None:
        """Update adaptive rate based on success/failure"""
        
        if not self.config.enable_adaptive:
            return
        
        if success:
            self.success_count += 1
            self.error_count = max(0, self.error_count - 1)
        else:
            self.error_count += 1
            self.success_count = max(0, self.success_count - 1)
        
        # Adjust rate based on success ratio
        total_requests = self.success_count + self.error_count
        if total_requests >= 10:
            success_ratio = self.success_count / total_requests
            
            if success_ratio > 0.95:
                # High success rate, increase rate slightly
                self.adaptive_rate = min(
                    self.config.requests_per_second * 1.5,
                    self.adaptive_rate * 1.1
                )
            elif success_ratio < 0.8:
                # Low success rate, decrease rate
                self.adaptive_rate = max(
                    self.config.requests_per_second * 0.5,
                    self.adaptive_rate * 0.9
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        
        current_time = time.time()
        recent_requests = [t for t in self.request_times if t > current_time - 60]
        
        return {
            "current_tokens": self.tokens,
            "adaptive_rate": self.adaptive_rate,
            "requests_last_minute": len(recent_requests),
            "requests_last_hour": len(self.request_times),
            "success_count": self.success_count,
            "error_count": self.error_count
        }

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.next_attempt_time = None
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            # Execute the function with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful execution"""
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self) -> None:
        """Handle failed execution"""
        
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.next_attempt_time = time.time() + self.config.recovery_timeout
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "next_attempt_time": self.next_attempt_time
        }

class BaseAPIConnector(ABC):
    """Base class for API connectors"""
    
    def __init__(
        self,
        api_type: APIType,
        base_url: str,
        auth_manager: AuthenticationManager,
        security_manager: SecurityManager,
        performance_monitor: PerformanceMonitor,
        rate_limit_config: Optional[RateLimitConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        **kwargs
    ):
        self.api_type = api_type
        self.base_url = base_url.rstrip('/')
        self.auth_manager = auth_manager
        self.security_manager = security_manager
        self.performance_monitor = performance_monitor
        
        # Configuration
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        
        # Components
        self.rate_limiter = RateLimiter(self.rate_limit_config)
        self.circuit_breaker = CircuitBreaker(self.circuit_breaker_config)
        
        # Session and state
        self.session = None
        self.connector_id = f"{api_type.value}_{uuid.uuid4().hex[:8]}"
        self.created_at = datetime.now(timezone.utc)
        
        # Metrics
        self.metrics = {
            "requests_made": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "total_response_time": 0.0,
            "average_response_time": 0.0,
            "rate_limited_requests": 0,
            "circuit_breaker_trips": 0
        }
        
        # Additional configuration
        self.default_headers = kwargs.get("default_headers", {})
        self.ssl_verify = kwargs.get("ssl_verify", True)
        self.proxy = kwargs.get("proxy")
        self.auth_config = kwargs.get("auth_config", {})
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get HTTP session with configuration"""
        
        if not self.session:
            connector = aiohttp.TCPConnector(
                ssl=ssl.create_default_context() if self.ssl_verify else False,
                limit=100,
                limit_per_host=30
            )
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self.default_headers
            )
        
        return self.session
    
    async def _authenticate_request(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Add authentication to request headers"""
        
        auth_method = self.auth_config.get("method", AuthenticationMethod.NONE)
        
        if auth_method == AuthenticationMethod.NONE:
            return headers
        
        # Use authentication manager for credential retrieval
        credential_id = self.auth_config.get("credential_id")
        if credential_id:
            auth_result = await self.auth_manager.authenticate(credential_id)
            if auth_result["success"]:
                headers.update(auth_result.get("headers", {}))
        
        # Handle specific authentication methods
        if auth_method == AuthenticationMethod.API_KEY:
            api_key = self.auth_config.get("api_key")
            key_header = self.auth_config.get("key_header", "X-API-Key")
            if api_key:
                headers[key_header] = api_key
        
        elif auth_method == AuthenticationMethod.BEARER_TOKEN:
            token = self.auth_config.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        elif auth_method == AuthenticationMethod.BASIC_AUTH:
            username = self.auth_config.get("username")
            password = self.auth_config.get("password")
            if username and password:
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {credentials}"
        
        elif auth_method == AuthenticationMethod.CUSTOM_HEADER:
            custom_headers = self.auth_config.get("custom_headers", {})
            headers.update(custom_headers)
        
        return headers
    
    async def _make_request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """Make HTTP request with retry logic"""
        
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # Rate limiting
                if not await self.rate_limiter.acquire():
                    self.metrics["rate_limited_requests"] += 1
                    await asyncio.sleep(0.1)  # Brief wait for rate limiting
                    continue
                
                # Make request through circuit breaker
                response = await self.circuit_breaker.call(
                    self._make_raw_request,
                    method,
                    url,
                    **kwargs
                )
                
                # Check if we should retry based on status code
                if response.status in self.retry_config.retry_on_status_codes and attempt < self.retry_config.max_retries:
                    await response.close()
                    await self._wait_for_retry(attempt)
                    continue
                
                return response
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.retry_config.max_retries:
                    await self._wait_for_retry(attempt)
                    continue
                else:
                    break
        
        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise Exception("Request failed after all retries")
    
    async def _make_raw_request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """Make raw HTTP request"""
        
        session = await self._get_session()
        
        # Authenticate request
        headers = kwargs.get("headers", {}).copy()
        headers = await self._authenticate_request(headers)
        kwargs["headers"] = headers
        
        # Add proxy if configured
        if self.proxy:
            kwargs["proxy"] = self.proxy
        
        return await session.request(method, url, **kwargs)
    
    async def _wait_for_retry(self, attempt: int) -> None:
        """Wait before retry with exponential backoff"""
        
        delay = min(
            self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
            self.retry_config.max_delay
        )
        
        # Add jitter to prevent thundering herd
        if self.retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        await asyncio.sleep(delay)
    
    @abstractmethod
    async def execute_request(
        self,
        endpoint: APIEndpoint,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute API request"""
        pass
    
    def get_connector_info(self) -> Dict[str, Any]:
        """Get connector information"""
        
        return {
            "connector_id": self.connector_id,
            "api_type": self.api_type.value,
            "base_url": self.base_url,
            "created_at": self.created_at.isoformat(),
            "metrics": self.metrics.copy(),
            "rate_limiter_stats": self.rate_limiter.get_stats(),
            "circuit_breaker_state": self.circuit_breaker.get_state()
        }
    
    async def close(self) -> None:
        """Close connector and cleanup resources"""
        
        if self.session:
            await self.session.close()
            self.session = None

class RESTAPIConnector(BaseAPIConnector):
    """REST API connector with full HTTP method support"""
    
    def __init__(self, base_url: str, **kwargs):
        super().__init__(APIType.REST, base_url, **kwargs)
        self.endpoints = {}
    
    def register_endpoint(
        self,
        name: str,
        endpoint: APIEndpoint
    ) -> None:
        """Register a named endpoint"""
        
        self.endpoints[name] = endpoint
    
    async def execute_request(
        self,
        endpoint: APIEndpoint,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute REST API request"""
        
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            # Build URL
            url = urljoin(self.base_url, endpoint.url.lstrip('/'))
            
            # Prepare request parameters
            headers = {**endpoint.headers}
            params = {**endpoint.query_params}
            
            # Add custom parameters
            if kwargs.get("headers"):
                headers.update(kwargs["headers"])
            if kwargs.get("params"):
                params.update(kwargs["params"])
            
            # Prepare request body
            request_body = None
            if data and endpoint.method in [HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.PATCH]:
                if endpoint.body_template:
                    # Apply template transformation
                    request_body = self._apply_template(endpoint.body_template, data)
                else:
                    request_body = data
            
            # Make request
            response = await self._make_request_with_retry(
                method=endpoint.method.value,
                url=url,
                headers=headers,
                params=params,
                json=request_body,
                timeout=aiohttp.ClientTimeout(total=endpoint.timeout)
            )
            
            # Process response
            response_data = await self._process_response(response, endpoint)
            
            # Update metrics
            response_time = time.time() - start_time
            self._update_metrics(True, response_time)
            self.rate_limiter.update_adaptive_rate(True)
            
            # Track performance
            await self.performance_monitor.track_api_request(
                api_type="rest",
                endpoint=endpoint.url,
                method=endpoint.method.value,
                response_time=response_time,
                status_code=response.status,
                success=True
            )
            
            return {
                "success": True,
                "request_id": request_id,
                "status_code": response.status,
                "data": response_data,
                "response_time": response_time,
                "headers": dict(response.headers)
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_metrics(False, response_time)
            self.rate_limiter.update_adaptive_rate(False)
            
            await self.performance_monitor.track_api_request(
                api_type="rest",
                endpoint=endpoint.url,
                method=endpoint.method.value,
                response_time=response_time,
                success=False,
                error=str(e)
            )
            
            logger.error(f"REST API request failed: {e}")
            
            return {
                "success": False,
                "request_id": request_id,
                "error": str(e),
                "response_time": response_time
            }
    
    async def _process_response(
        self,
        response: aiohttp.ClientResponse,
        endpoint: APIEndpoint
    ) -> Any:
        """Process API response"""
        
        content_type = response.headers.get("Content-Type", "").lower()
        
        try:
            if "application/json" in content_type:
                data = await response.json()
            elif "text/" in content_type:
                data = await response.text()
            else:
                data = await response.read()
            
            # Apply response transformation if configured
            if endpoint.response_transform:
                data = self._apply_response_transform(data, endpoint.response_transform)
            
            return data
            
        finally:
            await response.close()
    
    def _apply_template(self, template: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply template transformation to request data"""
        
        try:
            # Simple template substitution (can be enhanced with Jinja2)
            import string
            template_obj = string.Template(template)
            result_str = template_obj.safe_substitute(data)
            return json.loads(result_str)
        except Exception as e:
            logger.warning(f"Template application failed: {e}")
            return data
    
    def _apply_response_transform(self, data: Any, transform: str) -> Any:
        """Apply transformation to response data"""
        
        try:
            # Simple JSONPath-like transformation (can be enhanced)
            if transform.startswith("$."):
                # Extract nested data
                keys = transform[2:].split(".")
                result = data
                for key in keys:
                    if isinstance(result, dict):
                        result = result.get(key)
                    elif isinstance(result, list) and key.isdigit():
                        result = result[int(key)]
                    else:
                        break
                return result
            else:
                return data
        except Exception as e:
            logger.warning(f"Response transformation failed: {e}")
            return data
    
    def _update_metrics(self, success: bool, response_time: float) -> None:
        """Update connector metrics"""
        
        self.metrics["requests_made"] += 1
        self.metrics["total_response_time"] += response_time
        
        if success:
            self.metrics["requests_successful"] += 1
        else:
            self.metrics["requests_failed"] += 1
        
        # Update average response time
        self.metrics["average_response_time"] = (
            self.metrics["total_response_time"] / self.metrics["requests_made"]
        )
    
    async def get(self, endpoint_name: str, **kwargs) -> Dict[str, Any]:
        """Convenience method for GET requests"""
        
        if endpoint_name not in self.endpoints:
            raise ValueError(f"Endpoint '{endpoint_name}' not registered")
        
        endpoint = self.endpoints[endpoint_name]
        endpoint.method = HTTPMethod.GET
        
        return await self.execute_request(endpoint, **kwargs)
    
    async def post(self, endpoint_name: str, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Convenience method for POST requests"""
        
        if endpoint_name not in self.endpoints:
            raise ValueError(f"Endpoint '{endpoint_name}' not registered")
        
        endpoint = self.endpoints[endpoint_name]
        endpoint.method = HTTPMethod.POST
        
        return await self.execute_request(endpoint, data, **kwargs)
    
    async def put(self, endpoint_name: str, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Convenience method for PUT requests"""
        
        if endpoint_name not in self.endpoints:
            raise ValueError(f"Endpoint '{endpoint_name}' not registered")
        
        endpoint = self.endpoints[endpoint_name]
        endpoint.method = HTTPMethod.PUT
        
        return await self.execute_request(endpoint, data, **kwargs)
    
    async def delete(self, endpoint_name: str, **kwargs) -> Dict[str, Any]:
        """Convenience method for DELETE requests"""
        
        if endpoint_name not in self.endpoints:
            raise ValueError(f"Endpoint '{endpoint_name}' not registered")
        
        endpoint = self.endpoints[endpoint_name]
        endpoint.method = HTTPMethod.DELETE
        
        return await self.execute_request(endpoint, **kwargs)cla
ss GraphQLAPIConnector(BaseAPIConnector):
    """GraphQL API connector with query, mutation, and subscription support"""
    
    def __init__(self, base_url: str, **kwargs):
        super().__init__(APIType.GRAPHQL, base_url, **kwargs)
        self.schema = None
        self.introspection_query = """
        query IntrospectionQuery {
            __schema {
                types {
                    name
                    kind
                    description
                    fields {
                        name
                        type {
                            name
                            kind
                        }
                    }
                }
            }
        }
        """
    
    async def introspect_schema(self) -> Dict[str, Any]:
        """Introspect GraphQL schema"""
        
        try:
            result = await self.execute_query(self.introspection_query)
            
            if result["success"]:
                self.schema = result["data"]
                return {
                    "success": True,
                    "schema": self.schema
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Schema introspection failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def execute_request(
        self,
        endpoint: APIEndpoint,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute GraphQL request"""
        
        # GraphQL uses POST method for all operations
        endpoint.method = HTTPMethod.POST
        
        return await self.execute_query(
            query=data.get("query") if data else "",
            variables=data.get("variables") if data else {},
            operation_name=data.get("operationName") if data else None,
            **kwargs
        )
    
    async def execute_query(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute GraphQL query or mutation"""
        
        request_id = f"gql_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            # Prepare GraphQL request
            graphql_request = {
                "query": query,
                "variables": variables or {},
            }
            
            if operation_name:
                graphql_request["operationName"] = operation_name
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                **kwargs.get("headers", {})
            }
            
            # Make request
            response = await self._make_request_with_retry(
                method="POST",
                url=self.base_url,
                headers=headers,
                json=graphql_request
            )
            
            # Process response
            response_data = await response.json()
            
            # Check for GraphQL errors
            success = "errors" not in response_data
            
            # Update metrics
            response_time = time.time() - start_time
            self._update_metrics(success, response_time)
            self.rate_limiter.update_adaptive_rate(success)
            
            # Track performance
            await self.performance_monitor.track_api_request(
                api_type="graphql",
                endpoint=self.base_url,
                method="POST",
                response_time=response_time,
                status_code=response.status,
                success=success
            )
            
            result = {
                "success": success,
                "request_id": request_id,
                "status_code": response.status,
                "response_time": response_time,
                "headers": dict(response.headers)
            }
            
            if success:
                result["data"] = response_data.get("data")
            else:
                result["errors"] = response_data.get("errors", [])
                result["error"] = "GraphQL execution errors"
            
            await response.close()
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_metrics(False, response_time)
            self.rate_limiter.update_adaptive_rate(False)
            
            await self.performance_monitor.track_api_request(
                api_type="graphql",
                endpoint=self.base_url,
                method="POST",
                response_time=response_time,
                success=False,
                error=str(e)
            )
            
            logger.error(f"GraphQL request failed: {e}")
            
            return {
                "success": False,
                "request_id": request_id,
                "error": str(e),
                "response_time": response_time
            }
    
    async def execute_mutation(
        self,
        mutation: str,
        variables: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute GraphQL mutation"""
        
        return await self.execute_query(
            query=mutation,
            variables=variables,
            **kwargs
        )
    
    def _update_metrics(self, success: bool, response_time: float) -> None:
        """Update connector metrics"""
        
        self.metrics["requests_made"] += 1
        self.metrics["total_response_time"] += response_time
        
        if success:
            self.metrics["requests_successful"] += 1
        else:
            self.metrics["requests_failed"] += 1
        
        # Update average response time
        self.metrics["average_response_time"] = (
            self.metrics["total_response_time"] / self.metrics["requests_made"]
        )

class SOAPAPIConnector(BaseAPIConnector):
    """SOAP API connector with WSDL parsing and envelope handling"""
    
    def __init__(self, base_url: str, wsdl_url: Optional[str] = None, **kwargs):
        super().__init__(APIType.SOAP, base_url, **kwargs)
        self.wsdl_url = wsdl_url or f"{base_url}?wsdl"
        self.wsdl_schema = None
        self.soap_actions = {}
        self.namespace_map = {}
    
    async def load_wsdl(self) -> Dict[str, Any]:
        """Load and parse WSDL schema"""
        
        try:
            response = await self._make_request_with_retry(
                method="GET",
                url=self.wsdl_url
            )
            
            wsdl_content = await response.text()
            await response.close()
            
            # Parse WSDL XML
            root = ET.fromstring(wsdl_content)
            
            # Extract namespace mappings
            self.namespace_map = self._extract_namespaces(root)
            
            # Extract SOAP actions and operations
            self.soap_actions = self._extract_soap_actions(root)
            
            self.wsdl_schema = {
                "namespaces": self.namespace_map,
                "operations": list(self.soap_actions.keys()),
                "raw_wsdl": wsdl_content
            }
            
            return {
                "success": True,
                "wsdl_schema": self.wsdl_schema
            }
            
        except Exception as e:
            logger.error(f"WSDL loading failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_namespaces(self, root: ET.Element) -> Dict[str, str]:
        """Extract namespace mappings from WSDL"""
        
        namespaces = {}
        
        # Get namespaces from root element
        for prefix, uri in root.attrib.items():
            if prefix.startswith("xmlns"):
                ns_prefix = prefix.split(":")[-1] if ":" in prefix else "default"
                namespaces[ns_prefix] = uri
        
        return namespaces
    
    def _extract_soap_actions(self, root: ET.Element) -> Dict[str, Dict[str, Any]]:
        """Extract SOAP actions from WSDL"""
        
        actions = {}
        
        # Find all operation elements
        for operation in root.iter():
            if operation.tag.endswith("operation"):
                op_name = operation.get("name")
                if op_name:
                    # Find SOAP action
                    soap_action = None
                    for child in operation.iter():
                        if child.tag.endswith("operation") and child.get("soapAction"):
                            soap_action = child.get("soapAction")
                            break
                    
                    actions[op_name] = {
                        "soap_action": soap_action,
                        "operation_name": op_name
                    }
        
        return actions
    
    async def execute_request(
        self,
        endpoint: APIEndpoint,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute SOAP request"""
        
        operation_name = data.get("operation") if data else None
        soap_body = data.get("body", {}) if data else {}
        
        return await self.execute_soap_operation(
            operation_name=operation_name,
            parameters=soap_body,
            **kwargs
        )
    
    async def execute_soap_operation(
        self,
        operation_name: str,
        parameters: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute SOAP operation"""
        
        request_id = f"soap_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            # Check if operation exists
            if operation_name not in self.soap_actions:
                return {
                    "success": False,
                    "request_id": request_id,
                    "error": f"Operation '{operation_name}' not found in WSDL"
                }
            
            # Build SOAP envelope
            soap_envelope = self._build_soap_envelope(operation_name, parameters)
            
            # Prepare headers
            soap_action = self.soap_actions[operation_name].get("soap_action", "")
            headers = {
                "Content-Type": "text/xml; charset=utf-8",
                "SOAPAction": f'"{soap_action}"',
                **kwargs.get("headers", {})
            }
            
            # Make request
            response = await self._make_request_with_retry(
                method="POST",
                url=self.base_url,
                headers=headers,
                data=soap_envelope
            )
            
            # Process response
            response_text = await response.text()
            response_data = self._parse_soap_response(response_text)
            
            # Check for SOAP faults
            success = "fault" not in response_data
            
            # Update metrics
            response_time = time.time() - start_time
            self._update_metrics(success, response_time)
            self.rate_limiter.update_adaptive_rate(success)
            
            # Track performance
            await self.performance_monitor.track_api_request(
                api_type="soap",
                endpoint=self.base_url,
                method="POST",
                response_time=response_time,
                status_code=response.status,
                success=success
            )
            
            result = {
                "success": success,
                "request_id": request_id,
                "status_code": response.status,
                "response_time": response_time,
                "headers": dict(response.headers),
                "operation": operation_name
            }
            
            if success:
                result["data"] = response_data
            else:
                result["fault"] = response_data.get("fault")
                result["error"] = "SOAP fault occurred"
            
            await response.close()
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_metrics(False, response_time)
            self.rate_limiter.update_adaptive_rate(False)
            
            await self.performance_monitor.track_api_request(
                api_type="soap",
                endpoint=self.base_url,
                method="POST",
                response_time=response_time,
                success=False,
                error=str(e)
            )
            
            logger.error(f"SOAP request failed: {e}")
            
            return {
                "success": False,
                "request_id": request_id,
                "error": str(e),
                "response_time": response_time,
                "operation": operation_name
            }
    
    def _build_soap_envelope(
        self,
        operation_name: str,
        parameters: Dict[str, Any]
    ) -> str:
        """Build SOAP envelope for operation"""
        
        # Get target namespace
        target_ns = self.namespace_map.get("tns", "http://tempuri.org/")
        
        # Build parameter XML
        param_xml = ""
        for key, value in parameters.items():
            param_xml += f"<{key}>{self._escape_xml(str(value))}</{key}>"
        
        # Build SOAP envelope
        soap_envelope = f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/" xmlns:tns="{target_ns}">
    <soap:Header/>
    <soap:Body>
        <tns:{operation_name}>
            {param_xml}
        </tns:{operation_name}>
    </soap:Body>
</soap:Envelope>"""
        
        return soap_envelope
    
    def _parse_soap_response(self, response_text: str) -> Dict[str, Any]:
        """Parse SOAP response XML"""
        
        try:
            root = ET.fromstring(response_text)
            
            # Check for SOAP fault
            fault_elem = root.find(".//{http://schemas.xmlsoap.org/soap/envelope/}Fault")
            if fault_elem is not None:
                fault_code = fault_elem.find(".//faultcode")
                fault_string = fault_elem.find(".//faultstring")
                
                return {
                    "fault": {
                        "code": fault_code.text if fault_code is not None else "Unknown",
                        "string": fault_string.text if fault_string is not None else "Unknown fault"
                    }
                }
            
            # Extract response data from body
            body_elem = root.find(".//{http://schemas.xmlsoap.org/soap/envelope/}Body")
            if body_elem is not None:
                # Convert XML to dictionary
                return self._xml_to_dict(body_elem)
            
            return {"raw_response": response_text}
            
        except ET.ParseError as e:
            logger.error(f"SOAP response parsing failed: {e}")
            return {"parse_error": str(e), "raw_response": response_text}
    
    def _xml_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """Convert XML element to dictionary"""
        
        result = {}
        
        # Handle element text
        if element.text and element.text.strip():
            result["text"] = element.text.strip()
        
        # Handle attributes
        if element.attrib:
            result["attributes"] = element.attrib
        
        # Handle child elements
        children = {}
        for child in element:
            child_name = child.tag.split("}")[-1]  # Remove namespace
            child_data = self._xml_to_dict(child)
            
            if child_name in children:
                # Multiple children with same name - convert to list
                if not isinstance(children[child_name], list):
                    children[child_name] = [children[child_name]]
                children[child_name].append(child_data)
            else:
                children[child_name] = child_data
        
        if children:
            result.update(children)
        
        return result
    
    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters"""
        
        return (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&apos;"))
    
    def _update_metrics(self, success: bool, response_time: float) -> None:
        """Update connector metrics"""
        
        self.metrics["requests_made"] += 1
        self.metrics["total_response_time"] += response_time
        
        if success:
            self.metrics["requests_successful"] += 1
        else:
            self.metrics["requests_failed"] += 1
        
        # Update average response time
        self.metrics["average_response_time"] = (
            self.metrics["total_response_time"] / self.metrics["requests_made"]
        )

class APIIntegrationManager:
    """Manager for API integration connectors"""
    
    def __init__(
        self,
        auth_manager: AuthenticationManager,
        security_manager: SecurityManager,
        performance_monitor: PerformanceMonitor
    ):
        self.auth_manager = auth_manager
        self.security_manager = security_manager
        self.performance_monitor = performance_monitor
        self.connectors = {}
        self.connector_configs = {}
    
    async def create_connector(
        self,
        api_type: APIType,
        base_url: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create API connector"""
        
        connector_id = f"{api_type.value}_{uuid.uuid4().hex[:8]}"
        
        try:
            # Common configuration
            common_config = {
                "auth_manager": self.auth_manager,
                "security_manager": self.security_manager,
                "performance_monitor": self.performance_monitor,
                **config
            }
            
            # Create connector based on type
            if api_type == APIType.REST:
                connector = RESTAPIConnector(base_url, **common_config)
            elif api_type == APIType.GRAPHQL:
                connector = GraphQLAPIConnector(base_url, **common_config)
            elif api_type == APIType.SOAP:
                connector = SOAPAPIConnector(base_url, **common_config)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported API type: {api_type.value}",
                    "connector_id": connector_id
                }
            
            # Store connector
            self.connectors[connector_id] = connector
            self.connector_configs[connector_id] = {
                "api_type": api_type,
                "base_url": base_url,
                "config": config,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Created {api_type.value} connector {connector_id} for {base_url}")
            
            return {
                "success": True,
                "connector_id": connector_id,
                "api_type": api_type.value,
                "base_url": base_url,
                "connector_info": connector.get_connector_info()
            }
            
        except Exception as e:
            logger.error(f"Failed to create {api_type.value} connector: {e}")
            return {
                "success": False,
                "error": str(e),
                "connector_id": connector_id
            }
    
    async def get_connector(self, connector_id: str) -> Optional[BaseAPIConnector]:
        """Get connector by ID"""
        return self.connectors.get(connector_id)
    
    async def list_connectors(self) -> Dict[str, Any]:
        """List all connectors"""
        
        connector_list = []
        for connector_id, connector in self.connectors.items():
            connector_info = connector.get_connector_info()
            connector_info["connector_id"] = connector_id
            connector_info["config"] = self.connector_configs.get(connector_id, {})
            connector_list.append(connector_info)
        
        return {
            "connectors": connector_list,
            "total_count": len(connector_list)
        }
    
    async def remove_connector(self, connector_id: str) -> Dict[str, Any]:
        """Remove connector"""
        
        try:
            if connector_id not in self.connectors:
                return {
                    "success": False,
                    "error": "Connector not found",
                    "connector_id": connector_id
                }
            
            connector = self.connectors[connector_id]
            await connector.close()
            
            del self.connectors[connector_id]
            del self.connector_configs[connector_id]
            
            logger.info(f"Removed connector {connector_id}")
            
            return {
                "success": True,
                "connector_id": connector_id,
                "message": "Connector removed successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to remove connector {connector_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "connector_id": connector_id
            }
    
    async def execute_request(
        self,
        connector_id: str,
        endpoint: APIEndpoint,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute request on specific connector"""
        
        try:
            connector = self.connectors.get(connector_id)
            if not connector:
                return {
                    "success": False,
                    "error": "Connector not found",
                    "connector_id": connector_id
                }
            
            return await connector.execute_request(endpoint, data, **kwargs)
            
        except Exception as e:
            logger.error(f"Failed to execute request: {e}")
            return {
                "success": False,
                "error": str(e),
                "connector_id": connector_id
            }
    
    async def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from all connectors"""
        
        total_metrics = {
            "total_connectors": len(self.connectors),
            "connectors_by_type": {},
            "total_requests": 0,
            "total_successful": 0,
            "total_failed": 0,
            "average_response_time": 0.0,
            "total_rate_limited": 0,
            "total_circuit_breaker_trips": 0
        }
        
        response_times = []
        
        for connector in self.connectors.values():
            api_type = connector.api_type.value
            metrics = connector.metrics
            
            # Count by API type
            if api_type not in total_metrics["connectors_by_type"]:
                total_metrics["connectors_by_type"][api_type] = 0
            total_metrics["connectors_by_type"][api_type] += 1
            
            # Aggregate metrics
            total_metrics["total_requests"] += metrics["requests_made"]
            total_metrics["total_successful"] += metrics["requests_successful"]
            total_metrics["total_failed"] += metrics["requests_failed"]
            total_metrics["total_rate_limited"] += metrics["rate_limited_requests"]
            total_metrics["total_circuit_breaker_trips"] += metrics["circuit_breaker_trips"]
            
            if metrics["requests_made"] > 0:
                response_times.append(metrics["average_response_time"])
        
        # Calculate overall average response time
        if response_times:
            total_metrics["average_response_time"] = sum(response_times) / len(response_times)
        
        return total_metrics
    
    async def close_all(self) -> None:
        """Close all connectors"""
        
        for connector in self.connectors.values():
            try:
                await connector.close()
            except Exception as e:
                logger.error(f"Error closing connector: {e}")
        
        self.connectors.clear()
        self.connector_configs.clear()