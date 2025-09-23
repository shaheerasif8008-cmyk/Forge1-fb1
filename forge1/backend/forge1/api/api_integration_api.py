# forge1/backend/forge1/api/api_integration_api.py

"""
FastAPI endpoints for API integration layer
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
import logging

from forge1.integrations.api_integration_layer import (
    APIType,
    HTTPMethod,
    AuthenticationMethod,
    RateLimitConfig,
    RetryConfig,
    CircuitBreakerConfig,
    APIEndpoint,
    APIIntegrationManager
)
from forge1.auth.authentication_manager import AuthenticationManager
from forge1.core.security_manager import SecurityManager
from forge1.core.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class RateLimitConfigModel(BaseModel):
    """Rate limiting configuration model"""
    requests_per_second: float = Field(10.0, gt=0, description="Requests per second limit")
    requests_per_minute: int = Field(600, gt=0, description="Requests per minute limit")
    requests_per_hour: int = Field(36000, gt=0, description="Requests per hour limit")
    burst_size: int = Field(20, gt=0, description="Burst size for token bucket")
    enable_adaptive: bool = Field(True, description="Enable adaptive rate limiting")

class RetryConfigModel(BaseModel):
    """Retry configuration model"""
    max_retries: int = Field(3, ge=0, le=10, description="Maximum number of retries")
    base_delay: float = Field(1.0, gt=0, description="Base delay between retries in seconds")
    max_delay: float = Field(60.0, gt=0, description="Maximum delay between retries in seconds")
    exponential_base: float = Field(2.0, gt=1, description="Exponential backoff base")
    jitter: bool = Field(True, description="Add jitter to retry delays")
    retry_on_status_codes: List[int] = Field(
        default=[429, 500, 502, 503, 504],
        description="HTTP status codes to retry on"
    )

class CircuitBreakerConfigModel(BaseModel):
    """Circuit breaker configuration model"""
    failure_threshold: int = Field(5, gt=0, description="Number of failures to open circuit")
    recovery_timeout: float = Field(60.0, gt=0, description="Time to wait before attempting recovery")
    success_threshold: int = Field(3, gt=0, description="Number of successes to close circuit")
    timeout: float = Field(30.0, gt=0, description="Request timeout in seconds")

class AuthConfigModel(BaseModel):
    """Authentication configuration model"""
    method: str = Field(AuthenticationMethod.NONE.value, description="Authentication method")
    credential_id: Optional[str] = Field(None, description="Credential ID from auth manager")
    api_key: Optional[str] = Field(None, description="API key for API key authentication")
    key_header: Optional[str] = Field("X-API-Key", description="Header name for API key")
    token: Optional[str] = Field(None, description="Bearer token")
    username: Optional[str] = Field(None, description="Username for basic auth")
    password: Optional[str] = Field(None, description="Password for basic auth")
    custom_headers: Optional[Dict[str, str]] = Field(default_factory=dict, description="Custom headers")
    
    @validator('method')
    def validate_method(cls, v):
        try:
            AuthenticationMethod(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid authentication method: {v}")

class APIEndpointModel(BaseModel):
    """API endpoint model"""
    url: str = Field(..., description="Endpoint URL path")
    method: str = Field(HTTPMethod.GET.value, description="HTTP method")
    headers: Dict[str, str] = Field(default_factory=dict, description="Request headers")
    query_params: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    body_template: Optional[str] = Field(None, description="Request body template")
    response_transform: Optional[str] = Field(None, description="Response transformation rule")
    timeout: float = Field(30.0, gt=0, description="Request timeout in seconds")
    
    @validator('method')
    def validate_method(cls, v):
        try:
            HTTPMethod(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid HTTP method: {v}")

class ConnectorCreateRequest(BaseModel):
    """API connector creation request"""
    api_type: str = Field(..., description="API type (rest, graphql, soap)")
    base_url: str = Field(..., description="Base URL for the API")
    auth_config: Optional[AuthConfigModel] = Field(None, description="Authentication configuration")
    rate_limit_config: Optional[RateLimitConfigModel] = Field(None, description="Rate limiting configuration")
    retry_config: Optional[RetryConfigModel] = Field(None, description="Retry configuration")
    circuit_breaker_config: Optional[CircuitBreakerConfigModel] = Field(None, description="Circuit breaker configuration")
    default_headers: Optional[Dict[str, str]] = Field(default_factory=dict, description="Default headers")
    ssl_verify: bool = Field(True, description="Verify SSL certificates")
    proxy: Optional[str] = Field(None, description="Proxy URL")
    wsdl_url: Optional[str] = Field(None, description="WSDL URL for SOAP APIs")
    
    @validator('api_type')
    def validate_api_type(cls, v):
        try:
            APIType(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid API type: {v}")

class APIRequestModel(BaseModel):
    """API request execution model"""
    endpoint: APIEndpointModel = Field(..., description="API endpoint configuration")
    data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Request data")
    headers: Optional[Dict[str, str]] = Field(default_factory=dict, description="Additional headers")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional query parameters")

class GraphQLRequestModel(BaseModel):
    """GraphQL request model"""
    query: str = Field(..., description="GraphQL query or mutation")
    variables: Optional[Dict[str, Any]] = Field(default_factory=dict, description="GraphQL variables")
    operation_name: Optional[str] = Field(None, description="Operation name")

class SOAPRequestModel(BaseModel):
    """SOAP request model"""
    operation: str = Field(..., description="SOAP operation name")
    parameters: Dict[str, Any] = Field(..., description="SOAP operation parameters")

class EndpointRegistrationModel(BaseModel):
    """Endpoint registration model"""
    name: str = Field(..., description="Endpoint name")
    endpoint: APIEndpointModel = Field(..., description="Endpoint configuration")

# Global API integration manager instance
api_integration_manager: Optional[APIIntegrationManager] = None

def get_api_integration_manager() -> APIIntegrationManager:
    """Get API integration manager instance"""
    global api_integration_manager
    if api_integration_manager is None:
        # In production, these would be injected via dependency injection
        auth_manager = AuthenticationManager()
        security_manager = SecurityManager()
        performance_monitor = PerformanceMonitor()
        
        api_integration_manager = APIIntegrationManager(
            auth_manager=auth_manager,
            security_manager=security_manager,
            performance_monitor=performance_monitor
        )
    
    return api_integration_manager

# Create API router
router = APIRouter(prefix="/api/v1/integrations", tags=["api-integration"])

@router.post("/connectors", response_model=Dict[str, Any])
async def create_api_connector(
    request: ConnectorCreateRequest,
    manager: APIIntegrationManager = Depends(get_api_integration_manager)
) -> Dict[str, Any]:
    """Create API integration connector"""
    
    try:
        # Convert Pydantic models to configuration dictionaries
        config = {}
        
        if request.auth_config:
            config["auth_config"] = request.auth_config.dict(exclude_none=True)
        
        if request.rate_limit_config:
            rate_config = RateLimitConfig(**request.rate_limit_config.dict())
            config["rate_limit_config"] = rate_config
        
        if request.retry_config:
            retry_config = RetryConfig(**request.retry_config.dict())
            config["retry_config"] = retry_config
        
        if request.circuit_breaker_config:
            cb_config = CircuitBreakerConfig(**request.circuit_breaker_config.dict())
            config["circuit_breaker_config"] = cb_config
        
        # Add other configuration
        config.update({
            "default_headers": request.default_headers,
            "ssl_verify": request.ssl_verify,
            "proxy": request.proxy
        })
        
        # Add SOAP-specific configuration
        if request.api_type == APIType.SOAP.value and request.wsdl_url:
            config["wsdl_url"] = request.wsdl_url
        
        # Create connector
        api_type = APIType(request.api_type)
        result = await manager.create_connector(api_type, request.base_url, config)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create API connector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/connectors", response_model=Dict[str, Any])
async def list_api_connectors(
    manager: APIIntegrationManager = Depends(get_api_integration_manager)
) -> Dict[str, Any]:
    """List all API integration connectors"""
    
    try:
        return await manager.list_connectors()
        
    except Exception as e:
        logger.error(f"Failed to list API connectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/connectors/{connector_id}", response_model=Dict[str, Any])
async def get_api_connector(
    connector_id: str,
    manager: APIIntegrationManager = Depends(get_api_integration_manager)
) -> Dict[str, Any]:
    """Get specific API connector information"""
    
    try:
        connector = await manager.get_connector(connector_id)
        
        if not connector:
            raise HTTPException(status_code=404, detail="Connector not found")
        
        return connector.get_connector_info()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get API connector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/connectors/{connector_id}", response_model=Dict[str, Any])
async def remove_api_connector(
    connector_id: str,
    manager: APIIntegrationManager = Depends(get_api_integration_manager)
) -> Dict[str, Any]:
    """Remove API integration connector"""
    
    try:
        result = await manager.remove_connector(connector_id)
        
        if not result["success"]:
            if "not found" in result["error"].lower():
                raise HTTPException(status_code=404, detail=result["error"])
            else:
                raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove API connector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/connectors/{connector_id}/request", response_model=Dict[str, Any])
async def execute_api_request(
    connector_id: str,
    request: APIRequestModel,
    background_tasks: BackgroundTasks,
    manager: APIIntegrationManager = Depends(get_api_integration_manager)
) -> Dict[str, Any]:
    """Execute API request on specific connector"""
    
    try:
        # Convert Pydantic model to APIEndpoint
        endpoint_dict = request.endpoint.dict()
        endpoint_dict["method"] = HTTPMethod(endpoint_dict["method"])
        endpoint = APIEndpoint(**endpoint_dict)
        
        # Execute request
        result = await manager.execute_request(
            connector_id=connector_id,
            endpoint=endpoint,
            data=request.data,
            headers=request.headers,
            params=request.params
        )
        
        if not result["success"]:
            if "not found" in result["error"].lower():
                raise HTTPException(status_code=404, detail=result["error"])
            else:
                raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to execute API request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/connectors/{connector_id}/graphql", response_model=Dict[str, Any])
async def execute_graphql_request(
    connector_id: str,
    request: GraphQLRequestModel,
    manager: APIIntegrationManager = Depends(get_api_integration_manager)
) -> Dict[str, Any]:
    """Execute GraphQL request on specific connector"""
    
    try:
        connector = await manager.get_connector(connector_id)
        
        if not connector:
            raise HTTPException(status_code=404, detail="Connector not found")
        
        if connector.api_type != APIType.GRAPHQL:
            raise HTTPException(status_code=400, detail="Connector is not a GraphQL connector")
        
        # Execute GraphQL request
        result = await connector.execute_query(
            query=request.query,
            variables=request.variables,
            operation_name=request.operation_name
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "GraphQL request failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute GraphQL request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/connectors/{connector_id}/soap", response_model=Dict[str, Any])
async def execute_soap_request(
    connector_id: str,
    request: SOAPRequestModel,
    manager: APIIntegrationManager = Depends(get_api_integration_manager)
) -> Dict[str, Any]:
    """Execute SOAP request on specific connector"""
    
    try:
        connector = await manager.get_connector(connector_id)
        
        if not connector:
            raise HTTPException(status_code=404, detail="Connector not found")
        
        if connector.api_type != APIType.SOAP:
            raise HTTPException(status_code=400, detail="Connector is not a SOAP connector")
        
        # Execute SOAP request
        result = await connector.execute_soap_operation(
            operation_name=request.operation,
            parameters=request.parameters
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "SOAP request failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute SOAP request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/connectors/{connector_id}/endpoints", response_model=Dict[str, Any])
async def register_endpoint(
    connector_id: str,
    request: EndpointRegistrationModel,
    manager: APIIntegrationManager = Depends(get_api_integration_manager)
) -> Dict[str, Any]:
    """Register named endpoint on REST connector"""
    
    try:
        connector = await manager.get_connector(connector_id)
        
        if not connector:
            raise HTTPException(status_code=404, detail="Connector not found")
        
        if connector.api_type != APIType.REST:
            raise HTTPException(status_code=400, detail="Endpoint registration only supported for REST connectors")
        
        # Convert Pydantic model to APIEndpoint
        endpoint_dict = request.endpoint.dict()
        endpoint_dict["method"] = HTTPMethod(endpoint_dict["method"])
        endpoint = APIEndpoint(**endpoint_dict)
        
        # Register endpoint
        connector.register_endpoint(request.name, endpoint)
        
        return {
            "success": True,
            "message": f"Endpoint '{request.name}' registered successfully",
            "endpoint_name": request.name,
            "connector_id": connector_id
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to register endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/connectors/{connector_id}/endpoints", response_model=Dict[str, Any])
async def list_endpoints(
    connector_id: str,
    manager: APIIntegrationManager = Depends(get_api_integration_manager)
) -> Dict[str, Any]:
    """List registered endpoints on REST connector"""
    
    try:
        connector = await manager.get_connector(connector_id)
        
        if not connector:
            raise HTTPException(status_code=404, detail="Connector not found")
        
        if connector.api_type != APIType.REST:
            raise HTTPException(status_code=400, detail="Endpoint listing only supported for REST connectors")
        
        endpoints = []
        for name, endpoint in connector.endpoints.items():
            endpoints.append({
                "name": name,
                "url": endpoint.url,
                "method": endpoint.method.value,
                "headers": endpoint.headers,
                "query_params": endpoint.query_params,
                "timeout": endpoint.timeout
            })
        
        return {
            "endpoints": endpoints,
            "total_count": len(endpoints),
            "connector_id": connector_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list endpoints: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/connectors/{connector_id}/schema", response_model=Dict[str, Any])
async def get_api_schema(
    connector_id: str,
    manager: APIIntegrationManager = Depends(get_api_integration_manager)
) -> Dict[str, Any]:
    """Get API schema (GraphQL introspection or SOAP WSDL)"""
    
    try:
        connector = await manager.get_connector(connector_id)
        
        if not connector:
            raise HTTPException(status_code=404, detail="Connector not found")
        
        if connector.api_type == APIType.GRAPHQL:
            # GraphQL schema introspection
            result = await connector.introspect_schema()
            
            if not result["success"]:
                raise HTTPException(status_code=400, detail=result["error"])
            
            return result
            
        elif connector.api_type == APIType.SOAP:
            # SOAP WSDL loading
            result = await connector.load_wsdl()
            
            if not result["success"]:
                raise HTTPException(status_code=400, detail=result["error"])
            
            return result
            
        else:
            raise HTTPException(status_code=400, detail="Schema introspection not supported for this API type")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get API schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=Dict[str, Any])
async def get_integration_metrics(
    manager: APIIntegrationManager = Depends(get_api_integration_manager)
) -> Dict[str, Any]:
    """Get aggregated API integration metrics"""
    
    try:
        return await manager.get_aggregated_metrics()
        
    except Exception as e:
        logger.error(f"Failed to get integration metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api-types", response_model=Dict[str, Any])
async def get_supported_api_types() -> Dict[str, Any]:
    """Get list of supported API types"""
    
    api_types = []
    for api_type in APIType:
        api_types.append({
            "name": api_type.value,
            "display_name": api_type.value.upper(),
            "description": _get_api_type_description(api_type)
        })
    
    return {
        "api_types": api_types,
        "total_count": len(api_types)
    }

@router.get("/auth-methods", response_model=Dict[str, Any])
async def get_authentication_methods() -> Dict[str, Any]:
    """Get list of supported authentication methods"""
    
    auth_methods = []
    for auth_method in AuthenticationMethod:
        auth_methods.append({
            "name": auth_method.value,
            "display_name": auth_method.value.replace("_", " ").title(),
            "description": _get_auth_method_description(auth_method)
        })
    
    return {
        "auth_methods": auth_methods,
        "total_count": len(auth_methods)
    }

def _get_api_type_description(api_type: APIType) -> str:
    """Get description for API type"""
    
    descriptions = {
        APIType.REST: "RESTful APIs using HTTP methods (GET, POST, PUT, DELETE, etc.)",
        APIType.GRAPHQL: "GraphQL APIs with query, mutation, and subscription support",
        APIType.SOAP: "SOAP APIs with WSDL-based service definitions",
        APIType.WEBHOOK: "Webhook endpoints for receiving HTTP callbacks"
    }
    
    return descriptions.get(api_type, "API integration type")

def _get_auth_method_description(auth_method: AuthenticationMethod) -> str:
    """Get description for authentication method"""
    
    descriptions = {
        AuthenticationMethod.NONE: "No authentication required",
        AuthenticationMethod.API_KEY: "API key authentication via custom header",
        AuthenticationMethod.BEARER_TOKEN: "Bearer token authentication via Authorization header",
        AuthenticationMethod.BASIC_AUTH: "HTTP Basic authentication with username and password",
        AuthenticationMethod.OAUTH2: "OAuth 2.0 authentication flow",
        AuthenticationMethod.CUSTOM_HEADER: "Custom header-based authentication",
        AuthenticationMethod.HMAC_SIGNATURE: "HMAC signature-based authentication",
        AuthenticationMethod.MUTUAL_TLS: "Mutual TLS certificate-based authentication"
    }
    
    return descriptions.get(auth_method, "Authentication method")