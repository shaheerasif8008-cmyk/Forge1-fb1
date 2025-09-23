# forge1/backend/tests/test_api_integration_layer.py

"""
Test suite for API integration layer
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from forge1.integrations.api_integration_layer import (
    APIType,
    HTTPMethod,
    AuthenticationMethod,
    CircuitBreakerState,
    RateLimitConfig,
    RetryConfig,
    CircuitBreakerConfig,
    APIEndpoint,
    RateLimiter,
    CircuitBreaker,
    RESTAPIConnector,
    GraphQLAPIConnector,
    SOAPAPIConnector,
    APIIntegrationManager
)
from forge1.auth.authentication_manager import AuthenticationManager
from forge1.core.security_manager import SecurityManager
from forge1.core.performance_monitor import PerformanceMonitor

@pytest.fixture
def mock_auth_manager():
    """Mock authentication manager"""
    auth_manager = Mock(spec=AuthenticationManager)
    auth_manager.authenticate = AsyncMock(return_value={
        "success": True,
        "headers": {"Authorization": "Bearer test-token"}
    })
    return auth_manager

@pytest.fixture
def mock_security_manager():
    """Mock security manager"""
    return Mock(spec=SecurityManager)

@pytest.fixture
def mock_performance_monitor():
    """Mock performance monitor"""
    monitor = Mock(spec=PerformanceMonitor)
    monitor.track_api_request = AsyncMock()
    return monitor

@pytest.fixture
def rate_limit_config():
    """Rate limit configuration"""
    return RateLimitConfig(
        requests_per_second=5.0,
        requests_per_minute=300,
        burst_size=10
    )

@pytest.fixture
def retry_config():
    """Retry configuration"""
    return RetryConfig(
        max_retries=2,
        base_delay=0.1,
        max_delay=1.0
    )

@pytest.fixture
def circuit_breaker_config():
    """Circuit breaker configuration"""
    return CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=1.0,
        success_threshold=2
    )

@pytest.fixture
def rest_connector(mock_auth_manager, mock_security_manager, mock_performance_monitor, 
                  rate_limit_config, retry_config, circuit_breaker_config):
    """REST API connector fixture"""
    return RESTAPIConnector(
        base_url="https://api.example.com",
        auth_manager=mock_auth_manager,
        security_manager=mock_security_manager,
        performance_monitor=mock_performance_monitor,
        rate_limit_config=rate_limit_config,
        retry_config=retry_config,
        circuit_breaker_config=circuit_breaker_config,
        auth_config={
            "method": AuthenticationMethod.BEARER_TOKEN,
            "token": "test-token"
        }
    )

@pytest.fixture
def graphql_connector(mock_auth_manager, mock_security_manager, mock_performance_monitor):
    """GraphQL API connector fixture"""
    return GraphQLAPIConnector(
        base_url="https://api.example.com/graphql",
        auth_manager=mock_auth_manager,
        security_manager=mock_security_manager,
        performance_monitor=mock_performance_monitor
    )

@pytest.fixture
def soap_connector(mock_auth_manager, mock_security_manager, mock_performance_monitor):
    """SOAP API connector fixture"""
    return SOAPAPIConnector(
        base_url="https://api.example.com/soap",
        wsdl_url="https://api.example.com/soap?wsdl",
        auth_manager=mock_auth_manager,
        security_manager=mock_security_manager,
        performance_monitor=mock_performance_monitor
    )

@pytest.fixture
def api_manager(mock_auth_manager, mock_security_manager, mock_performance_monitor):
    """API integration manager fixture"""
    return APIIntegrationManager(
        auth_manager=mock_auth_manager,
        security_manager=mock_security_manager,
        performance_monitor=mock_performance_monitor
    )

class TestRateLimiter:
    """Test rate limiter functionality"""
    
    @pytest.mark.asyncio
    async def test_token_acquisition(self, rate_limit_config):
        """Test token acquisition"""
        
        limiter = RateLimiter(rate_limit_config)
        
        # Should be able to acquire tokens up to burst size
        for _ in range(rate_limit_config.burst_size):
            assert await limiter.acquire() is True
        
        # Should be rate limited after burst
        assert await limiter.acquire() is False
    
    @pytest.mark.asyncio
    async def test_token_replenishment(self, rate_limit_config):
        """Test token replenishment over time"""
        
        limiter = RateLimiter(rate_limit_config)
        
        # Exhaust tokens
        for _ in range(rate_limit_config.burst_size):
            await limiter.acquire()
        
        # Wait for token replenishment
        await asyncio.sleep(0.3)  # Should replenish ~1.5 tokens
        
        # Should be able to acquire token again
        assert await limiter.acquire() is True
    
    def test_adaptive_rate_adjustment(self, rate_limit_config):
        """Test adaptive rate adjustment"""
        
        limiter = RateLimiter(rate_limit_config)
        original_rate = limiter.adaptive_rate
        
        # Simulate high success rate
        for _ in range(20):
            limiter.update_adaptive_rate(True)
        
        # Rate should increase
        assert limiter.adaptive_rate > original_rate
        
        # Simulate high error rate
        for _ in range(20):
            limiter.update_adaptive_rate(False)
        
        # Rate should decrease
        assert limiter.adaptive_rate < original_rate
    
    def test_rate_limiter_stats(self, rate_limit_config):
        """Test rate limiter statistics"""
        
        limiter = RateLimiter(rate_limit_config)
        
        # Update some metrics
        limiter.update_adaptive_rate(True)
        limiter.update_adaptive_rate(False)
        
        stats = limiter.get_stats()
        
        assert "current_tokens" in stats
        assert "adaptive_rate" in stats
        assert "success_count" in stats
        assert "error_count" in stats
        assert stats["success_count"] == 1
        assert stats["error_count"] == 1

class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self, circuit_breaker_config):
        """Test circuit breaker in closed state"""
        
        breaker = CircuitBreaker(circuit_breaker_config)
        
        async def success_func():
            return "success"
        
        # Should execute successfully
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self, circuit_breaker_config):
        """Test circuit breaker opens after failures"""
        
        breaker = CircuitBreaker(circuit_breaker_config)
        
        async def failure_func():
            raise Exception("Test failure")
        
        # Trigger failures to open circuit
        for _ in range(circuit_breaker_config.failure_threshold):
            with pytest.raises(Exception):
                await breaker.call(failure_func)
        
        # Circuit should be open
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Further calls should fail immediately
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await breaker.call(failure_func)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self, circuit_breaker_config):
        """Test circuit breaker recovery through half-open state"""
        
        breaker = CircuitBreaker(circuit_breaker_config)
        
        # Force circuit to open
        breaker.state = CircuitBreakerState.OPEN
        breaker.last_failure_time = time.time() - circuit_breaker_config.recovery_timeout - 1
        
        async def success_func():
            return "success"
        
        # Should transition to half-open and succeed
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitBreakerState.HALF_OPEN
        
        # After enough successes, should close
        for _ in range(circuit_breaker_config.success_threshold - 1):
            await breaker.call(success_func)
        
        assert breaker.state == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_state(self, circuit_breaker_config):
        """Test circuit breaker state reporting"""
        
        breaker = CircuitBreaker(circuit_breaker_config)
        
        state = breaker.get_state()
        
        assert "state" in state
        assert "failure_count" in state
        assert "success_count" in state
        assert state["state"] == CircuitBreakerState.CLOSED.value

class TestRESTAPIConnector:
    """Test REST API connector functionality"""
    
    @pytest.mark.asyncio
    async def test_rest_get_request(self, rest_connector):
        """Test REST GET request"""
        
        endpoint = APIEndpoint(
            url="/users/123",
            method=HTTPMethod.GET,
            headers={"Accept": "application/json"}
        )
        
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={"id": 123, "name": "Test User"})
        mock_response.close = AsyncMock()
        
        with patch.object(rest_connector, '_make_request_with_retry', return_value=mock_response):
            result = await rest_connector.execute_request(endpoint)
            
            assert result["success"] is True
            assert result["status_code"] == 200
            assert result["data"]["id"] == 123
            assert "response_time" in result
    
    @pytest.mark.asyncio
    async def test_rest_post_request(self, rest_connector):
        """Test REST POST request"""
        
        endpoint = APIEndpoint(
            url="/users",
            method=HTTPMethod.POST,
            headers={"Content-Type": "application/json"}
        )
        
        data = {"name": "New User", "email": "user@example.com"}
        
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status = 201
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={"id": 124, **data})
        mock_response.close = AsyncMock()
        
        with patch.object(rest_connector, '_make_request_with_retry', return_value=mock_response):
            result = await rest_connector.execute_request(endpoint, data)
            
            assert result["success"] is True
            assert result["status_code"] == 201
            assert result["data"]["name"] == "New User"
    
    @pytest.mark.asyncio
    async def test_rest_request_with_template(self, rest_connector):
        """Test REST request with body template"""
        
        endpoint = APIEndpoint(
            url="/users",
            method=HTTPMethod.POST,
            body_template='{"user": {"name": "$name", "email": "$email"}}'
        )
        
        data = {"name": "Test User", "email": "test@example.com"}
        
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status = 201
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.close = AsyncMock()
        
        with patch.object(rest_connector, '_make_request_with_retry', return_value=mock_response):
            result = await rest_connector.execute_request(endpoint, data)
            
            assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_rest_request_failure(self, rest_connector):
        """Test REST request failure handling"""
        
        endpoint = APIEndpoint(url="/error")
        
        with patch.object(rest_connector, '_make_request_with_retry', side_effect=Exception("Network error")):
            result = await rest_connector.execute_request(endpoint)
            
            assert result["success"] is False
            assert "Network error" in result["error"]
            assert "response_time" in result
    
    def test_endpoint_registration(self, rest_connector):
        """Test endpoint registration"""
        
        endpoint = APIEndpoint(url="/test", method=HTTPMethod.GET)
        rest_connector.register_endpoint("test_endpoint", endpoint)
        
        assert "test_endpoint" in rest_connector.endpoints
        assert rest_connector.endpoints["test_endpoint"] == endpoint
    
    @pytest.mark.asyncio
    async def test_convenience_methods(self, rest_connector):
        """Test convenience methods (get, post, etc.)"""
        
        endpoint = APIEndpoint(url="/test")
        rest_connector.register_endpoint("test", endpoint)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.close = AsyncMock()
        
        with patch.object(rest_connector, '_make_request_with_retry', return_value=mock_response):
            # Test GET
            result = await rest_connector.get("test")
            assert result["success"] is True
            
            # Test POST
            result = await rest_connector.post("test", {"data": "test"})
            assert result["success"] is True
            
            # Test PUT
            result = await rest_connector.put("test", {"data": "test"})
            assert result["success"] is True
            
            # Test DELETE
            result = await rest_connector.delete("test")
            assert result["success"] is True

class TestGraphQLAPIConnector:
    """Test GraphQL API connector functionality"""
    
    @pytest.mark.asyncio
    async def test_graphql_query(self, graphql_connector):
        """Test GraphQL query execution"""
        
        query = """
        query GetUser($id: ID!) {
            user(id: $id) {
                id
                name
                email
            }
        }
        """
        
        variables = {"id": "123"}
        
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "data": {
                "user": {
                    "id": "123",
                    "name": "Test User",
                    "email": "test@example.com"
                }
            }
        })
        mock_response.close = AsyncMock()
        
        with patch.object(graphql_connector, '_make_request_with_retry', return_value=mock_response):
            result = await graphql_connector.execute_query(query, variables)
            
            assert result["success"] is True
            assert result["status_code"] == 200
            assert result["data"]["user"]["id"] == "123"
    
    @pytest.mark.asyncio
    async def test_graphql_mutation(self, graphql_connector):
        """Test GraphQL mutation execution"""
        
        mutation = """
        mutation CreateUser($input: UserInput!) {
            createUser(input: $input) {
                id
                name
            }
        }
        """
        
        variables = {"input": {"name": "New User", "email": "new@example.com"}}
        
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "data": {
                "createUser": {
                    "id": "124",
                    "name": "New User"
                }
            }
        })
        mock_response.close = AsyncMock()
        
        with patch.object(graphql_connector, '_make_request_with_retry', return_value=mock_response):
            result = await graphql_connector.execute_mutation(mutation, variables)
            
            assert result["success"] is True
            assert result["data"]["createUser"]["name"] == "New User"
    
    @pytest.mark.asyncio
    async def test_graphql_errors(self, graphql_connector):
        """Test GraphQL error handling"""
        
        query = "query { invalidField }"
        
        # Mock HTTP response with GraphQL errors
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "errors": [
                {
                    "message": "Cannot query field 'invalidField'",
                    "locations": [{"line": 1, "column": 9}]
                }
            ]
        })
        mock_response.close = AsyncMock()
        
        with patch.object(graphql_connector, '_make_request_with_retry', return_value=mock_response):
            result = await graphql_connector.execute_query(query)
            
            assert result["success"] is False
            assert "errors" in result
            assert len(result["errors"]) == 1
    
    @pytest.mark.asyncio
    async def test_schema_introspection(self, graphql_connector):
        """Test GraphQL schema introspection"""
        
        # Mock introspection response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "data": {
                "__schema": {
                    "types": [
                        {
                            "name": "User",
                            "kind": "OBJECT",
                            "fields": [
                                {"name": "id", "type": {"name": "ID", "kind": "SCALAR"}},
                                {"name": "name", "type": {"name": "String", "kind": "SCALAR"}}
                            ]
                        }
                    ]
                }
            }
        })
        mock_response.close = AsyncMock()
        
        with patch.object(graphql_connector, '_make_request_with_retry', return_value=mock_response):
            result = await graphql_connector.introspect_schema()
            
            assert result["success"] is True
            assert "schema" in result
            assert graphql_connector.schema is not None

class TestSOAPAPIConnector:
    """Test SOAP API connector functionality"""
    
    @pytest.mark.asyncio
    async def test_wsdl_loading(self, soap_connector):
        """Test WSDL loading and parsing"""
        
        wsdl_content = """<?xml version="1.0" encoding="utf-8"?>
<definitions xmlns="http://schemas.xmlsoap.org/wsdl/" 
             xmlns:tns="http://tempuri.org/"
             targetNamespace="http://tempuri.org/">
    <portType name="TestPortType">
        <operation name="GetUser">
            <input message="tns:GetUserRequest"/>
            <output message="tns:GetUserResponse"/>
        </operation>
    </portType>
    <binding name="TestBinding" type="tns:TestPortType">
        <soap:binding xmlns:soap="http://schemas.xmlsoap.org/wsdl/soap/" transport="http://schemas.xmlsoap.org/soap/http"/>
        <operation name="GetUser">
            <soap:operation xmlns:soap="http://schemas.xmlsoap.org/wsdl/soap/" soapAction="http://tempuri.org/GetUser"/>
        </operation>
    </binding>
</definitions>"""
        
        # Mock WSDL response
        mock_response = Mock()
        mock_response.text = AsyncMock(return_value=wsdl_content)
        mock_response.close = AsyncMock()
        
        with patch.object(soap_connector, '_make_request_with_retry', return_value=mock_response):
            result = await soap_connector.load_wsdl()
            
            assert result["success"] is True
            assert "wsdl_schema" in result
            assert "GetUser" in soap_connector.soap_actions
    
    @pytest.mark.asyncio
    async def test_soap_operation_execution(self, soap_connector):
        """Test SOAP operation execution"""
        
        # Setup SOAP actions
        soap_connector.soap_actions = {
            "GetUser": {
                "soap_action": "http://tempuri.org/GetUser",
                "operation_name": "GetUser"
            }
        }
        
        parameters = {"userId": "123"}
        
        # Mock SOAP response
        soap_response = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
    <soap:Body>
        <GetUserResponse xmlns="http://tempuri.org/">
            <User>
                <Id>123</Id>
                <Name>Test User</Name>
            </User>
        </GetUserResponse>
    </soap:Body>
</soap:Envelope>"""
        
        mock_response = Mock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=soap_response)
        mock_response.headers = {"Content-Type": "text/xml"}
        mock_response.close = AsyncMock()
        
        with patch.object(soap_connector, '_make_request_with_retry', return_value=mock_response):
            result = await soap_connector.execute_soap_operation("GetUser", parameters)
            
            assert result["success"] is True
            assert result["status_code"] == 200
            assert result["operation"] == "GetUser"
            assert "data" in result
    
    @pytest.mark.asyncio
    async def test_soap_fault_handling(self, soap_connector):
        """Test SOAP fault handling"""
        
        # Setup SOAP actions
        soap_connector.soap_actions = {
            "GetUser": {
                "soap_action": "http://tempuri.org/GetUser",
                "operation_name": "GetUser"
            }
        }
        
        # Mock SOAP fault response
        fault_response = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
    <soap:Body>
        <soap:Fault>
            <faultcode>Client</faultcode>
            <faultstring>Invalid user ID</faultstring>
        </soap:Fault>
    </soap:Body>
</soap:Envelope>"""
        
        mock_response = Mock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value=fault_response)
        mock_response.headers = {"Content-Type": "text/xml"}
        mock_response.close = AsyncMock()
        
        with patch.object(soap_connector, '_make_request_with_retry', return_value=mock_response):
            result = await soap_connector.execute_soap_operation("GetUser", {"userId": "invalid"})
            
            assert result["success"] is False
            assert "fault" in result
            assert result["fault"]["code"] == "Client"
            assert result["fault"]["string"] == "Invalid user ID"
    
    def test_soap_envelope_building(self, soap_connector):
        """Test SOAP envelope building"""
        
        soap_connector.namespace_map = {"tns": "http://tempuri.org/"}
        
        envelope = soap_connector._build_soap_envelope(
            "GetUser",
            {"userId": "123", "includeDetails": "true"}
        )
        
        assert "GetUser" in envelope
        assert "userId" in envelope
        assert "123" in envelope
        assert "includeDetails" in envelope
        assert "true" in envelope
        assert "soap:Envelope" in envelope

class TestAPIIntegrationManager:
    """Test API integration manager functionality"""
    
    @pytest.mark.asyncio
    async def test_create_rest_connector(self, api_manager):
        """Test creating REST connector through manager"""
        
        config = {
            "auth_config": {
                "method": AuthenticationMethod.API_KEY,
                "api_key": "test-key"
            },
            "default_headers": {"User-Agent": "Forge1/1.0"}
        }
        
        result = await api_manager.create_connector(
            APIType.REST,
            "https://api.example.com",
            config
        )
        
        assert result["success"] is True
        assert result["api_type"] == "rest"
        assert result["base_url"] == "https://api.example.com"
        assert "connector_id" in result
        
        # Check connector is stored
        connector_id = result["connector_id"]
        connector = await api_manager.get_connector(connector_id)
        assert connector is not None
        assert connector.api_type == APIType.REST
    
    @pytest.mark.asyncio
    async def test_create_graphql_connector(self, api_manager):
        """Test creating GraphQL connector through manager"""
        
        result = await api_manager.create_connector(
            APIType.GRAPHQL,
            "https://api.example.com/graphql",
            {}
        )
        
        assert result["success"] is True
        assert result["api_type"] == "graphql"
    
    @pytest.mark.asyncio
    async def test_create_soap_connector(self, api_manager):
        """Test creating SOAP connector through manager"""
        
        config = {
            "wsdl_url": "https://api.example.com/soap?wsdl"
        }
        
        result = await api_manager.create_connector(
            APIType.SOAP,
            "https://api.example.com/soap",
            config
        )
        
        assert result["success"] is True
        assert result["api_type"] == "soap"
    
    @pytest.mark.asyncio
    async def test_list_connectors(self, api_manager):
        """Test listing all connectors"""
        
        # Create multiple connectors
        await api_manager.create_connector(APIType.REST, "https://rest.example.com", {})
        await api_manager.create_connector(APIType.GRAPHQL, "https://graphql.example.com", {})
        
        result = await api_manager.list_connectors()
        
        assert "connectors" in result
        assert result["total_count"] == 2
        assert len(result["connectors"]) == 2
        
        # Check connector types
        api_types = [conn["api_type"] for conn in result["connectors"]]
        assert "rest" in api_types
        assert "graphql" in api_types
    
    @pytest.mark.asyncio
    async def test_remove_connector(self, api_manager):
        """Test removing connector"""
        
        # Create a connector
        create_result = await api_manager.create_connector(
            APIType.REST,
            "https://api.example.com",
            {}
        )
        
        connector_id = create_result["connector_id"]
        
        # Remove the connector
        result = await api_manager.remove_connector(connector_id)
        
        assert result["success"] is True
        assert result["connector_id"] == connector_id
        
        # Check connector is removed
        connector = await api_manager.get_connector(connector_id)
        assert connector is None
    
    @pytest.mark.asyncio
    async def test_execute_request_through_manager(self, api_manager):
        """Test executing request through manager"""
        
        # Create connector
        create_result = await api_manager.create_connector(
            APIType.REST,
            "https://api.example.com",
            {}
        )
        
        connector_id = create_result["connector_id"]
        endpoint = APIEndpoint(url="/test", method=HTTPMethod.GET)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.close = AsyncMock()
        
        connector = await api_manager.get_connector(connector_id)
        
        with patch.object(connector, '_make_request_with_retry', return_value=mock_response):
            result = await api_manager.execute_request(connector_id, endpoint)
            
            assert result["success"] is True
            assert result["status_code"] == 200
    
    @pytest.mark.asyncio
    async def test_aggregated_metrics(self, api_manager):
        """Test getting aggregated metrics"""
        
        # Create connectors and simulate some requests
        rest_result = await api_manager.create_connector(
            APIType.REST,
            "https://api.example.com",
            {}
        )
        
        graphql_result = await api_manager.create_connector(
            APIType.GRAPHQL,
            "https://graphql.example.com",
            {}
        )
        
        # Simulate some metrics
        rest_connector = await api_manager.get_connector(rest_result["connector_id"])
        rest_connector.metrics["requests_made"] = 10
        rest_connector.metrics["requests_successful"] = 8
        rest_connector.metrics["requests_failed"] = 2
        
        graphql_connector = await api_manager.get_connector(graphql_result["connector_id"])
        graphql_connector.metrics["requests_made"] = 5
        graphql_connector.metrics["requests_successful"] = 5
        graphql_connector.metrics["requests_failed"] = 0
        
        # Get metrics
        metrics = await api_manager.get_aggregated_metrics()
        
        assert metrics["total_connectors"] == 2
        assert "rest" in metrics["connectors_by_type"]
        assert "graphql" in metrics["connectors_by_type"]
        assert metrics["total_requests"] == 15
        assert metrics["total_successful"] == 13
        assert metrics["total_failed"] == 2
    
    @pytest.mark.asyncio
    async def test_unsupported_api_type(self, api_manager):
        """Test creating connector for unsupported API type"""
        
        # This should be handled by enum validation, but test error handling
        with patch('forge1.integrations.api_integration_layer.APIType') as mock_api_type:
            mock_api_type.UNSUPPORTED = "unsupported"
            
            result = await api_manager.create_connector(
                mock_api_type.UNSUPPORTED,
                "https://api.example.com",
                {}
            )
            
            assert result["success"] is False
            assert "Unsupported API type" in result["error"]

if __name__ == "__main__":
    pytest.main([__file__])