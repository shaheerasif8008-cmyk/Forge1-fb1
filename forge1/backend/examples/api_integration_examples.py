# forge1/backend/examples/api_integration_examples.py

"""
Examples demonstrating the API Integration Layer functionality

This file contains comprehensive examples showing how to use the Forge 1
API Integration Layer for REST, GraphQL, and SOAP APIs with various
authentication methods, rate limiting, and error handling.
"""

import asyncio
import json
from typing import Dict, Any

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

async def example_rest_api_integration():
    """Example: REST API integration with authentication and rate limiting"""
    
    print("=== REST API Integration Example ===")
    
    # Initialize components
    auth_manager = AuthenticationManager()
    security_manager = SecurityManager()
    performance_monitor = PerformanceMonitor()
    
    # Create API integration manager
    manager = APIIntegrationManager(
        auth_manager=auth_manager,
        security_manager=security_manager,
        performance_monitor=performance_monitor
    )
    
    # Configure rate limiting and retry policies
    rate_config = RateLimitConfig(
        requests_per_second=5.0,
        requests_per_minute=300,
        burst_size=10,
        enable_adaptive=True
    )
    
    retry_config = RetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        exponential_base=2.0,
        jitter=True,
        retry_on_status_codes=[429, 500, 502, 503, 504]
    )
    
    circuit_breaker_config = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60.0,
        success_threshold=3,
        timeout=30.0
    )
    
    # Create REST API connector
    connector_config = {
        "auth_config": {
            "method": AuthenticationMethod.BEARER_TOKEN,
            "token": "your-api-token-here"
        },
        "rate_limit_config": rate_config,
        "retry_config": retry_config,
        "circuit_breaker_config": circuit_breaker_config,
        "default_headers": {
            "User-Agent": "Forge1-API-Client/1.0",
            "Accept": "application/json"
        }
    }
    
    result = await manager.create_connector(
        APIType.REST,
        "https://jsonplaceholder.typicode.com",
        connector_config
    )
    
    if result["success"]:
        connector_id = result["connector_id"]
        print(f"Created REST connector: {connector_id}")
        
        # Get the connector
        connector = await manager.get_connector(connector_id)
        
        # Register some endpoints
        endpoints = {
            "get_users": APIEndpoint(
                url="/users",
                method=HTTPMethod.GET,
                headers={"Accept": "application/json"}
            ),
            "get_user": APIEndpoint(
                url="/users/{id}",
                method=HTTPMethod.GET,
                headers={"Accept": "application/json"}
            ),
            "create_user": APIEndpoint(
                url="/users",
                method=HTTPMethod.POST,
                headers={"Content-Type": "application/json"},
                body_template='{"name": "$name", "email": "$email", "username": "$username"}'
            ),
            "update_user": APIEndpoint(
                url="/users/{id}",
                method=HTTPMethod.PUT,
                headers={"Content-Type": "application/json"}
            )
        }
        
        for name, endpoint in endpoints.items():
            connector.register_endpoint(name, endpoint)
            print(f"Registered endpoint: {name}")
        
        # Example 1: Get all users
        print("\n--- Getting all users ---")
        users_result = await connector.get("get_users")
        if users_result["success"]:
            print(f"Retrieved {len(users_result['data'])} users")
            print(f"Response time: {users_result['response_time']:.2f}s")
        else:
            print(f"Failed to get users: {users_result['error']}")
        
        # Example 2: Get specific user
        print("\n--- Getting specific user ---")
        user_endpoint = APIEndpoint(url="/users/1", method=HTTPMethod.GET)
        user_result = await connector.execute_request(user_endpoint)
        if user_result["success"]:
            user_data = user_result["data"]
            print(f"User: {user_data['name']} ({user_data['email']})")
        else:
            print(f"Failed to get user: {user_result['error']}")
        
        # Example 3: Create new user
        print("\n--- Creating new user ---")
        new_user_data = {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "username": "johndoe"
        }
        create_result = await connector.post("create_user", new_user_data)
        if create_result["success"]:
            print(f"Created user with ID: {create_result['data']['id']}")
        else:
            print(f"Failed to create user: {create_result['error']}")
        
        # Get connector metrics
        print("\n--- Connector Metrics ---")
        connector_info = connector.get_connector_info()
        metrics = connector_info["metrics"]
        print(f"Total requests: {metrics['requests_made']}")
        print(f"Successful requests: {metrics['requests_successful']}")
        print(f"Failed requests: {metrics['requests_failed']}")
        print(f"Average response time: {metrics['average_response_time']:.2f}s")
        
        # Rate limiter stats
        rate_stats = connector_info["rate_limiter_stats"]
        print(f"Current tokens: {rate_stats['current_tokens']:.1f}")
        print(f"Adaptive rate: {rate_stats['adaptive_rate']:.1f} req/s")
        
        # Circuit breaker state
        cb_state = connector_info["circuit_breaker_state"]
        print(f"Circuit breaker state: {cb_state['state']}")
        
        # Cleanup
        await manager.remove_connector(connector_id)
        print(f"\nRemoved connector: {connector_id}")
    
    else:
        print(f"Failed to create REST connector: {result['error']}")

async def example_graphql_api_integration():
    """Example: GraphQL API integration"""
    
    print("\n=== GraphQL API Integration Example ===")
    
    # Initialize components
    auth_manager = AuthenticationManager()
    security_manager = SecurityManager()
    performance_monitor = PerformanceMonitor()
    
    manager = APIIntegrationManager(
        auth_manager=auth_manager,
        security_manager=security_manager,
        performance_monitor=performance_monitor
    )
    
    # Create GraphQL connector (using a public GraphQL API)
    connector_config = {
        "auth_config": {
            "method": AuthenticationMethod.NONE
        },
        "default_headers": {
            "User-Agent": "Forge1-GraphQL-Client/1.0"
        }
    }
    
    result = await manager.create_connector(
        APIType.GRAPHQL,
        "https://countries.trevorblades.com/",
        connector_config
    )
    
    if result["success"]:
        connector_id = result["connector_id"]
        print(f"Created GraphQL connector: {connector_id}")
        
        connector = await manager.get_connector(connector_id)
        
        # Example 1: Simple query
        print("\n--- Simple GraphQL Query ---")
        query = """
        query GetCountries {
            countries {
                code
                name
                capital
                currency
            }
        }
        """
        
        query_result = await connector.execute_query(query)
        if query_result["success"]:
            countries = query_result["data"]["countries"]
            print(f"Retrieved {len(countries)} countries")
            # Show first 3 countries
            for country in countries[:3]:
                print(f"  {country['name']} ({country['code']}) - Capital: {country['capital']}")
        else:
            print(f"Query failed: {query_result.get('error', 'Unknown error')}")
        
        # Example 2: Query with variables
        print("\n--- GraphQL Query with Variables ---")
        query_with_vars = """
        query GetCountry($code: ID!) {
            country(code: $code) {
                name
                capital
                currency
                languages {
                    name
                    native
                }
            }
        }
        """
        
        variables = {"code": "US"}
        
        var_result = await connector.execute_query(query_with_vars, variables)
        if var_result["success"]:
            country = var_result["data"]["country"]
            print(f"Country: {country['name']}")
            print(f"Capital: {country['capital']}")
            print(f"Currency: {country['currency']}")
            print("Languages:")
            for lang in country['languages']:
                print(f"  {lang['name']} ({lang['native']})")
        else:
            print(f"Query with variables failed: {var_result.get('error', 'Unknown error')}")
        
        # Example 3: Schema introspection
        print("\n--- GraphQL Schema Introspection ---")
        schema_result = await connector.introspect_schema()
        if schema_result["success"]:
            schema = schema_result["schema"]
            print("Schema introspection successful")
            print(f"Schema data available: {bool(schema)}")
        else:
            print(f"Schema introspection failed: {schema_result['error']}")
        
        # Cleanup
        await manager.remove_connector(connector_id)
        print(f"\nRemoved GraphQL connector: {connector_id}")
    
    else:
        print(f"Failed to create GraphQL connector: {result['error']}")

async def example_soap_api_integration():
    """Example: SOAP API integration (mock example)"""
    
    print("\n=== SOAP API Integration Example ===")
    
    # Initialize components
    auth_manager = AuthenticationManager()
    security_manager = SecurityManager()
    performance_monitor = PerformanceMonitor()
    
    manager = APIIntegrationManager(
        auth_manager=auth_manager,
        security_manager=security_manager,
        performance_monitor=performance_monitor
    )
    
    # Create SOAP connector (using a mock SOAP service)
    connector_config = {
        "wsdl_url": "http://www.dneonline.com/calculator.asmx?wsdl",
        "auth_config": {
            "method": AuthenticationMethod.NONE
        }
    }
    
    result = await manager.create_connector(
        APIType.SOAP,
        "http://www.dneonline.com/calculator.asmx",
        connector_config
    )
    
    if result["success"]:
        connector_id = result["connector_id"]
        print(f"Created SOAP connector: {connector_id}")
        
        connector = await manager.get_connector(connector_id)
        
        # Example 1: Load WSDL
        print("\n--- Loading WSDL ---")
        wsdl_result = await connector.load_wsdl()
        if wsdl_result["success"]:
            wsdl_schema = wsdl_result["wsdl_schema"]
            print("WSDL loaded successfully")
            print(f"Available operations: {wsdl_schema['operations']}")
            print(f"Namespaces: {list(wsdl_schema['namespaces'].keys())}")
        else:
            print(f"WSDL loading failed: {wsdl_result['error']}")
            # Continue with mock operations for demonstration
            connector.soap_actions = {
                "Add": {"soap_action": "http://tempuri.org/Add", "operation_name": "Add"},
                "Subtract": {"soap_action": "http://tempuri.org/Subtract", "operation_name": "Subtract"}
            }
        
        # Example 2: Execute SOAP operation
        print("\n--- Executing SOAP Operation ---")
        if "Add" in connector.soap_actions:
            add_params = {"intA": 10, "intB": 5}
            add_result = await connector.execute_soap_operation("Add", add_params)
            
            if add_result["success"]:
                print(f"SOAP Add operation successful")
                print(f"Response time: {add_result['response_time']:.2f}s")
                print(f"Result data: {add_result.get('data', 'No data')}")
            else:
                if "fault" in add_result:
                    fault = add_result["fault"]
                    print(f"SOAP Fault - Code: {fault['code']}, Message: {fault['string']}")
                else:
                    print(f"SOAP operation failed: {add_result['error']}")
        else:
            print("Add operation not available in WSDL")
        
        # Cleanup
        await manager.remove_connector(connector_id)
        print(f"\nRemoved SOAP connector: {connector_id}")
    
    else:
        print(f"Failed to create SOAP connector: {result['error']}")

async def example_advanced_features():
    """Example: Advanced features like rate limiting and circuit breaker"""
    
    print("\n=== Advanced Features Example ===")
    
    # Initialize components
    auth_manager = AuthenticationManager()
    security_manager = SecurityManager()
    performance_monitor = PerformanceMonitor()
    
    manager = APIIntegrationManager(
        auth_manager=auth_manager,
        security_manager=security_manager,
        performance_monitor=performance_monitor
    )
    
    # Create connector with aggressive rate limiting for demonstration
    rate_config = RateLimitConfig(
        requests_per_second=2.0,  # Very low for demonstration
        burst_size=3,
        enable_adaptive=True
    )
    
    circuit_breaker_config = CircuitBreakerConfig(
        failure_threshold=2,  # Low threshold for demonstration
        recovery_timeout=5.0,
        success_threshold=2
    )
    
    connector_config = {
        "rate_limit_config": rate_config,
        "circuit_breaker_config": circuit_breaker_config,
        "auth_config": {"method": AuthenticationMethod.NONE}
    }
    
    result = await manager.create_connector(
        APIType.REST,
        "https://httpbin.org",
        connector_config
    )
    
    if result["success"]:
        connector_id = result["connector_id"]
        print(f"Created connector with advanced features: {connector_id}")
        
        connector = await manager.get_connector(connector_id)
        
        # Example 1: Rate limiting demonstration
        print("\n--- Rate Limiting Demonstration ---")
        endpoint = APIEndpoint(url="/get", method=HTTPMethod.GET)
        
        for i in range(5):
            print(f"Request {i+1}:")
            result = await connector.execute_request(endpoint)
            
            if result["success"]:
                print(f"  Success - Response time: {result['response_time']:.2f}s")
            else:
                print(f"  Failed: {result['error']}")
            
            # Check rate limiter stats
            stats = connector.rate_limiter.get_stats()
            print(f"  Tokens remaining: {stats['current_tokens']:.1f}")
            
            # Small delay between requests
            await asyncio.sleep(0.1)
        
        # Example 2: Circuit breaker demonstration (simulate failures)
        print("\n--- Circuit Breaker Demonstration ---")
        
        # This would normally cause failures, but httpbin.org is reliable
        # So we'll just show the circuit breaker state
        cb_state = connector.circuit_breaker.get_state()
        print(f"Circuit breaker state: {cb_state['state']}")
        print(f"Failure count: {cb_state['failure_count']}")
        
        # Example 3: Aggregated metrics
        print("\n--- Aggregated Metrics ---")
        metrics = await manager.get_aggregated_metrics()
        print(f"Total connectors: {metrics['total_connectors']}")
        print(f"Total requests: {metrics['total_requests']}")
        print(f"Success rate: {metrics['total_successful'] / max(metrics['total_requests'], 1) * 100:.1f}%")
        print(f"Average response time: {metrics['average_response_time']:.2f}s")
        
        # Cleanup
        await manager.remove_connector(connector_id)
        print(f"\nRemoved connector: {connector_id}")
    
    else:
        print(f"Failed to create connector: {result['error']}")

async def example_authentication_methods():
    """Example: Different authentication methods"""
    
    print("\n=== Authentication Methods Example ===")
    
    # Initialize components
    auth_manager = AuthenticationManager()
    security_manager = SecurityManager()
    performance_monitor = PerformanceMonitor()
    
    manager = APIIntegrationManager(
        auth_manager=auth_manager,
        security_manager=security_manager,
        performance_monitor=performance_monitor
    )
    
    # Example authentication configurations
    auth_examples = [
        {
            "name": "API Key Authentication",
            "config": {
                "method": AuthenticationMethod.API_KEY,
                "api_key": "your-api-key-here",
                "key_header": "X-API-Key"
            }
        },
        {
            "name": "Bearer Token Authentication",
            "config": {
                "method": AuthenticationMethod.BEARER_TOKEN,
                "token": "your-bearer-token-here"
            }
        },
        {
            "name": "Basic Authentication",
            "config": {
                "method": AuthenticationMethod.BASIC_AUTH,
                "username": "your-username",
                "password": "your-password"
            }
        },
        {
            "name": "Custom Header Authentication",
            "config": {
                "method": AuthenticationMethod.CUSTOM_HEADER,
                "custom_headers": {
                    "X-Custom-Auth": "custom-auth-value",
                    "X-Client-ID": "client-123"
                }
            }
        }
    ]
    
    for auth_example in auth_examples:
        print(f"\n--- {auth_example['name']} ---")
        
        connector_config = {
            "auth_config": auth_example["config"]
        }
        
        result = await manager.create_connector(
            APIType.REST,
            "https://httpbin.org",
            connector_config
        )
        
        if result["success"]:
            connector_id = result["connector_id"]
            print(f"Created connector with {auth_example['name']}: {connector_id}")
            
            # Test the authentication (httpbin.org will echo back headers)
            connector = await manager.get_connector(connector_id)
            endpoint = APIEndpoint(url="/headers", method=HTTPMethod.GET)
            
            test_result = await connector.execute_request(endpoint)
            if test_result["success"]:
                headers = test_result["data"].get("headers", {})
                print("Request headers sent:")
                for header, value in headers.items():
                    if any(auth_header in header.lower() for auth_header in ["auth", "key", "token", "custom"]):
                        print(f"  {header}: {value}")
            else:
                print(f"Test request failed: {test_result['error']}")
            
            # Cleanup
            await manager.remove_connector(connector_id)
            print(f"Removed connector: {connector_id}")
        
        else:
            print(f"Failed to create connector: {result['error']}")

async def main():
    """Run all examples"""
    
    print("Forge 1 API Integration Layer Examples")
    print("=" * 50)
    
    try:
        # Run examples
        await example_rest_api_integration()
        await example_graphql_api_integration()
        await example_soap_api_integration()
        await example_advanced_features()
        await example_authentication_methods()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nExample execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())