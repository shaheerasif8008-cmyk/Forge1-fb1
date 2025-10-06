# forge1/backend/tests/performance/test_employee_performance.py
"""
Performance tests for Employee Lifecycle System

Tests system performance under various load conditions to validate
scalability requirements and identify performance bottlenecks.
"""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

import httpx

# Performance test configuration
API_BASE_URL = "http://localhost:8000"
PERFORMANCE_TIMEOUT = 60.0


class TestEmployeePerformance:
    """Performance test suite for employee lifecycle system"""
    
    @pytest.fixture
    def api_headers(self):
        """Standard API headers for testing"""
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer perf_test_token",
            "X-Tenant-ID": "perf_test_client",
            "X-Client-ID": "perf_test_client",
            "X-User-ID": "perf_test_user"
        }
    
    @pytest.fixture
    def sample_client_data(self):
        """Sample client data for performance testing"""
        return {
            "name": "Performance Test Corporation",
            "industry": "Technology",
            "tier": "enterprise",
            "max_employees": 100,
            "allowed_models": ["gpt-4", "gpt-3.5-turbo"],
            "security_level": "high",
            "compliance_requirements": ["SOC2"]
        }
    
    @pytest.fixture
    def sample_employee_requirements(self):
        """Sample employee requirements for performance testing"""
        return {
            "role": "Performance Test Agent",
            "industry": "Technology",
            "expertise_areas": ["customer_service"],
            "communication_style": "friendly",
            "tools_needed": ["email", "chat"],
            "knowledge_domains": ["product_docs"],
            "personality_traits": {
                "empathy_level": 0.8,
                "patience_level": 0.9
            },
            "model_preferences": {
                "primary_model": "gpt-3.5-turbo",  # Faster for performance tests
                "temperature": 0.7,
                "max_tokens": 500  # Smaller for faster responses
            }
        }
    
    async def measure_response_time(self, func, *args, **kwargs):
        """Measure response time of an async function"""
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        return result, (end_time - start_time) * 1000  # Return result and time in ms
    
    @pytest.mark.asyncio
    async def test_employee_creation_performance(self, api_headers, sample_client_data, sample_employee_requirements):
        """Test performance of employee creation under load"""
        async with httpx.AsyncClient(timeout=PERFORMANCE_TIMEOUT) as client:
            # Setup: Create client
            client_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients",
                json=sample_client_data,
                headers=api_headers
            )
            
            assert client_response.status_code == 200
            client_id = client_response.json()["id"]
            
            # Performance test: Create multiple employees
            num_employees = 10
            response_times = []
            
            for i in range(num_employees):
                requirements = sample_employee_requirements.copy()
                requirements["role"] = f"Performance Test Agent {i}"
                
                start_time = time.time()
                
                response = await client.post(
                    f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees",
                    json=requirements,
                    headers=api_headers
                )
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                
                assert response.status_code == 200
                response_times.append(response_time)
            
            # Analyze performance
            avg_response_time = statistics.mean(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            
            print(f"\nEmployee Creation Performance:")
            print(f"  Employees created: {num_employees}")
            print(f"  Average response time: {avg_response_time:.2f}ms")
            print(f"  Min response time: {min_response_time:.2f}ms")
            print(f"  Max response time: {max_response_time:.2f}ms")
            print(f"  95th percentile: {p95_response_time:.2f}ms")
            
            # Performance assertions
            assert avg_response_time < 5000  # Average should be under 5 seconds
            assert p95_response_time < 10000  # 95th percentile should be under 10 seconds
    
    @pytest.mark.asyncio
    async def test_concurrent_employee_interactions(self, api_headers, sample_client_data, sample_employee_requirements):
        """Test performance of concurrent employee interactions"""
        async with httpx.AsyncClient(timeout=PERFORMANCE_TIMEOUT) as client:
            # Setup: Create client and employee
            client_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients",
                json=sample_client_data,
                headers=api_headers
            )
            client_id = client_response.json()["id"]
            
            employee_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees",
                json=sample_employee_requirements,
                headers=api_headers
            )
            employee_id = employee_response.json()["id"]
            
            # Performance test: Concurrent interactions
            num_concurrent = 20
            messages = [f"Performance test message {i}" for i in range(num_concurrent)]
            
            async def single_interaction(message_id, message):
                start_time = time.time()
                
                response = await client.post(
                    f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees/{employee_id}/interact",
                    json={
                        "message": message,
                        "session_id": f"perf_session_{message_id}",
                        "include_memory": False  # Disable for performance
                    },
                    headers=api_headers
                )
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                
                return {
                    "message_id": message_id,
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "success": response.status_code == 200
                }
            
            # Execute concurrent interactions
            start_time = time.time()
            
            tasks = [
                single_interaction(i, messages[i]) 
                for i in range(num_concurrent)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = (time.time() - start_time) * 1000
            
            # Analyze results
            successful_results = [r for r in results if isinstance(r, dict) and r["success"]]
            failed_results = [r for r in results if not (isinstance(r, dict) and r.get("success", False))]
            
            if successful_results:
                response_times = [r["response_time"] for r in successful_results]
                avg_response_time = statistics.mean(response_times)
                max_response_time = max(response_times)
                throughput = len(successful_results) / (total_time / 1000)  # requests per second
                
                print(f"\nConcurrent Interactions Performance:")
                print(f"  Concurrent requests: {num_concurrent}")
                print(f"  Successful requests: {len(successful_results)}")
                print(f"  Failed requests: {len(failed_results)}")
                print(f"  Total time: {total_time:.2f}ms")
                print(f"  Average response time: {avg_response_time:.2f}ms")
                print(f"  Max response time: {max_response_time:.2f}ms")
                print(f"  Throughput: {throughput:.2f} requests/second")
                
                # Performance assertions
                assert len(successful_results) >= num_concurrent * 0.9  # 90% success rate
                assert avg_response_time < 10000  # Average under 10 seconds
                assert throughput > 1.0  # At least 1 request per second
            else:
                pytest.fail("No successful concurrent interactions")
    
    @pytest.mark.asyncio
    async def test_memory_search_performance(self, api_headers, sample_client_data, sample_employee_requirements):
        """Test performance of memory search operations"""
        async with httpx.AsyncClient(timeout=PERFORMANCE_TIMEOUT) as client:
            # Setup: Create client and employee with interactions
            client_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients",
                json=sample_client_data,
                headers=api_headers
            )
            client_id = client_response.json()["id"]
            
            employee_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees",
                json=sample_employee_requirements,
                headers=api_headers
            )
            employee_id = employee_response.json()["id"]
            
            # Create interactions for memory
            num_interactions = 50
            for i in range(num_interactions):
                await client.post(
                    f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees/{employee_id}/interact",
                    json={
                        "message": f"Memory test interaction {i} about topic {i % 5}",
                        "session_id": f"memory_session_{i}",
                        "include_memory": True
                    },
                    headers=api_headers
                )
            
            # Performance test: Memory searches
            search_queries = [
                "topic 0", "topic 1", "topic 2", "topic 3", "topic 4",
                "interaction", "memory", "test", "about", "help"
            ]
            
            search_times = []
            
            for query in search_queries:
                start_time = time.time()
                
                response = await client.get(
                    f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees/{employee_id}/memory",
                    params={"query": query, "limit": 10},
                    headers=api_headers
                )
                
                end_time = time.time()
                search_time = (end_time - start_time) * 1000
                
                assert response.status_code == 200
                search_times.append(search_time)
            
            # Analyze performance
            avg_search_time = statistics.mean(search_times)
            max_search_time = max(search_times)
            
            print(f"\nMemory Search Performance:")
            print(f"  Interactions in memory: {num_interactions}")
            print(f"  Search queries: {len(search_queries)}")
            print(f"  Average search time: {avg_search_time:.2f}ms")
            print(f"  Max search time: {max_search_time:.2f}ms")
            
            # Performance assertions
            assert avg_search_time < 1000  # Average under 1 second
            assert max_search_time < 3000  # Max under 3 seconds
    
    @pytest.mark.asyncio
    async def test_analytics_performance(self, api_headers, sample_client_data, sample_employee_requirements):
        """Test performance of analytics operations"""
        async with httpx.AsyncClient(timeout=PERFORMANCE_TIMEOUT) as client:
            # Setup: Create client and employee
            client_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients",
                json=sample_client_data,
                headers=api_headers
            )
            client_id = client_response.json()["id"]
            
            employee_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees",
                json=sample_employee_requirements,
                headers=api_headers
            )
            employee_id = employee_response.json()["id"]
            
            # Create some interactions for analytics
            num_interactions = 20
            for i in range(num_interactions):
                await client.post(
                    f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees/{employee_id}/interact",
                    json={
                        "message": f"Analytics test message {i}",
                        "session_id": f"analytics_session_{i}",
                        "include_memory": True
                    },
                    headers=api_headers
                )
            
            # Performance test: Analytics operations
            analytics_operations = [
                ("employee_metrics", f"/api/v1/analytics/employees/{client_id}/{employee_id}/metrics"),
                ("employee_health", f"/api/v1/analytics/employees/{client_id}/{employee_id}/health"),
                ("client_usage", f"/api/v1/analytics/clients/{client_id}/usage"),
                ("dashboard", f"/api/v1/analytics/clients/{client_id}/dashboard")
            ]
            
            operation_times = {}
            
            for operation_name, endpoint in analytics_operations:
                start_time = time.time()
                
                response = await client.get(
                    f"{API_BASE_URL}{endpoint}",
                    headers=api_headers
                )
                
                end_time = time.time()
                operation_time = (end_time - start_time) * 1000
                
                assert response.status_code == 200
                operation_times[operation_name] = operation_time
            
            print(f"\nAnalytics Performance:")
            for operation, time_ms in operation_times.items():
                print(f"  {operation}: {time_ms:.2f}ms")
            
            # Performance assertions
            for operation, time_ms in operation_times.items():
                assert time_ms < 5000, f"{operation} took too long: {time_ms}ms"
    
    @pytest.mark.asyncio
    async def test_system_scalability(self, api_headers, sample_client_data, sample_employee_requirements):
        """Test system scalability with multiple clients and employees"""
        async with httpx.AsyncClient(timeout=PERFORMANCE_TIMEOUT) as client:
            # Performance test: Multiple clients with multiple employees
            num_clients = 3
            employees_per_client = 5
            
            client_ids = []
            employee_ids = []
            
            # Create multiple clients
            for i in range(num_clients):
                client_data = sample_client_data.copy()
                client_data["name"] = f"Scalability Test Client {i}"
                
                response = await client.post(
                    f"{API_BASE_URL}/api/v1/employees/clients",
                    json=client_data,
                    headers=api_headers
                )
                
                assert response.status_code == 200
                client_ids.append(response.json()["id"])
            
            # Create multiple employees per client
            for client_id in client_ids:
                for j in range(employees_per_client):
                    requirements = sample_employee_requirements.copy()
                    requirements["role"] = f"Scalability Test Employee {j}"
                    
                    response = await client.post(
                        f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees",
                        json=requirements,
                        headers=api_headers
                    )
                    
                    assert response.status_code == 200
                    employee_ids.append((client_id, response.json()["id"]))
            
            # Test concurrent interactions across all employees
            async def test_interaction(client_id, employee_id, message_id):
                start_time = time.time()
                
                response = await client.post(
                    f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees/{employee_id}/interact",
                    json={
                        "message": f"Scalability test message {message_id}",
                        "session_id": f"scale_session_{message_id}",
                        "include_memory": False
                    },
                    headers=api_headers
                )
                
                end_time = time.time()
                
                return {
                    "client_id": client_id,
                    "employee_id": employee_id,
                    "message_id": message_id,
                    "response_time": (end_time - start_time) * 1000,
                    "success": response.status_code == 200
                }
            
            # Execute concurrent interactions across all employees
            tasks = []
            for i, (client_id, employee_id) in enumerate(employee_ids):
                tasks.append(test_interaction(client_id, employee_id, i))
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = (time.time() - start_time) * 1000
            
            # Analyze scalability results
            successful_results = [r for r in results if isinstance(r, dict) and r["success"]]
            
            if successful_results:
                response_times = [r["response_time"] for r in successful_results]
                avg_response_time = statistics.mean(response_times)
                throughput = len(successful_results) / (total_time / 1000)
                
                print(f"\nScalability Test Results:")
                print(f"  Clients: {num_clients}")
                print(f"  Employees per client: {employees_per_client}")
                print(f"  Total employees: {len(employee_ids)}")
                print(f"  Successful interactions: {len(successful_results)}")
                print(f"  Average response time: {avg_response_time:.2f}ms")
                print(f"  Total throughput: {throughput:.2f} requests/second")
                
                # Scalability assertions
                assert len(successful_results) >= len(employee_ids) * 0.8  # 80% success rate
                assert avg_response_time < 15000  # Average under 15 seconds for scalability test
            else:
                pytest.fail("No successful scalability interactions")
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, api_headers):
        """Test cache performance and hit rates"""
        async with httpx.AsyncClient(timeout=PERFORMANCE_TIMEOUT) as client:
            # Test cache status endpoint performance
            cache_times = []
            
            for i in range(10):
                start_time = time.time()
                
                response = await client.get(
                    f"{API_BASE_URL}/api/v1/performance/cache/status",
                    headers=api_headers
                )
                
                end_time = time.time()
                cache_time = (end_time - start_time) * 1000
                
                # Cache might not be available in test environment
                if response.status_code == 200:
                    cache_times.append(cache_time)
            
            if cache_times:
                avg_cache_time = statistics.mean(cache_times)
                
                print(f"\nCache Performance:")
                print(f"  Average cache status time: {avg_cache_time:.2f}ms")
                
                # Cache performance assertions
                assert avg_cache_time < 100  # Cache operations should be very fast
    
    @pytest.mark.asyncio
    async def test_database_performance(self, api_headers):
        """Test database performance"""
        async with httpx.AsyncClient(timeout=PERFORMANCE_TIMEOUT) as client:
            # Test database status endpoint performance
            db_times = []
            
            for i in range(5):
                start_time = time.time()
                
                response = await client.get(
                    f"{API_BASE_URL}/api/v1/performance/database/status",
                    headers=api_headers
                )
                
                end_time = time.time()
                db_time = (end_time - start_time) * 1000
                
                # Database might not be available in test environment
                if response.status_code == 200:
                    db_times.append(db_time)
            
            if db_times:
                avg_db_time = statistics.mean(db_times)
                
                print(f"\nDatabase Performance:")
                print(f"  Average database status time: {avg_db_time:.2f}ms")
                
                # Database performance assertions
                assert avg_db_time < 1000  # Database operations should be under 1 second


if __name__ == "__main__":
    pytest.main([__file__])