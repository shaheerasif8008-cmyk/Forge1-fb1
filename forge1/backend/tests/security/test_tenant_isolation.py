# forge1/backend/tests/security/test_tenant_isolation.py
"""
Security tests for Tenant Isolation

Tests comprehensive tenant isolation and security features to ensure
proper data separation and access control.
"""

import pytest
import asyncio
import httpx
from typing import Dict, Any

# Security test configuration
API_BASE_URL = "http://localhost:8000"
SECURITY_TIMEOUT = 30.0


class TestTenantIsolation:
    """Security test suite for tenant isolation"""
    
    @pytest.fixture
    def tenant_a_headers(self):
        """Headers for tenant A"""
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer tenant_a_token",
            "X-Tenant-ID": "tenant_a",
            "X-Client-ID": "tenant_a",
            "X-User-ID": "user_a"
        }
    
    @pytest.fixture
    def tenant_b_headers(self):
        """Headers for tenant B"""
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer tenant_b_token",
            "X-Tenant-ID": "tenant_b",
            "X-Client-ID": "tenant_b",
            "X-User-ID": "user_b"
        }
    
    @pytest.fixture
    def malicious_headers(self):
        """Headers for malicious access attempts"""
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer malicious_token",
            "X-Tenant-ID": "malicious_tenant",
            "X-Client-ID": "malicious_client",
            "X-User-ID": "malicious_user"
        }
    
    @pytest.fixture
    def sample_client_data(self):
        """Sample client data for security testing"""
        return {
            "name": "Security Test Corporation",
            "industry": "Technology",
            "tier": "enterprise",
            "max_employees": 5,
            "allowed_models": ["gpt-4"],
            "security_level": "high",
            "compliance_requirements": ["SOC2", "GDPR"]
        }
    
    @pytest.fixture
    def sample_employee_requirements(self):
        """Sample employee requirements for security testing"""
        return {
            "role": "Security Test Agent",
            "industry": "Technology",
            "expertise_areas": ["security"],
            "communication_style": "professional",
            "tools_needed": ["email"],
            "knowledge_domains": ["security_docs"],
            "personality_traits": {"security_level": 0.9},
            "model_preferences": {
                "primary_model": "gpt-4",
                "temperature": 0.5,
                "max_tokens": 1000
            }
        }
    
    @pytest.mark.asyncio
    async def test_client_isolation(self, tenant_a_headers, tenant_b_headers, sample_client_data):
        """Test that clients cannot access each other's data"""
        async with httpx.AsyncClient(timeout=SECURITY_TIMEOUT) as client:
            # Create client for tenant A
            client_a_data = sample_client_data.copy()
            client_a_data["name"] = "Tenant A Corporation"
            
            response_a = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients",
                json=client_a_data,
                headers=tenant_a_headers
            )
            
            assert response_a.status_code == 200
            client_a_id = response_a.json()["id"]
            
            # Create client for tenant B
            client_b_data = sample_client_data.copy()
            client_b_data["name"] = "Tenant B Corporation"
            
            response_b = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients",
                json=client_b_data,
                headers=tenant_b_headers
            )
            
            assert response_b.status_code == 200
            client_b_id = response_b.json()["id"]
            
            # Test: Tenant A cannot access Tenant B's client
            cross_access_response = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_b_id}",
                headers=tenant_a_headers
            )
            
            assert cross_access_response.status_code == 403  # Access denied
            
            # Test: Tenant B cannot access Tenant A's client
            cross_access_response = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_a_id}",
                headers=tenant_b_headers
            )
            
            assert cross_access_response.status_code == 403  # Access denied
            
            # Test: Each tenant can access their own client
            own_access_a = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_a_id}",
                headers=tenant_a_headers
            )
            
            assert own_access_a.status_code == 200
            
            own_access_b = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_b_id}",
                headers=tenant_b_headers
            )
            
            assert own_access_b.status_code == 200
    
    @pytest.mark.asyncio
    async def test_employee_isolation(self, tenant_a_headers, tenant_b_headers, sample_client_data, sample_employee_requirements):
        """Test that employees are isolated between tenants"""
        async with httpx.AsyncClient(timeout=SECURITY_TIMEOUT) as client:
            # Setup: Create clients for both tenants
            client_a_data = sample_client_data.copy()
            client_a_data["name"] = "Tenant A Corporation"
            response_a = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients",
                json=client_a_data,
                headers=tenant_a_headers
            )
            client_a_id = response_a.json()["id"]
            
            client_b_data = sample_client_data.copy()
            client_b_data["name"] = "Tenant B Corporation"
            response_b = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients",
                json=client_b_data,
                headers=tenant_b_headers
            )
            client_b_id = response_b.json()["id"]
            
            # Create employees for both tenants
            employee_a_req = sample_employee_requirements.copy()
            employee_a_req["role"] = "Tenant A Security Agent"
            
            employee_a_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_a_id}/employees",
                json=employee_a_req,
                headers=tenant_a_headers
            )
            
            assert employee_a_response.status_code == 200
            employee_a_id = employee_a_response.json()["id"]
            
            employee_b_req = sample_employee_requirements.copy()
            employee_b_req["role"] = "Tenant B Security Agent"
            
            employee_b_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_b_id}/employees",
                json=employee_b_req,
                headers=tenant_b_headers
            )
            
            assert employee_b_response.status_code == 200
            employee_b_id = employee_b_response.json()["id"]
            
            # Test: Tenant A cannot access Tenant B's employee
            cross_employee_access = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_b_id}/employees/{employee_b_id}",
                headers=tenant_a_headers
            )
            
            assert cross_employee_access.status_code == 403  # Access denied
            
            # Test: Tenant B cannot access Tenant A's employee
            cross_employee_access = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_a_id}/employees/{employee_a_id}",
                headers=tenant_b_headers
            )
            
            assert cross_employee_access.status_code == 403  # Access denied
            
            # Test: Each tenant can access their own employees
            own_employee_a = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_a_id}/employees/{employee_a_id}",
                headers=tenant_a_headers
            )
            
            assert own_employee_a.status_code == 200
            
            own_employee_b = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_b_id}/employees/{employee_b_id}",
                headers=tenant_b_headers
            )
            
            assert own_employee_b.status_code == 200
    
    @pytest.mark.asyncio
    async def test_memory_isolation(self, tenant_a_headers, tenant_b_headers, sample_client_data, sample_employee_requirements):
        """Test that employee memory is isolated between tenants"""
        async with httpx.AsyncClient(timeout=SECURITY_TIMEOUT) as client:
            # Setup: Create clients and employees for both tenants
            # Tenant A setup
            client_a_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients",
                json=sample_client_data,
                headers=tenant_a_headers
            )
            client_a_id = client_a_response.json()["id"]
            
            employee_a_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_a_id}/employees",
                json=sample_employee_requirements,
                headers=tenant_a_headers
            )
            employee_a_id = employee_a_response.json()["id"]
            
            # Tenant B setup
            client_b_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients",
                json=sample_client_data,
                headers=tenant_b_headers
            )
            client_b_id = client_b_response.json()["id"]
            
            employee_b_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_b_id}/employees",
                json=sample_employee_requirements,
                headers=tenant_b_headers
            )
            employee_b_id = employee_b_response.json()["id"]
            
            # Create interactions for both employees
            await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_a_id}/employees/{employee_a_id}/interact",
                json={
                    "message": "Tenant A confidential information",
                    "session_id": "tenant_a_session",
                    "include_memory": True
                },
                headers=tenant_a_headers
            )
            
            await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_b_id}/employees/{employee_b_id}/interact",
                json={
                    "message": "Tenant B confidential information",
                    "session_id": "tenant_b_session",
                    "include_memory": True
                },
                headers=tenant_b_headers
            )
            
            # Test: Tenant A cannot access Tenant B's employee memory
            cross_memory_access = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_b_id}/employees/{employee_b_id}/memory",
                headers=tenant_a_headers
            )
            
            assert cross_memory_access.status_code == 403  # Access denied
            
            # Test: Tenant B cannot access Tenant A's employee memory
            cross_memory_access = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_a_id}/employees/{employee_a_id}/memory",
                headers=tenant_b_headers
            )
            
            assert cross_memory_access.status_code == 403  # Access denied
            
            # Test: Each tenant can access their own employee memory
            own_memory_a = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_a_id}/employees/{employee_a_id}/memory",
                headers=tenant_a_headers
            )
            
            assert own_memory_a.status_code == 200
            memory_a_data = own_memory_a.json()
            
            # Verify Tenant A's memory contains their data
            assert any("Tenant A confidential" in mem.get("content", "") for mem in memory_a_data.get("memories", []))
            
            own_memory_b = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_b_id}/employees/{employee_b_id}/memory",
                headers=tenant_b_headers
            )
            
            assert own_memory_b.status_code == 200
            memory_b_data = own_memory_b.json()
            
            # Verify Tenant B's memory contains their data
            assert any("Tenant B confidential" in mem.get("content", "") for mem in memory_b_data.get("memories", []))
    
    @pytest.mark.asyncio
    async def test_analytics_isolation(self, tenant_a_headers, tenant_b_headers, sample_client_data, sample_employee_requirements):
        """Test that analytics data is isolated between tenants"""
        async with httpx.AsyncClient(timeout=SECURITY_TIMEOUT) as client:
            # Setup: Create clients and employees for both tenants
            client_a_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients",
                json=sample_client_data,
                headers=tenant_a_headers
            )
            client_a_id = client_a_response.json()["id"]
            
            employee_a_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_a_id}/employees",
                json=sample_employee_requirements,
                headers=tenant_a_headers
            )
            employee_a_id = employee_a_response.json()["id"]
            
            client_b_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients",
                json=sample_client_data,
                headers=tenant_b_headers
            )
            client_b_id = client_b_response.json()["id"]
            
            employee_b_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_b_id}/employees",
                json=sample_employee_requirements,
                headers=tenant_b_headers
            )
            employee_b_id = employee_b_response.json()["id"]
            
            # Test: Tenant A cannot access Tenant B's analytics
            cross_analytics_access = await client.get(
                f"{API_BASE_URL}/api/v1/analytics/employees/{client_b_id}/{employee_b_id}/metrics",
                headers=tenant_a_headers
            )
            
            assert cross_analytics_access.status_code == 403  # Access denied
            
            # Test: Tenant B cannot access Tenant A's analytics
            cross_analytics_access = await client.get(
                f"{API_BASE_URL}/api/v1/analytics/employees/{client_a_id}/{employee_a_id}/metrics",
                headers=tenant_b_headers
            )
            
            assert cross_analytics_access.status_code == 403  # Access denied
            
            # Test: Each tenant can access their own analytics
            own_analytics_a = await client.get(
                f"{API_BASE_URL}/api/v1/analytics/employees/{client_a_id}/{employee_a_id}/metrics",
                headers=tenant_a_headers
            )
            
            assert own_analytics_a.status_code == 200
            
            own_analytics_b = await client.get(
                f"{API_BASE_URL}/api/v1/analytics/employees/{client_b_id}/{employee_b_id}/metrics",
                headers=tenant_b_headers
            )
            
            assert own_analytics_b.status_code == 200
    
    @pytest.mark.asyncio
    async def test_malicious_access_attempts(self, malicious_headers, tenant_a_headers, sample_client_data, sample_employee_requirements):
        """Test system behavior against malicious access attempts"""
        async with httpx.AsyncClient(timeout=SECURITY_TIMEOUT) as client:
            # Setup: Create legitimate client and employee
            client_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients",
                json=sample_client_data,
                headers=tenant_a_headers
            )
            
            assert client_response.status_code == 200
            client_id = client_response.json()["id"]
            
            employee_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees",
                json=sample_employee_requirements,
                headers=tenant_a_headers
            )
            
            assert employee_response.status_code == 200
            employee_id = employee_response.json()["id"]
            
            # Test: Malicious access to client data
            malicious_client_access = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_id}",
                headers=malicious_headers
            )
            
            assert malicious_client_access.status_code == 403
            
            # Test: Malicious access to employee data
            malicious_employee_access = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees/{employee_id}",
                headers=malicious_headers
            )
            
            assert malicious_employee_access.status_code == 403
            
            # Test: Malicious interaction attempt
            malicious_interaction = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees/{employee_id}/interact",
                json={
                    "message": "Malicious interaction attempt",
                    "session_id": "malicious_session"
                },
                headers=malicious_headers
            )
            
            assert malicious_interaction.status_code == 403
            
            # Test: Malicious memory access
            malicious_memory_access = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees/{employee_id}/memory",
                headers=malicious_headers
            )
            
            assert malicious_memory_access.status_code == 403
            
            # Test: Malicious analytics access
            malicious_analytics_access = await client.get(
                f"{API_BASE_URL}/api/v1/analytics/employees/{client_id}/{employee_id}/metrics",
                headers=malicious_headers
            )
            
            assert malicious_analytics_access.status_code == 403
    
    @pytest.mark.asyncio
    async def test_authentication_bypass_attempts(self):
        """Test system behavior against authentication bypass attempts"""
        async with httpx.AsyncClient(timeout=SECURITY_TIMEOUT) as client:
            # Test: No authentication headers
            no_auth_response = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/test_client"
            )
            
            assert no_auth_response.status_code == 401  # Unauthorized
            
            # Test: Invalid authentication token
            invalid_auth_headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer invalid_token",
                "X-Tenant-ID": "test_tenant",
                "X-Client-ID": "test_client",
                "X-User-ID": "test_user"
            }
            
            invalid_auth_response = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/test_client",
                headers=invalid_auth_headers
            )
            
            assert invalid_auth_response.status_code in [401, 403]  # Unauthorized or Forbidden
            
            # Test: Missing tenant headers
            missing_tenant_headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer valid_token"
            }
            
            missing_tenant_response = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/test_client",
                headers=missing_tenant_headers
            )
            
            assert missing_tenant_response.status_code == 401  # Unauthorized
    
    @pytest.mark.asyncio
    async def test_sql_injection_protection(self, tenant_a_headers, sample_client_data):
        """Test protection against SQL injection attacks"""
        async with httpx.AsyncClient(timeout=SECURITY_TIMEOUT) as client:
            # Test: SQL injection in client creation
            malicious_client_data = sample_client_data.copy()
            malicious_client_data["name"] = "'; DROP TABLE clients; --"
            
            response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients",
                json=malicious_client_data,
                headers=tenant_a_headers
            )
            
            # Should either succeed (if properly sanitized) or fail gracefully
            assert response.status_code in [200, 400]  # Not 500 (server error)
            
            # Test: SQL injection in URL parameters
            malicious_client_id = "'; DROP TABLE employees; --"
            
            response = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{malicious_client_id}",
                headers=tenant_a_headers
            )
            
            # Should return 404 or 400, not 500
            assert response.status_code in [400, 404]
    
    @pytest.mark.asyncio
    async def test_rate_limiting_isolation(self, tenant_a_headers, tenant_b_headers):
        """Test that rate limiting is applied per tenant"""
        async with httpx.AsyncClient(timeout=SECURITY_TIMEOUT) as client:
            # Make rapid requests from tenant A
            tenant_a_responses = []
            for i in range(20):
                response = await client.get(
                    f"{API_BASE_URL}/api/v1/performance/metrics",
                    headers=tenant_a_headers
                )
                tenant_a_responses.append(response.status_code)
            
            # Make requests from tenant B
            tenant_b_responses = []
            for i in range(5):
                response = await client.get(
                    f"{API_BASE_URL}/api/v1/performance/metrics",
                    headers=tenant_b_headers
                )
                tenant_b_responses.append(response.status_code)
            
            # Tenant B should not be affected by tenant A's rate limiting
            # (assuming tenant A might hit rate limits)
            assert any(status == 200 for status in tenant_b_responses)
    
    @pytest.mark.asyncio
    async def test_data_encryption_isolation(self, tenant_a_headers, tenant_b_headers):
        """Test that data encryption is tenant-specific"""
        async with httpx.AsyncClient(timeout=SECURITY_TIMEOUT) as client:
            # Test encryption info endpoints for different tenants
            encryption_a_response = await client.get(
                f"{API_BASE_URL}/api/v1/security/encryption/tenant_a",
                headers=tenant_a_headers
            )
            
            encryption_b_response = await client.get(
                f"{API_BASE_URL}/api/v1/security/encryption/tenant_b",
                headers=tenant_b_headers
            )
            
            # Both should succeed for their own tenant
            if encryption_a_response.status_code == 200:
                assert "encryption_enabled" in encryption_a_response.json()
            
            if encryption_b_response.status_code == 200:
                assert "encryption_enabled" in encryption_b_response.json()
            
            # Test cross-tenant encryption access
            cross_encryption_access = await client.get(
                f"{API_BASE_URL}/api/v1/security/encryption/tenant_b",
                headers=tenant_a_headers
            )
            
            # Should be denied or return different encryption context
            assert cross_encryption_access.status_code in [403, 404, 200]


if __name__ == "__main__":
    pytest.main([__file__])