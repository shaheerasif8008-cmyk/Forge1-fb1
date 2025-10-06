# forge1/backend/tests/e2e/test_employee_lifecycle_e2e.py
"""
End-to-End tests for Employee Lifecycle System

Tests the complete employee lifecycle from client onboarding through
employee interactions via the API endpoints.
"""

import pytest
import asyncio
import httpx
import json
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30.0


class TestEmployeeLifecycleE2E:
    """End-to-end test suite for employee lifecycle via API"""
    
    @pytest.fixture
    def api_headers(self):
        """Standard API headers for testing"""
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer test_token",
            "X-Tenant-ID": "e2e_test_client",
            "X-Client-ID": "e2e_test_client",
            "X-User-ID": "e2e_test_user"
        }
    
    @pytest.fixture
    def sample_client_data(self):
        """Sample client data for testing"""
        return {
            "name": "E2E Test Corporation",
            "industry": "Technology",
            "tier": "enterprise",
            "max_employees": 5,
            "allowed_models": ["gpt-4", "gpt-3.5-turbo"],
            "security_level": "high",
            "compliance_requirements": ["SOC2", "GDPR"]
        }
    
    @pytest.fixture
    def sample_employee_requirements(self):
        """Sample employee requirements for testing"""
        return {
            "role": "E2E Test Support Agent",
            "industry": "Technology",
            "expertise_areas": ["customer_service", "technical_support"],
            "communication_style": "friendly",
            "tools_needed": ["email", "chat", "knowledge_base"],
            "knowledge_domains": ["product_docs", "troubleshooting"],
            "personality_traits": {
                "empathy_level": 0.9,
                "patience_level": 0.95,
                "technical_depth": 0.7
            },
            "model_preferences": {
                "primary_model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1500
            }
        }
    
    @pytest.mark.asyncio
    async def test_complete_client_onboarding_e2e(self, api_headers, sample_client_data):
        """Test complete client onboarding via API"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Step 1: Onboard client
            response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients",
                json=sample_client_data,
                headers=api_headers
            )
            
            assert response.status_code == 200
            client_response = response.json()
            
            assert "id" in client_response
            assert client_response["name"] == "E2E Test Corporation"
            assert client_response["status"] == "active"
            
            client_id = client_response["id"]
            
            # Step 2: Retrieve client information
            get_response = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_id}",
                headers=api_headers
            )
            
            assert get_response.status_code == 200
            retrieved_client = get_response.json()
            
            assert retrieved_client["id"] == client_id
            assert retrieved_client["name"] == "E2E Test Corporation"
            
            return client_id
    
    @pytest.mark.asyncio
    async def test_complete_employee_creation_e2e(self, api_headers, sample_client_data, sample_employee_requirements):
        """Test complete employee creation via API"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Step 1: Create client first
            client_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients",
                json=sample_client_data,
                headers=api_headers
            )
            
            assert client_response.status_code == 200
            client_id = client_response.json()["id"]
            
            # Step 2: Create employee
            employee_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees",
                json=sample_employee_requirements,
                headers=api_headers
            )
            
            assert employee_response.status_code == 200
            employee_data = employee_response.json()
            
            assert "id" in employee_data
            assert employee_data["name"] is not None
            assert employee_data["role"] == "E2E Test Support Agent"
            assert employee_data["status"] == "active"
            
            employee_id = employee_data["id"]
            
            # Step 3: Retrieve employee information
            get_response = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees/{employee_id}",
                headers=api_headers
            )
            
            assert get_response.status_code == 200
            retrieved_employee = get_response.json()
            
            assert retrieved_employee["id"] == employee_id
            assert retrieved_employee["role"] == "E2E Test Support Agent"
            
            return client_id, employee_id
    
    @pytest.mark.asyncio
    async def test_employee_interaction_e2e(self, api_headers, sample_client_data, sample_employee_requirements):
        """Test complete employee interaction flow via API"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
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
            
            # Step 1: Interact with employee
            interaction_data = {
                "message": "Hello, I need help with my account setup",
                "session_id": "e2e_test_session",
                "context": {
                    "user_type": "new_customer",
                    "urgency": "normal"
                },
                "include_memory": True,
                "memory_limit": 10
            }
            
            interaction_response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees/{employee_id}/interact",
                json=interaction_data,
                headers=api_headers
            )
            
            assert interaction_response.status_code == 200
            response_data = interaction_response.json()
            
            assert "message" in response_data
            assert response_data["employee_id"] == employee_id
            assert "interaction_id" in response_data
            assert response_data["processing_time_ms"] >= 0
            
            interaction_id = response_data["interaction_id"]
            
            # Step 2: Verify memory was stored
            memory_response = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees/{employee_id}/memory?limit=5",
                headers=api_headers
            )
            
            assert memory_response.status_code == 200
            memory_data = memory_response.json()
            
            assert memory_data["total_count"] > 0
            assert len(memory_data["memories"]) > 0
            
            # Step 3: Test memory search
            search_response = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees/{employee_id}/memory?query=account&limit=5",
                headers=api_headers
            )
            
            assert search_response.status_code == 200
            search_data = search_response.json()
            
            assert len(search_data["memories"]) > 0
            
            return client_id, employee_id, interaction_id
    
    @pytest.mark.asyncio
    async def test_employee_configuration_management_e2e(self, api_headers, sample_client_data, sample_employee_requirements):
        """Test employee configuration management via API"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
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
            
            # Step 1: Update personality
            personality_update = {
                "communication_style": "professional",
                "creativity_level": 0.5,
                "empathy_level": 0.8
            }
            
            personality_response = await client.put(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees/{employee_id}/personality",
                json=personality_update,
                headers=api_headers
            )
            
            assert personality_response.status_code == 200
            personality_data = personality_response.json()
            
            assert personality_data["personality"]["communication_style"] == "professional"
            
            # Step 2: Update model preferences
            model_update = {
                "primary_model": "gpt-3.5-turbo",
                "temperature": 0.5,
                "max_tokens": 1000
            }
            
            model_response = await client.put(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees/{employee_id}/model-preferences",
                json=model_update,
                headers=api_headers
            )
            
            assert model_response.status_code == 200
            model_data = model_response.json()
            
            assert model_data["model_preferences"]["primary_model"] == "gpt-3.5-turbo"
            
            # Step 3: Update tool access
            tool_update = {
                "tools": ["email", "chat", "calendar", "crm"]
            }
            
            tool_response = await client.put(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees/{employee_id}/tools",
                json=tool_update,
                headers=api_headers
            )
            
            assert tool_response.status_code == 200
            tool_data = tool_response.json()
            
            assert set(tool_data["tool_access"]) == set(["email", "chat", "calendar", "crm"])
            
            # Step 4: Get complete configuration
            config_response = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees/{employee_id}/configuration",
                headers=api_headers
            )
            
            assert config_response.status_code == 200
            config_data = config_response.json()
            
            assert config_data["configuration"]["personality"]["communication_style"] == "professional"
            assert config_data["configuration"]["model_preferences"]["primary_model"] == "gpt-3.5-turbo"
            assert set(config_data["configuration"]["tools"]["available_tools"]) == set(["email", "chat", "calendar", "crm"])
    
    @pytest.mark.asyncio
    async def test_analytics_and_monitoring_e2e(self, api_headers, sample_client_data, sample_employee_requirements):
        """Test analytics and monitoring via API"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
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
            
            # Create some interactions for analytics
            messages = [
                "Hello, I need help with billing",
                "Can you explain your pricing plans?",
                "How do I upgrade my account?"
            ]
            
            interaction_ids = []
            for message in messages:
                interaction_response = await client.post(
                    f"{API_BASE_URL}/api/v1/employees/clients/{client_id}/employees/{employee_id}/interact",
                    json={
                        "message": message,
                        "session_id": "analytics_test_session",
                        "include_memory": True
                    },
                    headers=api_headers
                )
                
                if interaction_response.status_code == 200:
                    interaction_ids.append(interaction_response.json()["interaction_id"])
            
            # Step 1: Test employee metrics
            metrics_response = await client.get(
                f"{API_BASE_URL}/api/v1/analytics/employees/{client_id}/{employee_id}/metrics?days=1",
                headers=api_headers
            )
            
            assert metrics_response.status_code == 200
            metrics_data = metrics_response.json()
            
            assert metrics_data["employee_id"] == employee_id
            assert "metrics" in metrics_data
            assert metrics_data["metrics"]["total_interactions"] >= len(messages)
            
            # Step 2: Test employee health
            health_response = await client.get(
                f"{API_BASE_URL}/api/v1/analytics/employees/{client_id}/{employee_id}/health",
                headers=api_headers
            )
            
            assert health_response.status_code == 200
            health_data = health_response.json()
            
            assert health_data["employee_id"] == employee_id
            assert "overall_health_score" in health_data
            assert "health_status" in health_data
            
            # Step 3: Test client usage analytics
            usage_response = await client.get(
                f"{API_BASE_URL}/api/v1/analytics/clients/{client_id}/usage?days=1",
                headers=api_headers
            )
            
            assert usage_response.status_code == 200
            usage_data = usage_response.json()
            
            assert usage_data["client_id"] == client_id
            assert usage_data["summary"]["total_interactions"] >= len(messages)
            
            # Step 4: Test dashboard
            dashboard_response = await client.get(
                f"{API_BASE_URL}/api/v1/analytics/clients/{client_id}/dashboard?days=1",
                headers=api_headers
            )
            
            assert dashboard_response.status_code == 200
            dashboard_data = dashboard_response.json()
            
            assert "client_metrics" in dashboard_data
            assert "employee_health_summary" in dashboard_data
            
            # Step 5: Test feedback recording
            if interaction_ids:
                feedback_data = {
                    "interaction_id": interaction_ids[0],
                    "rating": 5,
                    "feedback_text": "Excellent service!"
                }
                
                feedback_response = await client.post(
                    f"{API_BASE_URL}/api/v1/analytics/employees/{client_id}/{employee_id}/feedback",
                    json=feedback_data,
                    headers=api_headers
                )
                
                assert feedback_response.status_code == 200
                feedback_result = feedback_response.json()
                
                assert feedback_result["rating"] == 5
    
    @pytest.mark.asyncio
    async def test_system_integration_e2e(self, api_headers, sample_client_data, sample_employee_requirements):
        """Test system integration endpoints via API"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Step 1: Check integration status
            status_response = await client.get(
                f"{API_BASE_URL}/api/v1/integration/status",
                headers=api_headers
            )
            
            assert status_response.status_code == 200
            status_data = status_response.json()
            
            assert "initialized" in status_data
            assert "components" in status_data
            
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
            
            # Step 2: Test integrated agent creation
            agent_response = await client.post(
                f"{API_BASE_URL}/api/v1/integration/agents/{client_id}/{employee_id}",
                json={"session_context": {"test_mode": True}},
                headers=api_headers
            )
            
            assert agent_response.status_code == 200
            agent_data = agent_response.json()
            
            assert agent_data["agent_created"] is True
            assert agent_data["employee_id"] == employee_id
            assert "capabilities" in agent_data
            
            # Step 3: Test integrated agent interaction
            interaction_response = await client.post(
                f"{API_BASE_URL}/api/v1/integration/agents/{client_id}/{employee_id}/interact",
                json={
                    "message": "Test integrated interaction",
                    "context": {"integration_test": True},
                    "session_context": {"test_mode": True}
                },
                headers=api_headers
            )
            
            assert interaction_response.status_code == 200
            interaction_data = interaction_response.json()
            
            assert "response" in interaction_data
            assert interaction_data["employee_id"] == employee_id
            assert "processing_time_ms" in interaction_data
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_e2e(self, api_headers):
        """Test performance monitoring endpoints via API"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Step 1: Test performance metrics
            metrics_response = await client.get(
                f"{API_BASE_URL}/api/v1/performance/metrics",
                headers=api_headers
            )
            
            assert metrics_response.status_code == 200
            metrics_data = metrics_response.json()
            
            assert "metrics" in metrics_data
            assert "timestamp" in metrics_data
            
            # Step 2: Test system health
            health_response = await client.get(
                f"{API_BASE_URL}/api/v1/performance/health",
                headers=api_headers
            )
            
            assert health_response.status_code == 200
            health_data = health_response.json()
            
            assert "overall_score" in health_data
            assert "health_status" in health_data
            
            # Step 3: Test cache status
            cache_response = await client.get(
                f"{API_BASE_URL}/api/v1/performance/cache/status",
                headers=api_headers
            )
            
            # Cache might not be available in test environment
            assert cache_response.status_code in [200, 503]
            
            # Step 4: Test database status
            db_response = await client.get(
                f"{API_BASE_URL}/api/v1/performance/database/status",
                headers=api_headers
            )
            
            # Database might not be available in test environment
            assert db_response.status_code in [200, 503]
    
    @pytest.mark.asyncio
    async def test_error_handling_e2e(self, api_headers):
        """Test API error handling"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Test 404 errors
            response = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/nonexistent_client",
                headers=api_headers
            )
            assert response.status_code == 404
            
            # Test 400 errors (bad request)
            response = await client.post(
                f"{API_BASE_URL}/api/v1/employees/clients",
                json={"invalid": "data"},
                headers=api_headers
            )
            assert response.status_code == 400
            
            # Test 403 errors (access denied)
            wrong_headers = api_headers.copy()
            wrong_headers["X-Client-ID"] = "unauthorized_client"
            
            response = await client.get(
                f"{API_BASE_URL}/api/v1/employees/clients/some_client",
                headers=wrong_headers
            )
            assert response.status_code == 403


if __name__ == "__main__":
    pytest.main([__file__])