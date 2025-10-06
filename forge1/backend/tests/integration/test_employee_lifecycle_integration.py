# forge1/backend/tests/integration/test_employee_lifecycle_integration.py
"""
Integration tests for Employee Lifecycle System

Tests the complete integration between all employee lifecycle components
including managers, memory systems, and API endpoints.
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any

from forge1.services.employee_manager import EmployeeManager
from forge1.services.client_manager import ClientManager
from forge1.services.employee_memory_manager import EmployeeMemoryManager
from forge1.services.employee_analytics_service import EmployeeAnalyticsService
from forge1.integrations.forge1_system_adapter import Forge1SystemIntegrator
from forge1.models.employee_models import (
    EmployeeRequirements, ClientInfo, EmployeeStatus, MemoryType
)


class TestEmployeeLifecycleIntegration:
    """Integration test suite for complete employee lifecycle"""
    
    @pytest.fixture
    async def integrated_system(self):
        """Set up integrated employee lifecycle system"""
        # Initialize all components
        employee_manager = EmployeeManager()
        client_manager = ClientManager()
        memory_manager = EmployeeMemoryManager()
        analytics_service = EmployeeAnalyticsService()
        system_integrator = Forge1SystemIntegrator()
        
        # Mock initialization for testing
        await employee_manager.initialize()
        await client_manager.initialize()
        await memory_manager.initialize()
        await analytics_service.initialize()
        await system_integrator.initialize(employee_manager, memory_manager)
        
        return {
            "employee_manager": employee_manager,
            "client_manager": client_manager,
            "memory_manager": memory_manager,
            "analytics_service": analytics_service,
            "system_integrator": system_integrator
        }
    
    @pytest.fixture
    def sample_client_info(self):
        """Sample client information for testing"""
        return ClientInfo(
            name="Test Corporation",
            industry="Technology",
            tier="enterprise",
            max_employees=10,
            allowed_models=["gpt-4", "gpt-3.5-turbo"],
            security_level="high",
            compliance_requirements=["SOC2", "GDPR"]
        )
    
    @pytest.fixture
    def sample_employee_requirements(self):
        """Sample employee requirements for testing"""
        return EmployeeRequirements(
            role="Customer Support Specialist",
            industry="Technology",
            expertise_areas=["customer_service", "technical_support"],
            communication_style="friendly",
            tools_needed=["email", "chat", "knowledge_base"],
            knowledge_domains=["product_docs", "troubleshooting"],
            personality_traits={
                "empathy_level": 0.9,
                "patience_level": 0.95,
                "technical_depth": 0.7
            },
            model_preferences={
                "primary_model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1500
            }
        )
    
    @pytest.mark.asyncio
    async def test_complete_client_onboarding_flow(self, integrated_system, sample_client_info):
        """Test complete client onboarding process"""
        client_manager = integrated_system["client_manager"]
        
        # Step 1: Onboard client
        client = await client_manager.onboard_client(sample_client_info)
        
        assert client is not None
        assert client.name == "Test Corporation"
        assert client.status.value == "active"
        assert client.configuration.max_employees == 10
        
        # Step 2: Verify client can be retrieved
        retrieved_client = await client_manager.get_client(client.id)
        assert retrieved_client is not None
        assert retrieved_client.id == client.id
    
    @pytest.mark.asyncio
    async def test_complete_employee_creation_flow(self, integrated_system, sample_client_info, sample_employee_requirements):
        """Test complete employee creation and setup process"""
        client_manager = integrated_system["client_manager"]
        employee_manager = integrated_system["employee_manager"]
        memory_manager = integrated_system["memory_manager"]
        
        # Step 1: Create client
        client = await client_manager.onboard_client(sample_client_info)
        
        # Step 2: Create employee
        employee = await employee_manager.create_employee(client.id, sample_employee_requirements)
        
        assert employee is not None
        assert employee.client_id == client.id
        assert employee.role == "Customer Support Specialist"
        assert employee.status == EmployeeStatus.ACTIVE
        
        # Step 3: Verify employee memory namespace was initialized
        # This would check that the memory system is properly set up
        memory_context = await memory_manager.get_employee_context(
            client.id, employee.id, limit=1
        )
        assert isinstance(memory_context, list)  # Should return empty list initially
        
        # Step 4: Verify employee can be loaded
        loaded_employee = await employee_manager.load_employee(client.id, employee.id)
        assert loaded_employee is not None
        assert loaded_employee.id == employee.id
    
    @pytest.mark.asyncio
    async def test_employee_interaction_flow(self, integrated_system, sample_client_info, sample_employee_requirements):
        """Test complete employee interaction process"""
        client_manager = integrated_system["client_manager"]
        employee_manager = integrated_system["employee_manager"]
        memory_manager = integrated_system["memory_manager"]
        system_integrator = integrated_system["system_integrator"]
        
        # Setup: Create client and employee
        client = await client_manager.onboard_client(sample_client_info)
        employee = await employee_manager.create_employee(client.id, sample_employee_requirements)
        
        # Step 1: Create integrated agent
        agent = await system_integrator.create_employee_agent(
            client.id, employee.id, {"session_id": "test_session"}
        )
        
        assert agent is not None
        assert agent.employee.id == employee.id
        
        # Step 2: Process interaction
        test_message = "Hello, I need help with setting up my account"
        response = await agent.process_message(test_message)
        
        assert response is not None
        assert response.employee_id == employee.id
        assert len(response.message) > 0
        assert response.processing_time_ms >= 0
        
        # Step 3: Verify interaction was stored in memory
        memory_context = await memory_manager.get_employee_context(
            client.id, employee.id, limit=5
        )
        assert len(memory_context) > 0
        
        # Step 4: Test memory search
        search_results = await memory_manager.search_employee_memory(
            client.id, employee.id, "account setup", limit=5
        )
        assert len(search_results) > 0
    
    @pytest.mark.asyncio
    async def test_multi_employee_isolation(self, integrated_system, sample_client_info, sample_employee_requirements):
        """Test that multiple employees have isolated memory and interactions"""
        client_manager = integrated_system["client_manager"]
        employee_manager = integrated_system["employee_manager"]
        memory_manager = integrated_system["memory_manager"]
        system_integrator = integrated_system["system_integrator"]
        
        # Setup: Create client and two employees
        client = await client_manager.onboard_client(sample_client_info)
        
        # Create first employee
        employee1 = await employee_manager.create_employee(client.id, sample_employee_requirements)
        
        # Create second employee with different role
        requirements2 = sample_employee_requirements
        requirements2.role = "Sales Assistant"
        employee2 = await employee_manager.create_employee(client.id, requirements2)
        
        # Create agents for both employees
        agent1 = await system_integrator.create_employee_agent(
            client.id, employee1.id, {"session_id": "session1"}
        )
        agent2 = await system_integrator.create_employee_agent(
            client.id, employee2.id, {"session_id": "session2"}
        )
        
        # Have different interactions with each employee
        response1 = await agent1.process_message("I need technical support")
        response2 = await agent2.process_message("I want to buy your product")
        
        # Verify responses are different and appropriate
        assert response1.employee_id == employee1.id
        assert response2.employee_id == employee2.id
        assert response1.message != response2.message
        
        # Verify memory isolation
        memory1 = await memory_manager.get_employee_context(client.id, employee1.id)
        memory2 = await memory_manager.get_employee_context(client.id, employee2.id)
        
        # Each employee should only have their own interactions
        assert len(memory1) == 1
        assert len(memory2) == 1
        assert memory1[0].content != memory2[0].content
    
    @pytest.mark.asyncio
    async def test_employee_configuration_updates(self, integrated_system, sample_client_info, sample_employee_requirements):
        """Test employee configuration updates and their effects"""
        client_manager = integrated_system["client_manager"]
        employee_manager = integrated_system["employee_manager"]
        
        # Setup: Create client and employee
        client = await client_manager.onboard_client(sample_client_info)
        employee = await employee_manager.create_employee(client.id, sample_employee_requirements)
        
        # Test personality update
        personality_updates = {
            "communication_style": "professional",
            "creativity_level": 0.5,
            "empathy_level": 0.6
        }
        
        updated_employee = await employee_manager.update_employee_personality(
            client.id, employee.id, personality_updates
        )
        
        assert updated_employee.personality.communication_style.value == "professional"
        assert updated_employee.personality.creativity_level == 0.5
        
        # Test model preferences update
        model_updates = {
            "primary_model": "gpt-3.5-turbo",
            "temperature": 0.5,
            "max_tokens": 1000
        }
        
        updated_employee = await employee_manager.update_employee_model_preferences(
            client.id, employee.id, model_updates
        )
        
        assert updated_employee.model_preferences.primary_model == "gpt-3.5-turbo"
        assert updated_employee.model_preferences.temperature == 0.5
        
        # Test tool access update
        new_tools = ["email", "chat", "calendar", "crm"]
        updated_employee = await employee_manager.update_employee_tool_access(
            client.id, employee.id, new_tools
        )
        
        assert set(updated_employee.tool_access) == set(new_tools)
    
    @pytest.mark.asyncio
    async def test_analytics_integration(self, integrated_system, sample_client_info, sample_employee_requirements):
        """Test analytics system integration with employee lifecycle"""
        client_manager = integrated_system["client_manager"]
        employee_manager = integrated_system["employee_manager"]
        analytics_service = integrated_system["analytics_service"]
        system_integrator = integrated_system["system_integrator"]
        
        # Setup: Create client and employee
        client = await client_manager.onboard_client(sample_client_info)
        employee = await employee_manager.create_employee(client.id, sample_employee_requirements)
        
        # Create some interactions
        agent = await system_integrator.create_employee_agent(
            client.id, employee.id, {"session_id": "analytics_test"}
        )
        
        # Process multiple interactions
        messages = [
            "Hello, I need help",
            "Can you explain your pricing?",
            "How do I reset my password?"
        ]
        
        for message in messages:
            await agent.process_message(message)
        
        # Test employee metrics
        metrics = await analytics_service.get_employee_metrics(client.id, employee.id, days=1)
        
        assert metrics.employee_id == employee.id
        assert metrics.total_interactions >= 3
        assert metrics.avg_response_time_ms > 0
        
        # Test client usage metrics
        usage_metrics = await analytics_service.get_client_usage_metrics(client.id, days=1)
        
        assert usage_metrics.client_id == client.id
        assert usage_metrics.total_interactions >= 3
        assert usage_metrics.active_employees >= 1
        
        # Test health monitoring
        health_status = await analytics_service.get_employee_health_status(client.id, employee.id)
        
        assert health_status["employee_id"] == employee.id
        assert "overall_health_score" in health_status
        assert "health_status" in health_status
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, integrated_system, sample_client_info):
        """Test system behavior under error conditions"""
        employee_manager = integrated_system["employee_manager"]
        
        # Test handling of invalid employee creation
        invalid_requirements = EmployeeRequirements(
            role="",  # Invalid empty role
            industry="Technology",
            expertise_areas=[],
            communication_style="invalid_style",  # Invalid style
            tools_needed=[],
            knowledge_domains=[],
            personality_traits={}
        )
        
        with pytest.raises(Exception):  # Should raise validation error
            await employee_manager.create_employee("nonexistent_client", invalid_requirements)
        
        # Test handling of non-existent employee operations
        with pytest.raises(Exception):
            await employee_manager.get_employee("client_001", "nonexistent_employee")
        
        # Test handling of invalid updates
        with pytest.raises(Exception):
            await employee_manager.update_employee_personality(
                "client_001", "nonexistent_employee", {"invalid_field": "value"}
            )
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, integrated_system, sample_client_info, sample_employee_requirements):
        """Test system behavior under concurrent operations"""
        client_manager = integrated_system["client_manager"]
        employee_manager = integrated_system["employee_manager"]
        system_integrator = integrated_system["system_integrator"]
        
        # Setup: Create client and employee
        client = await client_manager.onboard_client(sample_client_info)
        employee = await employee_manager.create_employee(client.id, sample_employee_requirements)
        
        # Create multiple concurrent agents
        agents = []
        for i in range(5):
            agent = await system_integrator.create_employee_agent(
                client.id, employee.id, {"session_id": f"concurrent_session_{i}"}
            )
            agents.append(agent)
        
        # Process concurrent interactions
        async def process_interaction(agent, message_id):
            return await agent.process_message(f"Concurrent message {message_id}")
        
        # Run concurrent interactions
        tasks = [
            process_interaction(agents[i], i) 
            for i in range(len(agents))
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all interactions completed successfully
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        assert len(successful_responses) == 5
        
        # Verify each response is unique
        response_ids = [r.interaction_id for r in successful_responses]
        assert len(set(response_ids)) == 5  # All unique


if __name__ == "__main__":
    pytest.main([__file__])