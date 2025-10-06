# forge1/backend/tests/unit/test_employee_manager.py
"""
Unit tests for EmployeeManager

Tests all core functionality of the EmployeeManager class including
employee creation, loading, updating, and lifecycle management.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from forge1.services.employee_manager import EmployeeManager
from forge1.services.client_manager import ClientManager
from forge1.services.employee_memory_manager import EmployeeMemoryManager
from forge1.models.employee_models import (
    Employee, EmployeeRequirements, EmployeeStatus, PersonalityConfig,
    ModelPreferences, CommunicationStyle, FormalityLevel, ExpertiseLevel,
    ResponseLength, EmployeeNotFoundError, ClientNotFoundError
)


class TestEmployeeManager:
    """Test suite for EmployeeManager"""
    
    @pytest.fixture
    async def employee_manager(self):
        """Create EmployeeManager instance with mocked dependencies"""
        db_manager = AsyncMock()
        client_manager = AsyncMock(spec=ClientManager)
        memory_manager = AsyncMock(spec=EmployeeMemoryManager)
        
        manager = EmployeeManager(
            db_manager=db_manager,
            client_manager=client_manager,
            memory_manager=memory_manager
        )
        
        # Mock initialization
        manager._initialized = True
        
        return manager
    
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
                "patience_level": 0.95
            }
        )
    
    @pytest.fixture
    def sample_employee(self):
        """Sample employee for testing"""
        personality = PersonalityConfig(
            communication_style=CommunicationStyle.FRIENDLY,
            formality_level=FormalityLevel.CASUAL,
            expertise_level=ExpertiseLevel.INTERMEDIATE,
            response_length=ResponseLength.DETAILED,
            creativity_level=0.7,
            empathy_level=0.8,
            custom_traits={"patience": "high"}
        )
        
        model_preferences = ModelPreferences(
            primary_model="gpt-4",
            fallback_models=["gpt-3.5-turbo"],
            temperature=0.7,
            max_tokens=2000
        )
        
        return Employee(
            id="emp_test_001",
            client_id="client_test_001",
            name="Test Employee",
            role="Customer Support Specialist",
            personality=personality,
            model_preferences=model_preferences,
            tool_access=["email", "chat"],
            knowledge_sources=["kb_001"],
            status=EmployeeStatus.ACTIVE
        )
    
    @pytest.mark.asyncio
    async def test_create_employee_success(self, employee_manager, sample_employee_requirements):
        """Test successful employee creation"""
        # Mock client manager response
        employee_manager.client_manager.create_employee_for_client.return_value = AsyncMock()
        employee_manager.client_manager.create_employee_for_client.return_value.id = "emp_test_001"
        employee_manager.client_manager.create_employee_for_client.return_value.name = "Test Employee"
        
        # Mock memory manager
        employee_manager.memory_manager.initialize_employee_namespace.return_value = AsyncMock()
        
        # Mock tenant validation
        with patch('forge1.services.employee_manager.get_current_tenant', return_value="client_test_001"):
            result = await employee_manager.create_employee("client_test_001", sample_employee_requirements)
        
        # Assertions
        employee_manager.client_manager.create_employee_for_client.assert_called_once_with(
            "client_test_001", sample_employee_requirements
        )
        employee_manager.memory_manager.initialize_employee_namespace.assert_called_once()
        assert result.id == "emp_test_001"
    
    @pytest.mark.asyncio
    async def test_create_employee_client_not_found(self, employee_manager, sample_employee_requirements):
        """Test employee creation with non-existent client"""
        # Mock client manager to raise exception
        employee_manager.client_manager.create_employee_for_client.side_effect = ClientNotFoundError("Client not found")
        
        with pytest.raises(ClientNotFoundError):
            await employee_manager.create_employee("nonexistent_client", sample_employee_requirements)
    
    @pytest.mark.asyncio
    async def test_load_employee_success(self, employee_manager, sample_employee):
        """Test successful employee loading"""
        # Mock database response
        employee_manager._load_employee_from_database = AsyncMock(return_value=sample_employee)
        employee_manager.memory_manager.get_employee_context = AsyncMock(return_value=[])
        
        with patch('forge1.services.employee_manager.get_current_tenant', return_value="client_test_001"):
            result = await employee_manager.load_employee("client_test_001", "emp_test_001")
        
        assert result is not None
        assert result.id == "emp_test_001"
        assert result.name == "Test Employee"
    
    @pytest.mark.asyncio
    async def test_load_employee_not_found(self, employee_manager):
        """Test loading non-existent employee"""
        # Mock database to return None
        employee_manager._load_employee_from_database = AsyncMock(return_value=None)
        
        with patch('forge1.services.employee_manager.get_current_tenant', return_value="client_test_001"):
            result = await employee_manager.load_employee("client_test_001", "nonexistent_emp")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_update_employee_success(self, employee_manager, sample_employee):
        """Test successful employee update"""
        # Mock get_employee to return sample employee
        employee_manager.get_employee = AsyncMock(return_value=sample_employee)
        
        # Mock database connection
        mock_conn = AsyncMock()
        employee_manager.db_manager.get_connection.return_value.__aenter__.return_value = mock_conn
        
        updates = {
            "name": "Updated Employee Name",
            "personality": {
                "communication_style": "professional"
            }
        }
        
        with patch('forge1.services.employee_manager.get_current_tenant', return_value="client_test_001"):
            result = await employee_manager.update_employee("client_test_001", "emp_test_001", updates)
        
        # Verify database update was called
        mock_conn.execute.assert_called()
        employee_manager.get_employee.assert_called()
    
    @pytest.mark.asyncio
    async def test_update_employee_not_found(self, employee_manager):
        """Test updating non-existent employee"""
        # Mock get_employee to return None
        employee_manager.get_employee = AsyncMock(return_value=None)
        
        updates = {"name": "Updated Name"}
        
        with pytest.raises(EmployeeNotFoundError):
            await employee_manager.update_employee("client_test_001", "nonexistent_emp", updates)
    
    @pytest.mark.asyncio
    async def test_list_employees_success(self, employee_manager, sample_employee):
        """Test successful employee listing"""
        # Mock client manager response
        employee_manager.client_manager.get_client_employees = AsyncMock(return_value=[sample_employee])
        
        with patch('forge1.services.employee_manager.get_current_tenant', return_value="client_test_001"):
            result = await employee_manager.list_employees("client_test_001")
        
        assert len(result) == 1
        assert result[0].id == "emp_test_001"
    
    @pytest.mark.asyncio
    async def test_delete_employee_archive(self, employee_manager, sample_employee):
        """Test employee archiving (soft delete)"""
        # Mock get_employee
        employee_manager.get_employee = AsyncMock(return_value=sample_employee)
        employee_manager.update_employee = AsyncMock(return_value=sample_employee)
        
        with patch('forge1.services.employee_manager.get_current_tenant', return_value="client_test_001"):
            result = await employee_manager.delete_employee("client_test_001", "emp_test_001", archive_memory=True)
        
        assert result is True
        employee_manager.update_employee.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_employee_hard_delete(self, employee_manager, sample_employee):
        """Test employee hard deletion"""
        # Mock get_employee
        employee_manager.get_employee = AsyncMock(return_value=sample_employee)
        
        # Mock database connection
        mock_conn = AsyncMock()
        employee_manager.db_manager.get_connection.return_value.__aenter__.return_value = mock_conn
        
        with patch('forge1.services.employee_manager.get_current_tenant', return_value="client_test_001"):
            result = await employee_manager.delete_employee("client_test_001", "emp_test_001", archive_memory=False)
        
        assert result is True
        mock_conn.execute.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_employee_stats_success(self, employee_manager, sample_employee):
        """Test getting employee statistics"""
        # Mock get_employee
        employee_manager.get_employee = AsyncMock(return_value=sample_employee)
        
        # Mock memory manager stats
        employee_manager.memory_manager.get_memory_stats = AsyncMock(return_value={
            "total_interactions": 100,
            "total_memories": 50
        })
        
        # Mock database stats
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {
            "avg_processing_time": 1500.0,
            "total_tokens": 10000,
            "total_cost": 5.50,
            "active_days": 15
        }
        employee_manager.db_manager.get_connection.return_value.__aenter__.return_value = mock_conn
        
        with patch('forge1.services.employee_manager.get_current_tenant', return_value="client_test_001"):
            result = await employee_manager.get_employee_stats("client_test_001", "emp_test_001")
        
        assert result["employee_id"] == "emp_test_001"
        assert result["employee_name"] == "Test Employee"
        assert "total_interactions" in result
        assert "average_processing_time_ms" in result
    
    @pytest.mark.asyncio
    async def test_search_employees_success(self, employee_manager):
        """Test employee search functionality"""
        # Mock database search results
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                "id": "emp_001",
                "name": "Customer Support Agent",
                "role": "Support Specialist",
                "communication_style": "friendly",
                "expertise_level": "intermediate",
                "tool_access": ["email", "chat"],
                "knowledge_sources": ["kb_001"],
                "status": "active",
                "created_at": datetime.now(timezone.utc),
                "last_interaction_at": datetime.now(timezone.utc),
                "rank": 0.8
            }
        ]
        employee_manager.db_manager.get_connection.return_value.__aenter__.return_value = mock_conn
        
        with patch('forge1.services.employee_manager.get_current_tenant', return_value="client_test_001"):
            result = await employee_manager.search_employees("client_test_001", "support", limit=10)
        
        assert len(result) == 1
        assert result[0]["id"] == "emp_001"
        assert result[0]["name"] == "Customer Support Agent"
        assert "relevance_score" in result[0]
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, employee_manager, sample_employee):
        """Test employee caching functionality"""
        # Mock database load
        employee_manager._load_employee_from_database = AsyncMock(return_value=sample_employee)
        employee_manager.memory_manager.get_employee_context = AsyncMock(return_value=[])
        
        with patch('forge1.services.employee_manager.get_current_tenant', return_value="client_test_001"):
            # First load - should hit database
            result1 = await employee_manager.load_employee("client_test_001", "emp_test_001")
            
            # Second load - should hit cache
            result2 = await employee_manager.load_employee("client_test_001", "emp_test_001")
        
        # Database should only be called once due to caching
        assert employee_manager._load_employee_from_database.call_count == 1
        assert result1.id == result2.id
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, employee_manager):
        """Test metrics tracking functionality"""
        initial_metrics = employee_manager.get_metrics()
        
        # Simulate some operations
        employee_manager._metrics["employees_created"] = 5
        employee_manager._metrics["employees_loaded"] = 10
        employee_manager._metrics["cache_hits"] = 8
        employee_manager._metrics["cache_misses"] = 2
        
        metrics = employee_manager.get_metrics()
        
        assert metrics["employees_created"] == 5
        assert metrics["employees_loaded"] == 10
        assert metrics["cache_hit_rate"] == 0.8  # 8/(8+2)
    
    def test_cache_key_generation(self, employee_manager):
        """Test cache key generation"""
        key = employee_manager._get_cache_key("client_001", "emp_001")
        assert key == "client_001:emp_001"
    
    @pytest.mark.asyncio
    async def test_tenant_validation(self, employee_manager):
        """Test tenant access validation"""
        with patch('forge1.services.employee_manager.get_current_tenant', return_value="client_001"):
            # Should not raise exception for matching tenant
            await employee_manager._validate_tenant_access("client_001")
        
        with patch('forge1.services.employee_manager.get_current_tenant', return_value="client_002"):
            # Should raise exception for mismatched tenant
            with pytest.raises(Exception):  # TenantIsolationError
                await employee_manager._validate_tenant_access("client_001")


if __name__ == "__main__":
    pytest.main([__file__])