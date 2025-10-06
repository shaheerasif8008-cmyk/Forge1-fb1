"""
Integration tests for all vertical AI employees
Comprehensive test suite validating end-to-end functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta

from forge1.backend.forge1.verticals.registry import VerticalEmployeeRegistry, VerticalType
from forge1.backend.forge1.verticals.cx.employee import CXAIEmployee
from forge1.backend.forge1.verticals.revops.employee import RevOpsAIEmployee


class TestVerticalEmployeeRegistry:
    """Test the vertical employee registry"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for testing"""
        return {
            "workflow_engine": Mock(),
            "memory_manager": Mock(),
            "model_router": Mock(),
            "metrics_collector": Mock(),
            "secret_manager": Mock()
        }
    
    @pytest.fixture
    def registry(self):
        """Create a fresh registry instance"""
        return VerticalEmployeeRegistry()
    
    def test_supported_verticals(self, registry):
        """Test that all expected verticals are supported"""
        supported = registry.get_supported_verticals()
        
        expected_verticals = [
            VerticalType.CUSTOMER_EXPERIENCE,
            VerticalType.REVENUE_OPERATIONS,
            VerticalType.FINANCE_FPA,
            VerticalType.LEGAL,
            VerticalType.IT_OPERATIONS,
            VerticalType.SOFTWARE_ENGINEERING
        ]
        
        for vertical in expected_verticals:
            assert vertical in supported
    
    def test_create_cx_employee(self, registry, mock_dependencies):
        """Test creating a CX employee"""
        employee = registry.create_employee(
            vertical_type=VerticalType.CUSTOMER_EXPERIENCE,
            employee_id="CX-001",
            tenant_id="TENANT-123",
            **mock_dependencies
        )
        
        assert isinstance(employee, CXAIEmployee)
        assert employee.employee_id == "CX-001"
        assert employee.tenant_id == "TENANT-123"
    
    def test_create_revops_employee(self, registry, mock_dependencies):
        """Test creating a RevOps employee"""
        employee = registry.create_employee(
            vertical_type=VerticalType.REVENUE_OPERATIONS,
            employee_id="REVOPS-001",
            tenant_id="TENANT-123",
            **mock_dependencies
        )
        
        assert isinstance(employee, RevOpsAIEmployee)
        assert employee.employee_id == "REVOPS-001"
    
    def test_employee_registration(self, registry, mock_dependencies):
        """Test that employees are properly registered"""
        employee = registry.create_employee(
            vertical_type=VerticalType.CUSTOMER_EXPERIENCE,
            employee_id="CX-002",
            tenant_id="TENANT-123",
            **mock_dependencies
        )
        
        # Should be able to retrieve the employee
        retrieved = registry.get_employee("CX-002")
        assert retrieved is employee
        
        # Should appear in listings
        all_employees = registry.list_employees()
        assert "CX-002" in all_employees
        
        # Should appear in filtered listings
        cx_employees = registry.list_employees(VerticalType.CUSTOMER_EXPERIENCE)
        assert "CX-002" in cx_employees
    
    def test_employee_removal(self, registry, mock_dependencies):
        """Test employee removal from registry"""
        employee = registry.create_employee(
            vertical_type=VerticalType.CUSTOMER_EXPERIENCE,
            employee_id="CX-003",
            tenant_id="TENANT-123",
            **mock_dependencies
        )
        
        # Remove the employee
        removed = registry.remove_employee("CX-003")
        assert removed is True
        
        # Should no longer be retrievable
        retrieved = registry.get_employee("CX-003")
        assert retrieved is None
        
        # Removing non-existent employee should return False
        removed_again = registry.remove_employee("CX-003")
        assert removed_again is False
    
    def test_vertical_capabilities(self, registry):
        """Test that capabilities are properly defined for all verticals"""
        for vertical_type in registry.get_supported_verticals():
            capabilities = registry.get_vertical_capabilities(vertical_type)
            
            # Each vertical should have required fields
            assert "name" in capabilities
            assert "description" in capabilities
            assert "key_features" in capabilities
            assert "integrations" in capabilities
            assert "sla_targets" in capabilities
            
            # Key features should be a non-empty list
            assert isinstance(capabilities["key_features"], list)
            assert len(capabilities["key_features"]) > 0
            
            # Integrations should be a non-empty list
            assert isinstance(capabilities["integrations"], list)
            assert len(capabilities["integrations"]) > 0
            
            # SLA targets should be a non-empty dict
            assert isinstance(capabilities["sla_targets"], dict)
            assert len(capabilities["sla_targets"]) > 0


class TestVerticalEmployeePerformance:
    """Test performance characteristics of vertical employees"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies with realistic responses"""
        workflow_engine = Mock()
        memory_manager = Mock()
        model_router = Mock()
        metrics_collector = Mock()
        secret_manager = Mock()
        
        # Mock realistic AI responses
        model_router.generate_response = AsyncMock(
            return_value="Priority: high, Category: technical, Sentiment: negative"
        )
        
        # Mock memory operations
        memory_manager.retrieve_context = AsyncMock(return_value=[])
        memory_manager.store_context = AsyncMock()
        
        return {
            "workflow_engine": workflow_engine,
            "memory_manager": memory_manager,
            "model_router": model_router,
            "metrics_collector": metrics_collector,
            "secret_manager": secret_manager
        }
    
    @pytest.mark.asyncio
    async def test_cx_employee_performance(self, mock_dependencies):
        """Test CX employee performance under load"""
        registry = VerticalEmployeeRegistry()
        
        cx_employee = registry.create_employee(
            vertical_type=VerticalType.CUSTOMER_EXPERIENCE,
            employee_id="CX-PERF-001",
            tenant_id="TENANT-123",
            **mock_dependencies
        )
        
        # Mock the playbooks to avoid external dependencies
        cx_employee.playbooks.triage_ticket = AsyncMock()
        cx_employee.playbooks.resolve_ticket = AsyncMock()
        cx_employee._get_customer_profile = AsyncMock()
        cx_employee._update_external_ticket = AsyncMock(return_value=True)
        cx_employee._get_interaction_history = AsyncMock(return_value=[])
        cx_employee._store_interaction_memory = AsyncMock()
        
        # Test ticket processing performance
        ticket_data = {
            "id": "TICKET-PERF-001",
            "customer_id": "CUST-123",
            "subject": "Performance test ticket",
            "description": "Testing response time",
            "priority": "normal",
            "status": "new",
            "category": "general",
            "created_at": datetime.utcnow().isoformat()
        }
        
        start_time = datetime.utcnow()
        result = await cx_employee.handle_ticket(ticket_data)
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Should meet SLA targets
        assert result["sla_met"] is True
        assert processing_time < 5.0  # Should be under 5 seconds
        assert "response" in result
    
    @pytest.mark.asyncio
    async def test_revops_employee_performance(self, mock_dependencies):
        """Test RevOps employee performance"""
        registry = VerticalEmployeeRegistry()
        
        revops_employee = registry.create_employee(
            vertical_type=VerticalType.REVENUE_OPERATIONS,
            employee_id="REVOPS-PERF-001",
            tenant_id="TENANT-123",
            **mock_dependencies
        )
        
        # Mock the playbooks
        revops_employee.playbooks.pipeline_hygiene_analysis = AsyncMock()
        revops_employee._store_hygiene_results = AsyncMock()
        revops_employee._generate_hygiene_action_items = AsyncMock(return_value=[])
        
        # Test pipeline hygiene performance
        start_time = datetime.utcnow()
        result = await revops_employee.pipeline_hygiene_check()
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Should complete quickly
        assert processing_time < 10.0  # Should be under 10 seconds
        assert result["analysis_type"] == "pipeline_hygiene"
    
    @pytest.mark.asyncio
    async def test_concurrent_employee_operations(self, mock_dependencies):
        """Test multiple employees operating concurrently"""
        registry = VerticalEmployeeRegistry()
        
        # Create multiple employees
        employees = []
        for i in range(3):
            employee = registry.create_employee(
                vertical_type=VerticalType.CUSTOMER_EXPERIENCE,
                employee_id=f"CX-CONCURRENT-{i}",
                tenant_id="TENANT-123",
                **mock_dependencies
            )
            employees.append(employee)
        
        # Mock operations for all employees
        for employee in employees:
            employee.playbooks.triage_ticket = AsyncMock()
            employee.playbooks.resolve_ticket = AsyncMock()
            employee._get_customer_profile = AsyncMock()
            employee._update_external_ticket = AsyncMock(return_value=True)
            employee._get_interaction_history = AsyncMock(return_value=[])
            employee._store_interaction_memory = AsyncMock()
        
        # Create concurrent tasks
        tasks = []
        for i, employee in enumerate(employees):
            ticket_data = {
                "id": f"TICKET-CONCURRENT-{i}",
                "customer_id": f"CUST-{i}",
                "subject": f"Concurrent test ticket {i}",
                "description": "Testing concurrent processing",
                "priority": "normal",
                "status": "new",
                "category": "general",
                "created_at": datetime.utcnow().isoformat()
            }
            task = employee.handle_ticket(ticket_data)
            tasks.append(task)
        
        # Execute all tasks concurrently
        start_time = datetime.utcnow()
        results = await asyncio.gather(*tasks)
        total_time = (datetime.utcnow() - start_time).total_seconds()
        
        # All should complete successfully
        assert len(results) == 3
        for result in results:
            assert "ticket_id" in result
            assert result.get("sla_met") is True
        
        # Concurrent execution should be efficient
        assert total_time < 15.0  # Should complete all within 15 seconds


class TestVerticalEmployeeIntegration:
    """Test integration scenarios between vertical employees"""
    
    @pytest.mark.asyncio
    async def test_cross_vertical_collaboration(self):
        """Test collaboration between different vertical employees"""
        # This would test scenarios where multiple verticals work together
        # For example: CX escalating to Legal for contract issues
        pass
    
    @pytest.mark.asyncio
    async def test_tenant_isolation(self):
        """Test that employees properly isolate tenant data"""
        # This would test multi-tenancy isolation
        pass
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self):
        """Test that performance metrics are properly collected"""
        # This would test metrics collection across all verticals
        pass


if __name__ == "__main__":
    pytest.main([__file__])