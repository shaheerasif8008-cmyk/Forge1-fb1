"""
MCAE Integration Unit Tests

Comprehensive unit tests for all MCAE integration components including
adapters, context injection, tool permissions, and error handling.
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from forge1.integrations.mcae_adapter import MCAEAdapter, MCAEIntegrationError
from forge1.integrations.workflow_context_injector import (
    WorkflowContextInjector, context_injector, create_context, inject_context
)
from forge1.integrations.tool_permission_enforcer import (
    ToolPermissionEnforcer, PermissionLevel, ToolCategory, PermissionDeniedError
)
from forge1.integrations.mcae_error_handler import MCAEErrorHandler, ErrorCategory, ErrorSeverity
from forge1.integrations.forge1_agent_factory import Forge1AgentFactory
from forge1.integrations.forge1_model_client import Forge1ModelClient
from forge1.integrations.forge1_memory_store import Forge1MemoryStore
from forge1.models.employee_models import Employee, PersonalityConfig, ModelPreferences
from forge1.models.workflow_models import WorkflowContext, WorkflowType, CollaborationMode


class TestMCAEAdapter:
    """Test suite for MCAEAdapter"""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing"""
        employee_manager = Mock()
        model_router = Mock()
        memory_manager = Mock()
        
        return employee_manager, model_router, memory_manager
    
    @pytest.fixture
    async def mcae_adapter(self, mock_components):
        """Create MCAEAdapter instance for testing"""
        employee_manager, model_router, memory_manager = mock_components
        
        adapter = MCAEAdapter(employee_manager, model_router, memory_manager)
        
        # Mock the agent factory initialization
        with patch('forge1.integrations.mcae_adapter.Forge1AgentFactory') as mock_factory:
            mock_factory.return_value.initialize = AsyncMock()
            await adapter.initialize()
        
        return adapter
    
    @pytest.mark.asyncio
    async def test_adapter_initialization(self, mock_components):
        """Test MCAE adapter initialization"""
        employee_manager, model_router, memory_manager = mock_components
        
        adapter = MCAEAdapter(employee_manager, model_router, memory_manager)
        
        with patch('forge1.integrations.mcae_adapter.Forge1AgentFactory') as mock_factory:
            mock_factory.return_value.initialize = AsyncMock()
            
            await adapter.initialize()
            
            assert adapter._initialized == True
            assert adapter.agent_factory is not None
    
    @pytest.mark.asyncio
    async def test_employee_workflow_registration(self, mcae_adapter):
        """Test employee workflow registration"""
        # Create mock employee
        employee = Mock()
        employee.client_id = "test_tenant"
        employee.id = "test_employee"
        employee.name = "Test Employee"
        employee.role = "Test Role"
        
        # Mock agent factory
        mcae_adapter.agent_factory = Mock()
        mcae_adapter.agent_factory.create_agents_for_employee = AsyncMock(
            return_value={"test_agent": Mock()}
        )
        
        # Register workflow
        workflow_id = await mcae_adapter.register_employee_workflow(employee)
        
        # Verify workflow registration
        assert workflow_id is not None
        assert workflow_id in mcae_adapter.active_workflows
        assert mcae_adapter.employee_workflows[employee.id] == workflow_id
        
        workflow_context = mcae_adapter.active_workflows[workflow_id]
        assert workflow_context["employee"] == employee
        assert workflow_context["tenant_id"] == employee.client_id
        assert workflow_context["status"] == "registered"
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, mcae_adapter):
        """Test workflow execution through MCAE"""
        # Set up workflow
        employee = Mock()
        employee.client_id = "test_tenant"
        employee.id = "test_employee"
        
        mcae_adapter.agent_factory = Mock()
        mcae_adapter.agent_factory.create_agents_for_employee = AsyncMock(
            return_value={"group_chat_manager": Mock()}
        )
        
        workflow_id = await mcae_adapter.register_employee_workflow(employee)
        
        # Mock group chat manager
        mock_manager = Mock()
        mock_manager.handle_input_task = AsyncMock(return_value=Mock(id="plan_123", summary="Test plan"))
        
        mcae_adapter.active_workflows[workflow_id]["agent_instances"] = {
            "group_chat_manager": mock_manager
        }
        
        # Mock memory store
        mock_memory = Mock()
        mock_memory.get_steps_by_plan = AsyncMock(return_value=[])
        mock_memory.get_data_by_type = AsyncMock(return_value=[])
        
        with patch('forge1.integrations.mcae_adapter.Forge1MemoryStore', return_value=mock_memory):
            # Execute workflow
            result = await mcae_adapter.execute_workflow(workflow_id, "Test task")
            
            # Verify execution
            assert result is not None
            assert result["workflow_id"] == workflow_id
            assert result["status"] == "completed"
            assert result["tenant_id"] == employee.client_id
    
    @pytest.mark.asyncio
    async def test_workflow_cleanup(self, mcae_adapter):
        """Test workflow cleanup"""
        # Create workflow
        employee = Mock()
        employee.client_id = "test_tenant"
        employee.id = "test_employee"
        
        mcae_adapter.agent_factory = Mock()
        mcae_adapter.agent_factory.create_agents_for_employee = AsyncMock(
            return_value={"test_agent": Mock()}
        )
        
        workflow_id = await mcae_adapter.register_employee_workflow(employee)
        
        # Verify workflow exists
        assert workflow_id in mcae_adapter.active_workflows
        
        # Cleanup workflow
        success = await mcae_adapter.cleanup_workflow(workflow_id)
        
        # Verify cleanup
        assert success == True
        assert workflow_id not in mcae_adapter.active_workflows
        assert employee.id not in mcae_adapter.employee_workflows


class TestWorkflowContextInjector:
    """Test suite for WorkflowContextInjector"""
    
    @pytest.fixture
    def context_injector_instance(self):
        """Create WorkflowContextInjector instance for testing"""
        return WorkflowContextInjector()
    
    def test_create_workflow_context(self, context_injector_instance):
        """Test workflow context creation"""
        tenant_id = "test_tenant"
        employee_id = "test_employee"
        session_id = "test_session"
        user_id = "test_user"
        
        context = context_injector_instance.create_workflow_context(
            tenant_id, employee_id, session_id, user_id
        )
        
        assert context["tenant_id"] == tenant_id
        assert context["employee_id"] == employee_id
        assert context["session_id"] == session_id
        assert context["user_id"] == user_id
        assert "context_id" in context
        assert "validation_token" in context
        assert "created_at" in context
    
    def test_context_injection_and_retrieval(self, context_injector_instance):
        """Test context injection and retrieval"""
        context = {
            "tenant_id": "test_tenant",
            "employee_id": "test_employee",
            "session_id": "test_session",
            "user_id": "test_user"
        }
        
        # Inject context
        context_injector_instance.inject_context(context)
        
        # Retrieve context
        retrieved_context = context_injector_instance.get_current_context()
        
        assert retrieved_context == context
        assert context_injector_instance.get_current_tenant_id() == "test_tenant"
        assert context_injector_instance.get_current_employee_id() == "test_employee"
    
    def test_context_validation_success(self, context_injector_instance):
        """Test successful context validation"""
        tenant_id = "test_tenant"
        employee_id = "test_employee"
        
        # Create and inject context
        context = context_injector_instance.create_workflow_context(
            tenant_id, employee_id, "session", "user"
        )
        context_injector_instance.inject_context(context)
        
        # Validate context
        is_valid = context_injector_instance.validate_context(tenant_id, employee_id)
        
        assert is_valid == True
    
    def test_context_validation_tenant_violation(self, context_injector_instance):
        """Test context validation with tenant violation"""
        # Create and inject context for one tenant
        context = context_injector_instance.create_workflow_context(
            "tenant_1", "employee_1", "session", "user"
        )
        context_injector_instance.inject_context(context)
        
        # Try to validate with different tenant
        with pytest.raises(Exception):  # Should raise TenantIsolationViolationError
            context_injector_instance.validate_context("tenant_2", "employee_1")
    
    def test_context_manager(self, context_injector_instance):
        """Test context manager functionality"""
        context1 = {"tenant_id": "tenant_1", "employee_id": "emp_1"}
        context2 = {"tenant_id": "tenant_2", "employee_id": "emp_2"}
        
        # Set initial context
        context_injector_instance.inject_context(context1)
        
        # Use context manager
        with context_injector_instance.with_context(context2):
            current_context = context_injector_instance.get_current_context()
            assert current_context == context2
        
        # Context should be restored
        current_context = context_injector_instance.get_current_context()
        assert current_context == context1


class TestToolPermissionEnforcer:
    """Test suite for ToolPermissionEnforcer"""
    
    @pytest.fixture
    def permission_enforcer_instance(self):
        """Create ToolPermissionEnforcer instance for testing"""
        return ToolPermissionEnforcer()
    
    def test_tool_registration(self, permission_enforcer_instance):
        """Test tool registration"""
        tool_name = "test_tool"
        category = ToolCategory.DOCUMENT
        permission = PermissionLevel.READ
        
        permission_enforcer_instance.register_tool(
            tool_name, category, permission, description="Test tool"
        )
        
        assert tool_name in permission_enforcer_instance.tool_registry
        
        tool_info = permission_enforcer_instance.tool_registry[tool_name]
        assert tool_info["category"] == category
        assert tool_info["required_permission"] == permission
        assert tool_info["description"] == "Test tool"
    
    def test_employee_permission_setting(self, permission_enforcer_instance):
        """Test setting employee permissions"""
        tenant_id = "test_tenant"
        employee_id = "test_employee"
        permissions = {
            "document_parser": PermissionLevel.READ,
            "vector_search": PermissionLevel.WRITE
        }
        
        permission_enforcer_instance.set_employee_permissions(
            tenant_id, employee_id, permissions
        )
        
        key = f"{tenant_id}:{employee_id}"
        assert key in permission_enforcer_instance.employee_permissions
        assert permission_enforcer_instance.employee_permissions[key] == permissions
    
    @pytest.mark.asyncio
    async def test_permission_check_success(self, permission_enforcer_instance):
        """Test successful permission check"""
        # Set up context
        context = {
            "tenant_id": "test_tenant",
            "employee_id": "test_employee",
            "session_id": "test_session",
            "user_id": "test_user"
        }
        
        # Register tool
        permission_enforcer_instance.register_tool(
            "test_tool", ToolCategory.DOCUMENT, PermissionLevel.READ
        )
        
        # Set employee permissions
        permission_enforcer_instance.set_employee_permissions(
            "test_tenant", "test_employee", {"test_tool": PermissionLevel.READ}
        )
        
        # Mock context injection
        with patch('forge1.integrations.tool_permission_enforcer.get_context', return_value=context):
            # Check permission
            has_permission = await permission_enforcer_instance.check_tool_permission("test_tool")
            
            assert has_permission == True
    
    @pytest.mark.asyncio
    async def test_permission_check_denied(self, permission_enforcer_instance):
        """Test permission check denial"""
        # Set up context
        context = {
            "tenant_id": "test_tenant",
            "employee_id": "test_employee",
            "session_id": "test_session",
            "user_id": "test_user"
        }
        
        # Register tool requiring WRITE permission
        permission_enforcer_instance.register_tool(
            "test_tool", ToolCategory.DOCUMENT, PermissionLevel.WRITE
        )
        
        # Set employee with only READ permission
        permission_enforcer_instance.set_employee_permissions(
            "test_tenant", "test_employee", {"test_tool": PermissionLevel.READ}
        )
        
        # Mock context injection
        with patch('forge1.integrations.tool_permission_enforcer.get_context', return_value=context):
            # Check permission - should be denied
            with pytest.raises(PermissionDeniedError):
                await permission_enforcer_instance.check_tool_permission("test_tool")
    
    def test_get_allowed_tools(self, permission_enforcer_instance):
        """Test getting allowed tools for employee"""
        tenant_id = "test_tenant"
        employee_id = "test_employee"
        
        # Register tools
        permission_enforcer_instance.register_tool(
            "allowed_tool", ToolCategory.DOCUMENT, PermissionLevel.READ
        )
        permission_enforcer_instance.register_tool(
            "denied_tool", ToolCategory.DOCUMENT, PermissionLevel.ADMIN
        )
        
        # Set employee permissions
        permission_enforcer_instance.set_employee_permissions(
            tenant_id, employee_id, {"allowed_tool": PermissionLevel.READ}
        )
        
        # Get allowed tools
        allowed_tools = permission_enforcer_instance.get_employee_allowed_tools(tenant_id, employee_id)
        
        # Should only include the allowed tool
        assert len(allowed_tools) == 1
        assert allowed_tools[0]["name"] == "allowed_tool"


class TestMCAEErrorHandler:
    """Test suite for MCAEErrorHandler"""
    
    @pytest.fixture
    def error_handler_instance(self):
        """Create MCAEErrorHandler instance for testing"""
        return MCAEErrorHandler()
    
    @pytest.mark.asyncio
    async def test_error_classification(self, error_handler_instance):
        """Test error classification"""
        from forge1.integrations.mcae_adapter import TenantIsolationViolationError
        
        error = TenantIsolationViolationError("Test tenant violation")
        context = {"tenant_id": "test_tenant", "employee_id": "test_employee"}
        
        # Handle error
        result = await error_handler_instance.handle_error(error, context, "test_operation")
        
        # Verify error classification
        assert result["error_handled"] == True
        error_info = result["error_info"]
        assert error_info["category"] == ErrorCategory.TENANT_ISOLATION
        assert error_info["severity"] == ErrorSeverity.CRITICAL
    
    @pytest.mark.asyncio
    async def test_tenant_violation_handling(self, error_handler_instance):
        """Test tenant isolation violation handling"""
        from forge1.integrations.mcae_adapter import TenantIsolationViolationError
        
        error = TenantIsolationViolationError("Cross-tenant access attempt")
        context = {"tenant_id": "test_tenant", "employee_id": "test_employee"}
        
        # Handle error
        result = await error_handler_instance.handle_error(error, context, "test_operation")
        
        # Verify security handling
        assert result["error_handled"] == True
        recovery_result = result["recovery_result"]
        assert recovery_result["action"] == "security_abort"
        assert recovery_result["violation_logged"] == True
    
    @pytest.mark.asyncio
    async def test_fallback_recovery(self, error_handler_instance):
        """Test fallback recovery mechanism"""
        from forge1.integrations.mcae_adapter import WorkflowExecutionError
        
        error = WorkflowExecutionError("MCAE execution failed")
        context = {"tenant_id": "test_tenant", "employee_id": "test_employee"}
        
        # Handle error
        result = await error_handler_instance.handle_error(error, context, "workflow_execution")
        
        # Verify fallback
        assert result["error_handled"] == True
        recovery_result = result["recovery_result"]
        assert recovery_result["action"] == "fallback"
        assert recovery_result["fallback_system"] == "forge1_native"
    
    def test_error_statistics(self, error_handler_instance):
        """Test error statistics tracking"""
        # Simulate some errors
        error_handler_instance.error_stats["total_errors"] = 10
        error_handler_instance.error_stats["tenant_violations"] = 2
        error_handler_instance.error_stats["recovery_attempts"] = 8
        error_handler_instance.error_stats["successful_recoveries"] = 6
        
        # Get statistics
        stats = error_handler_instance.get_error_stats()
        
        assert stats["total_errors"] == 10
        assert stats["tenant_violations"] == 2
        assert stats["recovery_success_rate"] == 75.0  # 6/8 * 100
        assert stats["tenant_violation_rate"] == 20.0  # 2/10 * 100


class TestForge1AgentFactory:
    """Test suite for Forge1AgentFactory"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for agent factory"""
        employee_manager = Mock()
        model_router = Mock()
        memory_manager = Mock()
        
        return employee_manager, model_router, memory_manager
    
    @pytest.fixture
    def agent_factory(self, mock_dependencies):
        """Create Forge1AgentFactory instance for testing"""
        employee_manager, model_router, memory_manager = mock_dependencies
        
        factory = Forge1AgentFactory(
            tenant_id="test_tenant",
            employee_manager=employee_manager,
            model_router=model_router,
            memory_manager=memory_manager
        )
        
        return factory
    
    def test_role_to_agent_type_mapping(self, agent_factory):
        """Test role to agent type mapping"""
        # Test direct mappings
        assert agent_factory._get_agent_type_for_role("hr") == agent_factory.role_to_agent_type["hr"]
        assert agent_factory._get_agent_type_for_role("marketing") == agent_factory.role_to_agent_type["marketing"]
        
        # Test fuzzy matching
        from models.messages_kernel import AgentType
        assert agent_factory._get_agent_type_for_role("human resources") == AgentType.HR
        assert agent_factory._get_agent_type_for_role("legal assistant") == AgentType.GENERIC
    
    def test_system_message_generation(self, agent_factory):
        """Test system message generation"""
        from models.messages_kernel import AgentType
        
        # Create mock employee
        employee = Mock()
        employee.name = "Test Employee"
        employee.role = "Legal Assistant"
        employee.client_id = "test_tenant"
        employee.tool_access = ["document_parser", "vector_search"]
        
        # Mock personality
        personality = Mock()
        personality.communication_style.value = "professional"
        personality.formality_level.value = "formal"
        personality.expertise_level.value = "expert"
        personality.creativity_level = 0.8
        personality.empathy_level = 0.7
        employee.personality = personality
        
        # Generate system message
        system_message = agent_factory._generate_system_message(employee, AgentType.GENERIC)
        
        # Verify message content
        assert "Test Employee" in system_message
        assert "Legal Assistant" in system_message
        assert "professional" in system_message
        assert "formal" in system_message
        assert "document_parser" in system_message
        assert "test_tenant" in system_message


class TestForge1ModelClient:
    """Test suite for Forge1ModelClient"""
    
    @pytest.fixture
    def mock_model_router(self):
        """Create mock model router"""
        router = Mock()
        router.get_optimal_client = AsyncMock()
        router.release_client = AsyncMock()
        
        return router
    
    @pytest.fixture
    def model_client(self, mock_model_router):
        """Create Forge1ModelClient instance for testing"""
        return Forge1ModelClient(
            tenant_id="test_tenant",
            employee_id="test_employee",
            model_router=mock_model_router
        )
    
    @pytest.mark.asyncio
    async def test_model_client_initialization(self, model_client):
        """Test model client initialization"""
        employee_config = {
            "model_preferences": {
                "temperature": 0.8,
                "max_tokens": 1000,
                "primary_model": "gpt-4"
            }
        }
        
        await model_client.initialize(employee_config)
        
        assert model_client.default_temperature == 0.8
        assert model_client.default_max_tokens == 1000
        assert model_client.preferred_model == "gpt-4"
    
    @pytest.mark.asyncio
    async def test_generate_request(self, model_client, mock_model_router):
        """Test generate request through model router"""
        # Mock model client response
        mock_client = Mock()
        mock_client.generate = AsyncMock(return_value="Generated response")
        mock_model_router.get_optimal_client.return_value = mock_client
        
        # Make request
        response = await model_client.generate("Test prompt")
        
        # Verify request
        assert response == "Generated response"
        mock_model_router.get_optimal_client.assert_called_once()
        mock_client.generate.assert_called_once()
        mock_model_router.release_client.assert_called_once_with(mock_client)
    
    @pytest.mark.asyncio
    async def test_chat_request(self, model_client, mock_model_router):
        """Test chat request through model router"""
        # Mock model client response
        mock_client = Mock()
        mock_client.chat = AsyncMock(return_value="Chat response")
        mock_model_router.get_optimal_client.return_value = mock_client
        
        messages = [{"role": "user", "content": "Hello"}]
        
        # Make request
        response = await model_client.chat(messages)
        
        # Verify request
        assert response == "Chat response"
        mock_client.chat.assert_called_once_with(messages, temperature=0.7, max_tokens=2000)
    
    def test_metrics_tracking(self, model_client):
        """Test metrics tracking"""
        # Simulate some requests
        model_client.request_count = 5
        model_client.total_tokens = 1000
        model_client.total_cost = 0.05
        
        metrics = model_client.get_metrics()
        
        assert metrics["request_count"] == 5
        assert metrics["total_tokens"] == 1000
        assert metrics["total_cost"] == 0.05
        assert metrics["tenant_id"] == "test_tenant"
        assert metrics["employee_id"] == "test_employee"


class TestForge1MemoryStore:
    """Test suite for Forge1MemoryStore"""
    
    @pytest.fixture
    def mock_memory_manager(self):
        """Create mock memory manager"""
        manager = Mock()
        manager._initialized = True
        manager.store_memory = AsyncMock(return_value="memory_123")
        manager.search_memories = AsyncMock()
        manager.update_memory = AsyncMock(return_value=True)
        
        return manager
    
    @pytest.fixture
    def memory_store(self, mock_memory_manager):
        """Create Forge1MemoryStore instance for testing"""
        return Forge1MemoryStore(
            session_id="test_session",
            user_id="test_user",
            tenant_id="test_tenant",
            employee_id="test_employee",
            memory_manager=mock_memory_manager
        )
    
    @pytest.mark.asyncio
    async def test_memory_store_initialization(self, memory_store, mock_memory_manager):
        """Test memory store initialization"""
        with patch('forge1.integrations.forge1_memory_store.set_current_tenant') as mock_set_tenant:
            await memory_store.initialize()
            
            mock_set_tenant.assert_called_once_with("test_tenant")
    
    @pytest.mark.asyncio
    async def test_add_item(self, memory_store, mock_memory_manager):
        """Test adding item to memory store"""
        # Create mock item
        item = Mock()
        item.data_type = "test_item"
        item.id = "item_123"
        
        # Mock conversion method
        memory_store._convert_to_memory_context = AsyncMock(return_value=Mock())
        
        # Add item
        memory_id = await memory_store.add_item(item)
        
        # Verify storage
        assert memory_id == "memory_123"
        mock_memory_manager.store_memory.assert_called_once()
    
    def test_cache_key_generation(self, memory_store):
        """Test cache key generation"""
        cache_key = memory_store._get_cache_key("test_type", "test_id")
        
        expected_key = "test_tenant_test_employee_test_type_test_id"
        assert cache_key == expected_key
    
    def test_tenant_isolation_in_tags(self, memory_store):
        """Test tenant isolation in memory tags"""
        item = Mock()
        item.data_type = "test_item"
        item.id = "item_123"
        
        tags = memory_store._get_tags_for_item(item)
        
        # Verify tenant isolation tags
        assert "tenant:test_tenant" in tags
        assert "employee:test_employee" in tags
        assert "session:test_session" in tags
        assert "data_type:test_item" in tags


# Integration test fixtures and utilities
@pytest.fixture
def mock_employee():
    """Create a mock employee for testing"""
    employee = Mock()
    employee.id = "test_employee_123"
    employee.client_id = "test_tenant_456"
    employee.name = "Test Employee"
    employee.role = "Legal Assistant"
    employee.workflow_id = None
    
    # Mock personality
    personality = Mock()
    personality.communication_style.value = "professional"
    personality.formality_level.value = "formal"
    personality.expertise_level.value = "expert"
    personality.creativity_level = 0.7
    personality.empathy_level = 0.8
    employee.personality = personality
    
    # Mock model preferences
    model_prefs = Mock()
    model_prefs.temperature = 0.7
    model_prefs.max_tokens = 2000
    model_prefs.primary_model = "gpt-4"
    employee.model_preferences = model_prefs
    
    employee.tool_access = ["document_parser", "vector_search"]
    employee.is_active = Mock(return_value=True)
    
    return employee


@pytest.fixture
def integration_test_context():
    """Create integration test context"""
    return {
        "tenant_id": "integration_test_tenant",
        "employee_id": "integration_test_employee",
        "session_id": "integration_test_session",
        "user_id": "integration_test_user"
    }


# Performance and stress tests
class TestIntegrationPerformance:
    """Performance tests for MCAE integration"""
    
    @pytest.mark.asyncio
    async def test_context_injection_performance(self):
        """Test context injection performance"""
        import time
        
        injector = WorkflowContextInjector()
        
        # Measure context creation and injection time
        start_time = time.time()
        
        for i in range(1000):
            context = injector.create_workflow_context(
                f"tenant_{i}", f"employee_{i}", f"session_{i}", f"user_{i}"
            )
            injector.inject_context(context)
            injector.clear_context()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete 1000 operations in reasonable time (< 1 second)
        assert execution_time < 1.0
        
        print(f"Context injection performance: {execution_time:.3f}s for 1000 operations")
    
    @pytest.mark.asyncio
    async def test_permission_check_performance(self):
        """Test permission checking performance"""
        import time
        
        enforcer = ToolPermissionEnforcer()
        
        # Register tools
        for i in range(100):
            enforcer.register_tool(
                f"tool_{i}", ToolCategory.DOCUMENT, PermissionLevel.READ
            )
        
        # Set permissions
        permissions = {f"tool_{i}": PermissionLevel.READ for i in range(100)}
        enforcer.set_employee_permissions("test_tenant", "test_employee", permissions)
        
        # Mock context
        context = {
            "tenant_id": "test_tenant",
            "employee_id": "test_employee",
            "session_id": "test_session",
            "user_id": "test_user"
        }
        
        # Measure permission checking time
        start_time = time.time()
        
        with patch('forge1.integrations.tool_permission_enforcer.get_context', return_value=context):
            for i in range(100):
                try:
                    await enforcer.check_tool_permission(f"tool_{i}")
                except:
                    pass  # Ignore errors for performance test
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete 100 permission checks quickly (< 0.1 second)
        assert execution_time < 0.1
        
        print(f"Permission check performance: {execution_time:.3f}s for 100 checks")


if __name__ == "__main__":
    """Run tests with pytest"""
    pytest.main([__file__, "-v", "--tb=short"])