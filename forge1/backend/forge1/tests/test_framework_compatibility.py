# forge1/backend/forge1/tests/test_framework_compatibility.py
"""
Tests for Unified Framework Compatibility Layer

Comprehensive tests for the framework adapter and unified APIs.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from forge1.integrations.framework_adapter import (
    FrameworkCompatibilityLayer,
    FrameworkType,
    UnifiedTaskType,
    FrameworkCapability
)
from forge1.core.quality_assurance import QualityLevel


class TestFrameworkCompatibilityLayer:
    """Test cases for Framework Compatibility Layer"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for compatibility layer"""
        return {
            "memory_manager": Mock(),
            "model_router": Mock(),
            "performance_monitor": Mock(),
            "quality_assurance": Mock(),
            "security_manager": Mock(),
            "conflict_resolution": Mock()
        }
    
    @pytest.fixture
    def compatibility_layer(self, mock_dependencies):
        """Create compatibility layer for testing"""
        return FrameworkCompatibilityLayer(**mock_dependencies)
    
    def test_initialization(self, compatibility_layer):
        """Test compatibility layer initialization"""
        
        assert compatibility_layer.memory_manager is not None
        assert compatibility_layer.model_router is not None
        assert compatibility_layer.performance_monitor is not None
        assert compatibility_layer.quality_assurance is not None
        assert compatibility_layer.security_manager is not None
        assert compatibility_layer.conflict_resolution is not None
        
        # Check framework adapters initialized
        assert len(compatibility_layer.framework_adapters) > 0
        assert FrameworkType.LANGCHAIN in compatibility_layer.framework_adapters
        assert FrameworkType.CREWAI in compatibility_layer.framework_adapters
        assert FrameworkType.AUTOGEN in compatibility_layer.framework_adapters
        assert FrameworkType.HAYSTACK in compatibility_layer.framework_adapters
        
        # Check framework capabilities
        assert len(compatibility_layer.framework_capabilities) > 0
        
        # Check initial metrics
        metrics = compatibility_layer.compatibility_metrics
        assert metrics["framework_switches"] == 0
        assert metrics["unified_tasks_executed"] == 0
        assert metrics["conflicts_resolved"] == 0
    
    def test_framework_capabilities_mapping(self, compatibility_layer):
        """Test framework capabilities mapping"""
        
        capabilities = compatibility_layer.framework_capabilities
        
        # Check LangChain capabilities
        langchain_caps = capabilities[FrameworkType.LANGCHAIN]
        assert FrameworkCapability.AGENT_CREATION in langchain_caps
        assert FrameworkCapability.CHAIN_EXECUTION in langchain_caps
        assert FrameworkCapability.TOOL_INTEGRATION in langchain_caps
        
        # Check CrewAI capabilities
        crewai_caps = capabilities[FrameworkType.CREWAI]
        assert FrameworkCapability.MULTI_AGENT_COORDINATION in crewai_caps
        assert FrameworkCapability.WORKFLOW_ORCHESTRATION in crewai_caps
        
        # Check AutoGen capabilities
        autogen_caps = capabilities[FrameworkType.AUTOGEN]
        assert FrameworkCapability.CONVERSATION_HANDLING in autogen_caps
        assert FrameworkCapability.MULTI_AGENT_COORDINATION in autogen_caps
        
        # Check Haystack capabilities
        haystack_caps = capabilities[FrameworkType.HAYSTACK]
        assert FrameworkCapability.DOCUMENT_PROCESSING in haystack_caps
    
    def test_framework_supports_task(self, compatibility_layer):
        """Test framework task support checking"""
        
        # Test LangChain supports agent execution
        assert compatibility_layer._framework_supports_task(
            FrameworkType.LANGCHAIN, UnifiedTaskType.AGENT_EXECUTION
        )
        
        # Test CrewAI supports workflow orchestration
        assert compatibility_layer._framework_supports_task(
            FrameworkType.CREWAI, UnifiedTaskType.WORKFLOW_ORCHESTRATION
        )
        
        # Test AutoGen supports conversation management
        assert compatibility_layer._framework_supports_task(
            FrameworkType.AUTOGEN, UnifiedTaskType.CONVERSATION_MANAGEMENT
        )
        
        # Test Haystack supports document processing
        assert compatibility_layer._framework_supports_task(
            FrameworkType.HAYSTACK, UnifiedTaskType.DOCUMENT_PROCESSING
        )
        
        # Test negative case - LangChain doesn't support document processing
        assert not compatibility_layer._framework_supports_task(
            FrameworkType.LANGCHAIN, UnifiedTaskType.DOCUMENT_PROCESSING
        )
    
    @pytest.mark.asyncio
    async def test_select_optimal_framework(self, compatibility_layer):
        """Test optimal framework selection"""
        
        # Test with preferred framework
        selected = await compatibility_layer._select_optimal_framework(
            UnifiedTaskType.AGENT_EXECUTION,
            {},
            preferred_framework=FrameworkType.LANGCHAIN
        )
        assert selected == FrameworkType.LANGCHAIN
        
        # Test automatic selection for conversation management
        selected = await compatibility_layer._select_optimal_framework(
            UnifiedTaskType.CONVERSATION_MANAGEMENT,
            {}
        )
        assert selected == FrameworkType.AUTOGEN  # Should prefer AutoGen for conversations
        
        # Test automatic selection for workflow orchestration
        selected = await compatibility_layer._select_optimal_framework(
            UnifiedTaskType.WORKFLOW_ORCHESTRATION,
            {}
        )
        assert selected == FrameworkType.CREWAI  # Should prefer CrewAI for workflows
        
        # Test automatic selection for document processing
        selected = await compatibility_layer._select_optimal_framework(
            UnifiedTaskType.DOCUMENT_PROCESSING,
            {}
        )
        assert selected == FrameworkType.HAYSTACK  # Should prefer Haystack for documents
    
    @pytest.mark.asyncio
    async def test_execute_unified_task_langchain(self, compatibility_layer, mock_dependencies):
        """Test unified task execution with LangChain"""
        
        # Mock LangChain adapter methods
        langchain_adapter = compatibility_layer.framework_adapters[FrameworkType.LANGCHAIN]
        langchain_adapter.create_enhanced_agent = AsyncMock(return_value={
            "agent_id": "test_agent_123",
            "status": "created"
        })
        langchain_adapter.execute_agent_task = AsyncMock(return_value={
            "execution_id": "exec_123",
            "result": "Task completed successfully",
            "status": "completed"
        })
        
        # Mock memory manager
        mock_dependencies["memory_manager"].store_memory = AsyncMock()
        
        task_config = {
            "agent_config": {"model": "gpt-4"},
            "task": "Analyze the given data",
            "context": {"data_type": "sales"}
        }
        
        result = await compatibility_layer.execute_unified_task(
            task_type=UnifiedTaskType.AGENT_EXECUTION,
            task_config=task_config,
            preferred_framework=FrameworkType.LANGCHAIN
        )
        
        assert result["status"] == "completed"
        assert result["framework_used"] == "langchain"
        assert "task_id" in result
        assert "execution_time" in result
        assert "result" in result
        
        # Verify adapter methods were called
        langchain_adapter.create_enhanced_agent.assert_called_once()
        langchain_adapter.execute_agent_task.assert_called_once()
        
        # Verify memory storage
        mock_dependencies["memory_manager"].store_memory.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_unified_task_crewai(self, compatibility_layer, mock_dependencies):
        """Test unified task execution with CrewAI"""
        
        # Mock CrewAI adapter methods
        crewai_adapter = compatibility_layer.framework_adapters[FrameworkType.CREWAI]
        crewai_adapter.create_enhanced_agent = AsyncMock(return_value=Mock(agent_id="agent_123"))
        crewai_adapter.create_enhanced_task = AsyncMock(return_value=Mock(task_id="task_123"))
        crewai_adapter.create_enhanced_crew = AsyncMock(return_value=Mock(crew_id="crew_123"))
        crewai_adapter.execute_workflow = AsyncMock(return_value={
            "crew_id": "crew_123",
            "status": "completed",
            "result": "Workflow completed"
        })
        
        # Mock memory manager
        mock_dependencies["memory_manager"].store_memory = AsyncMock()
        
        task_config = {
            "agents": [
                {"role": "Analyst", "goal": "Analyze data", "backstory": "Expert analyst"}
            ],
            "tasks": [
                {"description": "Analyze sales data", "expected_output": "Analysis report"}
            ],
            "context": {"workflow_type": "analysis"}
        }
        
        result = await compatibility_layer.execute_unified_task(
            task_type=UnifiedTaskType.WORKFLOW_ORCHESTRATION,
            task_config=task_config,
            preferred_framework=FrameworkType.CREWAI
        )
        
        assert result["status"] == "completed"
        assert result["framework_used"] == "crewai"
        assert "task_id" in result
        assert "execution_time" in result
        assert "result" in result
        
        # Verify adapter methods were called
        crewai_adapter.create_enhanced_agent.assert_called_once()
        crewai_adapter.create_enhanced_task.assert_called_once()
        crewai_adapter.create_enhanced_crew.assert_called_once()
        crewai_adapter.execute_workflow.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_unified_task_autogen(self, compatibility_layer, mock_dependencies):
        """Test unified task execution with AutoGen"""
        
        # Mock AutoGen adapter methods
        autogen_adapter = compatibility_layer.framework_adapters[FrameworkType.AUTOGEN]
        autogen_adapter.create_enhanced_agent = AsyncMock(return_value=Mock(agent_id="agent_123"))
        autogen_adapter.create_conversation = AsyncMock(return_value=Mock(conversation_id="conv_123"))
        autogen_adapter.start_conversation = AsyncMock(return_value={
            "conversation_id": "conv_123",
            "status": "completed",
            "result": "Conversation completed"
        })
        
        # Mock memory manager
        mock_dependencies["memory_manager"].store_memory = AsyncMock()
        
        task_config = {
            "agents": [
                {"name": "User", "agent_type": "user_proxy", "system_message": "User proxy"},
                {"name": "Assistant", "agent_type": "assistant", "system_message": "AI assistant"}
            ],
            "initial_message": "Hello, let's discuss the project",
            "max_rounds": 3,
            "context": {"conversation_type": "project_discussion"}
        }
        
        result = await compatibility_layer.execute_unified_task(
            task_type=UnifiedTaskType.CONVERSATION_MANAGEMENT,
            task_config=task_config,
            preferred_framework=FrameworkType.AUTOGEN
        )
        
        assert result["status"] == "completed"
        assert result["framework_used"] == "autogen"
        assert "task_id" in result
        assert "execution_time" in result
        assert "result" in result
        
        # Verify adapter methods were called
        assert autogen_adapter.create_enhanced_agent.call_count == 2  # Two agents created
        autogen_adapter.create_conversation.assert_called_once()
        autogen_adapter.start_conversation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_unified_task_haystack(self, compatibility_layer, mock_dependencies):
        """Test unified task execution with Haystack"""
        
        # Mock Haystack adapter methods
        haystack_adapter = compatibility_layer.framework_adapters[FrameworkType.HAYSTACK]
        mock_processor = Mock()
        mock_processor.process_document = AsyncMock(return_value={
            "processing_id": "proc_123",
            "result": {"content": "Extracted document content"},
            "status": "completed"
        })
        haystack_adapter.create_document_processor = AsyncMock(return_value=mock_processor)
        
        # Mock memory manager
        mock_dependencies["memory_manager"].store_memory = AsyncMock()
        
        task_config = {
            "document_path": "/path/to/document.pdf",
            "document_format": "pdf",
            "processor_config": {"extraction_type": "full_text"},
            "context": {"processing_type": "content_extraction"}
        }
        
        result = await compatibility_layer.execute_unified_task(
            task_type=UnifiedTaskType.DOCUMENT_PROCESSING,
            task_config=task_config,
            preferred_framework=FrameworkType.HAYSTACK
        )
        
        assert result["status"] == "completed"
        assert result["framework_used"] == "haystack"
        assert "task_id" in result
        assert "execution_time" in result
        assert "result" in result
        
        # Verify adapter methods were called
        haystack_adapter.create_document_processor.assert_called_once()
        mock_processor.process_document.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_unified_task_with_fallback(self, compatibility_layer, mock_dependencies):
        """Test unified task execution with fallback framework"""
        
        # Mock primary framework to fail
        langchain_adapter = compatibility_layer.framework_adapters[FrameworkType.LANGCHAIN]
        langchain_adapter.create_enhanced_agent = AsyncMock(side_effect=Exception("Primary framework failed"))
        
        # Mock fallback framework to succeed
        crewai_adapter = compatibility_layer.framework_adapters[FrameworkType.CREWAI]
        crewai_adapter.create_enhanced_agent = AsyncMock(return_value=Mock(agent_id="fallback_agent"))
        crewai_adapter.create_enhanced_task = AsyncMock(return_value=Mock(task_id="fallback_task"))
        crewai_adapter.create_enhanced_crew = AsyncMock(return_value=Mock(crew_id="fallback_crew"))
        crewai_adapter.execute_workflow = AsyncMock(return_value={
            "crew_id": "fallback_crew",
            "status": "completed",
            "result": "Fallback execution successful"
        })
        
        # Mock memory manager
        mock_dependencies["memory_manager"].store_memory = AsyncMock()
        
        task_config = {
            "agents": [{"role": "Agent", "goal": "Complete task", "backstory": "Test agent"}],
            "tasks": [{"description": "Test task", "expected_output": "Test output"}]
        }
        
        result = await compatibility_layer.execute_unified_task(
            task_type=UnifiedTaskType.WORKFLOW_ORCHESTRATION,
            task_config=task_config,
            preferred_framework=FrameworkType.LANGCHAIN,
            fallback_frameworks=[FrameworkType.CREWAI]
        )
        
        assert result["status"] == "completed"
        assert result["framework_used"] == "crewai"
        assert result["fallback_used"] == True
        assert "task_id" in result
        assert "result" in result
    
    @pytest.mark.asyncio
    async def test_switch_framework(self, compatibility_layer, mock_dependencies):
        """Test framework switching"""
        
        # Mock memory manager
        mock_dependencies["memory_manager"].retrieve_memories = AsyncMock(return_value=[
            {"content": "Previous conversation", "type": "conversation"}
        ])
        mock_dependencies["memory_manager"].store_memory = AsyncMock()
        
        session_id = "test_session_123"
        
        result = await compatibility_layer.switch_framework(
            session_id=session_id,
            from_framework=FrameworkType.LANGCHAIN,
            to_framework=FrameworkType.CREWAI,
            migration_config={"preserve_context": True}
        )
        
        assert result["status"] == "completed"
        assert result["session_id"] == session_id
        assert result["from_framework"] == "langchain"
        assert result["to_framework"] == "crewai"
        assert "switch_time" in result
        assert "migration_result" in result
        
        # Verify session is tracked
        assert session_id in compatibility_layer.active_sessions
        session_info = compatibility_layer.active_sessions[session_id]
        assert session_info["current_framework"] == FrameworkType.CREWAI
        assert session_info["previous_framework"] == FrameworkType.LANGCHAIN
        
        # Verify memory operations
        mock_dependencies["memory_manager"].retrieve_memories.assert_called_once()
        mock_dependencies["memory_manager"].store_memory.assert_called_once()
        
        # Check metrics updated
        assert compatibility_layer.compatibility_metrics["framework_switches"] == 1
    
    def test_get_compatibility_metrics(self, compatibility_layer):
        """Test getting compatibility metrics"""
        
        metrics = compatibility_layer.get_compatibility_metrics()
        
        assert "compatibility_metrics" in metrics
        assert "available_frameworks" in metrics
        assert "framework_capabilities" in metrics
        assert "active_sessions" in metrics
        assert "supported_task_types" in metrics
        
        # Check available frameworks
        frameworks = metrics["available_frameworks"]
        assert "langchain" in frameworks
        assert "crewai" in frameworks
        assert "autogen" in frameworks
        assert "haystack" in frameworks
        
        # Check supported task types
        task_types = metrics["supported_task_types"]
        assert "agent_execution" in task_types
        assert "workflow_orchestration" in task_types
        assert "conversation_management" in task_types
        assert "document_processing" in task_types
    
    def test_get_framework_status(self, compatibility_layer):
        """Test getting framework status"""
        
        status = compatibility_layer.get_framework_status()
        
        # Check all frameworks have status
        assert "langchain" in status
        assert "crewai" in status
        assert "autogen" in status
        assert "haystack" in status
        
        # Check status structure
        for framework_name, framework_status in status.items():
            assert "available" in framework_status
            assert "capabilities" in framework_status
            # Should have either metrics or error
            assert "metrics" in framework_status or "error" in framework_status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])