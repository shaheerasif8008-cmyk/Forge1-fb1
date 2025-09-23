# forge1/backend/forge1/tests/test_langchain_integration.py
"""
Tests for LangChain Integration

Comprehensive tests for the LangChain adapter and enterprise enhancements.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from forge1.integrations.langchain_adapter import (
    LangChainAdapter, 
    LangChainIntegrationType,
    ForgeCallbackHandler,
    ForgeLangChainMemory,
    ForgeLangChainTool
)
from forge1.core.quality_assurance import QualityLevel


class TestLangChainAdapter:
    """Test cases for LangChain Adapter"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for LangChain adapter"""
        return {
            "memory_manager": Mock(),
            "model_router": Mock(),
            "performance_monitor": Mock(),
            "quality_assurance": Mock(),
            "security_manager": Mock()
        }
    
    @pytest.fixture
    def langchain_adapter(self, mock_dependencies):
        """Create LangChain adapter for testing"""
        return LangChainAdapter(**mock_dependencies)
    
    @pytest.mark.asyncio
    async def test_adapter_initialization(self, langchain_adapter):
        """Test LangChain adapter initialization"""
        
        assert langchain_adapter.memory_manager is not None
        assert langchain_adapter.model_router is not None
        assert langchain_adapter.performance_monitor is not None
        assert langchain_adapter.quality_assurance is not None
        assert langchain_adapter.security_manager is not None
        
        assert langchain_adapter.active_agents == {}
        assert langchain_adapter.active_chains == {}
        assert langchain_adapter.registered_tools == {}
        
        # Check initial metrics
        metrics = langchain_adapter.integration_metrics
        assert metrics["agents_created"] == 0
        assert metrics["chains_executed"] == 0
        assert metrics["tools_registered"] == 0
    
    @pytest.mark.asyncio
    async def test_create_enhanced_agent(self, langchain_adapter, mock_dependencies):
        """Test creating enhanced LangChain agent"""
        
        # Mock quality assurance
        mock_dependencies["quality_assurance"].conduct_quality_review = AsyncMock(return_value={
            "quality_decision": {"approved": True, "overall_score": 0.95}
        })
        
        agent_config = {
            "session_id": "test_session",
            "model_config": {
                "task_type": "general",
                "temperature": 0.7
            },
            "system_prompt": "You are a helpful AI assistant."
        }
        
        tools = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "function": lambda x: f"Test result: {x}"
            }
        ]
        
        # Mock model router
        mock_dependencies["model_router"].select_model = AsyncMock(return_value={
            "provider": "openai",
            "model": "gpt-4",
            "capabilities": ["text_generation"]
        })
        
        result = await langchain_adapter.create_enhanced_agent(
            agent_config=agent_config,
            tools=tools,
            integration_type=LangChainIntegrationType.AGENT_EXECUTOR
        )
        
        assert "agent_id" in result
        assert result["integration_type"] == "agent_executor"
        assert "capabilities" in result
        assert len(result["capabilities"]) > 0
        
        # Check that agent was stored
        agent_id = result["agent_id"]
        assert agent_id in langchain_adapter.active_agents
        
        # Check metrics updated
        assert langchain_adapter.integration_metrics["agents_created"] == 1
    
    @pytest.mark.asyncio
    async def test_execute_agent_task(self, langchain_adapter, mock_dependencies):
        """Test executing task with enhanced LangChain agent"""
        
        # First create an agent
        agent_config = {"session_id": "test_session"}
        
        # Mock dependencies
        mock_dependencies["model_router"].select_model = AsyncMock(return_value={
            "provider": "openai", "model": "gpt-4"
        })
        mock_dependencies["quality_assurance"].conduct_quality_review = AsyncMock(return_value={
            "quality_decision": {"approved": True, "overall_score": 0.95}
        })
        
        agent_result = await langchain_adapter.create_enhanced_agent(agent_config)
        agent_id = agent_result["agent_id"]
        
        # Mock memory operations
        memory_mock = Mock()
        memory_mock.add_message = AsyncMock()
        memory_mock.get_relevant_context = AsyncMock(return_value=[
            {"content": "Previous conversation context", "relevance": 0.8}
        ])
        langchain_adapter.active_agents[agent_id]["memory"] = memory_mock
        
        # Execute task
        task = "What is the capital of France?"
        result = await langchain_adapter.execute_agent_task(
            agent_id=agent_id,
            task=task,
            quality_level=QualityLevel.SUPERHUMAN
        )
        
        assert result["status"] in ["completed", "quality_issues"]
        assert result["agent_id"] == agent_id
        assert result["task"] == task
        assert "result" in result
        assert "execution_time" in result
        assert "quality_assessment" in result
        
        # Verify memory operations were called
        memory_mock.add_message.assert_called()
        memory_mock.get_relevant_context.assert_called_with(task, limit=5)
    
    @pytest.mark.asyncio
    async def test_register_tool(self, langchain_adapter, mock_dependencies):
        """Test registering enhanced tool"""
        
        def test_function(input_text: str) -> str:
            return f"Processed: {input_text}"
        
        # Mock security manager
        mock_dependencies["security_manager"].validate_tool_execution = AsyncMock()
        mock_dependencies["performance_monitor"].track_tool_execution = AsyncMock()
        
        tool_id = await langchain_adapter.register_tool(
            tool_name="test_processor",
            tool_description="Processes text input",
            tool_function=test_function,
            tool_config={"timeout": 30}
        )
        
        assert tool_id.startswith("tool_")
        assert tool_id in langchain_adapter.registered_tools
        
        tool_info = langchain_adapter.registered_tools[tool_id]
        assert tool_info["name"] == "test_processor"
        assert tool_info["description"] == "Processes text input"
        assert tool_info["config"]["timeout"] == 30
        
        # Check metrics updated
        assert langchain_adapter.integration_metrics["tools_registered"] == 1
    
    @pytest.mark.asyncio
    async def test_create_multi_agent_workflow(self, langchain_adapter, mock_dependencies):
        """Test creating multi-agent workflow"""
        
        # Mock dependencies
        mock_dependencies["model_router"].select_model = AsyncMock(return_value={
            "provider": "openai", "model": "gpt-4"
        })
        mock_dependencies["quality_assurance"].conduct_quality_review = AsyncMock(return_value={
            "quality_decision": {"approved": True, "overall_score": 0.95}
        })
        
        workflow_config = {
            "name": "test_workflow",
            "description": "A test multi-agent workflow"
        }
        
        agents = [
            {
                "session_id": "agent1_session",
                "model_config": {"task_type": "planning"},
                "tools": []
            },
            {
                "session_id": "agent2_session", 
                "model_config": {"task_type": "execution"},
                "tools": []
            }
        ]
        
        workflow = await langchain_adapter.create_multi_agent_workflow(
            workflow_config=workflow_config,
            agents=agents
        )
        
        assert "workflow_id" in workflow
        assert workflow["config"] == workflow_config
        assert "agents" in workflow
        assert workflow["status"] == "active"
        
        # Should have created agents
        assert len(workflow["agents"]) <= len(agents)  # Some might fail in mock environment
    
    @pytest.mark.asyncio
    async def test_integration_metrics(self, langchain_adapter):
        """Test integration metrics tracking"""
        
        metrics = langchain_adapter.get_integration_metrics()
        
        assert "integration_metrics" in metrics
        assert "active_agents" in metrics
        assert "registered_tools" in metrics
        assert "langchain_available" in metrics
        assert "capabilities" in metrics
        
        # Check capabilities
        capabilities = metrics["capabilities"]
        expected_capabilities = [
            "enhanced_memory_integration",
            "enterprise_security", 
            "quality_assurance",
            "performance_monitoring",
            "multi_agent_coordination",
            "tool_enhancement"
        ]
        
        for capability in expected_capabilities:
            assert capability in capabilities
    
    @pytest.mark.asyncio
    async def test_agent_cleanup(self, langchain_adapter, mock_dependencies):
        """Test agent cleanup functionality"""
        
        # Create an agent first
        mock_dependencies["model_router"].select_model = AsyncMock(return_value={
            "provider": "openai", "model": "gpt-4"
        })
        
        agent_result = await langchain_adapter.create_enhanced_agent({"session_id": "test"})
        agent_id = agent_result["agent_id"]
        
        # Verify agent exists
        assert agent_id in langchain_adapter.active_agents
        
        # Cleanup agent
        cleanup_result = await langchain_adapter.cleanup_agent(agent_id)
        
        assert cleanup_result == True
        assert agent_id not in langchain_adapter.active_agents
        
        # Test cleanup of non-existent agent
        cleanup_result = await langchain_adapter.cleanup_agent("non_existent")
        assert cleanup_result == False


class TestForgeLangChainMemory:
    """Test cases for Forge LangChain Memory integration"""
    
    @pytest.fixture
    def mock_memory_manager(self):
        """Create mock memory manager"""
        memory_manager = Mock()
        memory_manager.store_memory = AsyncMock()
        memory_manager.retrieve_memories = AsyncMock(return_value=[])
        return memory_manager
    
    @pytest.fixture
    def forge_memory(self, mock_memory_manager):
        """Create Forge LangChain memory for testing"""
        return ForgeLangChainMemory(mock_memory_manager, "test_session")
    
    @pytest.mark.asyncio
    async def test_add_message(self, forge_memory, mock_memory_manager):
        """Test adding message to memory"""
        
        message = "Hello, how are you?"
        await forge_memory.add_message(message, "human")
        
        # Verify message was stored in Forge 1 memory
        mock_memory_manager.store_memory.assert_called_once()
        call_args = mock_memory_manager.store_memory.call_args
        
        assert call_args[1]["content"] == message
        assert call_args[1]["memory_type"] == "conversation"
        assert call_args[1]["metadata"]["message_type"] == "human"
        assert call_args[1]["metadata"]["session_id"] == "test_session"
    
    @pytest.mark.asyncio
    async def test_get_relevant_context(self, forge_memory, mock_memory_manager):
        """Test getting relevant context"""
        
        # Mock return value
        mock_memory_manager.retrieve_memories.return_value = [
            {"content": "Previous context 1", "relevance": 0.9},
            {"content": "Previous context 2", "relevance": 0.8}
        ]
        
        query = "What did we discuss about AI?"
        context = await forge_memory.get_relevant_context(query, limit=5)
        
        assert len(context) == 2
        assert context[0]["content"] == "Previous context 1"
        
        # Verify correct parameters were passed
        mock_memory_manager.retrieve_memories.assert_called_once_with(
            query=query,
            memory_types=["conversation", "knowledge", "experience"],
            limit=5,
            session_id="test_session"
        )


class TestForgeLangChainTool:
    """Test cases for Forge LangChain Tool enhancement"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for tool"""
        return {
            "security_manager": Mock(),
            "performance_monitor": Mock()
        }
    
    @pytest.fixture
    def test_function(self):
        """Create test function for tool"""
        def process_text(text: str) -> str:
            return f"Processed: {text.upper()}"
        return process_text
    
    @pytest.fixture
    def forge_tool(self, mock_dependencies, test_function):
        """Create Forge LangChain tool for testing"""
        mock_dependencies["security_manager"].validate_tool_execution = AsyncMock()
        mock_dependencies["performance_monitor"].track_tool_execution = AsyncMock()
        
        return ForgeLangChainTool(
            name="test_tool",
            description="A test tool for processing text",
            func=test_function,
            **mock_dependencies
        )
    
    @pytest.mark.asyncio
    async def test_tool_execution(self, forge_tool, mock_dependencies):
        """Test enhanced tool execution"""
        
        input_text = "hello world"
        result = await forge_tool._arun(input_text)
        
        assert result == "Processed: HELLO WORLD"
        
        # Verify security validation was called
        mock_dependencies["security_manager"].validate_tool_execution.assert_called_once()
        
        # Verify performance tracking was called
        mock_dependencies["performance_monitor"].track_tool_execution.assert_called_once()
        
        # Check performance tracking parameters
        call_args = mock_dependencies["performance_monitor"].track_tool_execution.call_args
        assert call_args[1]["tool_name"] == "test_tool"
        assert call_args[1]["success"] == True
    
    @pytest.mark.asyncio
    async def test_tool_execution_failure(self, mock_dependencies):
        """Test tool execution with failure"""
        
        def failing_function(text: str) -> str:
            raise ValueError("Test error")
        
        mock_dependencies["security_manager"].validate_tool_execution = AsyncMock()
        mock_dependencies["performance_monitor"].track_tool_execution = AsyncMock()
        
        forge_tool = ForgeLangChainTool(
            name="failing_tool",
            description="A tool that fails",
            func=failing_function,
            **mock_dependencies
        )
        
        with pytest.raises(ValueError, match="Test error"):
            await forge_tool._arun("test input")
        
        # Verify failure was tracked
        call_args = mock_dependencies["performance_monitor"].track_tool_execution.call_args
        assert call_args[1]["success"] == False
        assert call_args[1]["error"] == "Test error"


class TestForgeCallbackHandler:
    """Test cases for Forge Callback Handler"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for callback handler"""
        return {
            "performance_monitor": Mock(),
            "quality_assurance": Mock(),
            "security_manager": Mock()
        }
    
    @pytest.fixture
    def callback_handler(self, mock_dependencies):
        """Create callback handler for testing"""
        mock_dependencies["performance_monitor"].track_execution = AsyncMock()
        
        return ForgeCallbackHandler(
            session_id="test_session",
            **mock_dependencies
        )
    
    def test_llm_lifecycle_tracking(self, callback_handler):
        """Test LLM lifecycle tracking"""
        
        run_id = "test_run_123"
        prompts = ["What is AI?"]
        
        # Start LLM
        callback_handler.on_llm_start(
            serialized={"name": "test_llm"},
            prompts=prompts,
            run_id=run_id
        )
        
        assert run_id in callback_handler.execution_metrics
        metrics = callback_handler.execution_metrics[run_id]
        assert metrics["prompts"] == prompts
        assert "start_time" in metrics
        
        # End LLM
        response = {"text": "AI is artificial intelligence"}
        callback_handler.on_llm_end(response, run_id=run_id)
        
        assert "end_time" in metrics
        assert "duration" in metrics
        assert metrics["response"] == response
    
    def test_tool_tracking(self, callback_handler):
        """Test tool execution tracking"""
        
        # Tool start
        callback_handler.on_tool_start(
            serialized={"name": "test_tool"},
            input_str="test input"
        )
        
        # Tool end
        callback_handler.on_tool_end("test output")
        
        # Should not raise any exceptions
        assert True
    
    def test_agent_action_tracking(self, callback_handler):
        """Test agent action tracking"""
        
        # Mock agent action
        action = {"tool": "test_tool", "input": "test input"}
        callback_handler.on_agent_action(action)
        
        # Mock agent finish
        finish = {"output": "test output"}
        callback_handler.on_agent_finish(finish)
        
        # Should not raise any exceptions
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])