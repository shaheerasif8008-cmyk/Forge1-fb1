# forge1/backend/forge1/tests/test_autogen_integration.py
"""
Tests for AutoGen Integration

Comprehensive tests for the AutoGen adapter and enterprise enhancements.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock

from forge1.integrations.autogen_adapter import (
    AutoGenAdapter,
    ForgeAutoGenAgent,
    ForgeAutoGenConversation,
    AutoGenAgentType,
    AutoGenConversationType,
    ConversationStatus
)
from forge1.core.quality_assurance import QualityLevel


class TestAutoGenAdapter:
    """Test cases for AutoGen Adapter"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for AutoGen adapter"""
        return {
            "memory_manager": Mock(),
            "model_router": Mock(),
            "performance_monitor": Mock(),
            "quality_assurance": Mock(),
            "security_manager": Mock()
        }
    
    @pytest.fixture
    def autogen_adapter(self, mock_dependencies):
        """Create AutoGen adapter for testing"""
        return AutoGenAdapter(**mock_dependencies)
    
    @pytest.mark.asyncio
    async def test_adapter_initialization(self, autogen_adapter):
        """Test AutoGen adapter initialization"""
        
        assert autogen_adapter.memory_manager is not None
        assert autogen_adapter.model_router is not None
        assert autogen_adapter.performance_monitor is not None
        assert autogen_adapter.quality_assurance is not None
        assert autogen_adapter.security_manager is not None
        
        assert autogen_adapter.active_agents == {}
        assert autogen_adapter.active_conversations == {}
        
        # Check initial metrics
        metrics = autogen_adapter.integration_metrics
        assert metrics["agents_created"] == 0
        assert metrics["conversations_started"] == 0
        assert metrics["conversations_completed"] == 0
        assert metrics["messages_exchanged"] == 0
    
    @pytest.mark.asyncio
    async def test_create_enhanced_agent(self, autogen_adapter):
        """Test creating enhanced AutoGen agent"""
        
        agent = await autogen_adapter.create_enhanced_agent(
            name="Assistant Agent",
            agent_type=AutoGenAgentType.ASSISTANT,
            system_message="You are a helpful AI assistant specialized in data analysis.",
            llm_config={"model": "gpt-4", "temperature": 0.7},
            agent_config={"session_id": "test_session"}
        )
        
        assert isinstance(agent, ForgeAutoGenAgent)
        assert agent.name == "Assistant Agent"
        assert agent.agent_type == AutoGenAgentType.ASSISTANT
        assert agent.system_message == "You are a helpful AI assistant specialized in data analysis."
        assert agent.agent_id.startswith("autogen_agent_")
        assert agent.session_id == "test_session"
        
        # Check that agent was stored
        assert agent.agent_id in autogen_adapter.active_agents
        
        # Check metrics updated
        assert autogen_adapter.integration_metrics["agents_created"] == 1
    
    @pytest.mark.asyncio
    async def test_create_conversation(self, autogen_adapter):
        """Test creating enhanced AutoGen conversation"""
        
        # Create agents first
        agent1 = await autogen_adapter.create_enhanced_agent(
            name="User Proxy",
            agent_type=AutoGenAgentType.USER_PROXY,
            system_message="You represent the user in conversations."
        )
        
        agent2 = await autogen_adapter.create_enhanced_agent(
            name="Assistant",
            agent_type=AutoGenAgentType.ASSISTANT,
            system_message="You are a helpful assistant."
        )
        
        # Create conversation
        conversation = await autogen_adapter.create_conversation(
            conversation_type=AutoGenConversationType.TWO_AGENT,
            participants=[agent1, agent2],
            max_rounds=5,
            conversation_config={"timeout": 300}
        )
        
        assert isinstance(conversation, ForgeAutoGenConversation)
        assert conversation.conversation_type == AutoGenConversationType.TWO_AGENT
        assert len(conversation.participants) == 2
        assert conversation.max_rounds == 5
        assert conversation.status == ConversationStatus.INITIALIZED
        
        # Check that conversation was stored
        assert conversation.conversation_id in autogen_adapter.active_conversations
        
        # Check metrics updated
        assert autogen_adapter.integration_metrics["conversations_started"] == 1
    
    @pytest.mark.asyncio
    async def test_start_conversation(self, autogen_adapter, mock_dependencies):
        """Test starting AutoGen conversation"""
        
        # Mock dependencies
        mock_dependencies["security_manager"].validate_message = AsyncMock()
        mock_dependencies["memory_manager"].store_memory = AsyncMock()
        mock_dependencies["performance_monitor"].track_message_exchange = AsyncMock()
        mock_dependencies["quality_assurance"].conduct_quality_review = AsyncMock(return_value={
            "quality_assessment": {"overall_score": 0.95},
            "quality_decision": {"approved": True}
        })
        
        # Create agents and conversation
        agent1 = await autogen_adapter.create_enhanced_agent(
            name="User",
            agent_type=AutoGenAgentType.USER_PROXY,
            system_message="User proxy agent"
        )
        
        agent2 = await autogen_adapter.create_enhanced_agent(
            name="Assistant",
            agent_type=AutoGenAgentType.ASSISTANT,
            system_message="Assistant agent"
        )
        
        conversation = await autogen_adapter.create_conversation(
            conversation_type=AutoGenConversationType.TWO_AGENT,
            participants=[agent1, agent2],
            max_rounds=3
        )
        
        # Start conversation
        result = await autogen_adapter.start_conversation(
            conversation_id=conversation.conversation_id,
            initial_message="Hello, can you help me analyze some data?",
            context={"task_type": "data_analysis"}
        )
        
        assert "conversation_id" in result
        assert result["conversation_id"] == conversation.conversation_id
        assert "conversation_type" in result
        assert "status" in result
        assert "duration" in result
        assert "total_messages" in result
        
        # Check metrics updated
        if result.get("status") == "completed":
            assert autogen_adapter.integration_metrics["conversations_completed"] >= 1
        assert autogen_adapter.integration_metrics["messages_exchanged"] >= 0
    
    @pytest.mark.asyncio
    async def test_integration_metrics(self, autogen_adapter):
        """Test integration metrics tracking"""
        
        metrics = autogen_adapter.get_integration_metrics()
        
        assert "integration_metrics" in metrics
        assert "active_agents" in metrics
        assert "active_conversations" in metrics
        assert "autogen_available" in metrics
        assert "capabilities" in metrics
        
        # Check capabilities
        capabilities = metrics["capabilities"]
        expected_capabilities = [
            "conversation_pattern_integration",
            "enterprise_security",
            "quality_assurance",
            "performance_monitoring",
            "multi_agent_conversations",
            "group_chat_management"
        ]
        
        for capability in expected_capabilities:
            assert capability in capabilities
    
    @pytest.mark.asyncio
    async def test_agent_cleanup(self, autogen_adapter):
        """Test agent cleanup functionality"""
        
        # Create an agent
        agent = await autogen_adapter.create_enhanced_agent(
            name="Test Agent",
            agent_type=AutoGenAgentType.ASSISTANT,
            system_message="Test agent for cleanup"
        )
        
        agent_id = agent.agent_id
        
        # Verify agent exists
        assert agent_id in autogen_adapter.active_agents
        
        # Cleanup agent
        cleanup_result = await autogen_adapter.cleanup_agent(agent_id)
        
        assert cleanup_result == True
        assert agent_id not in autogen_adapter.active_agents
        
        # Test cleanup of non-existent agent
        cleanup_result = await autogen_adapter.cleanup_agent("non_existent")
        assert cleanup_result == False
    
    @pytest.mark.asyncio
    async def test_conversation_cleanup(self, autogen_adapter):
        """Test conversation cleanup functionality"""
        
        # Create agents and conversation
        agent1 = await autogen_adapter.create_enhanced_agent(
            name="Agent 1",
            agent_type=AutoGenAgentType.USER_PROXY,
            system_message="Test agent 1"
        )
        
        agent2 = await autogen_adapter.create_enhanced_agent(
            name="Agent 2",
            agent_type=AutoGenAgentType.ASSISTANT,
            system_message="Test agent 2"
        )
        
        conversation = await autogen_adapter.create_conversation(
            conversation_type=AutoGenConversationType.TWO_AGENT,
            participants=[agent1, agent2]
        )
        
        conversation_id = conversation.conversation_id
        
        # Verify conversation exists
        assert conversation_id in autogen_adapter.active_conversations
        
        # Cleanup conversation
        cleanup_result = await autogen_adapter.cleanup_conversation(conversation_id)
        
        assert cleanup_result == True
        assert conversation_id not in autogen_adapter.active_conversations
        
        # Test cleanup of non-existent conversation
        cleanup_result = await autogen_adapter.cleanup_conversation("non_existent")
        assert cleanup_result == False


class TestForgeAutoGenAgent:
    """Test cases for Forge AutoGen Agent"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for agent"""
        return {
            "model_router": Mock(),
            "memory_manager": Mock(),
            "performance_monitor": Mock(),
            "security_manager": Mock()
        }
    
    @pytest.fixture
    def forge_agent(self, mock_dependencies):
        """Create Forge AutoGen Agent for testing"""
        mock_dependencies["security_manager"].validate_message = AsyncMock()
        mock_dependencies["memory_manager"].store_memory = AsyncMock()
        mock_dependencies["performance_monitor"].track_message_exchange = AsyncMock()
        
        return ForgeAutoGenAgent(
            name="Test Assistant",
            agent_type=AutoGenAgentType.ASSISTANT,
            system_message="You are a test assistant agent.",
            **mock_dependencies
        )
    
    def test_agent_initialization(self, forge_agent):
        """Test agent initialization"""
        
        assert forge_agent.name == "Test Assistant"
        assert forge_agent.agent_type == AutoGenAgentType.ASSISTANT
        assert forge_agent.system_message == "You are a test assistant agent."
        assert forge_agent.agent_id.startswith("autogen_agent_")
        assert forge_agent.agent_metrics["messages_sent"] == 0
        assert forge_agent.agent_metrics["messages_received"] == 0
    
    @pytest.mark.asyncio
    async def test_send_message(self, forge_agent, mock_dependencies):
        """Test message sending between agents"""
        
        # Create recipient agent
        recipient = ForgeAutoGenAgent(
            name="Recipient Agent",
            agent_type=AutoGenAgentType.USER_PROXY,
            system_message="You are a recipient agent.",
            **mock_dependencies
        )
        
        message = "Hello, can you help me with a task?"
        context = {"task_type": "assistance"}
        
        result = await forge_agent.send_message(
            message=message,
            recipient=recipient,
            context=context
        )
        
        assert "message_id" in result
        assert result["sender"] == forge_agent.agent_id
        assert result["recipient"] == recipient.agent_id
        assert result["message"] == message
        assert "response" in result
        assert "response_time" in result
        assert result["status"] in ["completed", "failed"]
        
        # Verify security validation was called
        mock_dependencies["security_manager"].validate_message.assert_called_once()
        
        # Verify memory storage was called
        assert mock_dependencies["memory_manager"].store_memory.call_count >= 1
        
        # Verify performance tracking was called
        mock_dependencies["performance_monitor"].track_message_exchange.assert_called_once()
        
        # Check metrics updated
        assert forge_agent.agent_metrics["messages_sent"] == 1
    
    def test_get_agent_info(self, forge_agent):
        """Test getting agent information"""
        
        info = forge_agent.get_agent_info()
        
        assert info["agent_id"] == forge_agent.agent_id
        assert info["name"] == "Test Assistant"
        assert info["agent_type"] == "assistant"
        assert info["system_message"] == "You are a test assistant agent."
        assert "agent_metrics" in info
        assert "autogen_available" in info


class TestForgeAutoGenConversation:
    """Test cases for Forge AutoGen Conversation"""
    
    @pytest.fixture
    def mock_agents(self, mock_dependencies):
        """Create mock agents for conversation testing"""
        mock_deps = {
            "model_router": Mock(),
            "memory_manager": Mock(),
            "performance_monitor": Mock(),
            "security_manager": Mock()
        }
        
        mock_deps["security_manager"].validate_message = AsyncMock()
        mock_deps["memory_manager"].store_memory = AsyncMock()
        mock_deps["performance_monitor"].track_message_exchange = AsyncMock()
        
        agents = []
        for i in range(2):
            agent = ForgeAutoGenAgent(
                name=f"Test Agent {i+1}",
                agent_type=AutoGenAgentType.ASSISTANT if i == 0 else AutoGenAgentType.USER_PROXY,
                system_message=f"You are test agent {i+1}.",
                **mock_deps
            )
            agents.append(agent)
        
        return agents
    
    @pytest.fixture
    def forge_conversation(self, mock_agents):
        """Create Forge AutoGen Conversation for testing"""
        quality_assurance = Mock()
        quality_assurance.conduct_quality_review = AsyncMock(return_value={
            "quality_assessment": {"overall_score": 0.9},
            "quality_decision": {"approved": True}
        })
        
        return ForgeAutoGenConversation(
            conversation_type=AutoGenConversationType.TWO_AGENT,
            participants=mock_agents,
            quality_assurance=quality_assurance,
            max_rounds=3
        )
    
    def test_conversation_initialization(self, forge_conversation, mock_agents):
        """Test conversation initialization"""
        
        assert forge_conversation.conversation_type == AutoGenConversationType.TWO_AGENT
        assert len(forge_conversation.participants) == 2
        assert forge_conversation.max_rounds == 3
        assert forge_conversation.status == ConversationStatus.INITIALIZED
        assert forge_conversation.conversation_id.startswith("conv_")
        assert forge_conversation.round_count == 0
    
    @pytest.mark.asyncio
    async def test_start_two_agent_conversation(self, forge_conversation):
        """Test starting two-agent conversation"""
        
        initial_message = "Hello, let's discuss the project requirements."
        context = {"project": "test_project"}
        
        result = await forge_conversation.start_conversation(initial_message, context)
        
        assert "conversation_id" in result
        assert result["conversation_id"] == forge_conversation.conversation_id
        assert result["conversation_type"] == "two_agent"
        assert "status" in result
        assert "duration" in result
        assert "rounds_completed" in result
        assert "total_messages" in result
        
        # Check conversation status updated
        assert forge_conversation.status in [ConversationStatus.COMPLETED, ConversationStatus.FAILED]
        assert forge_conversation.round_count >= 0
    
    @pytest.mark.asyncio
    async def test_start_group_chat_conversation(self, mock_agents):
        """Test starting group chat conversation"""
        
        # Add a third agent for group chat
        third_agent = ForgeAutoGenAgent(
            name="Third Agent",
            agent_type=AutoGenAgentType.ASSISTANT,
            system_message="You are the third agent.",
            model_router=Mock(),
            memory_manager=Mock(),
            performance_monitor=Mock(),
            security_manager=Mock()
        )
        
        participants = mock_agents + [third_agent]
        
        conversation = ForgeAutoGenConversation(
            conversation_type=AutoGenConversationType.GROUP_CHAT,
            participants=participants,
            max_rounds=5
        )
        
        initial_message = "Let's start our group discussion about the new features."
        context = {"topic": "feature_discussion"}
        
        result = await conversation.start_conversation(initial_message, context)
        
        assert "conversation_id" in result
        assert result["conversation_type"] == "group_chat"
        assert "status" in result
        assert "total_messages" in result
    
    @pytest.mark.asyncio
    async def test_start_sequential_conversation(self, mock_agents):
        """Test starting sequential conversation"""
        
        # Add more agents for sequential conversation
        additional_agents = []
        for i in range(2):
            agent = ForgeAutoGenAgent(
                name=f"Sequential Agent {i+3}",
                agent_type=AutoGenAgentType.ASSISTANT,
                system_message=f"You are sequential agent {i+3}.",
                model_router=Mock(),
                memory_manager=Mock(),
                performance_monitor=Mock(),
                security_manager=Mock()
            )
            additional_agents.append(agent)
        
        participants = mock_agents + additional_agents
        
        conversation = ForgeAutoGenConversation(
            conversation_type=AutoGenConversationType.SEQUENTIAL,
            participants=participants,
            max_rounds=4
        )
        
        initial_message = "Please process this request sequentially."
        context = {"process_type": "sequential"}
        
        result = await conversation.start_conversation(initial_message, context)
        
        assert "conversation_id" in result
        assert result["conversation_type"] == "sequential"
        assert "status" in result
        assert "rounds_completed" in result
    
    def test_get_conversation_info(self, forge_conversation):
        """Test getting conversation information"""
        
        info = forge_conversation.get_conversation_info()
        
        assert info["conversation_id"] == forge_conversation.conversation_id
        assert info["conversation_type"] == "two_agent"
        assert info["status"] == "initialized"
        assert info["participants_count"] == 2
        assert info["max_rounds"] == 3
        assert info["current_round"] == 0
        assert "conversation_metrics" in info
        assert "autogen_available" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])