# forge1/backend/forge1/tests/test_crewai_integration.py
"""
Tests for CrewAI Integration

Comprehensive tests for the CrewAI adapter and enterprise enhancements.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock

from forge1.integrations.crewai_adapter import (
    CrewAIAdapter,
    ForgeCrewAgent,
    ForgeCrewTask,
    ForgeCrew,
    CrewAIWorkflowType,
    CrewAITaskStatus
)
from forge1.core.quality_assurance import QualityLevel


class TestCrewAIAdapter:
    """Test cases for CrewAI Adapter"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for CrewAI adapter"""
        return {
            "memory_manager": Mock(),
            "model_router": Mock(),
            "performance_monitor": Mock(),
            "quality_assurance": Mock(),
            "security_manager": Mock(),
            "compliance_engine": Mock()
        }
    
    @pytest.fixture
    def crewai_adapter(self, mock_dependencies):
        """Create CrewAI adapter for testing"""
        return CrewAIAdapter(**mock_dependencies)
    
    @pytest.mark.asyncio
    async def test_adapter_initialization(self, crewai_adapter):
        """Test CrewAI adapter initialization"""
        
        assert crewai_adapter.memory_manager is not None
        assert crewai_adapter.model_router is not None
        assert crewai_adapter.performance_monitor is not None
        assert crewai_adapter.quality_assurance is not None
        assert crewai_adapter.security_manager is not None
        
        assert crewai_adapter.active_crews == {}
        assert crewai_adapter.active_agents == {}
        assert crewai_adapter.active_tasks == {}
        
        # Check initial metrics
        metrics = crewai_adapter.integration_metrics
        assert metrics["crews_created"] == 0
        assert metrics["agents_created"] == 0
        assert metrics["tasks_created"] == 0
        assert metrics["workflows_executed"] == 0
    
    @pytest.mark.asyncio
    async def test_create_enhanced_agent(self, crewai_adapter):
        """Test creating enhanced CrewAI agent"""
        
        agent = await crewai_adapter.create_enhanced_agent(
            role="Data Analyst",
            goal="Analyze data and provide insights",
            backstory="Expert data analyst with 10 years of experience",
            tools=[],
            agent_config={"session_id": "test_session"}
        )
        
        assert isinstance(agent, ForgeCrewAgent)
        assert agent.role == "Data Analyst"
        assert agent.goal == "Analyze data and provide insights"
        assert agent.agent_id.startswith("crew_agent_")
        assert agent.session_id == "test_session"
        
        # Check that agent was stored
        assert agent.agent_id in crewai_adapter.active_agents
        
        # Check metrics updated
        assert crewai_adapter.integration_metrics["agents_created"] == 1
    
    @pytest.mark.asyncio
    async def test_create_enhanced_task(self, crewai_adapter):
        """Test creating enhanced CrewAI task"""
        
        # First create an agent
        agent = await crewai_adapter.create_enhanced_agent(
            role="Writer",
            goal="Write high-quality content",
            backstory="Professional content writer"
        )
        
        # Create task
        task = await crewai_adapter.create_enhanced_task(
            description="Write a blog post about AI trends",
            agent=agent,
            expected_output="A 1000-word blog post with insights on AI trends",
            quality_level=QualityLevel.SUPERHUMAN,
            compliance_requirements=["content_guidelines", "brand_voice"]
        )
        
        assert isinstance(task, ForgeCrewTask)
        assert task.description == "Write a blog post about AI trends"
        assert task.agent == agent
        assert task.quality_level == QualityLevel.SUPERHUMAN
        assert task.compliance_requirements == ["content_guidelines", "brand_voice"]
        assert task.status == CrewAITaskStatus.PENDING
        
        # Check that task was stored
        assert task.task_id in crewai_adapter.active_tasks
        
        # Check metrics updated
        assert crewai_adapter.integration_metrics["tasks_created"] == 1
    
    @pytest.mark.asyncio
    async def test_create_enhanced_crew(self, crewai_adapter, mock_dependencies):
        """Test creating enhanced CrewAI crew"""
        
        # Create agents
        agent1 = await crewai_adapter.create_enhanced_agent(
            role="Researcher",
            goal="Research topics thoroughly",
            backstory="Expert researcher"
        )
        
        agent2 = await crewai_adapter.create_enhanced_agent(
            role="Writer",
            goal="Write engaging content",
            backstory="Professional writer"
        )
        
        # Create tasks
        task1 = await crewai_adapter.create_enhanced_task(
            description="Research AI trends",
            agent=agent1,
            expected_output="Research report on AI trends"
        )
        
        task2 = await crewai_adapter.create_enhanced_task(
            description="Write article based on research",
            agent=agent2,
            expected_output="Well-written article"
        )
        
        # Create crew
        crew = await crewai_adapter.create_enhanced_crew(
            agents=[agent1, agent2],
            tasks=[task1, task2],
            workflow_type=CrewAIWorkflowType.SEQUENTIAL
        )
        
        assert isinstance(crew, ForgeCrew)
        assert len(crew.agents) == 2
        assert len(crew.tasks) == 2
        assert crew.workflow_type == CrewAIWorkflowType.SEQUENTIAL
        assert crew.status == "initialized"
        
        # Check that crew was stored
        assert crew.crew_id in crewai_adapter.active_crews
        
        # Check metrics updated
        assert crewai_adapter.integration_metrics["crews_created"] == 1
    
    @pytest.mark.asyncio
    async def test_execute_workflow(self, crewai_adapter, mock_dependencies):
        """Test executing CrewAI workflow"""
        
        # Mock dependencies
        mock_dependencies["security_manager"].validate_task_execution = AsyncMock()
        mock_dependencies["memory_manager"].store_memory = AsyncMock()
        mock_dependencies["performance_monitor"].track_agent_execution = AsyncMock()
        mock_dependencies["quality_assurance"].conduct_quality_review = AsyncMock(return_value={
            "quality_decision": {"approved": True, "overall_score": 0.95}
        })
        
        # Create a simple crew
        agent = await crewai_adapter.create_enhanced_agent(
            role="Test Agent",
            goal="Complete test tasks",
            backstory="Test agent for workflow execution"
        )
        
        task = await crewai_adapter.create_enhanced_task(
            description="Complete a test task",
            agent=agent,
            expected_output="Test completion result"
        )
        
        crew = await crewai_adapter.create_enhanced_crew(
            agents=[agent],
            tasks=[task],
            workflow_type=CrewAIWorkflowType.SEQUENTIAL
        )
        
        # Execute workflow
        result = await crewai_adapter.execute_workflow(
            crew_id=crew.crew_id,
            context={"test_context": "test_value"}
        )
        
        assert "crew_id" in result
        assert result["crew_id"] == crew.crew_id
        assert "execution_id" in result
        assert "workflow_type" in result
        assert "execution_time" in result
        assert result["workflow_type"] == "sequential"
        
        # Check metrics updated
        assert crewai_adapter.integration_metrics["workflows_executed"] == 1
    
    @pytest.mark.asyncio
    async def test_integration_metrics(self, crewai_adapter):
        """Test integration metrics tracking"""
        
        metrics = crewai_adapter.get_integration_metrics()
        
        assert "integration_metrics" in metrics
        assert "active_crews" in metrics
        assert "active_agents" in metrics
        assert "active_tasks" in metrics
        assert "crewai_available" in metrics
        assert "capabilities" in metrics
        
        # Check capabilities
        capabilities = metrics["capabilities"]
        expected_capabilities = [
            "enhanced_workflow_orchestration",
            "enterprise_compliance",
            "quality_assurance",
            "performance_monitoring",
            "multi_agent_coordination",
            "task_management"
        ]
        
        for capability in expected_capabilities:
            assert capability in capabilities
    
    @pytest.mark.asyncio
    async def test_crew_cleanup(self, crewai_adapter):
        """Test crew cleanup functionality"""
        
        # Create a crew
        agent = await crewai_adapter.create_enhanced_agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory"
        )
        
        task = await crewai_adapter.create_enhanced_task(
            description="Test task",
            agent=agent
        )
        
        crew = await crewai_adapter.create_enhanced_crew(
            agents=[agent],
            tasks=[task]
        )
        
        crew_id = crew.crew_id
        task_id = task.task_id
        
        # Verify crew and task exist
        assert crew_id in crewai_adapter.active_crews
        assert task_id in crewai_adapter.active_tasks
        
        # Cleanup crew
        cleanup_result = await crewai_adapter.cleanup_crew(crew_id)
        
        assert cleanup_result == True
        assert crew_id not in crewai_adapter.active_crews
        assert task_id not in crewai_adapter.active_tasks
        
        # Test cleanup of non-existent crew
        cleanup_result = await crewai_adapter.cleanup_crew("non_existent")
        assert cleanup_result == False


class TestForgeCrewAgent:
    """Test cases for Forge Crew Agent"""
    
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
        """Create Forge Crew Agent for testing"""
        mock_dependencies["security_manager"].validate_task_execution = AsyncMock()
        mock_dependencies["memory_manager"].store_memory = AsyncMock()
        mock_dependencies["performance_monitor"].track_agent_execution = AsyncMock()
        
        return ForgeCrewAgent(
            role="Test Agent",
            goal="Complete test tasks efficiently",
            backstory="A test agent designed for unit testing",
            **mock_dependencies
        )
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, forge_agent):
        """Test agent initialization"""
        
        assert forge_agent.role == "Test Agent"
        assert forge_agent.goal == "Complete test tasks efficiently"
        assert forge_agent.backstory == "A test agent designed for unit testing"
        assert forge_agent.agent_id.startswith("crew_agent_")
        assert forge_agent.execution_metrics["tasks_completed"] == 0
    
    @pytest.mark.asyncio
    async def test_execute_task(self, forge_agent, mock_dependencies):
        """Test task execution"""
        
        task = {
            "description": "Analyze sales data",
            "expected_output": "Sales analysis report"
        }
        
        context = {
            "data_source": "sales_database",
            "time_period": "Q1 2024"
        }
        
        result = await forge_agent.execute_task(task, context)
        
        assert "task_id" in result
        assert result["agent_id"] == forge_agent.agent_id
        assert "result" in result
        assert "execution_time" in result
        assert result["status"] in ["completed", "failed"]
        
        # Verify security validation was called
        mock_dependencies["security_manager"].validate_task_execution.assert_called_once()
        
        # Verify memory storage was called
        assert mock_dependencies["memory_manager"].store_memory.call_count >= 1
        
        # Verify performance tracking was called
        mock_dependencies["performance_monitor"].track_agent_execution.assert_called_once()
        
        # Check metrics updated
        assert forge_agent.execution_metrics["tasks_completed"] == 1
    
    def test_get_agent_info(self, forge_agent):
        """Test getting agent information"""
        
        info = forge_agent.get_agent_info()
        
        assert info["agent_id"] == forge_agent.agent_id
        assert info["role"] == "Test Agent"
        assert info["goal"] == "Complete test tasks efficiently"
        assert info["backstory"] == "A test agent designed for unit testing"
        assert "execution_metrics" in info
        assert "crewai_available" in info


class TestForgeCrewTask:
    """Test cases for Forge Crew Task"""
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent for task testing"""
        agent = Mock()
        agent.agent_id = "test_agent_123"
        agent.execute_task = AsyncMock(return_value={
            "task_id": "test_task",
            "result": "Task completed successfully",
            "execution_time": 1.5,
            "status": "completed"
        })
        return agent
    
    @pytest.fixture
    def forge_task(self, mock_agent):
        """Create Forge Crew Task for testing"""
        return ForgeCrewTask(
            description="Write a comprehensive market analysis report",
            agent=mock_agent,
            expected_output="A detailed market analysis with insights and recommendations",
            quality_level=QualityLevel.SUPERHUMAN,
            compliance_requirements=["data_privacy", "accuracy_standards"]
        )
    
    def test_task_initialization(self, forge_task, mock_agent):
        """Test task initialization"""
        
        assert forge_task.description == "Write a comprehensive market analysis report"
        assert forge_task.agent == mock_agent
        assert forge_task.expected_output == "A detailed market analysis with insights and recommendations"
        assert forge_task.quality_level == QualityLevel.SUPERHUMAN
        assert forge_task.compliance_requirements == ["data_privacy", "accuracy_standards"]
        assert forge_task.status == CrewAITaskStatus.PENDING
        assert forge_task.task_id.startswith("crew_task_")
    
    @pytest.mark.asyncio
    async def test_execute_task(self, forge_task, mock_agent):
        """Test task execution with quality assurance"""
        
        # Mock quality assurance
        quality_assurance = Mock()
        quality_assurance.conduct_quality_review = AsyncMock(return_value={
            "quality_decision": {"approved": True, "overall_score": 0.96}
        })
        
        result = await forge_task.execute(quality_assurance)
        
        assert result["task_id"] == forge_task.task_id
        assert result["status"] == "completed"
        assert "result" in result
        assert "quality_assessment" in result
        assert result["agent_id"] == mock_agent.agent_id
        
        # Verify agent execution was called
        mock_agent.execute_task.assert_called_once()
        
        # Verify quality assurance was called
        quality_assurance.conduct_quality_review.assert_called_once()
        
        # Check task status updated
        assert forge_task.status == CrewAITaskStatus.COMPLETED
        assert forge_task.result is not None
    
    @pytest.mark.asyncio
    async def test_execute_task_quality_failure(self, forge_task, mock_agent):
        """Test task execution with quality failure"""
        
        # Mock quality assurance with failure
        quality_assurance = Mock()
        quality_assurance.conduct_quality_review = AsyncMock(return_value={
            "quality_decision": {"approved": False, "overall_score": 0.65}
        })
        
        result = await forge_task.execute(quality_assurance)
        
        assert result["task_id"] == forge_task.task_id
        assert result["status"] == "escalated"
        
        # Check task status updated to escalated
        assert forge_task.status == CrewAITaskStatus.ESCALATED
    
    def test_get_task_info(self, forge_task):
        """Test getting task information"""
        
        info = forge_task.get_task_info()
        
        assert info["task_id"] == forge_task.task_id
        assert info["description"] == forge_task.description
        assert info["expected_output"] == forge_task.expected_output
        assert info["status"] == "pending"
        assert info["quality_level"] == "superhuman"
        assert info["compliance_requirements"] == ["data_privacy", "accuracy_standards"]


class TestForgeCrew:
    """Test cases for Forge Crew"""
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for crew testing"""
        agents = []
        for i in range(2):
            agent = Mock()
            agent.agent_id = f"test_agent_{i}"
            agent.execute_task = AsyncMock(return_value={
                "task_id": f"test_task_{i}",
                "result": f"Task {i} completed",
                "execution_time": 1.0,
                "status": "completed"
            })
            agents.append(agent)
        return agents
    
    @pytest.fixture
    def mock_tasks(self, mock_agents):
        """Create mock tasks for crew testing"""
        tasks = []
        for i, agent in enumerate(mock_agents):
            task = Mock()
            task.task_id = f"test_task_{i}"
            task.agent = agent
            task.description = f"Test task {i}"
            task.status = CrewAITaskStatus.PENDING
            task.execute = AsyncMock(return_value={
                "task_id": f"test_task_{i}",
                "status": "completed",
                "result": f"Task {i} result"
            })
            tasks.append(task)
        return tasks
    
    @pytest.fixture
    def forge_crew(self, mock_agents, mock_tasks):
        """Create Forge Crew for testing"""
        return ForgeCrew(
            agents=mock_agents,
            tasks=mock_tasks,
            workflow_type=CrewAIWorkflowType.SEQUENTIAL,
            quality_assurance=Mock(),
            compliance_engine=Mock()
        )
    
    def test_crew_initialization(self, forge_crew, mock_agents, mock_tasks):
        """Test crew initialization"""
        
        assert len(forge_crew.agents) == 2
        assert len(forge_crew.tasks) == 2
        assert forge_crew.workflow_type == CrewAIWorkflowType.SEQUENTIAL
        assert forge_crew.status == "initialized"
        assert forge_crew.crew_id.startswith("crew_")
        assert forge_crew.crew_metrics["executions_completed"] == 0
    
    @pytest.mark.asyncio
    async def test_kickoff_sequential_workflow(self, forge_crew, mock_tasks):
        """Test crew kickoff with sequential workflow"""
        
        context = {"test_context": "test_value"}
        
        result = await forge_crew.kickoff(context)
        
        assert "crew_id" in result
        assert result["crew_id"] == forge_crew.crew_id
        assert "execution_id" in result
        assert result["workflow_type"] == "sequential"
        assert "execution_time" in result
        assert result["status"] in ["completed", "failed"]
        
        # Verify tasks were executed
        for task in mock_tasks:
            task.execute.assert_called_once()
        
        # Check crew status updated
        assert forge_crew.status in ["completed", "failed"]
        assert forge_crew.crew_metrics["executions_completed"] == 1
    
    def test_get_crew_info(self, forge_crew):
        """Test getting crew information"""
        
        info = forge_crew.get_crew_info()
        
        assert info["crew_id"] == forge_crew.crew_id
        assert info["workflow_type"] == "sequential"
        assert info["agents_count"] == 2
        assert info["tasks_count"] == 2
        assert info["status"] == "initialized"
        assert "crew_metrics" in info
        assert "agents" in info
        assert "tasks" in info
        assert "crewai_available" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])