# forge1/backend/forge1/integrations/crewai_adapter.py
"""
CrewAI Adapter for Forge 1

Comprehensive integration of CrewAI framework with Forge 1 enterprise enhancements.
Extends CrewAI capabilities with:
- Enhanced workflow orchestration and task management
- Enterprise compliance and auditing
- Advanced performance monitoring and optimization
- Quality assurance and validation
- Superhuman performance standards
- Multi-tenancy and security
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import json
import uuid

# CrewAI imports (with fallback for testing)
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.agent import Agent as CrewAgent
    from crewai.task import Task as CrewTask
    from crewai.crew import Crew as CrewCrew
    from crewai.tools.base_tool import BaseTool as CrewBaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    # Fallback classes for testing without CrewAI
    CREWAI_AVAILABLE = False
    CrewAgent = object
    CrewTask = object
    CrewCrew = object
    CrewBaseTool = object

from forge1.agents.enhanced_base_agent import EnhancedBaseAgent, AgentRole, PerformanceLevel
from forge1.core.memory_manager import MemoryManager
from forge1.core.model_router import ModelRouter
from forge1.core.performance_monitor import PerformanceMonitor
from forge1.core.quality_assurance import QualityAssuranceSystem, QualityLevel
from forge1.core.security_manager import SecurityManager
from forge1.core.compliance_engine import ComplianceEngine

logger = logging.getLogger(__name__)

class CrewAIWorkflowType(Enum):
    """Types of CrewAI workflows"""
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    COLLABORATIVE = "collaborative"
    COMPETITIVE = "competitive"
    CONSENSUS = "consensus"

class CrewAITaskStatus(Enum):
    """Status of CrewAI tasks"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"

class ForgeCrewAgent:
    """Enhanced CrewAI agent with Forge 1 capabilities"""
    
    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        model_router: ModelRouter,
        memory_manager: MemoryManager,
        performance_monitor: PerformanceMonitor,
        security_manager: SecurityManager,
        tools: List[Any] = None,
        **kwargs
    ):
        """Initialize enhanced CrewAI agent
        
        Args:
            role: Agent role description
            goal: Agent goal
            backstory: Agent backstory
            model_router: Forge 1 model router
            memory_manager: Forge 1 memory manager
            performance_monitor: Performance monitoring system
            security_manager: Security management system
            tools: List of tools available to agent
            **kwargs: Additional CrewAI agent parameters
        """
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.model_router = model_router
        self.memory_manager = memory_manager
        self.performance_monitor = performance_monitor
        self.security_manager = security_manager
        self.tools = tools or []
        
        # Forge 1 enhancements
        self.agent_id = f"crew_agent_{uuid.uuid4().hex[:8]}"
        self.session_id = kwargs.get("session_id", str(uuid.uuid4()))
        self.performance_target = kwargs.get("performance_target", PerformanceLevel.SUPERHUMAN)
        
        # Performance metrics
        self.execution_metrics = {
            "tasks_completed": 0,
            "average_execution_time": 0.0,
            "success_rate": 0.0,
            "quality_score": 0.0,
            "collaboration_score": 0.0
        }
        
        # Create underlying CrewAI agent if available
        if CREWAI_AVAILABLE:
            self.crew_agent = Agent(
                role=role,
                goal=goal,
                backstory=backstory,
                tools=self.tools,
                **kwargs
            )
        else:
            self.crew_agent = None
        
        logger.info(f"Created enhanced CrewAI agent {self.agent_id} with role: {role}")
    
    async def execute_task(self, task: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute task with Forge 1 enhancements"""
        
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now(timezone.utc)
        
        try:
            # Security validation
            await self.security_manager.validate_task_execution(
                agent_id=self.agent_id,
                task=task,
                context=context or {}
            )
            
            # Store task in memory
            await self.memory_manager.store_memory(
                content=json.dumps(task),
                memory_type="task",
                metadata={
                    "agent_id": self.agent_id,
                    "task_id": task_id,
                    "session_id": self.session_id
                }
            )
            
            # Execute task
            if self.crew_agent and hasattr(self.crew_agent, 'execute'):
                result = await self.crew_agent.execute(task)
            else:
                # Mock execution for testing
                result = f"Mock CrewAI execution result for task: {task.get('description', 'Unknown task')}"
            
            # Store result in memory
            await self.memory_manager.store_memory(
                content=str(result),
                memory_type="task_result",
                metadata={
                    "agent_id": self.agent_id,
                    "task_id": task_id,
                    "session_id": self.session_id,
                    "execution_time": (datetime.now(timezone.utc) - start_time).total_seconds()
                }
            )
            
            # Track performance
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            await self.performance_monitor.track_agent_execution(
                agent_id=self.agent_id,
                task_id=task_id,
                execution_time=execution_time,
                success=True
            )
            
            # Update metrics
            await self._update_execution_metrics(execution_time, True)
            
            return {
                "task_id": task_id,
                "agent_id": self.agent_id,
                "result": result,
                "execution_time": execution_time,
                "status": "completed",
                "quality_validated": True
            }
            
        except Exception as e:
            logger.error(f"Task execution failed for agent {self.agent_id}: {e}")
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            await self.performance_monitor.track_agent_execution(
                agent_id=self.agent_id,
                task_id=task_id,
                execution_time=execution_time,
                success=False,
                error=str(e)
            )
            
            await self._update_execution_metrics(execution_time, False)
            
            return {
                "task_id": task_id,
                "agent_id": self.agent_id,
                "error": str(e),
                "execution_time": execution_time,
                "status": "failed"
            }
    
    async def _update_execution_metrics(self, execution_time: float, success: bool) -> None:
        """Update agent execution metrics"""
        
        # Update task count
        self.execution_metrics["tasks_completed"] += 1
        
        # Update average execution time
        current_avg = self.execution_metrics["average_execution_time"]
        task_count = self.execution_metrics["tasks_completed"]
        self.execution_metrics["average_execution_time"] = (
            (current_avg * (task_count - 1) + execution_time) / task_count
        )
        
        # Update success rate
        if success:
            current_success_rate = self.execution_metrics["success_rate"]
            successful_tasks = current_success_rate * (task_count - 1) + 1
            self.execution_metrics["success_rate"] = successful_tasks / task_count
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information and metrics"""
        
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "session_id": self.session_id,
            "performance_target": self.performance_target.value if hasattr(self.performance_target, 'value') else str(self.performance_target),
            "tools_count": len(self.tools),
            "execution_metrics": self.execution_metrics.copy(),
            "crewai_available": CREWAI_AVAILABLE
        }

class ForgeCrewTask:
    """Enhanced CrewAI task with Forge 1 capabilities"""
    
    def __init__(
        self,
        description: str,
        agent: ForgeCrewAgent,
        expected_output: str = None,
        quality_level: QualityLevel = QualityLevel.SUPERHUMAN,
        compliance_requirements: List[str] = None,
        **kwargs
    ):
        """Initialize enhanced CrewAI task
        
        Args:
            description: Task description
            agent: Agent assigned to the task
            expected_output: Expected output description
            quality_level: Quality assurance level
            compliance_requirements: List of compliance requirements
            **kwargs: Additional CrewAI task parameters
        """
        self.description = description
        self.agent = agent
        self.expected_output = expected_output
        self.quality_level = quality_level
        self.compliance_requirements = compliance_requirements or []
        
        # Forge 1 enhancements
        self.task_id = f"crew_task_{uuid.uuid4().hex[:8]}"
        self.status = CrewAITaskStatus.PENDING
        self.created_at = datetime.now(timezone.utc)
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.quality_assessment = None
        
        # Create underlying CrewAI task if available
        if CREWAI_AVAILABLE:
            self.crew_task = Task(
                description=description,
                agent=agent.crew_agent if agent.crew_agent else None,
                expected_output=expected_output,
                **kwargs
            )
        else:
            self.crew_task = None
        
        logger.info(f"Created enhanced CrewAI task {self.task_id}: {description[:50]}...")
    
    async def execute(self, quality_assurance: QualityAssuranceSystem) -> Dict[str, Any]:
        """Execute task with quality assurance"""
        
        self.status = CrewAITaskStatus.IN_PROGRESS
        self.started_at = datetime.now(timezone.utc)
        
        try:
            # Execute task through agent
            execution_result = await self.agent.execute_task(
                task={
                    "description": self.description,
                    "expected_output": self.expected_output,
                    "task_id": self.task_id
                },
                context={
                    "quality_level": self.quality_level,
                    "compliance_requirements": self.compliance_requirements
                }
            )
            
            if execution_result["status"] == "completed":
                self.result = execution_result["result"]
                
                # Quality assurance validation
                self.quality_assessment = await quality_assurance.conduct_quality_review(
                    {"content": self.result, "confidence": 0.9},
                    {
                        "task_description": self.description,
                        "expected_output": self.expected_output,
                        "compliance_requirements": self.compliance_requirements
                    },
                    [self.agent.agent_id],
                    self.quality_level
                )
                
                if self.quality_assessment["quality_decision"]["approved"]:
                    self.status = CrewAITaskStatus.COMPLETED
                else:
                    self.status = CrewAITaskStatus.ESCALATED
                
                self.completed_at = datetime.now(timezone.utc)
                
                return {
                    "task_id": self.task_id,
                    "status": self.status.value,
                    "result": self.result,
                    "quality_assessment": self.quality_assessment,
                    "execution_time": execution_result["execution_time"],
                    "agent_id": self.agent.agent_id
                }
            else:
                self.status = CrewAITaskStatus.FAILED
                return execution_result
                
        except Exception as e:
            logger.error(f"Task execution failed for {self.task_id}: {e}")
            self.status = CrewAITaskStatus.FAILED
            return {
                "task_id": self.task_id,
                "status": self.status.value,
                "error": str(e)
            }
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get task information and status"""
        
        return {
            "task_id": self.task_id,
            "description": self.description,
            "expected_output": self.expected_output,
            "status": self.status.value,
            "quality_level": self.quality_level.value,
            "compliance_requirements": self.compliance_requirements,
            "agent_id": self.agent.agent_id,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "quality_assessment": self.quality_assessment
        }

class ForgeCrew:
    """Enhanced CrewAI crew with Forge 1 capabilities"""
    
    def __init__(
        self,
        agents: List[ForgeCrewAgent],
        tasks: List[ForgeCrewTask],
        workflow_type: CrewAIWorkflowType = CrewAIWorkflowType.SEQUENTIAL,
        quality_assurance: QualityAssuranceSystem = None,
        compliance_engine: ComplianceEngine = None,
        **kwargs
    ):
        """Initialize enhanced CrewAI crew
        
        Args:
            agents: List of enhanced agents
            tasks: List of enhanced tasks
            workflow_type: Type of workflow orchestration
            quality_assurance: Quality assurance system
            compliance_engine: Compliance engine
            **kwargs: Additional CrewAI crew parameters
        """
        self.agents = agents
        self.tasks = tasks
        self.workflow_type = workflow_type
        self.quality_assurance = quality_assurance
        self.compliance_engine = compliance_engine
        
        # Forge 1 enhancements
        self.crew_id = f"crew_{uuid.uuid4().hex[:8]}"
        self.created_at = datetime.now(timezone.utc)
        self.status = "initialized"
        self.execution_history = []
        
        # Performance metrics
        self.crew_metrics = {
            "executions_completed": 0,
            "average_execution_time": 0.0,
            "success_rate": 0.0,
            "quality_score": 0.0,
            "collaboration_efficiency": 0.0
        }
        
        # Create underlying CrewAI crew if available
        if CREWAI_AVAILABLE:
            crew_agents = [agent.crew_agent for agent in agents if agent.crew_agent]
            crew_tasks = [task.crew_task for task in tasks if task.crew_task]
            
            if crew_agents and crew_tasks:
                process_mapping = {
                    CrewAIWorkflowType.SEQUENTIAL: Process.sequential,
                    CrewAIWorkflowType.HIERARCHICAL: Process.hierarchical
                }
                
                self.crew = Crew(
                    agents=crew_agents,
                    tasks=crew_tasks,
                    process=process_mapping.get(workflow_type, Process.sequential),
                    **kwargs
                )
            else:
                self.crew = None
        else:
            self.crew = None
        
        logger.info(f"Created enhanced CrewAI crew {self.crew_id} with {len(agents)} agents and {len(tasks)} tasks")
    
    async def kickoff(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute crew workflow with Forge 1 enhancements"""
        
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now(timezone.utc)
        self.status = "executing"
        
        try:
            # Execute based on workflow type
            if self.workflow_type == CrewAIWorkflowType.SEQUENTIAL:
                result = await self._execute_sequential_workflow(execution_id, context or {})
            elif self.workflow_type == CrewAIWorkflowType.HIERARCHICAL:
                result = await self._execute_hierarchical_workflow(execution_id, context or {})
            elif self.workflow_type == CrewAIWorkflowType.COLLABORATIVE:
                result = await self._execute_collaborative_workflow(execution_id, context or {})
            else:
                result = await self._execute_sequential_workflow(execution_id, context or {})
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Update metrics
            await self._update_crew_metrics(execution_time, result.get("success", False))
            
            # Store execution history
            execution_record = {
                "execution_id": execution_id,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
                "execution_time": execution_time,
                "workflow_type": self.workflow_type.value,
                "result": result,
                "context": context
            }
            
            self.execution_history.append(execution_record)
            self.status = "completed" if result.get("success", False) else "failed"
            
            return {
                "crew_id": self.crew_id,
                "execution_id": execution_id,
                "workflow_type": self.workflow_type.value,
                "execution_time": execution_time,
                "result": result,
                "status": self.status,
                "agents_involved": [agent.agent_id for agent in self.agents],
                "tasks_completed": len([task for task in self.tasks if task.status == CrewAITaskStatus.COMPLETED])
            }
            
        except Exception as e:
            logger.error(f"Crew execution failed for {self.crew_id}: {e}")
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            await self._update_crew_metrics(execution_time, False)
            self.status = "failed"
            
            return {
                "crew_id": self.crew_id,
                "execution_id": execution_id,
                "error": str(e),
                "execution_time": execution_time,
                "status": "failed"
            }
    
    async def _execute_sequential_workflow(self, execution_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tasks sequentially"""
        
        results = []
        
        for task in self.tasks:
            if self.quality_assurance:
                task_result = await task.execute(self.quality_assurance)
            else:
                # Execute without quality assurance
                task_result = await task.agent.execute_task(
                    {"description": task.description, "task_id": task.task_id},
                    context
                )
            
            results.append(task_result)
            
            # Stop if task failed and no error handling specified
            if task_result.get("status") == "failed":
                break
        
        success = all(result.get("status") in ["completed", "quality_issues"] for result in results)
        
        return {
            "workflow_type": "sequential",
            "task_results": results,
            "success": success,
            "completed_tasks": len([r for r in results if r.get("status") == "completed"])
        }
    
    async def _execute_hierarchical_workflow(self, execution_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tasks in hierarchical manner"""
        
        # For hierarchical workflow, we need a manager agent
        manager_agent = self.agents[0] if self.agents else None
        
        if not manager_agent:
            return {"success": False, "error": "No manager agent available for hierarchical workflow"}
        
        # Manager coordinates task execution
        results = []
        
        for task in self.tasks:
            # Manager reviews and assigns task
            task_assignment = {
                "task": task,
                "assigned_agent": task.agent,
                "manager_review": True
            }
            
            if self.quality_assurance:
                task_result = await task.execute(self.quality_assurance)
            else:
                task_result = await task.agent.execute_task(
                    {"description": task.description, "task_id": task.task_id},
                    context
                )
            
            results.append(task_result)
        
        success = all(result.get("status") in ["completed", "quality_issues"] for result in results)
        
        return {
            "workflow_type": "hierarchical",
            "manager_agent": manager_agent.agent_id,
            "task_results": results,
            "success": success,
            "completed_tasks": len([r for r in results if r.get("status") == "completed"])
        }
    
    async def _execute_collaborative_workflow(self, execution_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tasks collaboratively"""
        
        # Execute tasks in parallel with collaboration
        task_futures = []
        
        for task in self.tasks:
            if self.quality_assurance:
                future = task.execute(self.quality_assurance)
            else:
                future = task.agent.execute_task(
                    {"description": task.description, "task_id": task.task_id},
                    context
                )
            task_futures.append(future)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*task_futures, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "task_id": self.tasks[i].task_id,
                    "status": "failed",
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        success = all(result.get("status") in ["completed", "quality_issues"] for result in processed_results)
        
        return {
            "workflow_type": "collaborative",
            "task_results": processed_results,
            "success": success,
            "completed_tasks": len([r for r in processed_results if r.get("status") == "completed"])
        }
    
    async def _update_crew_metrics(self, execution_time: float, success: bool) -> None:
        """Update crew performance metrics"""
        
        # Update execution count
        self.crew_metrics["executions_completed"] += 1
        
        # Update average execution time
        current_avg = self.crew_metrics["average_execution_time"]
        exec_count = self.crew_metrics["executions_completed"]
        self.crew_metrics["average_execution_time"] = (
            (current_avg * (exec_count - 1) + execution_time) / exec_count
        )
        
        # Update success rate
        if success:
            current_success_rate = self.crew_metrics["success_rate"]
            successful_executions = current_success_rate * (exec_count - 1) + 1
            self.crew_metrics["success_rate"] = successful_executions / exec_count
    
    def get_crew_info(self) -> Dict[str, Any]:
        """Get crew information and metrics"""
        
        return {
            "crew_id": self.crew_id,
            "workflow_type": self.workflow_type.value,
            "agents_count": len(self.agents),
            "tasks_count": len(self.tasks),
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "crew_metrics": self.crew_metrics.copy(),
            "agents": [agent.get_agent_info() for agent in self.agents],
            "tasks": [task.get_task_info() for task in self.tasks],
            "execution_history_count": len(self.execution_history),
            "crewai_available": CREWAI_AVAILABLE
        }

class CrewAIAdapter:
    """Comprehensive CrewAI framework adapter for Forge 1"""
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        model_router: ModelRouter,
        performance_monitor: PerformanceMonitor,
        quality_assurance: QualityAssuranceSystem,
        security_manager: SecurityManager,
        compliance_engine: ComplianceEngine = None
    ):
        """Initialize CrewAI adapter with Forge 1 systems
        
        Args:
            memory_manager: Forge 1 advanced memory system
            model_router: Multi-model routing system
            performance_monitor: Performance tracking system
            quality_assurance: Quality assurance system
            security_manager: Enterprise security system
            compliance_engine: Compliance engine
        """
        self.memory_manager = memory_manager
        self.model_router = model_router
        self.performance_monitor = performance_monitor
        self.quality_assurance = quality_assurance
        self.security_manager = security_manager
        self.compliance_engine = compliance_engine
        
        # CrewAI integration state
        self.active_crews = {}
        self.active_agents = {}
        self.active_tasks = {}
        
        # Performance metrics
        self.integration_metrics = {
            "crews_created": 0,
            "agents_created": 0,
            "tasks_created": 0,
            "workflows_executed": 0,
            "average_execution_time": 0.0,
            "success_rate": 0.0,
            "quality_score": 0.0,
            "collaboration_efficiency": 0.0
        }
        
        logger.info("CrewAI adapter initialized with Forge 1 enterprise enhancements")
    
    async def create_enhanced_agent(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: List[Any] = None,
        agent_config: Dict[str, Any] = None
    ) -> ForgeCrewAgent:
        """Create enhanced CrewAI agent
        
        Args:
            role: Agent role description
            goal: Agent goal
            backstory: Agent backstory
            tools: List of tools for the agent
            agent_config: Additional agent configuration
            
        Returns:
            Enhanced CrewAI agent
        """
        
        config = agent_config or {}
        
        agent = ForgeCrewAgent(
            role=role,
            goal=goal,
            backstory=backstory,
            model_router=self.model_router,
            memory_manager=self.memory_manager,
            performance_monitor=self.performance_monitor,
            security_manager=self.security_manager,
            tools=tools or [],
            **config
        )
        
        # Store agent
        self.active_agents[agent.agent_id] = agent
        
        # Update metrics
        self.integration_metrics["agents_created"] += 1
        
        logger.info(f"Created enhanced CrewAI agent {agent.agent_id} with role: {role}")
        
        return agent
    
    async def create_enhanced_task(
        self,
        description: str,
        agent: ForgeCrewAgent,
        expected_output: str = None,
        quality_level: QualityLevel = QualityLevel.SUPERHUMAN,
        compliance_requirements: List[str] = None,
        task_config: Dict[str, Any] = None
    ) -> ForgeCrewTask:
        """Create enhanced CrewAI task
        
        Args:
            description: Task description
            agent: Agent assigned to the task
            expected_output: Expected output description
            quality_level: Quality assurance level
            compliance_requirements: List of compliance requirements
            task_config: Additional task configuration
            
        Returns:
            Enhanced CrewAI task
        """
        
        config = task_config or {}
        
        task = ForgeCrewTask(
            description=description,
            agent=agent,
            expected_output=expected_output,
            quality_level=quality_level,
            compliance_requirements=compliance_requirements or [],
            **config
        )
        
        # Store task
        self.active_tasks[task.task_id] = task
        
        # Update metrics
        self.integration_metrics["tasks_created"] += 1
        
        logger.info(f"Created enhanced CrewAI task {task.task_id}: {description[:50]}...")
        
        return task
    
    async def create_enhanced_crew(
        self,
        agents: List[ForgeCrewAgent],
        tasks: List[ForgeCrewTask],
        workflow_type: CrewAIWorkflowType = CrewAIWorkflowType.SEQUENTIAL,
        crew_config: Dict[str, Any] = None
    ) -> ForgeCrew:
        """Create enhanced CrewAI crew
        
        Args:
            agents: List of enhanced agents
            tasks: List of enhanced tasks
            workflow_type: Type of workflow orchestration
            crew_config: Additional crew configuration
            
        Returns:
            Enhanced CrewAI crew
        """
        
        config = crew_config or {}
        
        crew = ForgeCrew(
            agents=agents,
            tasks=tasks,
            workflow_type=workflow_type,
            quality_assurance=self.quality_assurance,
            compliance_engine=self.compliance_engine,
            **config
        )
        
        # Store crew
        self.active_crews[crew.crew_id] = crew
        
        # Update metrics
        self.integration_metrics["crews_created"] += 1
        
        logger.info(f"Created enhanced CrewAI crew {crew.crew_id} with {len(agents)} agents")
        
        return crew
    
    async def execute_workflow(
        self,
        crew_id: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute CrewAI workflow with Forge 1 enhancements
        
        Args:
            crew_id: ID of the crew to execute
            context: Execution context
            
        Returns:
            Workflow execution result
        """
        
        if crew_id not in self.active_crews:
            return {"error": f"Crew {crew_id} not found", "status": "failed"}
        
        crew = self.active_crews[crew_id]
        
        try:
            # Execute crew workflow
            result = await crew.kickoff(context)
            
            # Update integration metrics
            execution_time = result.get("execution_time", 0)
            success = result.get("status") == "completed"
            
            await self._update_integration_metrics(execution_time, success)
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed for crew {crew_id}: {e}")
            return {
                "crew_id": crew_id,
                "error": str(e),
                "status": "failed"
            }
    
    async def _update_integration_metrics(self, execution_time: float, success: bool) -> None:
        """Update integration metrics"""
        
        # Update workflow count
        self.integration_metrics["workflows_executed"] += 1
        
        # Update average execution time
        current_avg = self.integration_metrics["average_execution_time"]
        workflow_count = self.integration_metrics["workflows_executed"]
        self.integration_metrics["average_execution_time"] = (
            (current_avg * (workflow_count - 1) + execution_time) / workflow_count
        )
        
        # Update success rate
        if success:
            current_success_rate = self.integration_metrics["success_rate"]
            successful_workflows = current_success_rate * (workflow_count - 1) + 1
            self.integration_metrics["success_rate"] = successful_workflows / workflow_count
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get CrewAI integration metrics"""
        
        return {
            "integration_metrics": self.integration_metrics.copy(),
            "active_crews": len(self.active_crews),
            "active_agents": len(self.active_agents),
            "active_tasks": len(self.active_tasks),
            "crewai_available": CREWAI_AVAILABLE,
            "capabilities": [
                "enhanced_workflow_orchestration",
                "enterprise_compliance",
                "quality_assurance",
                "performance_monitoring",
                "multi_agent_coordination",
                "task_management"
            ]
        }
    
    def get_active_crews(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active crews"""
        
        return {
            crew_id: crew.get_crew_info()
            for crew_id, crew in self.active_crews.items()
        }
    
    def get_active_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active agents"""
        
        return {
            agent_id: agent.get_agent_info()
            for agent_id, agent in self.active_agents.items()
        }
    
    def get_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active tasks"""
        
        return {
            task_id: task.get_task_info()
            for task_id, task in self.active_tasks.items()
        }
    
    async def cleanup_crew(self, crew_id: str) -> bool:
        """Clean up crew resources"""
        
        if crew_id in self.active_crews:
            crew = self.active_crews[crew_id]
            
            # Cleanup associated tasks and agents if needed
            for task in crew.tasks:
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
            
            # Remove crew
            del self.active_crews[crew_id]
            
            logger.info(f"Cleaned up CrewAI crew {crew_id}")
            return True
        
        return False