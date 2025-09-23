# forge1/backend/forge1/integrations/framework_adapter.py
"""
Unified Framework Compatibility Layer for Forge 1

Provides seamless framework switching and unified APIs that abstract complexity
while preserving each framework's strengths. Includes conflict resolution
mechanisms for framework interactions.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable, Type
from enum import Enum
import json
import uuid

from forge1.integrations.langchain_adapter import LangChainAdapter, LangChainIntegrationType
from forge1.integrations.crewai_adapter import CrewAIAdapter, CrewAIWorkflowType
from forge1.integrations.autogen_adapter import AutoGenAdapter, AutoGenConversationType
from forge1.integrations.haystack_adapter import HaystackLlamaIndexAdapter, DocumentProcessingType
from forge1.core.memory_manager import MemoryManager
from forge1.core.model_router import ModelRouter
from forge1.core.performance_monitor import PerformanceMonitor
from forge1.core.quality_assurance import QualityAssuranceSystem, QualityLevel
from forge1.core.security_manager import SecurityManager
from forge1.core.conflict_resolution import ConflictResolutionSystem, ConflictType

logger = logging.getLogger(__name__)

class FrameworkType(Enum):
    """Supported framework types"""
    LANGCHAIN = "langchain"
    CREWAI = "crewai"
    AUTOGEN = "autogen"
    HAYSTACK = "haystack"
    LLAMAINDEX = "llamaindex"

class UnifiedTaskType(Enum):
    """Unified task types across frameworks"""
    AGENT_EXECUTION = "agent_execution"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    CONVERSATION_MANAGEMENT = "conversation_management"
    DOCUMENT_PROCESSING = "document_processing"
    MULTI_AGENT_COORDINATION = "multi_agent_coordination"

class FrameworkCapability(Enum):
    """Framework capabilities"""
    AGENT_CREATION = "agent_creation"
    CHAIN_EXECUTION = "chain_execution"
    TOOL_INTEGRATION = "tool_integration"
    MEMORY_MANAGEMENT = "memory_management"
    CONVERSATION_HANDLING = "conversation_handling"
    DOCUMENT_PROCESSING = "document_processing"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    MULTI_AGENT_COORDINATION = "multi_agent_coordination"

class FrameworkCompatibilityLayer:
    """Unified framework compatibility layer for seamless framework switching"""
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        model_router: ModelRouter,
        performance_monitor: PerformanceMonitor,
        quality_assurance: QualityAssuranceSystem,
        security_manager: SecurityManager,
        conflict_resolution: ConflictResolutionSystem
    ):
        """Initialize unified framework compatibility layer
        
        Args:
            memory_manager: Forge 1 advanced memory system
            model_router: Multi-model routing system
            performance_monitor: Performance tracking system
            quality_assurance: Quality assurance system
            security_manager: Enterprise security system
            conflict_resolution: Conflict resolution system
        """
        self.memory_manager = memory_manager
        self.model_router = model_router
        self.performance_monitor = performance_monitor
        self.quality_assurance = quality_assurance
        self.security_manager = security_manager
        self.conflict_resolution = conflict_resolution
        
        # Initialize framework adapters
        self.framework_adapters = {}
        self._initialize_framework_adapters()
        
        # Framework capabilities mapping
        self.framework_capabilities = self._initialize_framework_capabilities()
        
        # Active framework sessions
        self.active_sessions = {}
        
        # Performance metrics
        self.compatibility_metrics = {
            "framework_switches": 0,
            "unified_tasks_executed": 0,
            "conflicts_resolved": 0,
            "average_execution_time": 0.0,
            "success_rate": 0.0,
            "framework_utilization": {}
        }
        
        logger.info("Unified Framework Compatibility Layer initialized")
    
    def _initialize_framework_adapters(self) -> None:
        """Initialize all framework adapters"""
        
        try:
            # LangChain adapter
            self.framework_adapters[FrameworkType.LANGCHAIN] = LangChainAdapter(
                memory_manager=self.memory_manager,
                model_router=self.model_router,
                performance_monitor=self.performance_monitor,
                quality_assurance=self.quality_assurance,
                security_manager=self.security_manager
            )
            
            # CrewAI adapter
            self.framework_adapters[FrameworkType.CREWAI] = CrewAIAdapter(
                memory_manager=self.memory_manager,
                model_router=self.model_router,
                performance_monitor=self.performance_monitor,
                quality_assurance=self.quality_assurance,
                security_manager=self.security_manager
            )
            
            # AutoGen adapter
            self.framework_adapters[FrameworkType.AUTOGEN] = AutoGenAdapter(
                memory_manager=self.memory_manager,
                model_router=self.model_router,
                performance_monitor=self.performance_monitor,
                quality_assurance=self.quality_assurance,
                security_manager=self.security_manager
            )
            
            # Haystack/LlamaIndex adapter
            self.framework_adapters[FrameworkType.HAYSTACK] = HaystackLlamaIndexAdapter(
                memory_manager=self.memory_manager,
                model_router=self.model_router,
                performance_monitor=self.performance_monitor,
                quality_assurance=self.quality_assurance,
                security_manager=self.security_manager
            )
            
            logger.info(f"Initialized {len(self.framework_adapters)} framework adapters")
            
        except Exception as e:
            logger.error(f"Failed to initialize framework adapters: {e}")
    
    def _initialize_framework_capabilities(self) -> Dict[FrameworkType, List[FrameworkCapability]]:
        """Initialize framework capabilities mapping"""
        
        return {
            FrameworkType.LANGCHAIN: [
                FrameworkCapability.AGENT_CREATION,
                FrameworkCapability.CHAIN_EXECUTION,
                FrameworkCapability.TOOL_INTEGRATION,
                FrameworkCapability.MEMORY_MANAGEMENT
            ],
            FrameworkType.CREWAI: [
                FrameworkCapability.MULTI_AGENT_COORDINATION,
                FrameworkCapability.WORKFLOW_ORCHESTRATION,
                FrameworkCapability.AGENT_CREATION
            ],
            FrameworkType.AUTOGEN: [
                FrameworkCapability.CONVERSATION_HANDLING,
                FrameworkCapability.MULTI_AGENT_COORDINATION,
                FrameworkCapability.AGENT_CREATION
            ],
            FrameworkType.HAYSTACK: [
                FrameworkCapability.DOCUMENT_PROCESSING,
                FrameworkCapability.TOOL_INTEGRATION
            ]
        }
    
    async def execute_unified_task(
        self,
        task_type: UnifiedTaskType,
        task_config: Dict[str, Any],
        preferred_framework: Optional[FrameworkType] = None,
        fallback_frameworks: Optional[List[FrameworkType]] = None
    ) -> Dict[str, Any]:
        """Execute task using unified API with automatic framework selection
        
        Args:
            task_type: Type of unified task
            task_config: Task configuration
            preferred_framework: Preferred framework to use
            fallback_frameworks: Fallback frameworks if preferred fails
            
        Returns:
            Unified task execution result
        """
        
        task_id = f"unified_task_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now(timezone.utc)
        
        try:
            # Select optimal framework
            selected_framework = await self._select_optimal_framework(
                task_type, task_config, preferred_framework
            )
            
            if not selected_framework:
                return {
                    "task_id": task_id,
                    "error": "No suitable framework found for task",
                    "status": "failed"
                }
            
            # Execute task with selected framework
            result = await self._execute_with_framework(
                selected_framework, task_type, task_config, task_id
            )
            
            # Handle framework conflicts if multiple results
            if isinstance(result, list) and len(result) > 1:
                result = await self._resolve_framework_conflicts(result, task_config)
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Update metrics
            await self._update_compatibility_metrics(
                selected_framework, execution_time, result.get("status") == "completed"
            )
            
            # Store execution in memory
            await self.memory_manager.store_memory(
                content=json.dumps(result),
                memory_type="unified_task_execution",
                metadata={
                    "task_id": task_id,
                    "task_type": task_type.value,
                    "framework_used": selected_framework.value,
                    "execution_time": execution_time
                }
            )
            
            return {
                "task_id": task_id,
                "task_type": task_type.value,
                "framework_used": selected_framework.value,
                "execution_time": execution_time,
                "result": result,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Unified task execution failed: {e}")
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Try fallback frameworks
            if fallback_frameworks:
                for fallback_framework in fallback_frameworks:
                    try:
                        fallback_result = await self._execute_with_framework(
                            fallback_framework, task_type, task_config, task_id
                        )
                        
                        logger.info(f"Fallback to {fallback_framework.value} succeeded")
                        
                        return {
                            "task_id": task_id,
                            "task_type": task_type.value,
                            "framework_used": fallback_framework.value,
                            "fallback_used": True,
                            "execution_time": execution_time,
                            "result": fallback_result,
                            "status": "completed"
                        }
                        
                    except Exception as fallback_error:
                        logger.warning(f"Fallback to {fallback_framework.value} failed: {fallback_error}")
                        continue
            
            return {
                "task_id": task_id,
                "task_type": task_type.value,
                "error": str(e),
                "execution_time": execution_time,
                "status": "failed"
            }
    
    async def _select_optimal_framework(
        self,
        task_type: UnifiedTaskType,
        task_config: Dict[str, Any],
        preferred_framework: Optional[FrameworkType] = None
    ) -> Optional[FrameworkType]:
        """Select optimal framework for task execution"""
        
        if preferred_framework and preferred_framework in self.framework_adapters:
            # Check if preferred framework supports the task
            if self._framework_supports_task(preferred_framework, task_type):
                return preferred_framework
        
        # Find frameworks that support the task type
        suitable_frameworks = []
        
        for framework_type, capabilities in self.framework_capabilities.items():
            if self._task_type_matches_capabilities(task_type, capabilities):
                suitable_frameworks.append(framework_type)
        
        if not suitable_frameworks:
            return None
        
        # Select based on performance metrics and task requirements
        best_framework = await self._rank_frameworks_for_task(
            suitable_frameworks, task_type, task_config
        )
        
        return best_framework
    
    def _framework_supports_task(self, framework: FrameworkType, task_type: UnifiedTaskType) -> bool:
        """Check if framework supports the task type"""
        
        framework_caps = self.framework_capabilities.get(framework, [])
        return self._task_type_matches_capabilities(task_type, framework_caps)
    
    def _task_type_matches_capabilities(
        self,
        task_type: UnifiedTaskType,
        capabilities: List[FrameworkCapability]
    ) -> bool:
        """Check if task type matches framework capabilities"""
        
        task_capability_mapping = {
            UnifiedTaskType.AGENT_EXECUTION: [FrameworkCapability.AGENT_CREATION],
            UnifiedTaskType.WORKFLOW_ORCHESTRATION: [FrameworkCapability.WORKFLOW_ORCHESTRATION],
            UnifiedTaskType.CONVERSATION_MANAGEMENT: [FrameworkCapability.CONVERSATION_HANDLING],
            UnifiedTaskType.DOCUMENT_PROCESSING: [FrameworkCapability.DOCUMENT_PROCESSING],
            UnifiedTaskType.MULTI_AGENT_COORDINATION: [FrameworkCapability.MULTI_AGENT_COORDINATION]
        }
        
        required_capabilities = task_capability_mapping.get(task_type, [])
        
        return any(cap in capabilities for cap in required_capabilities)
    
    async def _rank_frameworks_for_task(
        self,
        frameworks: List[FrameworkType],
        task_type: UnifiedTaskType,
        task_config: Dict[str, Any]
    ) -> FrameworkType:
        """Rank frameworks based on suitability for the task"""
        
        framework_scores = {}
        
        for framework in frameworks:
            score = 0.0
            
            # Base capability score
            score += 1.0
            
            # Performance history score
            utilization = self.compatibility_metrics["framework_utilization"].get(framework.value, {})
            success_rate = utilization.get("success_rate", 0.8)
            avg_time = utilization.get("average_time", 1.0)
            
            score += success_rate * 0.5
            score += (1.0 / max(avg_time, 0.1)) * 0.3
            
            # Task-specific preferences
            if task_type == UnifiedTaskType.CONVERSATION_MANAGEMENT and framework == FrameworkType.AUTOGEN:
                score += 0.5
            elif task_type == UnifiedTaskType.WORKFLOW_ORCHESTRATION and framework == FrameworkType.CREWAI:
                score += 0.5
            elif task_type == UnifiedTaskType.DOCUMENT_PROCESSING and framework == FrameworkType.HAYSTACK:
                score += 0.5
            elif task_type == UnifiedTaskType.AGENT_EXECUTION and framework == FrameworkType.LANGCHAIN:
                score += 0.3
            
            framework_scores[framework] = score
        
        # Return framework with highest score
        return max(framework_scores.items(), key=lambda x: x[1])[0]
    
    async def _execute_with_framework(
        self,
        framework: FrameworkType,
        task_type: UnifiedTaskType,
        task_config: Dict[str, Any],
        task_id: str
    ) -> Dict[str, Any]:
        """Execute task with specific framework"""
        
        adapter = self.framework_adapters.get(framework)
        if not adapter:
            raise ValueError(f"Framework adapter not available: {framework.value}")
        
        # Map unified task to framework-specific execution
        if framework == FrameworkType.LANGCHAIN:
            return await self._execute_langchain_task(adapter, task_type, task_config, task_id)
        elif framework == FrameworkType.CREWAI:
            return await self._execute_crewai_task(adapter, task_type, task_config, task_id)
        elif framework == FrameworkType.AUTOGEN:
            return await self._execute_autogen_task(adapter, task_type, task_config, task_id)
        elif framework == FrameworkType.HAYSTACK:
            return await self._execute_haystack_task(adapter, task_type, task_config, task_id)
        else:
            raise ValueError(f"Unsupported framework: {framework.value}")
    
    async def _execute_langchain_task(
        self,
        adapter: LangChainAdapter,
        task_type: UnifiedTaskType,
        task_config: Dict[str, Any],
        task_id: str
    ) -> Dict[str, Any]:
        """Execute task using LangChain adapter"""
        
        if task_type == UnifiedTaskType.AGENT_EXECUTION:
            # Create and execute LangChain agent
            agent_result = await adapter.create_enhanced_agent(
                agent_config=task_config.get("agent_config", {}),
                tools=task_config.get("tools", []),
                integration_type=LangChainIntegrationType.AGENT_EXECUTOR
            )
            
            if "error" in agent_result:
                return agent_result
            
            # Execute task with agent
            execution_result = await adapter.execute_agent_task(
                agent_id=agent_result["agent_id"],
                task=task_config.get("task", ""),
                context=task_config.get("context", {}),
                quality_level=task_config.get("quality_level", QualityLevel.SUPERHUMAN)
            )
            
            return execution_result
        
        else:
            return {"error": f"Task type {task_type.value} not supported by LangChain adapter"}
    
    async def _execute_crewai_task(
        self,
        adapter: CrewAIAdapter,
        task_type: UnifiedTaskType,
        task_config: Dict[str, Any],
        task_id: str
    ) -> Dict[str, Any]:
        """Execute task using CrewAI adapter"""
        
        if task_type == UnifiedTaskType.WORKFLOW_ORCHESTRATION:
            # Create agents and tasks
            agents = []
            for agent_config in task_config.get("agents", []):
                agent = await adapter.create_enhanced_agent(
                    role=agent_config.get("role", "Agent"),
                    goal=agent_config.get("goal", "Complete assigned tasks"),
                    backstory=agent_config.get("backstory", "Professional agent"),
                    tools=agent_config.get("tools", [])
                )
                agents.append(agent)
            
            tasks = []
            for i, task_config_item in enumerate(task_config.get("tasks", [])):
                agent = agents[i % len(agents)] if agents else None
                if agent:
                    task = await adapter.create_enhanced_task(
                        description=task_config_item.get("description", ""),
                        agent=agent,
                        expected_output=task_config_item.get("expected_output", "")
                    )
                    tasks.append(task)
            
            if not agents or not tasks:
                return {"error": "No agents or tasks created for workflow"}
            
            # Create and execute crew
            crew = await adapter.create_enhanced_crew(
                agents=agents,
                tasks=tasks,
                workflow_type=CrewAIWorkflowType.SEQUENTIAL
            )
            
            execution_result = await adapter.execute_workflow(
                crew_id=crew.crew_id,
                context=task_config.get("context", {})
            )
            
            return execution_result
        
        else:
            return {"error": f"Task type {task_type.value} not supported by CrewAI adapter"}
    
    async def _execute_autogen_task(
        self,
        adapter: AutoGenAdapter,
        task_type: UnifiedTaskType,
        task_config: Dict[str, Any],
        task_id: str
    ) -> Dict[str, Any]:
        """Execute task using AutoGen adapter"""
        
        if task_type == UnifiedTaskType.CONVERSATION_MANAGEMENT:
            # Create agents
            agents = []
            for agent_config in task_config.get("agents", []):
                agent = await adapter.create_enhanced_agent(
                    name=agent_config.get("name", "Agent"),
                    agent_type=agent_config.get("agent_type", "assistant"),
                    system_message=agent_config.get("system_message", "You are a helpful assistant.")
                )
                agents.append(agent)
            
            if len(agents) < 2:
                return {"error": "At least 2 agents required for conversation"}
            
            # Create conversation
            conversation = await adapter.create_conversation(
                conversation_type=AutoGenConversationType.TWO_AGENT,
                participants=agents,
                max_rounds=task_config.get("max_rounds", 5)
            )
            
            # Start conversation
            execution_result = await adapter.start_conversation(
                conversation_id=conversation.conversation_id,
                initial_message=task_config.get("initial_message", "Hello"),
                context=task_config.get("context", {})
            )
            
            return execution_result
        
        else:
            return {"error": f"Task type {task_type.value} not supported by AutoGen adapter"}
    
    async def _execute_haystack_task(
        self,
        adapter: HaystackLlamaIndexAdapter,
        task_type: UnifiedTaskType,
        task_config: Dict[str, Any],
        task_id: str
    ) -> Dict[str, Any]:
        """Execute task using Haystack adapter"""
        
        if task_type == UnifiedTaskType.DOCUMENT_PROCESSING:
            # Create document processor
            processor = await adapter.create_document_processor(
                processor_type=DocumentProcessingType.EXTRACTION,
                processor_config=task_config.get("processor_config", {})
            )
            
            # Process document
            execution_result = await processor.process_document(
                document_path=task_config.get("document_path", ""),
                document_format=task_config.get("document_format", "txt"),
                context=task_config.get("context", {})
            )
            
            return execution_result
        
        else:
            return {"error": f"Task type {task_type.value} not supported by Haystack adapter"}
    
    async def _resolve_framework_conflicts(
        self,
        conflicting_results: List[Dict[str, Any]],
        task_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve conflicts between framework results"""
        
        if not self.conflict_resolution:
            # Simple fallback - return first result
            return conflicting_results[0] if conflicting_results else {}
        
        # Use conflict resolution system
        agents_involved = [f"framework_{i}" for i in range(len(conflicting_results))]
        
        resolution_result = await self.conflict_resolution.detect_and_resolve_conflict(
            agents_involved=agents_involved,
            conflicting_results=conflicting_results,
            context=task_config,
            conflict_type=ConflictType.RESULT_DISAGREEMENT
        )
        
        if resolution_result["final_result"]["resolved"]:
            self.compatibility_metrics["conflicts_resolved"] += 1
            return resolution_result["final_result"]["resolution"]
        else:
            # Fallback to first result
            return conflicting_results[0] if conflicting_results else {}
    
    async def _update_compatibility_metrics(
        self,
        framework: FrameworkType,
        execution_time: float,
        success: bool
    ) -> None:
        """Update compatibility layer metrics"""
        
        # Update general metrics
        self.compatibility_metrics["unified_tasks_executed"] += 1
        
        # Update average execution time
        current_avg = self.compatibility_metrics["average_execution_time"]
        task_count = self.compatibility_metrics["unified_tasks_executed"]
        self.compatibility_metrics["average_execution_time"] = (
            (current_avg * (task_count - 1) + execution_time) / task_count
        )
        
        # Update success rate
        if success:
            current_success_rate = self.compatibility_metrics["success_rate"]
            successful_tasks = current_success_rate * (task_count - 1) + 1
            self.compatibility_metrics["success_rate"] = successful_tasks / task_count
        
        # Update framework-specific utilization
        framework_key = framework.value
        if framework_key not in self.compatibility_metrics["framework_utilization"]:
            self.compatibility_metrics["framework_utilization"][framework_key] = {
                "usage_count": 0,
                "success_count": 0,
                "total_time": 0.0,
                "average_time": 0.0,
                "success_rate": 0.0
            }
        
        fw_metrics = self.compatibility_metrics["framework_utilization"][framework_key]
        fw_metrics["usage_count"] += 1
        fw_metrics["total_time"] += execution_time
        fw_metrics["average_time"] = fw_metrics["total_time"] / fw_metrics["usage_count"]
        
        if success:
            fw_metrics["success_count"] += 1
        
        fw_metrics["success_rate"] = fw_metrics["success_count"] / fw_metrics["usage_count"]
    
    async def switch_framework(
        self,
        session_id: str,
        from_framework: FrameworkType,
        to_framework: FrameworkType,
        migration_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Switch from one framework to another for a session"""
        
        switch_id = f"switch_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now(timezone.utc)
        
        try:
            # Validate frameworks
            if from_framework not in self.framework_adapters or to_framework not in self.framework_adapters:
                return {
                    "switch_id": switch_id,
                    "error": "Invalid framework specified",
                    "status": "failed"
                }
            
            # Get session state from source framework
            session_state = await self._extract_session_state(session_id, from_framework)
            
            # Migrate state to target framework
            migration_result = await self._migrate_session_state(
                session_id, session_state, to_framework, migration_config or {}
            )
            
            # Update session tracking
            self.active_sessions[session_id] = {
                "current_framework": to_framework,
                "previous_framework": from_framework,
                "switched_at": datetime.now(timezone.utc),
                "migration_result": migration_result
            }
            
            # Update metrics
            self.compatibility_metrics["framework_switches"] += 1
            
            switch_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return {
                "switch_id": switch_id,
                "session_id": session_id,
                "from_framework": from_framework.value,
                "to_framework": to_framework.value,
                "switch_time": switch_time,
                "migration_result": migration_result,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Framework switch failed: {e}")
            switch_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return {
                "switch_id": switch_id,
                "session_id": session_id,
                "error": str(e),
                "switch_time": switch_time,
                "status": "failed"
            }
    
    async def _extract_session_state(self, session_id: str, framework: FrameworkType) -> Dict[str, Any]:
        """Extract session state from framework"""
        
        # Get relevant memories for the session
        memories = await self.memory_manager.retrieve_memories(
            query="",
            memory_types=["conversation", "task_execution", "agent_state"],
            session_id=session_id,
            limit=100
        )
        
        return {
            "session_id": session_id,
            "framework": framework.value,
            "memories": memories,
            "extracted_at": datetime.now(timezone.utc).isoformat()
        }
    
    async def _migrate_session_state(
        self,
        session_id: str,
        session_state: Dict[str, Any],
        target_framework: FrameworkType,
        migration_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Migrate session state to target framework"""
        
        # Store migration information in memory
        await self.memory_manager.store_memory(
            content=json.dumps(session_state),
            memory_type="framework_migration",
            metadata={
                "session_id": session_id,
                "target_framework": target_framework.value,
                "migration_config": migration_config
            }
        )
        
        return {
            "migrated": True,
            "target_framework": target_framework.value,
            "memories_migrated": len(session_state.get("memories", [])),
            "migration_time": datetime.now(timezone.utc).isoformat()
        }
    
    def get_compatibility_metrics(self) -> Dict[str, Any]:
        """Get compatibility layer metrics"""
        
        return {
            "compatibility_metrics": self.compatibility_metrics.copy(),
            "available_frameworks": [fw.value for fw in self.framework_adapters.keys()],
            "framework_capabilities": {
                fw.value: [cap.value for cap in caps]
                for fw, caps in self.framework_capabilities.items()
            },
            "active_sessions": len(self.active_sessions),
            "supported_task_types": [task.value for task in UnifiedTaskType]
        }
    
    def get_framework_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all frameworks"""
        
        status = {}
        
        for framework_type, adapter in self.framework_adapters.items():
            try:
                if hasattr(adapter, 'get_integration_metrics'):
                    metrics = adapter.get_integration_metrics()
                else:
                    metrics = {"status": "available"}
                
                status[framework_type.value] = {
                    "available": True,
                    "metrics": metrics,
                    "capabilities": [cap.value for cap in self.framework_capabilities.get(framework_type, [])]
                }
                
            except Exception as e:
                status[framework_type.value] = {
                    "available": False,
                    "error": str(e),
                    "capabilities": []
                }
        
        return status
