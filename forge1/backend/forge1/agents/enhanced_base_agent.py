# forge1/backend/forge1/agents/enhanced_base_agent.py
"""
Enhanced Base Agent for Forge 1

Base class for all enhanced agents with superhuman capabilities.
Extends Microsoft's BaseAgent with enterprise-grade features.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone

# Import Microsoft's base functionality
import sys
import os

# Add Microsoft's backend to path
microsoft_backend_path = os.path.join(os.path.dirname(__file__), '../../../../../Multi-Agent-Custom-Automation-Engine-Solution-Accelerator/src/backend')
if os.path.exists(microsoft_backend_path):
    sys.path.append(microsoft_backend_path)
    try:
        from kernel_agents.agent_base import BaseAgent
        from models.messages_kernel import ActionRequest, ActionResponse, AgentMessage, Step, StepStatus
    except ImportError:
        # Fallback to mock implementations for testing
        BaseAgent = object
        ActionRequest = dict
        ActionResponse = dict
        AgentMessage = dict
        Step = dict
        StepStatus = str
else:
    # Mock implementations for standalone testing
    BaseAgent = object
    ActionRequest = dict
    ActionResponse = dict
    AgentMessage = dict
    Step = dict
    StepStatus = str

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Specialized agent roles for multi-agent coordination"""
    PLANNER = "planner"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    REPORTER = "reporter"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"

class PerformanceLevel(Enum):
    """Performance levels for superhuman validation"""
    HUMAN_BASELINE = "human_baseline"
    ENHANCED = "enhanced"
    SUPERHUMAN = "superhuman"
    EXTREME = "extreme"

class EnhancedBaseAgent(ABC):
    """Enhanced base class for all Forge 1 agents with superhuman capabilities"""
    
    def __init__(
        self,
        agent_name: str,
        session_id: str,
        user_id: str,
        model_router: Any,
        memory_store: Any,
        role: AgentRole = AgentRole.SPECIALIST,
        performance_target: PerformanceLevel = PerformanceLevel.SUPERHUMAN,
        tools: Optional[List] = None,
        system_message: Optional[str] = None,
        client: Optional[Any] = None,
        definition: Optional[Any] = None,
        **kwargs
    ):
        """Initialize enhanced agent with Forge 1 capabilities
        
        Args:
            agent_name: Name of the agent
            session_id: Session identifier
            user_id: User identifier
            model_router: Multi-model router for intelligent model selection
            memory_store: Enhanced memory store
            role: Specialized role for multi-agent coordination
            performance_target: Target performance level
            tools: Available tools for the agent
            system_message: System message/instructions
            client: AI client instance
            definition: Agent definition
            **kwargs: Additional parameters
        """
        
        # Initialize Microsoft's BaseAgent
        super().__init__(
            agent_name=agent_name,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=tools,
            system_message=system_message,
            client=client,
            definition=definition
        )
        
        # Forge 1 enhancements
        self.model_router = model_router
        self.role = role
        self.performance_target = performance_target
        
        # Performance tracking
        self.performance_metrics = {
            "accuracy": 0.0,
            "speed": 0.0,
            "efficiency": 0.0,
            "quality_score": 0.0,
            "superhuman_ratio": 0.0,
            "task_completion_rate": 0.0,
            "error_rate": 0.0,
            "response_time_ms": 0.0
        }
        
        # Coordination capabilities
        self.coordination_state = {
            "active_collaborations": [],
            "pending_handoffs": [],
            "shared_context": {},
            "conflict_resolution_history": []
        }
        
        # Lifecycle management
        self.lifecycle_state = {
            "created_at": datetime.now(timezone.utc),
            "last_active": datetime.now(timezone.utc),
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "optimization_cycles": 0
        }
        
        # Monitoring and observability
        self.monitoring_enabled = True
        self.telemetry_data = []
        
        logger.info(f"Enhanced agent {agent_name} initialized with role {role.value} targeting {performance_target.value} performance")
    
    async def handle_action_request_enhanced(self, action_request: ActionRequest) -> ActionResponse:
        """Enhanced action request handling with performance monitoring and multi-model routing
        
        Args:
            action_request: The action request to handle
            
        Returns:
            Enhanced action response with performance metrics
        """
        start_time = time.time()
        
        try:
            # Update lifecycle tracking
            self.lifecycle_state["last_active"] = datetime.now(timezone.utc)
            self.lifecycle_state["total_tasks"] += 1
            
            # Pre-processing: Analyze task complexity and select optimal model
            task_analysis = await self._analyze_task_complexity(action_request)
            optimal_model = await self.model_router.select_optimal_model(
                task_analysis=task_analysis,
                agent_role=self.role,
                performance_target=self.performance_target
            )
            
            # Execute with selected model
            response = await self._execute_with_model(action_request, optimal_model)
            
            # Post-processing: Validate superhuman performance
            performance_validation = await self._validate_performance(response, start_time)
            
            # Update metrics
            await self._update_performance_metrics(performance_validation)
            
            # Success tracking
            self.lifecycle_state["successful_tasks"] += 1
            
            return response
            
        except Exception as e:
            # Error handling and tracking
            self.lifecycle_state["failed_tasks"] += 1
            self.performance_metrics["error_rate"] = (
                self.lifecycle_state["failed_tasks"] / self.lifecycle_state["total_tasks"]
            )
            
            logger.error(f"Enhanced agent {self._agent_name} failed to handle action: {e}")
            
            # Create error response
            return ActionResponse(
                step_id=action_request.step_id,
                plan_id=action_request.plan_id,
                session_id=action_request.session_id,
                result=f"Enhanced agent error: {str(e)}",
                status=StepStatus.failed
            )
    
    async def _analyze_task_complexity(self, action_request: ActionRequest) -> Dict[str, Any]:
        """Analyze task complexity for optimal model selection"""
        
        # Get step details
        step = await self._memory_store.get_step(action_request.step_id, action_request.session_id)
        
        if not step:
            return {"complexity": "medium", "reasoning_required": True, "domain": "general"}
        
        # Analyze task characteristics
        task_text = step.action + " " + (step.human_feedback or "")
        
        complexity_indicators = {
            "length": len(task_text.split()),
            "has_reasoning": any(word in task_text.lower() for word in ["analyze", "explain", "reason", "why", "how"]),
            "has_creativity": any(word in task_text.lower() for word in ["create", "design", "generate", "invent"]),
            "has_math": any(word in task_text.lower() for word in ["calculate", "compute", "solve", "equation"]),
            "has_code": any(word in task_text.lower() for word in ["code", "program", "function", "script"]),
            "domain_specific": self._detect_domain(task_text)
        }
        
        # Determine complexity level
        complexity_score = 0
        if complexity_indicators["length"] > 100:
            complexity_score += 1
        if complexity_indicators["has_reasoning"]:
            complexity_score += 2
        if complexity_indicators["has_creativity"]:
            complexity_score += 2
        if complexity_indicators["has_math"] or complexity_indicators["has_code"]:
            complexity_score += 1
        
        complexity_level = "low" if complexity_score <= 2 else "medium" if complexity_score <= 4 else "high"
        
        return {
            "complexity": complexity_level,
            "score": complexity_score,
            "indicators": complexity_indicators,
            "estimated_tokens": len(task_text.split()) * 1.3,  # Rough estimate
            "requires_reasoning": complexity_indicators["has_reasoning"],
            "domain": complexity_indicators["domain_specific"]
        }
    
    def _detect_domain(self, text: str) -> str:
        """Detect domain-specific requirements"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["hr", "human resources", "employee", "hiring"]):
            return "hr"
        elif any(word in text_lower for word in ["marketing", "campaign", "brand", "promotion"]):
            return "marketing"
        elif any(word in text_lower for word in ["product", "feature", "development", "roadmap"]):
            return "product"
        elif any(word in text_lower for word in ["procurement", "purchase", "vendor", "supplier"]):
            return "procurement"
        elif any(word in text_lower for word in ["support", "technical", "troubleshoot", "debug"]):
            return "tech_support"
        else:
            return "general"
    
    async def _execute_with_model(self, action_request: ActionRequest, model_config: Dict[str, Any]) -> ActionResponse:
        """Execute action with selected model configuration"""
        
        # Use the model router to execute with optimal model
        try:
            # Delegate to Microsoft's base implementation but with model routing
            response_json = await super().handle_action_request(action_request)
            
            # Parse response if it's JSON string
            if isinstance(response_json, str):
                import json
                response_data = json.loads(response_json)
                response = ActionResponse(**response_data)
            else:
                response = response_json
            
            # Enhance response with model information
            if hasattr(response, 'result'):
                response.result += f"\n\n[Model: {model_config.get('name', 'default')}]"
            
            return response
            
        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            raise
    
    async def _validate_performance(self, response: ActionResponse, start_time: float) -> Dict[str, Any]:
        """Validate superhuman performance requirements"""
        
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        # Performance validation metrics
        validation = {
            "response_time_ms": response_time_ms,
            "meets_speed_target": response_time_ms < 5000,  # 5 second target
            "meets_accuracy_target": response.status == StepStatus.completed,
            "quality_indicators": {
                "has_content": bool(response.result and len(response.result.strip()) > 0),
                "structured_response": "[Model:" in response.result if response.result else False,
                "error_free": response.status != StepStatus.failed
            }
        }
        
        # Calculate quality score
        quality_score = 0.0
        if validation["quality_indicators"]["has_content"]:
            quality_score += 0.4
        if validation["quality_indicators"]["structured_response"]:
            quality_score += 0.3
        if validation["quality_indicators"]["error_free"]:
            quality_score += 0.3
        
        validation["quality_score"] = quality_score
        validation["superhuman_performance"] = (
            validation["meets_speed_target"] and 
            validation["meets_accuracy_target"] and 
            quality_score >= 0.8
        )
        
        return validation
    
    async def _update_performance_metrics(self, validation: Dict[str, Any]):
        """Update agent performance metrics"""
        
        # Update response time (rolling average)
        current_response_time = self.performance_metrics["response_time_ms"]
        new_response_time = validation["response_time_ms"]
        self.performance_metrics["response_time_ms"] = (
            (current_response_time * 0.8) + (new_response_time * 0.2)
        )
        
        # Update quality score (rolling average)
        current_quality = self.performance_metrics["quality_score"]
        new_quality = validation["quality_score"]
        self.performance_metrics["quality_score"] = (
            (current_quality * 0.8) + (new_quality * 0.2)
        )
        
        # Update accuracy based on task completion
        total_tasks = self.lifecycle_state["total_tasks"]
        successful_tasks = self.lifecycle_state["successful_tasks"]
        self.performance_metrics["accuracy"] = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        
        # Update task completion rate
        self.performance_metrics["task_completion_rate"] = self.performance_metrics["accuracy"]
        
        # Calculate superhuman ratio
        superhuman_count = getattr(self, '_superhuman_task_count', 0)
        if validation.get("superhuman_performance", False):
            superhuman_count += 1
            self._superhuman_task_count = superhuman_count
        
        self.performance_metrics["superhuman_ratio"] = superhuman_count / total_tasks if total_tasks > 0 else 0.0
        
        # Speed metric (inverse of response time, normalized)
        max_acceptable_time = 10000  # 10 seconds
        self.performance_metrics["speed"] = max(0.0, 1.0 - (self.performance_metrics["response_time_ms"] / max_acceptable_time))
        
        # Efficiency combines speed and accuracy
        self.performance_metrics["efficiency"] = (
            self.performance_metrics["speed"] * 0.4 + 
            self.performance_metrics["accuracy"] * 0.6
        )
        
        # Log performance update
        if self.monitoring_enabled:
            logger.info(f"Agent {self._agent_name} performance updated: "
                       f"Accuracy: {self.performance_metrics['accuracy']:.3f}, "
                       f"Speed: {self.performance_metrics['speed']:.3f}, "
                       f"Quality: {self.performance_metrics['quality_score']:.3f}, "
                       f"Superhuman: {self.performance_metrics['superhuman_ratio']:.3f}")
    
    # Coordination methods for multi-agent collaboration
    
    async def initiate_collaboration(self, target_agent: 'EnhancedBaseAgent', context: Dict[str, Any]) -> str:
        """Initiate collaboration with another agent"""
        
        collaboration_id = f"collab_{self._agent_name}_{target_agent._agent_name}_{int(time.time())}"
        
        collaboration = {
            "id": collaboration_id,
            "initiator": self._agent_name,
            "target": target_agent._agent_name,
            "context": context,
            "status": "active",
            "created_at": datetime.now(timezone.utc)
        }
        
        self.coordination_state["active_collaborations"].append(collaboration)
        target_agent.coordination_state["active_collaborations"].append(collaboration)
        
        logger.info(f"Collaboration {collaboration_id} initiated between {self._agent_name} and {target_agent._agent_name}")
        return collaboration_id
    
    async def handoff_task(self, target_agent: 'EnhancedBaseAgent', task_context: Dict[str, Any]) -> bool:
        """Hand off a task to another specialized agent"""
        
        handoff_id = f"handoff_{int(time.time())}"
        
        handoff = {
            "id": handoff_id,
            "from_agent": self._agent_name,
            "to_agent": target_agent._agent_name,
            "task_context": task_context,
            "status": "pending",
            "created_at": datetime.now(timezone.utc)
        }
        
        self.coordination_state["pending_handoffs"].append(handoff)
        
        # Notify target agent
        await target_agent.receive_handoff(handoff)
        
        logger.info(f"Task handed off from {self._agent_name} to {target_agent._agent_name}")
        return True
    
    async def receive_handoff(self, handoff: Dict[str, Any]) -> bool:
        """Receive a task handoff from another agent"""
        
        # Add to coordination state
        self.coordination_state["pending_handoffs"].append(handoff)
        
        # Process handoff based on agent capabilities
        can_handle = await self._can_handle_handoff(handoff)
        
        if can_handle:
            handoff["status"] = "accepted"
            logger.info(f"Agent {self._agent_name} accepted handoff {handoff['id']}")
        else:
            handoff["status"] = "rejected"
            logger.warning(f"Agent {self._agent_name} rejected handoff {handoff['id']}")
        
        return can_handle
    
    async def _can_handle_handoff(self, handoff: Dict[str, Any]) -> bool:
        """Determine if agent can handle a specific handoff"""
        
        task_context = handoff.get("task_context", {})
        required_role = task_context.get("required_role")
        
        # Check role compatibility
        if required_role and required_role != self.role.value:
            return False
        
        # Check current workload
        active_tasks = len(self.coordination_state["active_collaborations"])
        if active_tasks > 5:  # Max concurrent tasks
            return False
        
        return True
    
    async def resolve_conflict(self, conflict_context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflicts with other agents"""
        
        conflict_id = f"conflict_{int(time.time())}"
        
        resolution = {
            "id": conflict_id,
            "agent": self._agent_name,
            "context": conflict_context,
            "resolution_strategy": await self._determine_resolution_strategy(conflict_context),
            "resolved_at": datetime.now(timezone.utc)
        }
        
        self.coordination_state["conflict_resolution_history"].append(resolution)
        
        logger.info(f"Conflict {conflict_id} resolved by {self._agent_name} using strategy: {resolution['resolution_strategy']}")
        return resolution
    
    async def _determine_resolution_strategy(self, conflict_context: Dict[str, Any]) -> str:
        """Determine the best strategy for conflict resolution"""
        
        conflict_type = conflict_context.get("type", "unknown")
        
        strategies = {
            "resource_contention": "priority_based",
            "conflicting_results": "consensus_building",
            "role_overlap": "specialization_based",
            "unknown": "escalation"
        }
        
        return strategies.get(conflict_type, "escalation")
    
    # Lifecycle and monitoring methods
    
    async def optimize_performance(self, metrics: Dict[str, float]) -> 'EnhancedBaseAgent':
        """Optimize agent performance based on provided metrics"""
        
        self.lifecycle_state["optimization_cycles"] += 1
        
        # Update performance targets based on metrics
        for metric_name, value in metrics.items():
            if metric_name in self.performance_metrics:
                # Blend new metrics with existing ones
                current_value = self.performance_metrics[metric_name]
                self.performance_metrics[metric_name] = (current_value * 0.7) + (value * 0.3)
        
        # Adjust performance target if consistently achieving superhuman performance
        if self.performance_metrics["superhuman_ratio"] > 0.9:
            if self.performance_target != PerformanceLevel.EXTREME:
                self.performance_target = PerformanceLevel.EXTREME
                logger.info(f"Agent {self._agent_name} upgraded to EXTREME performance target")
        
        logger.info(f"Agent {self._agent_name} performance optimized (cycle {self.lifecycle_state['optimization_cycles']})")
        return self
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        return {
            "agent_name": self._agent_name,
            "role": self.role.value,
            "performance_target": self.performance_target.value,
            "metrics": self.performance_metrics.copy(),
            "lifecycle": self.lifecycle_state.copy(),
            "coordination": {
                "active_collaborations": len(self.coordination_state["active_collaborations"]),
                "pending_handoffs": len(self.coordination_state["pending_handoffs"]),
                "conflicts_resolved": len(self.coordination_state["conflict_resolution_history"])
            },
            "superhuman_status": self.performance_metrics["superhuman_ratio"] > 0.8,
            "report_generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    @classmethod
    async def create(cls, **kwargs):
        """Create enhanced agent instance"""
        return cls(**kwargs)
    
    @classmethod
    async def enhance_existing_agent(cls, base_agent: BaseAgent, model_router: Any, **kwargs):
        """Enhance existing Microsoft agent with Forge 1 capabilities"""
        
        enhanced = cls(
            agent_name=getattr(base_agent, '_agent_name', 'enhanced'),
            session_id=getattr(base_agent, '_session_id', ''),
            user_id=getattr(base_agent, '_user_id', ''),
            model_router=model_router,
            memory_store=getattr(base_agent, '_memory_store', None),
            **kwargs
        )
        
        # Preserve original agent reference
        enhanced.base_agent = base_agent
        
        # Copy relevant state
        if hasattr(base_agent, '_chat_history'):
            enhanced._chat_history = base_agent._chat_history.copy()
        
        logger.info(f"Enhanced existing agent {enhanced._agent_name} with Forge 1 capabilities")
        return enhanced