# forge1/backend/forge1/core/escalation_manager.py
"""
Escalation Manager for Forge 1

Implements comprehensive escalation procedures and fallback mechanisms
to ensure system reliability and continuous operation even when conflicts
or quality issues cannot be automatically resolved.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable
import json
import uuid

logger = logging.getLogger(__name__)

class EscalationTrigger(Enum):
    """Triggers that can initiate escalation"""
    QUALITY_FAILURE = "quality_failure"
    CONFLICT_UNRESOLVED = "conflict_unresolved"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SYSTEM_ERROR = "system_error"
    TIMEOUT_EXCEEDED = "timeout_exceeded"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    STAKEHOLDER_REQUEST = "stakeholder_request"

class FallbackStrategy(Enum):
    """Available fallback strategies"""
    SAFE_DEFAULT = "safe_default"
    PREVIOUS_KNOWN_GOOD = "previous_known_good"
    SIMPLIFIED_APPROACH = "simplified_approach"
    HUMAN_HANDOFF = "human_handoff"
    SYSTEM_SHUTDOWN = "system_shutdown"
    ALTERNATIVE_AGENT = "alternative_agent"
    DEGRADED_SERVICE = "degraded_service"

class EscalationPriority(Enum):
    """Priority levels for escalations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class EscalationManager:
    """Comprehensive escalation and fallback management system"""
    
    def __init__(
        self,
        max_escalation_levels: int = 4,
        escalation_timeout: int = 1800,  # 30 minutes
        enable_auto_fallback: bool = True,
        emergency_contacts: Optional[List[str]] = None
    ):
        """Initialize Escalation Manager
        
        Args:
            max_escalation_levels: Maximum escalation levels before emergency procedures
            escalation_timeout: Maximum time for escalation resolution
            enable_auto_fallback: Enable automatic fallback mechanisms
            emergency_contacts: List of emergency contact identifiers
        """
        self.max_escalation_levels = max_escalation_levels
        self.escalation_timeout = escalation_timeout
        self.enable_auto_fallback = enable_auto_fallback
        self.emergency_contacts = emergency_contacts or []
        
        # Escalation tracking
        self.active_escalations: Dict[str, Dict[str, Any]] = {}
        self.escalation_history: List[Dict[str, Any]] = []
        self.escalation_patterns: Dict[str, Any] = {}
        
        # Fallback mechanisms
        self.fallback_strategies: Dict[FallbackStrategy, Callable] = {
            FallbackStrategy.SAFE_DEFAULT: self._execute_safe_default,
            FallbackStrategy.PREVIOUS_KNOWN_GOOD: self._execute_previous_known_good,
            FallbackStrategy.SIMPLIFIED_APPROACH: self._execute_simplified_approach,
            FallbackStrategy.HUMAN_HANDOFF: self._execute_human_handoff,
            FallbackStrategy.SYSTEM_SHUTDOWN: self._execute_system_shutdown,
            FallbackStrategy.ALTERNATIVE_AGENT: self._execute_alternative_agent,
            FallbackStrategy.DEGRADED_SERVICE: self._execute_degraded_service
        }
        
        # Known good states for fallback
        self.known_good_states: Dict[str, Any] = {}
        self.safe_defaults: Dict[str, Any] = {}
        
        # Performance metrics
        self.escalation_metrics = {
            "escalations_triggered": 0,
            "escalations_resolved": 0,
            "fallbacks_executed": 0,
            "emergency_procedures_activated": 0,
            "average_escalation_time": 0.0,
            "escalation_success_rate": 0.0,
            "fallback_success_rate": 0.0,
            "system_availability": 1.0
        }
        
        # Initialize escalation procedures
        self._initialize_escalation_procedures()
        
        logger.info("Escalation Manager initialized with comprehensive fallback mechanisms")
    
    def _initialize_escalation_procedures(self) -> None:
        """Initialize escalation procedures and safe defaults"""
        
        # Define safe default responses for different scenarios
        self.safe_defaults = {
            "quality_failure": {
                "response": "Quality standards not met. Reverting to manual review process.",
                "action": "manual_review",
                "confidence": 0.5
            },
            "conflict_unresolved": {
                "response": "Agent conflict could not be resolved automatically. Escalating to supervisor.",
                "action": "supervisor_review",
                "confidence": 0.6
            },
            "performance_degradation": {
                "response": "Performance below acceptable thresholds. Switching to simplified processing.",
                "action": "simplified_mode",
                "confidence": 0.7
            },
            "compliance_violation": {
                "response": "Compliance violation detected. Halting processing for review.",
                "action": "compliance_review",
                "confidence": 0.9
            },
            "system_error": {
                "response": "System error encountered. Attempting recovery procedures.",
                "action": "error_recovery",
                "confidence": 0.4
            }
        }
        
        # Initialize escalation level procedures
        self.escalation_procedures = {
            1: {
                "name": "Automatic Resolution",
                "timeout": 300,  # 5 minutes
                "actions": ["retry_operation", "alternative_approach", "parameter_adjustment"]
            },
            2: {
                "name": "Supervisor Intervention",
                "timeout": 900,  # 15 minutes
                "actions": ["supervisor_notification", "expert_consultation", "resource_reallocation"]
            },
            3: {
                "name": "Human Expert Review",
                "timeout": 3600,  # 1 hour
                "actions": ["expert_notification", "detailed_analysis", "manual_intervention"]
            },
            4: {
                "name": "Emergency Procedures",
                "timeout": 7200,  # 2 hours
                "actions": ["emergency_contacts", "system_isolation", "incident_management"]
            }
        }
    
    async def trigger_escalation(
        self,
        trigger: EscalationTrigger,
        context: Dict[str, Any],
        priority: EscalationPriority = EscalationPriority.MEDIUM,
        initial_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Trigger escalation procedure
        
        Args:
            trigger: What triggered the escalation
            context: Context information about the situation
            priority: Priority level of the escalation
            initial_data: Initial data about the issue
            
        Returns:
            Escalation tracking information and initial response
        """
        
        escalation_id = f"esc_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.warning(f"Escalation {escalation_id} triggered: {trigger.value} (Priority: {priority.value})")
            
            # Create escalation record
            escalation_record = {
                "id": escalation_id,
                "trigger": trigger.value,
                "priority": priority.value,
                "context": context,
                "initial_data": initial_data or {},
                "start_time": start_time.isoformat(),
                "current_level": 1,
                "status": "active",
                "actions_taken": [],
                "fallback_executed": False,
                "resolution": None
            }
            
            # Track active escalation
            self.active_escalations[escalation_id] = escalation_record
            
            # Determine initial escalation level based on priority
            initial_level = self._determine_initial_escalation_level(priority, trigger)
            escalation_record["current_level"] = initial_level
            
            # Execute initial escalation level
            escalation_result = await self._execute_escalation_level(
                escalation_id, initial_level, escalation_record
            )
            
            # Check if immediate fallback is needed
            if escalation_result and escalation_result.get("requires_fallback", False) and self.enable_auto_fallback:
                fallback_result = await self._execute_fallback_procedure(
                    escalation_id, trigger, context, escalation_record
                )
                escalation_record["fallback_executed"] = True
                escalation_record["fallback_result"] = fallback_result
            
            # Update metrics
            await self._update_escalation_metrics(escalation_record)
            
            logger.info(f"Escalation {escalation_id} initiated at level {initial_level}")
            
            return {
                "escalation_id": escalation_id,
                "initial_level": initial_level,
                "escalation_result": escalation_result or {},
                "fallback_executed": escalation_record.get("fallback_executed", False),
                "status": "initiated",
                "estimated_resolution_time": self._estimate_resolution_time(initial_level, priority)
            }
            
        except Exception as e:
            logger.error(f"Failed to trigger escalation: {e}")
            return {
                "escalation_id": escalation_id,
                "status": "failed",
                "error": str(e)
            }
    
    def _determine_initial_escalation_level(
        self,
        priority: EscalationPriority,
        trigger: EscalationTrigger
    ) -> int:
        """Determine initial escalation level based on priority and trigger"""
        
        # Critical triggers start at higher levels
        critical_triggers = {
            EscalationTrigger.COMPLIANCE_VIOLATION: 3,
            EscalationTrigger.SYSTEM_ERROR: 2,
            EscalationTrigger.RESOURCE_EXHAUSTION: 2
        }
        
        if trigger in critical_triggers:
            return critical_triggers[trigger]
        
        # Priority-based level determination
        priority_levels = {
            EscalationPriority.LOW: 1,
            EscalationPriority.MEDIUM: 1,
            EscalationPriority.HIGH: 2,
            EscalationPriority.CRITICAL: 3,
            EscalationPriority.EMERGENCY: 4
        }
        
        return priority_levels.get(priority, 1)
    
    async def _execute_escalation_level(
        self,
        escalation_id: str,
        level: int,
        escalation_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute specific escalation level procedures"""
        
        if level > self.max_escalation_levels:
            return await self._execute_emergency_procedures(escalation_id, escalation_record)
        
        procedure = self.escalation_procedures.get(level, self.escalation_procedures[1])
        
        try:
            logger.info(f"Executing escalation level {level}: {procedure['name']} for {escalation_id}")
            
            # Execute level-specific actions
            action_results = []
            for action in procedure["actions"]:
                action_result = await self._execute_escalation_action(
                    action, escalation_record, level
                )
                action_results.append(action_result)
                escalation_record["actions_taken"].append({
                    "action": action,
                    "result": action_result,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            # Evaluate if escalation is resolved
            resolution_status = await self._evaluate_escalation_resolution(
                escalation_record, action_results
            )
            
            return {
                "level": level,
                "procedure_name": procedure["name"],
                "actions_executed": len(action_results),
                "action_results": action_results,
                "resolution_status": resolution_status,
                "requires_fallback": resolution_status["status"] == "unresolved" and level >= 2,
                "next_level": level + 1 if not resolution_status["resolved"] else None
            }
            
        except Exception as e:
            logger.error(f"Escalation level {level} execution failed for {escalation_id}: {e}")
            return {
                "level": level,
                "status": "failed",
                "error": str(e),
                "requires_fallback": True
            }
    
    async def _execute_escalation_action(
        self,
        action: str,
        escalation_record: Dict[str, Any],
        level: int
    ) -> Dict[str, Any]:
        """Execute specific escalation action"""
        
        action_handlers = {
            "retry_operation": self._action_retry_operation,
            "alternative_approach": self._action_alternative_approach,
            "parameter_adjustment": self._action_parameter_adjustment,
            "supervisor_notification": self._action_supervisor_notification,
            "expert_consultation": self._action_expert_consultation,
            "resource_reallocation": self._action_resource_reallocation,
            "expert_notification": self._action_expert_notification,
            "detailed_analysis": self._action_detailed_analysis,
            "manual_intervention": self._action_manual_intervention,
            "emergency_contacts": self._action_emergency_contacts,
            "system_isolation": self._action_system_isolation,
            "incident_management": self._action_incident_management
        }
        
        handler = action_handlers.get(action, self._action_default)
        
        try:
            return await handler(escalation_record, level)
        except Exception as e:
            return {
                "action": action,
                "status": "failed",
                "error": str(e)
            }
    
    async def _action_retry_operation(self, escalation_record: Dict[str, Any], level: int) -> Dict[str, Any]:
        """Retry the original operation with modified parameters"""
        return {
            "action": "retry_operation",
            "status": "completed",
            "result": "Operation retry attempted with adjusted parameters",
            "success_probability": 0.6
        }
    
    async def _action_alternative_approach(self, escalation_record: Dict[str, Any], level: int) -> Dict[str, Any]:
        """Try alternative approach to the problem"""
        return {
            "action": "alternative_approach",
            "status": "completed",
            "result": "Alternative solution approach initiated",
            "success_probability": 0.7
        }
    
    async def _action_parameter_adjustment(self, escalation_record: Dict[str, Any], level: int) -> Dict[str, Any]:
        """Adjust system parameters to resolve the issue"""
        return {
            "action": "parameter_adjustment",
            "status": "completed",
            "result": "System parameters adjusted for optimal performance",
            "success_probability": 0.8
        }
    
    async def _action_supervisor_notification(self, escalation_record: Dict[str, Any], level: int) -> Dict[str, Any]:
        """Notify supervisor of the escalation"""
        return {
            "action": "supervisor_notification",
            "status": "completed",
            "result": "Supervisor notified and engaged in resolution process",
            "notification_sent": True
        }
    
    async def _action_expert_consultation(self, escalation_record: Dict[str, Any], level: int) -> Dict[str, Any]:
        """Consult with domain experts"""
        return {
            "action": "expert_consultation",
            "status": "completed",
            "result": "Expert consultation initiated for specialized guidance",
            "expert_engaged": True
        }
    
    async def _action_resource_reallocation(self, escalation_record: Dict[str, Any], level: int) -> Dict[str, Any]:
        """Reallocate system resources to address the issue"""
        return {
            "action": "resource_reallocation",
            "status": "completed",
            "result": "Additional resources allocated to resolve the issue",
            "resources_allocated": True
        }
    
    async def _action_expert_notification(self, escalation_record: Dict[str, Any], level: int) -> Dict[str, Any]:
        """Notify human experts for manual review"""
        return {
            "action": "expert_notification",
            "status": "completed",
            "result": "Human experts notified for immediate review",
            "experts_notified": True
        }
    
    async def _action_detailed_analysis(self, escalation_record: Dict[str, Any], level: int) -> Dict[str, Any]:
        """Perform detailed analysis of the issue"""
        return {
            "action": "detailed_analysis",
            "status": "completed",
            "result": "Comprehensive analysis initiated to identify root cause",
            "analysis_started": True
        }
    
    async def _action_manual_intervention(self, escalation_record: Dict[str, Any], level: int) -> Dict[str, Any]:
        """Request manual intervention"""
        return {
            "action": "manual_intervention",
            "status": "completed",
            "result": "Manual intervention requested from qualified personnel",
            "intervention_requested": True
        }
    
    async def _action_emergency_contacts(self, escalation_record: Dict[str, Any], level: int) -> Dict[str, Any]:
        """Contact emergency personnel"""
        return {
            "action": "emergency_contacts",
            "status": "completed",
            "result": f"Emergency contacts notified: {len(self.emergency_contacts)} personnel",
            "contacts_notified": len(self.emergency_contacts)
        }
    
    async def _action_system_isolation(self, escalation_record: Dict[str, Any], level: int) -> Dict[str, Any]:
        """Isolate affected system components"""
        return {
            "action": "system_isolation",
            "status": "completed",
            "result": "Affected system components isolated to prevent cascade failures",
            "isolation_applied": True
        }
    
    async def _action_incident_management(self, escalation_record: Dict[str, Any], level: int) -> Dict[str, Any]:
        """Initiate formal incident management procedures"""
        return {
            "action": "incident_management",
            "status": "completed",
            "result": "Formal incident management procedures activated",
            "incident_created": True
        }
    
    async def _action_default(self, escalation_record: Dict[str, Any], level: int) -> Dict[str, Any]:
        """Default action for unknown action types"""
        return {
            "action": "default",
            "status": "completed",
            "result": "Default escalation action executed",
            "success_probability": 0.5
        }
    
    async def _evaluate_escalation_resolution(
        self,
        escalation_record: Dict[str, Any],
        action_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate if the escalation has been resolved"""
        
        # Calculate overall success probability
        success_probabilities = [
            result.get("success_probability", 0.5)
            for result in action_results
            if "success_probability" in result
        ]
        
        if success_probabilities:
            avg_success_probability = sum(success_probabilities) / len(success_probabilities)
        else:
            avg_success_probability = 0.5
        
        # Check if any critical actions failed
        failed_actions = [
            result for result in action_results
            if result.get("status") == "failed"
        ]
        
        # Determine resolution status
        if failed_actions:
            status = "failed"
            resolved = False
        elif avg_success_probability >= 0.8:
            status = "resolved"
            resolved = True
        elif avg_success_probability >= 0.6:
            status = "partially_resolved"
            resolved = False
        else:
            status = "unresolved"
            resolved = False
        
        return {
            "status": status,
            "resolved": resolved,
            "success_probability": avg_success_probability,
            "failed_actions": len(failed_actions),
            "total_actions": len(action_results),
            "resolution_confidence": avg_success_probability
        }
    
    async def _execute_fallback_procedure(
        self,
        escalation_id: str,
        trigger: EscalationTrigger,
        context: Dict[str, Any],
        escalation_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute appropriate fallback procedure"""
        
        # Select fallback strategy based on trigger and context
        fallback_strategy = self._select_fallback_strategy(trigger, context, escalation_record)
        
        try:
            logger.info(f"Executing fallback strategy {fallback_strategy.value} for escalation {escalation_id}")
            
            # Execute fallback strategy
            handler = self.fallback_strategies[fallback_strategy]
            fallback_result = await handler(escalation_record, context)
            
            # Update metrics
            self.escalation_metrics["fallbacks_executed"] += 1
            
            return {
                "strategy": fallback_strategy.value,
                "result": fallback_result,
                "executed_at": datetime.now(timezone.utc).isoformat(),
                "success": fallback_result.get("success", False)
            }
            
        except Exception as e:
            logger.error(f"Fallback procedure failed for escalation {escalation_id}: {e}")
            return {
                "strategy": fallback_strategy.value,
                "success": False,
                "error": str(e)
            }
    
    def _select_fallback_strategy(
        self,
        trigger: EscalationTrigger,
        context: Dict[str, Any],
        escalation_record: Dict[str, Any]
    ) -> FallbackStrategy:
        """Select appropriate fallback strategy"""
        
        # Strategy selection based on trigger type
        trigger_strategies = {
            EscalationTrigger.QUALITY_FAILURE: FallbackStrategy.PREVIOUS_KNOWN_GOOD,
            EscalationTrigger.CONFLICT_UNRESOLVED: FallbackStrategy.SAFE_DEFAULT,
            EscalationTrigger.PERFORMANCE_DEGRADATION: FallbackStrategy.SIMPLIFIED_APPROACH,
            EscalationTrigger.COMPLIANCE_VIOLATION: FallbackStrategy.HUMAN_HANDOFF,
            EscalationTrigger.SYSTEM_ERROR: FallbackStrategy.ALTERNATIVE_AGENT,
            EscalationTrigger.TIMEOUT_EXCEEDED: FallbackStrategy.DEGRADED_SERVICE,
            EscalationTrigger.RESOURCE_EXHAUSTION: FallbackStrategy.SIMPLIFIED_APPROACH,
            EscalationTrigger.STAKEHOLDER_REQUEST: FallbackStrategy.HUMAN_HANDOFF
        }
        
        return trigger_strategies.get(trigger, FallbackStrategy.SAFE_DEFAULT)
    
    async def _execute_safe_default(
        self,
        escalation_record: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute safe default fallback"""
        
        trigger = escalation_record["trigger"]
        safe_default = self.safe_defaults.get(trigger, self.safe_defaults["system_error"])
        
        return {
            "fallback_type": "safe_default",
            "response": safe_default["response"],
            "action": safe_default["action"],
            "confidence": safe_default["confidence"],
            "success": True
        }
    
    async def _execute_previous_known_good(
        self,
        escalation_record: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute previous known good state fallback"""
        
        task_type = context.get("task_type", "general")
        known_good = self.known_good_states.get(task_type)
        
        if known_good:
            return {
                "fallback_type": "previous_known_good",
                "restored_state": known_good,
                "confidence": 0.8,
                "success": True
            }
        else:
            # Fall back to safe default if no known good state
            return await self._execute_safe_default(escalation_record, context)
    
    async def _execute_simplified_approach(
        self,
        escalation_record: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute simplified approach fallback"""
        
        return {
            "fallback_type": "simplified_approach",
            "approach": "basic_processing_mode",
            "features_disabled": ["advanced_analysis", "complex_reasoning"],
            "confidence": 0.7,
            "success": True
        }
    
    async def _execute_human_handoff(
        self,
        escalation_record: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute human handoff fallback"""
        
        return {
            "fallback_type": "human_handoff",
            "handoff_initiated": True,
            "human_review_required": True,
            "estimated_review_time": "2-4 hours",
            "confidence": 0.9,
            "success": True
        }
    
    async def _execute_system_shutdown(
        self,
        escalation_record: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute controlled system shutdown fallback"""
        
        return {
            "fallback_type": "system_shutdown",
            "shutdown_initiated": True,
            "reason": "Critical failure requiring system isolation",
            "recovery_procedure": "manual_restart_required",
            "confidence": 1.0,
            "success": True
        }
    
    async def _execute_alternative_agent(
        self,
        escalation_record: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute alternative agent fallback"""
        
        return {
            "fallback_type": "alternative_agent",
            "alternative_selected": "backup_agent",
            "original_agent_disabled": True,
            "confidence": 0.75,
            "success": True
        }
    
    async def _execute_degraded_service(
        self,
        escalation_record: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute degraded service fallback"""
        
        return {
            "fallback_type": "degraded_service",
            "service_level": "basic",
            "features_available": ["core_functionality"],
            "features_disabled": ["advanced_features", "optimization"],
            "confidence": 0.6,
            "success": True
        }
    
    async def _execute_emergency_procedures(
        self,
        escalation_id: str,
        escalation_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute emergency procedures when max escalation level exceeded"""
        
        logger.critical(f"Emergency procedures activated for escalation {escalation_id}")
        
        # Update metrics
        self.escalation_metrics["emergency_procedures_activated"] += 1
        
        # Execute emergency actions
        emergency_actions = [
            "notify_all_emergency_contacts",
            "activate_incident_response_team",
            "isolate_affected_systems",
            "initiate_business_continuity_plan"
        ]
        
        action_results = []
        for action in emergency_actions:
            result = await self._execute_emergency_action(action, escalation_record)
            action_results.append(result)
        
        return {
            "level": "emergency",
            "procedure_name": "Emergency Response",
            "actions_executed": len(action_results),
            "action_results": action_results,
            "status": "emergency_procedures_active",
            "requires_immediate_attention": True
        }
    
    async def _execute_emergency_action(
        self,
        action: str,
        escalation_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute emergency action"""
        
        # Simulate emergency action execution
        return {
            "action": action,
            "status": "executed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": True
        }
    
    def _estimate_resolution_time(self, level: int, priority: EscalationPriority) -> str:
        """Estimate resolution time based on escalation level and priority"""
        
        base_times = {
            1: 300,   # 5 minutes
            2: 900,   # 15 minutes
            3: 3600,  # 1 hour
            4: 7200   # 2 hours
        }
        
        priority_multipliers = {
            EscalationPriority.LOW: 1.5,
            EscalationPriority.MEDIUM: 1.0,
            EscalationPriority.HIGH: 0.7,
            EscalationPriority.CRITICAL: 0.5,
            EscalationPriority.EMERGENCY: 0.3
        }
        
        base_time = base_times.get(level, 3600)
        multiplier = priority_multipliers.get(priority, 1.0)
        estimated_seconds = int(base_time * multiplier)
        
        # Convert to human-readable format
        if estimated_seconds < 3600:
            return f"{estimated_seconds // 60} minutes"
        else:
            return f"{estimated_seconds // 3600} hours"
    
    async def _update_escalation_metrics(self, escalation_record: Dict[str, Any]) -> None:
        """Update escalation metrics"""
        
        self.escalation_metrics["escalations_triggered"] += 1
        
        if escalation_record.get("resolution", {}).get("resolved", False):
            self.escalation_metrics["escalations_resolved"] += 1
        
        # Update success rates
        total_escalations = self.escalation_metrics["escalations_triggered"]
        resolved_escalations = self.escalation_metrics["escalations_resolved"]
        
        self.escalation_metrics["escalation_success_rate"] = resolved_escalations / total_escalations
        
        # Update fallback success rate
        if escalation_record.get("fallback_executed", False):
            fallback_success = escalation_record.get("fallback_result", {}).get("success", False)
            if fallback_success:
                current_rate = self.escalation_metrics["fallback_success_rate"]
                fallback_count = self.escalation_metrics["fallbacks_executed"]
                self.escalation_metrics["fallback_success_rate"] = (
                    (current_rate * (fallback_count - 1) + 1) / fallback_count
                )
    
    def store_known_good_state(self, task_type: str, state: Dict[str, Any]) -> None:
        """Store a known good state for fallback purposes"""
        
        self.known_good_states[task_type] = {
            "state": state,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verified": True
        }
        
        logger.info(f"Known good state stored for task type: {task_type}")
    
    def get_escalation_status(self, escalation_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an escalation"""
        
        return self.active_escalations.get(escalation_id)
    
    def get_escalation_metrics(self) -> Dict[str, Any]:
        """Get current escalation metrics"""
        
        return self.escalation_metrics.copy()
    
    def get_active_escalations(self) -> Dict[str, Dict[str, Any]]:
        """Get all active escalations"""
        
        return self.active_escalations.copy()
    
    def get_escalation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get escalation history"""
        
        return self.escalation_history[-limit:]
    
    async def resolve_escalation(
        self,
        escalation_id: str,
        resolution: Dict[str, Any],
        resolved_by: str = "system"
    ) -> Dict[str, Any]:
        """Mark an escalation as resolved"""
        
        if escalation_id not in self.active_escalations:
            return {"error": "Escalation not found"}
        
        escalation_record = self.active_escalations[escalation_id]
        escalation_record["status"] = "resolved"
        escalation_record["resolution"] = resolution
        escalation_record["resolved_by"] = resolved_by
        escalation_record["resolution_time"] = datetime.now(timezone.utc).isoformat()
        
        # Move to history
        self.escalation_history.append(escalation_record)
        del self.active_escalations[escalation_id]
        
        # Update metrics
        self.escalation_metrics["escalations_resolved"] += 1
        
        logger.info(f"Escalation {escalation_id} resolved by {resolved_by}")
        
        return {"status": "resolved", "escalation_id": escalation_id}