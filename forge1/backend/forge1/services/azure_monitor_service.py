"""
Azure Monitor Business Telemetry Service

Provides business-specific telemetry tracking for Forge1 operations including
employee interactions, model usage, task execution, and business metrics.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from forge1.integrations.observability.azure_monitor_adapter import (
    azure_monitor_adapter, CustomEvent, CustomMetric
)
from forge1.integrations.base_adapter import ExecutionContext, TenantContext
from forge1.core.tenancy import get_current_tenant

logger = logging.getLogger(__name__)

class BusinessEventType(Enum):
    """Types of business events"""
    EMPLOYEE_CREATED = "employee_created"
    EMPLOYEE_INTERACTION = "employee_interaction"
    TASK_EXECUTED = "task_executed"
    MODEL_USAGE = "model_usage"
    TOOL_EXECUTION = "tool_execution"
    WORKFLOW_COMPLETED = "workflow_completed"
    POLICY_VIOLATION = "policy_violation"
    COMPLIANCE_CHECK = "compliance_check"
    BILLING_EVENT = "billing_event"
    PERFORMANCE_ALERT = "performance_alert"

@dataclass
class EmployeeInteraction:
    """Employee interaction data"""
    employee_id: str
    employee_name: str
    interaction_type: str
    user_message: str
    response_message: str
    processing_time_ms: float
    model_used: str
    tokens_used: int
    cost: float
    success: bool
    tenant_id: str
    user_id: str

@dataclass
class TaskExecution:
    """Task execution data"""
    task_id: str
    task_type: str
    description: str
    execution_time_ms: float
    success: bool
    error_message: Optional[str]
    tenant_id: str
    employee_id: Optional[str]
    user_id: str

@dataclass
class ModelUsage:
    """Model usage data"""
    model_name: str
    provider: str
    tokens_input: int
    tokens_output: int
    cost: float
    latency_ms: float
    tenant_id: str
    user_id: str
    request_type: str

class AzureMonitorBusinessService:
    """Service for tracking business-specific telemetry in Azure Monitor"""
    
    def __init__(self):
        self.adapter = azure_monitor_adapter
        
        # Business metrics tracking
        self._employee_interactions = 0
        self._tasks_executed = 0
        self._model_calls = 0
        self._total_cost = 0.0
        self._policy_violations = 0
        self._compliance_checks = 0
    
    async def track_employee_interaction(self, interaction: EmployeeInteraction, 
                                       context: Optional[ExecutionContext] = None) -> bool:
        """Track employee interaction event"""
        
        try:
            # Create execution context if not provided
            if not context:
                tenant_context = TenantContext(
                    tenant_id=interaction.tenant_id,
                    user_id=interaction.user_id,
                    employee_id=interaction.employee_id
                )
                context = ExecutionContext(
                    tenant_context=tenant_context,
                    request_id=f"interaction_{int(time.time())}"
                )
            
            # Track interaction event
            interaction_event = CustomEvent(
                name="forge1_employee_interaction",
                properties={
                    "employee_id": interaction.employee_id,
                    "employee_name": interaction.employee_name,
                    "interaction_type": interaction.interaction_type,
                    "model_used": interaction.model_used,
                    "success": str(interaction.success).lower(),
                    "tenant_id": interaction.tenant_id,
                    "user_id": interaction.user_id,
                    "message_length": str(len(interaction.user_message)),
                    "response_length": str(len(interaction.response_message))
                },
                measurements={
                    "processing_time_ms": interaction.processing_time_ms,
                    "tokens_used": float(interaction.tokens_used),
                    "cost": interaction.cost
                },
                tenant_id=interaction.tenant_id
            )
            
            success = await self.adapter.track_custom_event(interaction_event, context)
            
            if success:
                self._employee_interactions += 1
                self._total_cost += interaction.cost
                
                # Track individual metrics
                await self._track_interaction_metrics(interaction, context)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to track employee interaction: {e}")
            return False
    
    async def track_task_execution(self, task: TaskExecution, 
                                 context: Optional[ExecutionContext] = None) -> bool:
        """Track task execution event"""
        
        try:
            # Create execution context if not provided
            if not context:
                tenant_context = TenantContext(
                    tenant_id=task.tenant_id,
                    user_id=task.user_id,
                    employee_id=task.employee_id
                )
                context = ExecutionContext(
                    tenant_context=tenant_context,
                    request_id=f"task_{task.task_id}"
                )
            
            # Track task event
            task_event = CustomEvent(
                name="forge1_task_execution",
                properties={
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "description": task.description[:200],  # Limit description length
                    "success": str(task.success).lower(),
                    "error_message": task.error_message or "none",
                    "tenant_id": task.tenant_id,
                    "employee_id": task.employee_id or "none",
                    "user_id": task.user_id
                },
                measurements={
                    "execution_time_ms": task.execution_time_ms,
                    "success_flag": 1.0 if task.success else 0.0
                },
                tenant_id=task.tenant_id
            )
            
            success = await self.adapter.track_custom_event(task_event, context)
            
            if success:
                self._tasks_executed += 1
                
                # Track task metrics
                await self._track_task_metrics(task, context)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to track task execution: {e}")
            return False
    
    async def track_model_usage(self, usage: ModelUsage, 
                              context: Optional[ExecutionContext] = None) -> bool:
        """Track model usage event"""
        
        try:
            # Create execution context if not provided
            if not context:
                tenant_context = TenantContext(
                    tenant_id=usage.tenant_id,
                    user_id=usage.user_id
                )
                context = ExecutionContext(
                    tenant_context=tenant_context,
                    request_id=f"model_{int(time.time())}"
                )
            
            # Track model usage event
            model_event = CustomEvent(
                name="forge1_model_usage",
                properties={
                    "model_name": usage.model_name,
                    "provider": usage.provider,
                    "request_type": usage.request_type,
                    "tenant_id": usage.tenant_id,
                    "user_id": usage.user_id
                },
                measurements={
                    "tokens_input": float(usage.tokens_input),
                    "tokens_output": float(usage.tokens_output),
                    "total_tokens": float(usage.tokens_input + usage.tokens_output),
                    "cost": usage.cost,
                    "latency_ms": usage.latency_ms
                },
                tenant_id=usage.tenant_id
            )
            
            success = await self.adapter.track_custom_event(model_event, context)
            
            if success:
                self._model_calls += 1
                self._total_cost += usage.cost
                
                # Track model metrics
                await self._track_model_metrics(usage, context)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to track model usage: {e}")
            return False
    
    async def track_policy_violation(self, violation_type: str, resource: str, 
                                   tenant_id: str, user_id: str, 
                                   details: Dict[str, Any],
                                   context: Optional[ExecutionContext] = None) -> bool:
        """Track policy violation event"""
        
        try:
            # Create execution context if not provided
            if not context:
                tenant_context = TenantContext(tenant_id=tenant_id, user_id=user_id)
                context = ExecutionContext(
                    tenant_context=tenant_context,
                    request_id=f"violation_{int(time.time())}"
                )
            
            # Track violation event
            violation_event = CustomEvent(
                name="forge1_policy_violation",
                properties={
                    "violation_type": violation_type,
                    "resource": resource,
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "severity": details.get("severity", "medium"),
                    "policy_name": details.get("policy_name", "unknown"),
                    "action_taken": details.get("action_taken", "denied")
                },
                measurements={
                    "severity_score": float(details.get("severity_score", 1.0))
                },
                tenant_id=tenant_id
            )
            
            success = await self.adapter.track_custom_event(violation_event, context)
            
            if success:
                self._policy_violations += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to track policy violation: {e}")
            return False
    
    async def track_compliance_check(self, check_type: str, result: str, 
                                   tenant_id: str, details: Dict[str, Any],
                                   context: Optional[ExecutionContext] = None) -> bool:
        """Track compliance check event"""
        
        try:
            # Create execution context if not provided
            if not context:
                tenant_context = TenantContext(tenant_id=tenant_id)
                context = ExecutionContext(
                    tenant_context=tenant_context,
                    request_id=f"compliance_{int(time.time())}"
                )
            
            # Track compliance event
            compliance_event = CustomEvent(
                name="forge1_compliance_check",
                properties={
                    "check_type": check_type,
                    "result": result,
                    "tenant_id": tenant_id,
                    "regulation": details.get("regulation", "unknown"),
                    "data_classification": details.get("data_classification", "unclassified"),
                    "remediation_required": str(details.get("remediation_required", False)).lower()
                },
                measurements={
                    "compliance_score": float(details.get("compliance_score", 1.0))
                },
                tenant_id=tenant_id
            )
            
            success = await self.adapter.track_custom_event(compliance_event, context)
            
            if success:
                self._compliance_checks += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to track compliance check: {e}")
            return False
    
    async def track_billing_event(self, event_type: str, amount: float, 
                                tenant_id: str, details: Dict[str, Any],
                                context: Optional[ExecutionContext] = None) -> bool:
        """Track billing-related event"""
        
        try:
            # Create execution context if not provided
            if not context:
                tenant_context = TenantContext(tenant_id=tenant_id)
                context = ExecutionContext(
                    tenant_context=tenant_context,
                    request_id=f"billing_{int(time.time())}"
                )
            
            # Track billing event
            billing_event = CustomEvent(
                name="forge1_billing_event",
                properties={
                    "event_type": event_type,
                    "tenant_id": tenant_id,
                    "currency": details.get("currency", "USD"),
                    "billing_period": details.get("billing_period", "unknown"),
                    "resource_type": details.get("resource_type", "unknown")
                },
                measurements={
                    "amount": amount,
                    "quantity": float(details.get("quantity", 1.0))
                },
                tenant_id=tenant_id
            )
            
            success = await self.adapter.track_custom_event(billing_event, context)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to track billing event: {e}")
            return False
    
    async def _track_interaction_metrics(self, interaction: EmployeeInteraction, 
                                       context: ExecutionContext):
        """Track individual interaction metrics"""
        
        # Response time metric
        response_time_metric = CustomMetric(
            name="forge1_employee_response_time_ms",
            value=interaction.processing_time_ms,
            properties={
                "employee_id": interaction.employee_id,
                "model_used": interaction.model_used,
                "interaction_type": interaction.interaction_type,
                "tenant_id": interaction.tenant_id
            },
            tenant_id=interaction.tenant_id
        )
        
        await self.adapter.track_custom_metric(response_time_metric, context)
        
        # Cost metric
        cost_metric = CustomMetric(
            name="forge1_interaction_cost",
            value=interaction.cost,
            properties={
                "employee_id": interaction.employee_id,
                "model_used": interaction.model_used,
                "tenant_id": interaction.tenant_id
            },
            tenant_id=interaction.tenant_id
        )
        
        await self.adapter.track_custom_metric(cost_metric, context)
        
        # Token usage metric
        token_metric = CustomMetric(
            name="forge1_tokens_used",
            value=float(interaction.tokens_used),
            properties={
                "employee_id": interaction.employee_id,
                "model_used": interaction.model_used,
                "tenant_id": interaction.tenant_id
            },
            tenant_id=interaction.tenant_id
        )
        
        await self.adapter.track_custom_metric(token_metric, context)
    
    async def _track_task_metrics(self, task: TaskExecution, context: ExecutionContext):
        """Track individual task metrics"""
        
        # Task duration metric
        duration_metric = CustomMetric(
            name="forge1_task_duration_ms",
            value=task.execution_time_ms,
            properties={
                "task_type": task.task_type,
                "success": str(task.success).lower(),
                "tenant_id": task.tenant_id
            },
            tenant_id=task.tenant_id
        )
        
        await self.adapter.track_custom_metric(duration_metric, context)
        
        # Task success rate metric
        success_metric = CustomMetric(
            name="forge1_task_success_rate",
            value=1.0 if task.success else 0.0,
            properties={
                "task_type": task.task_type,
                "tenant_id": task.tenant_id
            },
            tenant_id=task.tenant_id
        )
        
        await self.adapter.track_custom_metric(success_metric, context)
    
    async def _track_model_metrics(self, usage: ModelUsage, context: ExecutionContext):
        """Track individual model metrics"""
        
        # Model latency metric
        latency_metric = CustomMetric(
            name="forge1_model_latency_ms",
            value=usage.latency_ms,
            properties={
                "model_name": usage.model_name,
                "provider": usage.provider,
                "tenant_id": usage.tenant_id
            },
            tenant_id=usage.tenant_id
        )
        
        await self.adapter.track_custom_metric(latency_metric, context)
        
        # Model cost metric
        cost_metric = CustomMetric(
            name="forge1_model_cost",
            value=usage.cost,
            properties={
                "model_name": usage.model_name,
                "provider": usage.provider,
                "tenant_id": usage.tenant_id
            },
            tenant_id=usage.tenant_id
        )
        
        await self.adapter.track_custom_metric(cost_metric, context)
    
    async def send_daily_summary(self, tenant_id: str) -> bool:
        """Send daily summary metrics"""
        
        try:
            tenant_context = TenantContext(tenant_id=tenant_id)
            context = ExecutionContext(
                tenant_context=tenant_context,
                request_id=f"daily_summary_{int(time.time())}"
            )
            
            # Send daily summary event
            summary_event = CustomEvent(
                name="forge1_daily_summary",
                properties={
                    "tenant_id": tenant_id,
                    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d")
                },
                measurements={
                    "employee_interactions": float(self._employee_interactions),
                    "tasks_executed": float(self._tasks_executed),
                    "model_calls": float(self._model_calls),
                    "total_cost": self._total_cost,
                    "policy_violations": float(self._policy_violations),
                    "compliance_checks": float(self._compliance_checks)
                },
                tenant_id=tenant_id
            )
            
            return await self.adapter.track_custom_event(summary_event, context)
            
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get business service statistics"""
        
        return {
            "employee_interactions": self._employee_interactions,
            "tasks_executed": self._tasks_executed,
            "model_calls": self._model_calls,
            "total_cost": self._total_cost,
            "policy_violations": self._policy_violations,
            "compliance_checks": self._compliance_checks,
            "average_cost_per_interaction": self._total_cost / max(1, self._employee_interactions),
            "task_success_rate": 1.0 - (self._policy_violations / max(1, self._tasks_executed))
        }

# Global Azure Monitor business service
azure_monitor_business_service = AzureMonitorBusinessService()