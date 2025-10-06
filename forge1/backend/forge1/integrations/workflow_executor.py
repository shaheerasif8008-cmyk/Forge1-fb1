"""
End-to-End Workflow Execution System

Orchestrates complete workflow execution from employee creation to task completion,
managing workflow state, progress tracking, and result aggregation.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from enum import Enum

from forge1.models.workflow_models import (
    WorkflowContext, WorkflowResult, WorkflowStatus, WorkflowType, 
    CollaborationMode, generate_workflow_id
)
from forge1.integrations.workflow_context_injector import (
    context_injector, create_context, inject_context, with_context
)
from forge1.integrations.tool_permission_enforcer import permission_enforcer
from forge1.integrations.mcae_adapter import MCAEAdapter
from forge1.integrations.mcae_error_handler import MCAEErrorHandler
from forge1.core.tenancy import get_current_tenant, set_current_tenant

logger = logging.getLogger(__name__)


class ExecutionPhase(str, Enum):
    """Workflow execution phases"""
    INITIALIZATION = "initialization"
    CONTEXT_SETUP = "context_setup"
    AGENT_CREATION = "agent_creation"
    TASK_EXECUTION = "task_execution"
    RESULT_PROCESSING = "result_processing"
    CLEANUP = "cleanup"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowExecutor:
    """
    Manages end-to-end workflow execution with state tracking,
    progress monitoring, and result aggregation.
    """
    
    def __init__(
        self,
        mcae_adapter: MCAEAdapter,
        error_handler: MCAEErrorHandler,
        employee_manager=None
    ):
        self.mcae_adapter = mcae_adapter
        self.error_handler = error_handler
        self.employee_manager = employee_manager
        
        # Active workflows
        self.active_workflows = {}
        self.workflow_history = {}
        
        # Execution hooks
        self.pre_execution_hooks = []
        self.post_execution_hooks = []
        self.phase_hooks = {}
        
        # Statistics
        self.stats = {
            "workflows_executed": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "average_execution_time": 0.0,
            "total_execution_time": 0.0,
            "workflows_by_type": {},
            "workflows_by_status": {}
        }
    
    async def execute_workflow(
        self,
        tenant_id: str,
        employee_id: str,
        task_description: str,
        workflow_type: WorkflowType = WorkflowType.STANDARD,
        collaboration_mode: CollaborationMode = CollaborationMode.SEQUENTIAL,
        context_data: Optional[Dict] = None,
        timeout_seconds: Optional[int] = None
    ) -> WorkflowResult:
        """
        Execute a complete workflow from start to finish.
        
        Args:
            tenant_id: Tenant ID for isolation
            employee_id: Employee ID to execute workflow with
            task_description: Description of the task to execute
            workflow_type: Type of workflow to execute
            collaboration_mode: How agents should collaborate
            context_data: Additional context data
            timeout_seconds: Maximum execution time
            
        Returns:
            WorkflowResult containing execution results and metadata
        """
        workflow_id = generate_workflow_id(tenant_id, employee_id)
        start_time = datetime.now(timezone.utc)
        
        # Create workflow context
        session_id = f"session_{workflow_id}"
        workflow_context = WorkflowContext(
            tenant_id=tenant_id,
            employee_id=employee_id,
            session_id=session_id,
            user_id=employee_id,
            task_description=task_description,
            workflow_type=workflow_type,
            collaboration_mode=collaboration_mode,
            custom_config=context_data or {},
            timeout_seconds=timeout_seconds
        )
        
        # Initialize workflow result
        workflow_result = WorkflowResult(
            workflow_id=workflow_id,
            session_id=session_id,
            status=WorkflowStatus.REGISTERED,
            started_at=start_time,
            tenant_id=tenant_id,
            employee_id=employee_id
        )
        
        # Track active workflow
        self.active_workflows[workflow_id] = {
            "context": workflow_context,
            "result": workflow_result,
            "phase": ExecutionPhase.INITIALIZATION,
            "start_time": start_time
        }
        
        try:
            # Execute workflow with context
            context_dict = workflow_context.to_dict()
            
            async with self._workflow_context_manager(context_dict):
                result = await self._execute_workflow_phases(
                    workflow_id, workflow_context, workflow_result
                )
            
            # Update statistics
            self._update_execution_stats(result, start_time)
            
            return result
            
        except Exception as e:
            # Handle execution error
            error_result = await self._handle_workflow_error(
                workflow_id, workflow_context, workflow_result, e
            )
            
            self._update_execution_stats(error_result, start_time)
            return error_result
            
        finally:
            # Move to history and cleanup
            if workflow_id in self.active_workflows:
                self.workflow_history[workflow_id] = self.active_workflows[workflow_id]
                del self.active_workflows[workflow_id]
    
    async def _execute_workflow_phases(
        self,
        workflow_id: str,
        context: WorkflowContext,
        result: WorkflowResult
    ) -> WorkflowResult:
        """Execute all workflow phases in sequence"""
        
        phases = [
            (ExecutionPhase.INITIALIZATION, self._phase_initialization),
            (ExecutionPhase.CONTEXT_SETUP, self._phase_context_setup),
            (ExecutionPhase.AGENT_CREATION, self._phase_agent_creation),
            (ExecutionPhase.TASK_EXECUTION, self._phase_task_execution),
            (ExecutionPhase.RESULT_PROCESSING, self._phase_result_processing),
            (ExecutionPhase.CLEANUP, self._phase_cleanup)
        ]
        
        for phase, phase_func in phases:
            try:
                # Update current phase
                self.active_workflows[workflow_id]["phase"] = phase
                
                # Run pre-phase hooks
                await self._run_phase_hooks("pre", phase, workflow_id, context, result)
                
                # Execute phase
                logger.debug(f"Executing phase {phase.value} for workflow {workflow_id}")
                await phase_func(workflow_id, context, result)
                
                # Run post-phase hooks
                await self._run_phase_hooks("post", phase, workflow_id, context, result)
                
            except Exception as e:
                logger.error(f"Phase {phase.value} failed for workflow {workflow_id}: {e}")
                result.status = WorkflowStatus.FAILED
                result.error = f"Phase {phase.value} failed: {str(e)}"
                raise
        
        # Mark as completed
        result.status = WorkflowStatus.COMPLETED
        result.completed_at = datetime.now(timezone.utc)
        result.execution_time_seconds = (
            result.completed_at - result.started_at
        ).total_seconds()
        
        self.active_workflows[workflow_id]["phase"] = ExecutionPhase.COMPLETED
        
        return result
    
    async def _phase_initialization(
        self,
        workflow_id: str,
        context: WorkflowContext,
        result: WorkflowResult
    ) -> None:
        """Initialize workflow execution environment"""
        
        # Set tenant context
        set_current_tenant(context.tenant_id)
        
        # Validate employee exists and is active
        if self.employee_manager:
            employee = await self.employee_manager.load_employee(
                context.tenant_id, context.employee_id
            )
            if not employee:
                raise ValueError(f"Employee {context.employee_id} not found")
            if not employee.is_active():
                raise ValueError(f"Employee {context.employee_id} is not active")
        
        # Run pre-execution hooks
        await self._run_execution_hooks("pre", workflow_id, context, result)
        
        logger.info(f"Initialized workflow {workflow_id} for employee {context.employee_id}")
    
    async def _phase_context_setup(
        self,
        workflow_id: str,
        context: WorkflowContext,
        result: WorkflowResult
    ) -> None:
        """Set up workflow context and permissions"""
        
        # Create and inject workflow context
        context_dict = context_injector.create_workflow_context(
            context.tenant_id,
            context.employee_id,
            context.session_id,
            context.user_id,
            context.custom_config
        )
        
        context_injector.inject_context(context_dict)
        
        # Set up tool permissions for employee
        if self.employee_manager:
            employee = await self.employee_manager.load_employee(
                context.tenant_id, context.employee_id
            )
            
            # Configure tool permissions based on employee configuration
            tool_permissions = {}
            for tool_name in employee.tool_access:
                tool_permissions[tool_name] = "write"  # Default permission level
            
            permission_enforcer.set_employee_permissions(
                context.tenant_id,
                context.employee_id,
                tool_permissions
            )
        
        logger.debug(f"Set up context for workflow {workflow_id}")
    
    async def _phase_agent_creation(
        self,
        workflow_id: str,
        context: WorkflowContext,
        result: WorkflowResult
    ) -> None:
        """Create and configure MCAE agents"""
        
        # Ensure MCAE adapter is available
        if not self.mcae_adapter:
            raise RuntimeError("MCAE adapter not available")
        
        # Check if employee is already registered with MCAE
        if self.employee_manager:
            employee = await self.employee_manager.load_employee(
                context.tenant_id, context.employee_id
            )
            
            if not employee.workflow_id:
                # Register employee with MCAE
                workflow_mcae_id = await self.mcae_adapter.register_employee_workflow(employee)
                employee.workflow_id = workflow_mcae_id
                
                # Update employee record
                await self.employee_manager._update_employee_workflow_id(
                    employee.id, workflow_mcae_id
                )
        
        logger.debug(f"Created agents for workflow {workflow_id}")
    
    async def _phase_task_execution(
        self,
        workflow_id: str,
        context: WorkflowContext,
        result: WorkflowResult
    ) -> None:
        """Execute the main task through MCAE"""
        
        result.status = WorkflowStatus.EXECUTING
        
        # Execute task through MCAE adapter
        mcae_result = await self.mcae_adapter.execute_workflow(
            workflow_id,
            context.task_description,
            context.to_dict()
        )
        
        # Store MCAE execution results
        result.result = mcae_result.get("result", {})
        result.agent_messages = mcae_result.get("agent_messages", [])
        result.execution_steps = mcae_result.get("steps", [])
        result.tokens_used = mcae_result.get("tokens_used", 0)
        
        # Calculate cost (rough estimate)
        result.cost = result.tokens_used * 0.00001  # $0.00001 per token
        
        logger.info(f"Executed task for workflow {workflow_id}")
    
    async def _phase_result_processing(
        self,
        workflow_id: str,
        context: WorkflowContext,
        result: WorkflowResult
    ) -> None:
        """Process and format workflow results"""
        
        # Add workflow metadata to results
        if result.result:
            result.result.update({
                "workflow_metadata": {
                    "workflow_id": workflow_id,
                    "workflow_type": context.workflow_type.value,
                    "collaboration_mode": context.collaboration_mode.value,
                    "tenant_id": context.tenant_id,
                    "employee_id": context.employee_id,
                    "execution_time": result.execution_time_seconds
                }
            })
        
        # Run post-execution hooks
        await self._run_execution_hooks("post", workflow_id, context, result)
        
        logger.debug(f"Processed results for workflow {workflow_id}")
    
    async def _phase_cleanup(
        self,
        workflow_id: str,
        context: WorkflowContext,
        result: WorkflowResult
    ) -> None:
        """Clean up workflow resources"""
        
        # Clear workflow context
        context_injector.clear_context()
        
        # Clean up any temporary resources
        # (In a real implementation, this might clean up temporary files, 
        #  close connections, etc.)
        
        logger.debug(f"Cleaned up workflow {workflow_id}")
    
    async def _handle_workflow_error(
        self,
        workflow_id: str,
        context: WorkflowContext,
        result: WorkflowResult,
        error: Exception
    ) -> WorkflowResult:
        """Handle workflow execution errors"""
        
        # Update workflow status
        result.status = WorkflowStatus.FAILED
        result.error = str(error)
        result.completed_at = datetime.now(timezone.utc)
        result.execution_time_seconds = (
            result.completed_at - result.started_at
        ).total_seconds()
        
        # Update phase
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["phase"] = ExecutionPhase.FAILED
        
        # Handle error through error handler
        try:
            error_context = {
                "tenant_id": context.tenant_id,
                "employee_id": context.employee_id,
                "workflow_id": workflow_id,
                "session_id": context.session_id
            }
            
            error_result = await self.error_handler.handle_error(
                error, error_context, "workflow_execution"
            )
            
            # If error handler suggests fallback, try it
            if error_result.get("recovery_result", {}).get("success"):
                logger.info(f"Error recovery successful for workflow {workflow_id}")
                # Could potentially retry or use fallback here
            
        except Exception as handler_error:
            logger.error(f"Error handler failed for workflow {workflow_id}: {handler_error}")
        
        logger.error(f"Workflow {workflow_id} failed: {error}")
        return result
    
    def _workflow_context_manager(self, context_dict: Dict[str, Any]):
        """Context manager for workflow execution"""
        return with_context(context_dict)
    
    async def _run_execution_hooks(
        self,
        hook_type: str,
        workflow_id: str,
        context: WorkflowContext,
        result: WorkflowResult
    ) -> None:
        """Run execution hooks"""
        
        hooks = self.pre_execution_hooks if hook_type == "pre" else self.post_execution_hooks
        
        for hook in hooks:
            try:
                await hook(workflow_id, context, result)
            except Exception as e:
                logger.error(f"Execution hook {hook.__name__} failed: {e}")
    
    async def _run_phase_hooks(
        self,
        hook_type: str,
        phase: ExecutionPhase,
        workflow_id: str,
        context: WorkflowContext,
        result: WorkflowResult
    ) -> None:
        """Run phase-specific hooks"""
        
        phase_key = f"{hook_type}_{phase.value}"
        hooks = self.phase_hooks.get(phase_key, [])
        
        for hook in hooks:
            try:
                await hook(workflow_id, context, result)
            except Exception as e:
                logger.error(f"Phase hook {hook.__name__} failed: {e}")
    
    def _update_execution_stats(self, result: WorkflowResult, start_time: datetime) -> None:
        """Update execution statistics"""
        
        self.stats["workflows_executed"] += 1
        
        if result.status == WorkflowStatus.COMPLETED:
            self.stats["successful_workflows"] += 1
        else:
            self.stats["failed_workflows"] += 1
        
        # Update execution time stats
        execution_time = result.execution_time_seconds or 0
        self.stats["total_execution_time"] += execution_time
        self.stats["average_execution_time"] = (
            self.stats["total_execution_time"] / self.stats["workflows_executed"]
        )
        
        # Update type and status stats
        workflow_type = getattr(result, 'workflow_type', 'unknown')
        if workflow_type not in self.stats["workflows_by_type"]:
            self.stats["workflows_by_type"][workflow_type] = 0
        self.stats["workflows_by_type"][workflow_type] += 1
        
        status = result.status.value
        if status not in self.stats["workflows_by_status"]:
            self.stats["workflows_by_status"][status] = 0
        self.stats["workflows_by_status"][status] += 1
    
    def add_execution_hook(self, hook: Callable, hook_type: str = "pre") -> None:
        """Add an execution hook"""
        
        if hook_type == "pre":
            self.pre_execution_hooks.append(hook)
        elif hook_type == "post":
            self.post_execution_hooks.append(hook)
        
        logger.info(f"Added {hook_type}-execution hook: {hook.__name__}")
    
    def add_phase_hook(self, phase: ExecutionPhase, hook: Callable, hook_type: str = "pre") -> None:
        """Add a phase-specific hook"""
        
        phase_key = f"{hook_type}_{phase.value}"
        if phase_key not in self.phase_hooks:
            self.phase_hooks[phase_key] = []
        
        self.phase_hooks[phase_key].append(hook)
        logger.info(f"Added {hook_type}-{phase.value} hook: {hook.__name__}")
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow"""
        
        if workflow_id in self.active_workflows:
            workflow_data = self.active_workflows[workflow_id]
            return {
                "workflow_id": workflow_id,
                "status": "active",
                "phase": workflow_data["phase"].value,
                "context": workflow_data["context"].to_dict(),
                "result_status": workflow_data["result"].status.value,
                "start_time": workflow_data["start_time"].isoformat()
            }
        
        if workflow_id in self.workflow_history:
            workflow_data = self.workflow_history[workflow_id]
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "phase": workflow_data["phase"].value,
                "result_status": workflow_data["result"].status.value,
                "start_time": workflow_data["start_time"].isoformat(),
                "execution_time": workflow_data["result"].execution_time_seconds
            }
        
        return None
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get list of currently active workflows"""
        
        active = []
        for workflow_id, workflow_data in self.active_workflows.items():
            active.append({
                "workflow_id": workflow_id,
                "tenant_id": workflow_data["context"].tenant_id,
                "employee_id": workflow_data["context"].employee_id,
                "phase": workflow_data["phase"].value,
                "status": workflow_data["result"].status.value,
                "start_time": workflow_data["start_time"].isoformat()
            })
        
        return active
    
    def get_stats(self) -> Dict[str, Any]:
        """Get workflow execution statistics"""
        
        return {
            **self.stats,
            "active_workflows": len(self.active_workflows),
            "workflow_history_size": len(self.workflow_history),
            "success_rate": (
                self.stats["successful_workflows"] / 
                max(self.stats["workflows_executed"], 1)
            ) * 100
        }
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow"""
        
        if workflow_id not in self.active_workflows:
            return False
        
        try:
            workflow_data = self.active_workflows[workflow_id]
            result = workflow_data["result"]
            
            # Update status
            result.status = WorkflowStatus.CANCELLED
            result.completed_at = datetime.now(timezone.utc)
            result.execution_time_seconds = (
                result.completed_at - result.started_at
            ).total_seconds()
            
            # Move to history
            self.workflow_history[workflow_id] = workflow_data
            del self.active_workflows[workflow_id]
            
            logger.info(f"Cancelled workflow {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel workflow {workflow_id}: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for workflow executor"""
        
        try:
            stats = self.get_stats()
            
            return {
                "status": "healthy",
                "stats": stats,
                "mcae_adapter_available": self.mcae_adapter is not None,
                "error_handler_available": self.error_handler is not None
            }
        except Exception as e:
            logger.error(f"Workflow executor health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }