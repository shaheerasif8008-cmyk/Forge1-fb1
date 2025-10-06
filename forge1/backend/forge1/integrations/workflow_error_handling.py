"""
Workflow Error Handling and Resilience Patterns

Implements comprehensive error handling, circuit breaker patterns, dead-letter queues,
and workflow status tracking for robust workflow execution in production environments.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import json
import uuid

# Forge1 imports
from forge1.core.memory_manager import MemoryManager
from forge1.core.audit_logger import AuditLogger
from forge1.integrations.workflows_adapter import WorkflowStatus, StepStatus

logger = logging.getLogger(__name__)

class ErrorCategory(Enum):
    """Categories of workflow errors"""
    TRANSIENT = "transient"  # Network timeouts, rate limits
    PERMISSION = "permission"  # RBAC violations
    DATA = "data"  # Invalid input, parsing failures
    SYSTEM = "system"  # Service unavailable
    BUSINESS = "business"  # Business logic errors

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class ErrorInfo:
    """Information about a workflow error"""
    error_id: str
    error_category: ErrorCategory
    error_message: str
    step_name: Optional[str]
    workflow_id: str
    tenant_id: str
    employee_id: str
    timestamp: datetime
    retry_count: int
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = None

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: int = 30  # Seconds before trying half-open
    success_threshold: int = 3  # Successes to close from half-open
    timeout_seconds: float = 30.0  # Request timeout

class CircuitBreaker:
    """Circuit breaker for external service calls"""
    
    def __init__(self, service_name: str, config: CircuitBreakerConfig):
        self.service_name = service_name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        
        # Check if circuit is open
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"Circuit breaker {self.service_name} moving to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker {self.service_name} is OPEN"
                )
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout_seconds
            )
            
            # Record success
            await self._record_success()
            return result
            
        except asyncio.TimeoutError:
            await self._record_failure()
            raise CircuitBreakerTimeoutError(
                f"Circuit breaker {self.service_name} timeout after {self.config.timeout_seconds}s"
            )
        except Exception as e:
            await self._record_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if not self.last_failure_time:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.recovery_timeout
    
    async def _record_success(self) -> None:
        """Record successful call"""
        self.last_success_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker {self.service_name} CLOSED after recovery")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0  # Reset failure count on success
    
    async def _record_failure(self) -> None:
        """Record failed call"""
        self.last_failure_time = time.time()
        self.failure_count += 1
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker {self.service_name} OPENED after {self.failure_count} failures")
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0
            logger.warning(f"Circuit breaker {self.service_name} returned to OPEN state")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "service_name": self.service_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time
        }

class DeadLetterQueue:
    """Dead letter queue for failed workflows"""
    
    def __init__(self, memory_manager: MemoryManager, audit_logger: AuditLogger):
        self.memory_manager = memory_manager
        self.audit_logger = audit_logger
    
    async def add_failed_workflow(
        self,
        workflow_id: str,
        error_info: ErrorInfo,
        workflow_context: Dict[str, Any]
    ) -> str:
        """Add failed workflow to dead letter queue"""
        try:
            dlq_entry_id = f"dlq_{uuid.uuid4().hex[:8]}"
            
            dlq_entry = {
                "dlq_entry_id": dlq_entry_id,
                "workflow_id": workflow_id,
                "error_info": asdict(error_info),
                "workflow_context": workflow_context,
                "added_at": datetime.now(timezone.utc).isoformat(),
                "retry_attempts": 0,
                "max_retries": 3,
                "status": "pending_review"
            }
            
            # Store in memory system
            await self.memory_manager.store_memory(
                content=dlq_entry,
                memory_type="dead_letter_queue",
                metadata={
                    "dlq_entry_id": dlq_entry_id,
                    "workflow_id": workflow_id,
                    "error_category": error_info.error_category.value,
                    "tenant_id": error_info.tenant_id,
                    "employee_id": error_info.employee_id
                }
            )
            
            # Log to audit system
            await self.audit_logger.log_error_event(
                event_type="workflow_dead_letter",
                user_id=error_info.employee_id,
                tenant_id=error_info.tenant_id,
                details={
                    "dlq_entry_id": dlq_entry_id,
                    "workflow_id": workflow_id,
                    "error_category": error_info.error_category.value,
                    "error_message": error_info.error_message,
                    "step_name": error_info.step_name
                }
            )
            
            logger.info(f"Added workflow {workflow_id} to dead letter queue: {dlq_entry_id}")
            return dlq_entry_id
            
        except Exception as e:
            logger.error(f"Failed to add workflow to dead letter queue: {e}")
            raise
    
    async def get_failed_workflows(
        self,
        tenant_id: Optional[str] = None,
        status: str = "pending_review"
    ) -> List[Dict[str, Any]]:
        """Get failed workflows from dead letter queue"""
        try:
            # Search for dead letter entries
            query_filters = {"status": status}
            if tenant_id:
                query_filters["tenant_id"] = tenant_id
            
            # This would use the memory manager's search functionality
            # Implementation depends on memory manager's query interface
            results = []  # Placeholder - would implement actual search
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get failed workflows: {e}")
            return []
    
    async def retry_workflow(self, dlq_entry_id: str) -> bool:
        """Retry a failed workflow from dead letter queue"""
        try:
            # This would retrieve the workflow context and retry execution
            # Implementation would depend on workflow adapter integration
            logger.info(f"Retrying workflow from DLQ entry: {dlq_entry_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to retry workflow {dlq_entry_id}: {e}")
            return False

class WorkflowStatusTracker:
    """Track workflow execution status and progress"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
    
    async def start_workflow_tracking(
        self,
        workflow_id: str,
        workflow_type: str,
        context: Dict[str, Any]
    ) -> None:
        """Start tracking a workflow"""
        try:
            workflow_status = {
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "status": WorkflowStatus.RUNNING.value,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "context": context,
                "steps": [],
                "current_step": None,
                "progress_percentage": 0.0,
                "estimated_completion": None
            }
            
            self.active_workflows[workflow_id] = workflow_status
            
            # Store in memory system
            await self.memory_manager.store_memory(
                content=workflow_status,
                memory_type="workflow_status",
                metadata={
                    "workflow_id": workflow_id,
                    "workflow_type": workflow_type,
                    "tenant_id": context.get("tenant_id"),
                    "employee_id": context.get("employee_id")
                }
            )
            
            logger.info(f"Started tracking workflow: {workflow_id}")
            
        except Exception as e:
            logger.error(f"Failed to start workflow tracking: {e}")
    
    async def update_step_status(
        self,
        workflow_id: str,
        step_name: str,
        status: StepStatus,
        output: Any = None,
        error: Optional[str] = None
    ) -> None:
        """Update status of a workflow step"""
        try:
            if workflow_id not in self.active_workflows:
                logger.warning(f"Workflow {workflow_id} not being tracked")
                return
            
            workflow_status = self.active_workflows[workflow_id]
            
            # Update step information
            step_info = {
                "step_name": step_name,
                "status": status.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "output": output,
                "error": error
            }
            
            # Find existing step or add new one
            existing_step = None
            for i, step in enumerate(workflow_status["steps"]):
                if step["step_name"] == step_name:
                    workflow_status["steps"][i] = step_info
                    existing_step = step_info
                    break
            
            if not existing_step:
                workflow_status["steps"].append(step_info)
            
            # Update current step and progress
            workflow_status["current_step"] = step_name
            workflow_status["progress_percentage"] = self._calculate_progress(workflow_status["steps"])
            
            # Update in memory system
            await self.memory_manager.update_memory(
                memory_id=workflow_id,  # Assuming workflow_id is used as memory_id
                updates={"content": workflow_status},
                user_id=workflow_status["context"].get("employee_id", "system")
            )
            
        except Exception as e:
            logger.error(f"Failed to update step status: {e}")
    
    async def complete_workflow(
        self,
        workflow_id: str,
        final_status: WorkflowStatus,
        final_output: Any = None,
        error: Optional[str] = None
    ) -> None:
        """Mark workflow as completed"""
        try:
            if workflow_id not in self.active_workflows:
                logger.warning(f"Workflow {workflow_id} not being tracked")
                return
            
            workflow_status = self.active_workflows[workflow_id]
            workflow_status["status"] = final_status.value
            workflow_status["completed_at"] = datetime.now(timezone.utc).isoformat()
            workflow_status["final_output"] = final_output
            workflow_status["error"] = error
            workflow_status["progress_percentage"] = 100.0 if final_status == WorkflowStatus.COMPLETED else workflow_status["progress_percentage"]
            
            # Calculate total execution time
            started_at = datetime.fromisoformat(workflow_status["started_at"].replace('Z', '+00:00'))
            completed_at = datetime.now(timezone.utc)
            execution_time = (completed_at - started_at).total_seconds()
            workflow_status["execution_time_seconds"] = execution_time
            
            # Update in memory system
            await self.memory_manager.update_memory(
                memory_id=workflow_id,
                updates={"content": workflow_status},
                user_id=workflow_status["context"].get("employee_id", "system")
            )
            
            # Remove from active tracking
            del self.active_workflows[workflow_id]
            
            logger.info(f"Completed workflow tracking: {workflow_id} ({final_status.value})")
            
        except Exception as e:
            logger.error(f"Failed to complete workflow tracking: {e}")
    
    def _calculate_progress(self, steps: List[Dict[str, Any]]) -> float:
        """Calculate workflow progress percentage"""
        if not steps:
            return 0.0
        
        completed_steps = sum(1 for step in steps if step["status"] == StepStatus.COMPLETED.value)
        return (completed_steps / len(steps)) * 100.0
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current workflow status"""
        try:
            if workflow_id in self.active_workflows:
                return self.active_workflows[workflow_id].copy()
            
            # Try to retrieve from memory system
            # This would use memory manager's retrieval functionality
            return None
            
        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")
            return None

class WorkflowErrorHandler:
    """Comprehensive workflow error handling system"""
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        audit_logger: AuditLogger
    ):
        self.memory_manager = memory_manager
        self.audit_logger = audit_logger
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.dead_letter_queue = DeadLetterQueue(memory_manager, audit_logger)
        self.status_tracker = WorkflowStatusTracker(memory_manager)
        
        # Error handling configuration
        self.retry_config = {
            ErrorCategory.TRANSIENT: {"max_retries": 3, "base_delay": 1.0, "max_delay": 60.0},
            ErrorCategory.PERMISSION: {"max_retries": 0, "base_delay": 0.0, "max_delay": 0.0},
            ErrorCategory.DATA: {"max_retries": 1, "base_delay": 0.5, "max_delay": 5.0},
            ErrorCategory.SYSTEM: {"max_retries": 2, "base_delay": 2.0, "max_delay": 30.0},
            ErrorCategory.BUSINESS: {"max_retries": 1, "base_delay": 1.0, "max_delay": 10.0}
        }
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service"""
        if service_name not in self.circuit_breakers:
            config = CircuitBreakerConfig()
            self.circuit_breakers[service_name] = CircuitBreaker(service_name, config)
        
        return self.circuit_breakers[service_name]
    
    async def handle_workflow_error(
        self,
        workflow_id: str,
        step_name: Optional[str],
        error: Exception,
        context: Dict[str, Any],
        retry_count: int = 0
    ) -> ErrorInfo:
        """Handle workflow error with appropriate strategy"""
        
        # Categorize error
        error_category = self._categorize_error(error)
        
        # Create error info
        error_info = ErrorInfo(
            error_id=f"error_{uuid.uuid4().hex[:8]}",
            error_category=error_category,
            error_message=str(error),
            step_name=step_name,
            workflow_id=workflow_id,
            tenant_id=context.get("tenant_id", "unknown"),
            employee_id=context.get("employee_id", "unknown"),
            timestamp=datetime.now(timezone.utc),
            retry_count=retry_count,
            stack_trace=self._get_stack_trace(error),
            context=context
        )
        
        # Log error
        await self._log_error(error_info)
        
        # Determine if retry is appropriate
        retry_config = self.retry_config.get(error_category)
        if retry_count < retry_config["max_retries"]:
            # Calculate delay for retry
            delay = min(
                retry_config["base_delay"] * (2 ** retry_count),
                retry_config["max_delay"]
            )
            
            logger.info(
                f"Will retry workflow {workflow_id} step {step_name} "
                f"in {delay}s (attempt {retry_count + 1}/{retry_config['max_retries'] + 1})"
            )
            
            # Update status tracker
            await self.status_tracker.update_step_status(
                workflow_id, step_name, StepStatus.RETRYING, error=str(error)
            )
            
        else:
            # Max retries exceeded - send to dead letter queue
            logger.error(
                f"Max retries exceeded for workflow {workflow_id} step {step_name}, "
                f"sending to dead letter queue"
            )
            
            await self.dead_letter_queue.add_failed_workflow(
                workflow_id, error_info, context
            )
            
            # Update status tracker
            await self.status_tracker.update_step_status(
                workflow_id, step_name, StepStatus.FAILED, error=str(error)
            )
        
        return error_info
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error for appropriate handling"""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Permission errors
        if "permission" in error_str or "rbac" in error_str or "403" in error_str:
            return ErrorCategory.PERMISSION
        
        # Transient errors
        if any(term in error_str for term in ["timeout", "connection", "network", "rate limit"]):
            return ErrorCategory.TRANSIENT
        
        # System errors
        if any(term in error_str for term in ["service unavailable", "502", "503", "504"]):
            return ErrorCategory.SYSTEM
        
        # Data errors
        if any(term in error_str for term in ["invalid", "parse", "format", "validation"]):
            return ErrorCategory.DATA
        
        # Default to business logic error
        return ErrorCategory.BUSINESS
    
    def _get_stack_trace(self, error: Exception) -> str:
        """Get formatted stack trace"""
        import traceback
        return traceback.format_exc()
    
    async def _log_error(self, error_info: ErrorInfo) -> None:
        """Log error to audit system"""
        try:
            await self.audit_logger.log_error_event(
                event_type="workflow_error",
                user_id=error_info.employee_id,
                tenant_id=error_info.tenant_id,
                details={
                    "error_id": error_info.error_id,
                    "error_category": error_info.error_category.value,
                    "error_message": error_info.error_message,
                    "workflow_id": error_info.workflow_id,
                    "step_name": error_info.step_name,
                    "retry_count": error_info.retry_count
                }
            )
        except Exception as e:
            logger.error(f"Failed to log error to audit system: {e}")

# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Circuit breaker is open"""
    pass

class CircuitBreakerTimeoutError(Exception):
    """Circuit breaker timeout"""
    pass