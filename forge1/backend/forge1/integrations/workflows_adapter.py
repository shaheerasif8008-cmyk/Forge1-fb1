"""
workflows-py Integration Adapter for Forge1

Provides workflows-py integration for sub-orchestration within MCAE stages.
Maintains tenant isolation, RBAC enforcement, and proper observability while
enabling explicit step graphs for complex multi-step processes.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import json
import traceback

# workflows-py imports
try:
    from workflows import workflow, step, Workflow, WorkflowContext
    from workflows.exceptions import WorkflowError, StepError
    WORKFLOWS_AVAILABLE = True
except ImportError:
    # Fallback for when workflows-py is not available
    WORKFLOWS_AVAILABLE = False
    
    # Mock classes for development
    class Workflow:
        pass
    
    class WorkflowContext:
        pass
    
    def workflow(cls):
        return cls
    
    def step(func):
        return func

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Forge1 imports
from forge1.core.tenancy import get_current_tenant, set_current_tenant
from forge1.core.model_router import ModelRouter
from forge1.core.memory_manager import MemoryManager
from forge1.core.security_manager import SecurityManager
from forge1.core.audit_logger import AuditLogger
from forge1.integrations.llamaindex_adapter import LlamaIndexAdapter, ExecutionContext

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StepStatus(Enum):
    """Individual step execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"

@dataclass
class HandoffPacket:
    """Standardized data packet passed between workflow steps"""
    tenant_id: str
    employee_id: str
    case_id: str
    summary: str
    citations: List[Dict[str, Any]]
    attachment_refs: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    step_outputs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.step_outputs is None:
            self.step_outputs = {}

@dataclass
class StepResult:
    """Result from individual step execution"""
    step_name: str
    status: StepStatus
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    retry_count: int = 0
    span_id: Optional[str] = None

@dataclass
class WorkflowResult:
    """Result from complete workflow execution"""
    workflow_id: str
    status: WorkflowStatus
    steps: List[StepResult]
    final_output: HandoffPacket
    execution_time_ms: float
    error: Optional[str] = None
    trace_id: Optional[str] = None

class WorkflowsIntegrationError(Exception):
    """Base exception for workflows-py integration errors"""
    pass

class WorkflowExecutionError(WorkflowsIntegrationError):
    """Workflow execution error"""
    pass

class StepExecutionError(WorkflowsIntegrationError):
    """Step execution error"""
    pass

class Forge1WorkflowContext(WorkflowContext):
    """Extended workflow context with Forge1 integration"""
    
    def __init__(
        self,
        execution_context: ExecutionContext,
        llamaindex_adapter: LlamaIndexAdapter,
        model_router: ModelRouter,
        memory_manager: MemoryManager,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.execution_context = execution_context
        self.llamaindex_adapter = llamaindex_adapter
        self.model_router = model_router
        self.memory_manager = memory_manager
        self.handoff_packet = None
        self.step_results = []
        
    def set_handoff_packet(self, packet: HandoffPacket):
        """Set the current handoff packet"""
        self.handoff_packet = packet
    
    def get_handoff_packet(self) -> HandoffPacket:
        """Get the current handoff packet"""
        return self.handoff_packet
    
    def add_step_result(self, result: StepResult):
        """Add a step result to the context"""
        self.step_results.append(result)

class Forge1Workflow:
    """Base class for Forge1-integrated workflows"""
    
    def __init__(
        self,
        workflow_id: str,
        context: Forge1WorkflowContext,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout_seconds: float = 300.0
    ):
        self.workflow_id = workflow_id
        self.context = context
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout_seconds = timeout_seconds
        self.status = WorkflowStatus.PENDING
        self.start_time = None
        self.end_time = None
        
    async def execute(self, input_data: Dict[str, Any]) -> WorkflowResult:
        """Execute the workflow with proper error handling and observability"""
        self.start_time = datetime.now(timezone.utc)
        self.status = WorkflowStatus.RUNNING
        
        with tracer.start_as_current_span(
            f"workflow_execution_{self.workflow_id}",
            attributes={
                "workflow.id": self.workflow_id,
                "tenant.id": self.context.execution_context.tenant_id,
                "employee.id": self.context.execution_context.employee_id,
                "case.id": self.context.execution_context.case_id
            }
        ) as span:
            try:
                # Set tenant context
                set_current_tenant(self.context.execution_context.tenant_id)
                
                # Initialize handoff packet
                initial_packet = HandoffPacket(
                    tenant_id=self.context.execution_context.tenant_id,
                    employee_id=self.context.execution_context.employee_id,
                    case_id=self.context.execution_context.case_id,
                    summary="Workflow started",
                    citations=[],
                    attachment_refs=[],
                    metadata=input_data
                )
                self.context.set_handoff_packet(initial_packet)
                
                # Execute workflow steps
                result = await self._execute_steps()
                
                self.end_time = datetime.now(timezone.utc)
                execution_time = (self.end_time - self.start_time).total_seconds() * 1000
                
                if result:
                    self.status = WorkflowStatus.COMPLETED
                    span.set_status(Status(StatusCode.OK))
                else:
                    self.status = WorkflowStatus.FAILED
                    span.set_status(Status(StatusCode.ERROR, "Workflow execution failed"))
                
                return WorkflowResult(
                    workflow_id=self.workflow_id,
                    status=self.status,
                    steps=self.context.step_results,
                    final_output=self.context.get_handoff_packet(),
                    execution_time_ms=execution_time,
                    trace_id=span.get_span_context().trace_id.to_bytes(16, 'big').hex()
                )
                
            except Exception as e:
                self.status = WorkflowStatus.FAILED
                self.end_time = datetime.now(timezone.utc)
                execution_time = (self.end_time - self.start_time).total_seconds() * 1000
                
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                
                logger.error(f"Workflow {self.workflow_id} failed: {e}")
                
                return WorkflowResult(
                    workflow_id=self.workflow_id,
                    status=self.status,
                    steps=self.context.step_results,
                    final_output=self.context.get_handoff_packet(),
                    execution_time_ms=execution_time,
                    error=str(e),
                    trace_id=span.get_span_context().trace_id.to_bytes(16, 'big').hex()
                )
    
    async def _execute_steps(self) -> bool:
        """Execute workflow steps - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _execute_steps")
    
    async def _execute_step_with_retry(
        self,
        step_name: str,
        step_func: Callable,
        *args,
        **kwargs
    ) -> StepResult:
        """Execute a single step with retry logic and observability"""
        
        with tracer.start_as_current_span(
            f"step_{step_name}",
            attributes={
                "step.name": step_name,
                "workflow.id": self.workflow_id,
                "tenant.id": self.context.execution_context.tenant_id,
                "employee.id": self.context.execution_context.employee_id
            }
        ) as span:
            
            start_time = datetime.now(timezone.utc)
            retry_count = 0
            last_error = None
            
            while retry_count <= self.max_retries:
                try:
                    # Execute step
                    output = await step_func(*args, **kwargs)
                    
                    execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    
                    result = StepResult(
                        step_name=step_name,
                        status=StepStatus.COMPLETED,
                        output=output,
                        execution_time_ms=execution_time,
                        retry_count=retry_count,
                        span_id=span.get_span_context().span_id.to_bytes(8, 'big').hex()
                    )
                    
                    span.set_status(Status(StatusCode.OK))
                    span.set_attribute("step.retry_count", retry_count)
                    
                    return result
                    
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    
                    logger.warning(
                        f"Step {step_name} failed (attempt {retry_count}/{self.max_retries + 1}): {e}"
                    )
                    
                    if retry_count <= self.max_retries:
                        # Wait before retry with exponential backoff
                        delay = self.retry_delay * (2 ** (retry_count - 1))
                        await asyncio.sleep(delay)
                        
                        span.add_event(
                            f"retry_attempt_{retry_count}",
                            {"error": str(e), "delay_seconds": delay}
                        )
                    else:
                        # Max retries exceeded
                        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                        
                        result = StepResult(
                            step_name=step_name,
                            status=StepStatus.FAILED,
                            error=str(last_error),
                            execution_time_ms=execution_time,
                            retry_count=retry_count - 1,
                            span_id=span.get_span_context().span_id.to_bytes(8, 'big').hex()
                        )
                        
                        span.set_status(Status(StatusCode.ERROR, str(last_error)))
                        span.record_exception(last_error)
                        
                        return result

class WorkflowsAdapter:
    """Main adapter for workflows-py integration with Forge1"""
    
    def __init__(
        self,
        llamaindex_adapter: LlamaIndexAdapter,
        model_router: ModelRouter,
        memory_manager: MemoryManager,
        security_manager: SecurityManager,
        audit_logger: AuditLogger
    ):
        self.llamaindex_adapter = llamaindex_adapter
        self.model_router = model_router
        self.memory_manager = memory_manager
        self.security_manager = security_manager
        self.audit_logger = audit_logger
        
        # Workflow registry
        self.workflows: Dict[str, Forge1Workflow] = {}
        self.workflow_definitions: Dict[str, type] = {}
        
        # Performance metrics
        self.metrics = {
            "workflows_created": 0,
            "workflows_executed": 0,
            "workflows_completed": 0,
            "workflows_failed": 0,
            "average_execution_time": 0.0,
            "total_steps_executed": 0,
            "steps_failed": 0
        }
        
        self._initialized = False
        logger.info("workflows-py adapter initialized")
    
    async def initialize(self) -> None:
        """Initialize the adapter"""
        if self._initialized:
            return
        
        if not WORKFLOWS_AVAILABLE:
            logger.warning("workflows-py not available - using mock implementation")
        
        self._initialized = True
        logger.info("workflows-py adapter initialized")
    
    async def register_workflow_definition(
        self,
        workflow_name: str,
        workflow_class: type
    ) -> None:
        """Register a workflow definition"""
        self.workflow_definitions[workflow_name] = workflow_class
        logger.info(f"Registered workflow definition: {workflow_name}")
    
    async def create_workflow(
        self,
        workflow_name: str,
        context: ExecutionContext,
        config: Dict[str, Any] = None
    ) -> str:
        """Create a new workflow instance"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Validate workflow definition exists
            if workflow_name not in self.workflow_definitions:
                raise WorkflowsIntegrationError(f"Workflow definition not found: {workflow_name}")
            
            # Generate workflow ID
            workflow_id = f"workflow_{workflow_name}_{uuid.uuid4().hex[:8]}"
            
            # Create Forge1 workflow context
            forge1_context = Forge1WorkflowContext(
                execution_context=context,
                llamaindex_adapter=self.llamaindex_adapter,
                model_router=self.model_router,
                memory_manager=self.memory_manager
            )
            
            # Create workflow instance
            workflow_class = self.workflow_definitions[workflow_name]
            workflow = workflow_class(
                workflow_id=workflow_id,
                context=forge1_context,
                **(config or {})
            )
            
            # Register workflow
            self.workflows[workflow_id] = workflow
            self.metrics["workflows_created"] += 1
            
            logger.info(f"Created workflow {workflow_name} with ID {workflow_id}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Failed to create workflow {workflow_name}: {e}")
            raise WorkflowsIntegrationError(f"Workflow creation failed: {e}")
    
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any]
    ) -> WorkflowResult:
        """Execute a workflow"""
        try:
            # Validate workflow exists
            if workflow_id not in self.workflows:
                raise WorkflowsIntegrationError(f"Workflow not found: {workflow_id}")
            
            workflow = self.workflows[workflow_id]
            
            # Set tenant context
            set_current_tenant(workflow.context.execution_context.tenant_id)
            
            # Execute workflow
            result = await workflow.execute(input_data)
            
            # Update metrics
            self.metrics["workflows_executed"] += 1
            if result.status == WorkflowStatus.COMPLETED:
                self.metrics["workflows_completed"] += 1
            else:
                self.metrics["workflows_failed"] += 1
            
            # Update execution time metrics
            current_avg = self.metrics["average_execution_time"]
            executed_count = self.metrics["workflows_executed"]
            self.metrics["average_execution_time"] = (
                (current_avg * (executed_count - 1) + result.execution_time_ms) / executed_count
            )
            
            # Update step metrics
            self.metrics["total_steps_executed"] += len(result.steps)
            self.metrics["steps_failed"] += sum(
                1 for step in result.steps if step.status == StepStatus.FAILED
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed for {workflow_id}: {e}")
            raise WorkflowExecutionError(f"Workflow execution failed: {e}")
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status"""
        if workflow_id not in self.workflows:
            return {"error": f"Workflow not found: {workflow_id}"}
        
        workflow = self.workflows[workflow_id]
        return {
            "workflow_id": workflow_id,
            "status": workflow.status.value,
            "tenant_id": workflow.context.execution_context.tenant_id,
            "employee_id": workflow.context.execution_context.employee_id,
            "case_id": workflow.context.execution_context.case_id,
            "start_time": workflow.start_time.isoformat() if workflow.start_time else None,
            "end_time": workflow.end_time.isoformat() if workflow.end_time else None,
            "step_count": len(workflow.context.step_results)
        }
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        if workflow.status == WorkflowStatus.RUNNING:
            workflow.status = WorkflowStatus.CANCELLED
            logger.info(f"Cancelled workflow {workflow_id}")
            return True
        
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter performance metrics"""
        return {
            **self.metrics,
            "active_workflows": len(self.workflows),
            "registered_definitions": len(self.workflow_definitions)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on workflows-py integration"""
        try:
            health_status = {
                "status": "healthy" if self._initialized else "unhealthy",
                "initialized": self._initialized,
                "workflows_available": WORKFLOWS_AVAILABLE,
                "active_workflows": len(self.workflows),
                "metrics": self.get_metrics()
            }
            
            if WORKFLOWS_AVAILABLE:
                try:
                    import workflows
                    health_status["workflows_version"] = getattr(workflows, '__version__', 'unknown')
                except Exception:
                    health_status["workflows_version"] = 'unknown'
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self._initialized
            }