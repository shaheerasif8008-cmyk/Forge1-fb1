"""
Workflow Models for MCAE Integration

Data models for MCAE workflow management, execution context,
and results within the Forge1 ecosystem.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
from decimal import Decimal
import uuid


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    REGISTERED = "registered"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowType(str, Enum):
    """Types of MCAE workflows"""
    STANDARD = "standard"
    LEGAL_INTAKE = "legal_intake"
    LEGAL_ANALYSIS = "legal_analysis"
    LEGAL_RESEARCH = "legal_research"
    CUSTOMER_SUPPORT = "customer_support"
    SALES_PROCESS = "sales_process"
    CONTENT_CREATION = "content_creation"
    DATA_ANALYSIS = "data_analysis"
    CUSTOM = "custom"


class CollaborationMode(str, Enum):
    """Agent collaboration modes"""
    SEQUENTIAL = "sequential"  # Agents work one after another
    PARALLEL = "parallel"      # Agents work simultaneously
    ADAPTIVE = "adaptive"      # Dynamic collaboration based on task
    HIERARCHICAL = "hierarchical"  # Manager-worker pattern


@dataclass
class WorkflowContext:
    """
    Context passed to MCAE workflows.
    
    Contains all necessary information for workflow execution
    including tenant isolation, employee configuration, and task details.
    """
    tenant_id: str
    employee_id: str
    session_id: str
    user_id: str
    task_description: str
    workflow_type: WorkflowType = WorkflowType.STANDARD
    collaboration_mode: CollaborationMode = CollaborationMode.SEQUENTIAL
    memory_context: List[Dict] = field(default_factory=list)
    tool_permissions: List[str] = field(default_factory=list)
    model_preferences: Optional[Dict] = None
    custom_config: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1-10, higher is more priority
    timeout_seconds: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "tenant_id": self.tenant_id,
            "employee_id": self.employee_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "task_description": self.task_description,
            "workflow_type": self.workflow_type.value,
            "collaboration_mode": self.collaboration_mode.value,
            "memory_context": self.memory_context,
            "tool_permissions": self.tool_permissions,
            "model_preferences": self.model_preferences,
            "custom_config": self.custom_config,
            "priority": self.priority,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class WorkflowResult:
    """
    Result from MCAE workflow execution.
    
    Contains execution results, metadata, and performance information.
    """
    workflow_id: str
    session_id: str
    status: WorkflowStatus
    result: Optional[Dict] = None
    error: Optional[str] = None
    agent_messages: List[Dict] = field(default_factory=list)
    execution_steps: List[Dict] = field(default_factory=list)
    tokens_used: int = 0
    cost: Decimal = field(default_factory=lambda: Decimal('0.00'))
    execution_time_seconds: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tenant_id: Optional[str] = None
    employee_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "workflow_id": self.workflow_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "agent_messages": self.agent_messages,
            "execution_steps": self.execution_steps,
            "tokens_used": self.tokens_used,
            "cost": float(self.cost),
            "execution_time_seconds": self.execution_time_seconds,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "tenant_id": self.tenant_id,
            "employee_id": self.employee_id
        }
    
    def is_successful(self) -> bool:
        """Check if workflow completed successfully"""
        return self.status == WorkflowStatus.COMPLETED and self.error is None
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the workflow result"""
        if self.status == WorkflowStatus.COMPLETED:
            return f"Workflow completed successfully in {self.execution_time_seconds:.2f}s"
        elif self.status == WorkflowStatus.FAILED:
            return f"Workflow failed: {self.error}"
        elif self.status == WorkflowStatus.EXECUTING:
            return "Workflow is currently executing"
        else:
            return f"Workflow status: {self.status.value}"


class WorkflowRequest(BaseModel):
    """Request model for workflow execution"""
    task_description: str = Field(..., min_length=1, max_length=10000)
    workflow_type: WorkflowType = Field(WorkflowType.STANDARD)
    collaboration_mode: CollaborationMode = Field(CollaborationMode.SEQUENTIAL)
    context: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(5, ge=1, le=10)
    timeout_seconds: Optional[int] = Field(None, ge=1, le=3600)
    
    @validator('task_description')
    def validate_task_description(cls, v):
        if not v.strip():
            raise ValueError('Task description cannot be empty')
        return v.strip()


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status queries"""
    workflow_id: str
    employee_id: str
    tenant_id: str
    status: WorkflowStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    error: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None


class WorkflowMetrics(BaseModel):
    """Metrics for workflow performance tracking"""
    total_workflows: int = 0
    successful_workflows: int = 0
    failed_workflows: int = 0
    average_execution_time: float = 0.0
    total_tokens_used: int = 0
    total_cost: Decimal = Decimal('0.00')
    workflows_by_type: Dict[str, int] = Field(default_factory=dict)
    workflows_by_status: Dict[str, int] = Field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate workflow success rate"""
        if self.total_workflows == 0:
            return 0.0
        return (self.successful_workflows / self.total_workflows) * 100
    
    @property
    def failure_rate(self) -> float:
        """Calculate workflow failure rate"""
        if self.total_workflows == 0:
            return 0.0
        return (self.failed_workflows / self.total_workflows) * 100


# Exception Classes for Workflow Operations
class WorkflowError(Exception):
    """Base exception for workflow operations"""
    pass


class WorkflowNotFoundError(WorkflowError):
    """Workflow not found or inaccessible"""
    pass


class WorkflowExecutionError(WorkflowError):
    """Workflow execution failed"""
    pass


class WorkflowTimeoutError(WorkflowError):
    """Workflow execution timed out"""
    pass


class WorkflowPermissionError(WorkflowError):
    """Insufficient permissions for workflow operation"""
    pass


# Utility functions
def generate_workflow_id(tenant_id: str, employee_id: str) -> str:
    """Generate a unique workflow ID"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"wf_{tenant_id}_{employee_id}_{timestamp}_{unique_id}"


def create_workflow_context(
    tenant_id: str,
    employee_id: str,
    task_description: str,
    **kwargs
) -> WorkflowContext:
    """Create a workflow context with default values"""
    session_id = kwargs.get('session_id', f"session_{uuid.uuid4().hex[:8]}")
    user_id = kwargs.get('user_id', employee_id)
    
    return WorkflowContext(
        tenant_id=tenant_id,
        employee_id=employee_id,
        session_id=session_id,
        user_id=user_id,
        task_description=task_description,
        workflow_type=WorkflowType(kwargs.get('workflow_type', 'standard')),
        collaboration_mode=CollaborationMode(kwargs.get('collaboration_mode', 'sequential')),
        memory_context=kwargs.get('memory_context', []),
        tool_permissions=kwargs.get('tool_permissions', []),
        model_preferences=kwargs.get('model_preferences'),
        custom_config=kwargs.get('custom_config', {}),
        priority=kwargs.get('priority', 5),
        timeout_seconds=kwargs.get('timeout_seconds')
    )