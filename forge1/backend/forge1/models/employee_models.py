# forge1/backend/forge1/models/employee_models.py
"""
AI Employee Lifecycle Data Models

Comprehensive data models for AI employee management including clients, employees,
interactions, and all related configurations with proper validation and type safety.

Requirements: 1.2, 4.1, 4.2, 4.3
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator, root_validator
from decimal import Decimal
import uuid


# Enums for status and configuration options
class ClientStatus(str, Enum):
    """Client account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class ClientTier(str, Enum):
    """Client service tier"""
    STANDARD = "standard"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class SecurityLevel(str, Enum):
    """Security classification levels"""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"


class EmployeeStatus(str, Enum):
    """AI Employee status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    ARCHIVED = "archived"


class CommunicationStyle(str, Enum):
    """Employee communication styles"""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    TECHNICAL = "technical"
    CASUAL = "casual"


class FormalityLevel(str, Enum):
    """Employee formality levels"""
    FORMAL = "formal"
    CASUAL = "casual"
    ADAPTIVE = "adaptive"


class ExpertiseLevel(str, Enum):
    """Employee expertise levels"""
    EXPERT = "expert"
    INTERMEDIATE = "intermediate"
    BEGINNER = "beginner"


class ResponseLength(str, Enum):
    """Employee response length preferences"""
    CONCISE = "concise"
    DETAILED = "detailed"
    ADAPTIVE = "adaptive"


class MemoryType(str, Enum):
    """Types of employee memory"""
    CONVERSATION = "conversation"
    TASK = "task"
    KNOWLEDGE = "knowledge"
    FEEDBACK = "feedback"
    SYSTEM = "system"


class SummaryType(str, Enum):
    """Types of memory summaries"""
    CONVERSATION = "conversation"
    DAILY = "daily"
    WEEKLY = "weekly"
    PROJECT = "project"
    RELATIONSHIP = "relationship"


class KnowledgeSourceType(str, Enum):
    """Types of knowledge sources"""
    DOCUMENT = "document"
    URL = "url"
    MANUAL = "manual"
    TRAINING = "training"
    API = "api"


# Core Configuration Classes
@dataclass
class PersonalityConfig:
    """Employee personality and behavior configuration"""
    communication_style: CommunicationStyle = CommunicationStyle.PROFESSIONAL
    formality_level: FormalityLevel = FormalityLevel.FORMAL
    expertise_level: ExpertiseLevel = ExpertiseLevel.EXPERT
    response_length: ResponseLength = ResponseLength.DETAILED
    creativity_level: float = 0.7
    empathy_level: float = 0.7
    custom_traits: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate personality configuration"""
        if not 0.0 <= self.creativity_level <= 1.0:
            raise ValueError("creativity_level must be between 0.0 and 1.0")
        if not 0.0 <= self.empathy_level <= 1.0:
            raise ValueError("empathy_level must be between 0.0 and 1.0")


@dataclass
class ModelPreferences:
    """Employee's preferred AI models and settings"""
    primary_model: str = "gpt-4"
    fallback_models: List[str] = field(default_factory=lambda: ["gpt-3.5-turbo"])
    temperature: float = 0.7
    max_tokens: int = 2000
    specialized_models: Dict[str, str] = field(default_factory=dict)  # task_type -> model
    
    def __post_init__(self):
        """Validate model preferences"""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass
class ClientConfiguration:
    """Client-specific configuration and limits"""
    max_employees: int = 10
    allowed_models: List[str] = field(default_factory=lambda: ["gpt-4", "gpt-3.5-turbo"])
    security_level: SecurityLevel = SecurityLevel.STANDARD
    compliance_requirements: List[str] = field(default_factory=list)
    custom_settings: Dict[str, Any] = field(default_factory=dict)


# Main Entity Classes
@dataclass
class Client:
    """Client entity with configuration and metadata"""
    id: str
    name: str
    industry: str
    tier: ClientTier
    configuration: ClientConfiguration
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: ClientStatus = ClientStatus.ACTIVE
    
    def can_create_employee(self) -> bool:
        """Check if client can create more employees"""
        # This would check against current employee count in production
        return self.status == ClientStatus.ACTIVE
    
    def is_model_allowed(self, model: str) -> bool:
        """Check if client is allowed to use specific model"""
        return model in self.configuration.allowed_models


@dataclass
class Employee:
    """AI Employee entity with complete configuration"""
    id: str
    client_id: str
    name: str
    role: str
    personality: PersonalityConfig
    model_preferences: ModelPreferences
    tool_access: List[str]
    knowledge_sources: List[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_interaction_at: Optional[datetime] = None
    status: EmployeeStatus = EmployeeStatus.ACTIVE
    memory_context: Optional[List['MemoryItem']] = None
    
    # MCAE Integration fields
    workflow_id: Optional[str] = None  # MCAE workflow identifier
    mcae_agent_config: Optional[Dict[str, Any]] = None  # MCAE-specific configuration
    
    def update_last_interaction(self):
        """Update last interaction timestamp"""
        self.last_interaction_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
    
    def is_active(self) -> bool:
        """Check if employee is active and available"""
        return self.status == EmployeeStatus.ACTIVE
    
    def get_system_prompt(self) -> str:
        """Generate system prompt based on employee configuration"""
        return f"""You are {self.name}, a {self.role} AI assistant.

Personality:
- Communication style: {self.personality.communication_style.value}
- Formality level: {self.personality.formality_level.value}
- Expertise level: {self.personality.expertise_level.value}
- Response length: {self.personality.response_length.value}
- Creativity level: {self.personality.creativity_level}
- Empathy level: {self.personality.empathy_level}

Available tools: {', '.join(self.tool_access) if self.tool_access else 'None'}
Knowledge sources: {', '.join(self.knowledge_sources) if self.knowledge_sources else 'General knowledge'}

Always maintain your personality and use your available tools when appropriate."""


@dataclass
class MemoryItem:
    """Individual memory item for employee context"""
    id: str
    content: str
    response: str
    timestamp: datetime
    memory_type: MemoryType = MemoryType.CONVERSATION
    importance_score: float = 0.5
    relevance_score: float = 0.0
    embedding: Optional[List[float]] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmployeeInteraction:
    """Single interaction with an AI employee"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    employee_id: str = ""
    client_id: str = ""
    session_id: Optional[str] = None
    message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time_ms: Optional[float] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    cost: Optional[Decimal] = None


@dataclass
class EmployeeResponse:
    """Response from an AI employee interaction"""
    message: str
    employee_id: str
    interaction_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model_used: str = "gpt-4"
    tokens_used: int = 0
    processing_time_ms: float = 0.0
    cost: Decimal = Decimal('0.00')
    embedding: Optional[List[float]] = None
    confidence_score: float = 1.0
    context_used: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)


# Pydantic Models for API Input/Output
class ClientInfo(BaseModel):
    """Input model for client creation"""
    name: str = Field(..., min_length=1, max_length=255, description="Client company name")
    industry: str = Field(..., min_length=1, max_length=100, description="Client industry")
    tier: ClientTier = Field(ClientTier.STANDARD, description="Service tier")
    max_employees: int = Field(10, ge=1, le=10000, description="Maximum number of employees")
    allowed_models: List[str] = Field(
        default=["gpt-4", "gpt-3.5-turbo"],
        description="List of allowed AI models"
    )
    security_level: SecurityLevel = Field(
        SecurityLevel.STANDARD,
        description="Security classification level"
    )
    compliance_requirements: List[str] = Field(
        default=[],
        description="List of compliance requirements (GDPR, HIPAA, etc.)"
    )
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Client name cannot be empty')
        return v.strip()


class EmployeeRequirements(BaseModel):
    """Input model for employee creation from client requirements"""
    role: str = Field(..., min_length=1, max_length=255, description="Employee role/job title")
    industry: str = Field(..., min_length=1, max_length=100, description="Industry context")
    expertise_areas: List[str] = Field(..., min_items=1, description="Areas of expertise")
    communication_style: CommunicationStyle = Field(
        CommunicationStyle.PROFESSIONAL,
        description="Preferred communication style"
    )
    formality_level: FormalityLevel = Field(
        FormalityLevel.FORMAL,
        description="Formality level"
    )
    expertise_level: ExpertiseLevel = Field(
        ExpertiseLevel.EXPERT,
        description="Expertise level"
    )
    response_length: ResponseLength = Field(
        ResponseLength.DETAILED,
        description="Preferred response length"
    )
    creativity_level: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Creativity level (0.0 to 1.0)"
    )
    empathy_level: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Empathy level (0.0 to 1.0)"
    )
    tools_needed: List[str] = Field(default=[], description="Required tools and integrations")
    knowledge_domains: List[str] = Field(default=[], description="Knowledge domains")
    personality_traits: Dict[str, Any] = Field(
        default={},
        description="Additional personality traits"
    )
    model_preferences: Optional[Dict[str, Any]] = Field(
        None,
        description="Specific model preferences"
    )
    
    # MCAE Integration fields
    workflow_type: Optional[str] = Field(
        "standard",
        description="Type of MCAE workflow to create"
    )
    collaboration_mode: Optional[str] = Field(
        "sequential",
        description="How agents collaborate (sequential, parallel, adaptive)"
    )
    
    @validator('role')
    def validate_role(cls, v):
        if not v.strip():
            raise ValueError('Role cannot be empty')
        return v.strip()
    
    @validator('expertise_areas')
    def validate_expertise_areas(cls, v):
        if not v or all(not area.strip() for area in v):
            raise ValueError('At least one expertise area is required')
        return [area.strip() for area in v if area.strip()]


class InteractionRequest(BaseModel):
    """Input model for employee interaction"""
    message: str = Field(..., min_length=1, max_length=10000, description="Message to employee")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    context: Dict[str, Any] = Field(default={}, description="Additional context")
    include_memory: bool = Field(True, description="Whether to include memory context")
    memory_limit: int = Field(10, ge=1, le=100, description="Number of memory items to include")
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()


class EmployeeUpdate(BaseModel):
    """Input model for updating employee configuration"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    personality: Optional[Dict[str, Any]] = None
    model_preferences: Optional[Dict[str, Any]] = None
    tool_access: Optional[List[str]] = None
    knowledge_sources: Optional[List[str]] = None
    status: Optional[EmployeeStatus] = None


class MemoryQuery(BaseModel):
    """Input model for querying employee memory"""
    query: Optional[str] = Field(None, description="Semantic search query")
    memory_types: List[MemoryType] = Field(default=[], description="Filter by memory types")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    include_embeddings: bool = Field(False, description="Include embedding vectors")
    min_importance: float = Field(0.0, ge=0.0, le=1.0, description="Minimum importance score")


# Response Models
class ClientResponse(BaseModel):
    """Response model for client operations"""
    id: str
    name: str
    industry: str
    tier: ClientTier
    max_employees: int
    current_employees: int = 0
    security_level: SecurityLevel
    status: ClientStatus
    created_at: datetime
    
    class Config:
        use_enum_values = True


class EmployeeResponse(BaseModel):
    """Response model for employee operations"""
    id: str
    client_id: str
    name: str
    role: str
    communication_style: CommunicationStyle
    expertise_level: ExpertiseLevel
    status: EmployeeStatus
    created_at: datetime
    last_interaction_at: Optional[datetime]
    
    class Config:
        use_enum_values = True


class InteractionResponse(BaseModel):
    """Response model for employee interactions"""
    message: str
    employee_id: str
    employee_name: str
    interaction_id: str
    session_id: Optional[str]
    timestamp: datetime
    model_used: str
    processing_time_ms: float
    tokens_used: int
    cost: Decimal
    confidence_score: float
    
    class Config:
        json_encoders = {
            Decimal: lambda v: float(v),
            datetime: lambda v: v.isoformat()
        }


class MemoryResponse(BaseModel):
    """Response model for memory queries"""
    memories: List[Dict[str, Any]]
    total_count: int
    query_time_ms: float
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Exception Classes
class EmployeeLifecycleError(Exception):
    """Base exception for employee lifecycle operations"""
    pass


class ClientNotFoundError(EmployeeLifecycleError):
    """Client not found or inaccessible"""
    pass


class EmployeeNotFoundError(EmployeeLifecycleError):
    """Employee not found or inaccessible"""
    pass


class EmployeeLimitExceededError(EmployeeLifecycleError):
    """Client has reached maximum employee limit"""
    pass


class InvalidConfigurationError(EmployeeLifecycleError):
    """Invalid employee or client configuration"""
    pass


class MemoryAccessError(EmployeeLifecycleError):
    """Error accessing employee memory"""
    pass


class TenantIsolationError(EmployeeLifecycleError):
    """Tenant isolation violation"""
    pass


# Utility Functions
def generate_employee_id(client_id: str) -> str:
    """Generate unique employee ID"""
    return f"{client_id}_emp_{uuid.uuid4().hex[:8]}"


def generate_client_id() -> str:
    """Generate unique client ID"""
    return f"client_{uuid.uuid4().hex[:8]}"


def generate_interaction_id() -> str:
    """Generate unique interaction ID"""
    return f"int_{uuid.uuid4().hex[:8]}"


def create_default_personality() -> PersonalityConfig:
    """Create default personality configuration"""
    return PersonalityConfig()


def create_default_model_preferences() -> ModelPreferences:
    """Create default model preferences"""
    return ModelPreferences()


def create_default_client_config() -> ClientConfiguration:
    """Create default client configuration"""
    return ClientConfiguration()


# Type aliases for convenience
EmployeeDict = Dict[str, Any]
ClientDict = Dict[str, Any]
InteractionDict = Dict[str, Any]
MemoryDict = Dict[str, Any]