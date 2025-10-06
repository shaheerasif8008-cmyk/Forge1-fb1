# forge1/backend/forge1/models/__init__.py
"""
Forge1 Data Models

Centralized imports for all data models used throughout the Forge1 platform.
"""

# Employee lifecycle models
from .employee_models import (
    # Enums
    ClientStatus,
    ClientTier,
    SecurityLevel,
    EmployeeStatus,
    CommunicationStyle,
    FormalityLevel,
    ExpertiseLevel,
    ResponseLength,
    MemoryType,
    SummaryType,
    KnowledgeSourceType,
    
    # Configuration classes
    PersonalityConfig,
    ModelPreferences,
    ClientConfiguration,
    
    # Entity classes
    Client,
    Employee,
    MemoryItem,
    EmployeeInteraction,
    EmployeeResponse,
    
    # Pydantic models for API
    ClientInfo,
    EmployeeRequirements,
    InteractionRequest,
    EmployeeUpdate,
    MemoryQuery,
    ClientResponse,
    EmployeeResponse as EmployeeAPIResponse,
    InteractionResponse,
    MemoryResponse,
    
    # Exceptions
    EmployeeLifecycleError,
    ClientNotFoundError,
    EmployeeNotFoundError,
    EmployeeLimitExceededError,
    InvalidConfigurationError,
    MemoryAccessError,
    TenantIsolationError,
    
    # Utility functions
    generate_employee_id,
    generate_client_id,
    generate_interaction_id,
    create_default_personality,
    create_default_model_preferences,
    create_default_client_config,
    
    # Type aliases
    EmployeeDict,
    ClientDict,
    InteractionDict,
    MemoryDict,
)

__all__ = [
    # Enums
    "ClientStatus",
    "ClientTier", 
    "SecurityLevel",
    "EmployeeStatus",
    "CommunicationStyle",
    "FormalityLevel",
    "ExpertiseLevel",
    "ResponseLength",
    "MemoryType",
    "SummaryType",
    "KnowledgeSourceType",
    
    # Configuration classes
    "PersonalityConfig",
    "ModelPreferences",
    "ClientConfiguration",
    
    # Entity classes
    "Client",
    "Employee",
    "MemoryItem",
    "EmployeeInteraction",
    "EmployeeResponse",
    
    # Pydantic models for API
    "ClientInfo",
    "EmployeeRequirements",
    "InteractionRequest",
    "EmployeeUpdate",
    "MemoryQuery",
    "ClientResponse",
    "EmployeeAPIResponse",
    "InteractionResponse",
    "MemoryResponse",
    
    # Exceptions
    "EmployeeLifecycleError",
    "ClientNotFoundError",
    "EmployeeNotFoundError",
    "EmployeeLimitExceededError",
    "InvalidConfigurationError",
    "MemoryAccessError",
    "TenantIsolationError",
    
    # Utility functions
    "generate_employee_id",
    "generate_client_id",
    "generate_interaction_id",
    "create_default_personality",
    "create_default_model_preferences",
    "create_default_client_config",
    
    # Type aliases
    "EmployeeDict",
    "ClientDict",
    "InteractionDict",
    "MemoryDict",
]