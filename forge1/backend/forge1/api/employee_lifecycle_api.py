# forge1/backend/forge1/api/employee_lifecycle_api.py
"""
Employee Lifecycle API Endpoints

REST API endpoints for complete AI employee lifecycle management including
client onboarding, employee creation, interactions, and management.

Requirements: 7.1, 7.2, 7.3, 7.4
"""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query, Header, Request
from fastapi.responses import JSONResponse

from forge1.core.tenancy import set_current_tenant, get_current_tenant
from forge1.services.client_manager import ClientManager
from forge1.services.employee_manager import EmployeeManager
from forge1.services.employee_memory_manager import EmployeeMemoryManager
from forge1.services.personality_manager import PersonalityManager
from forge1.models.employee_models import (
    # Input models
    ClientInfo, EmployeeRequirements, InteractionRequest, EmployeeUpdate, MemoryQuery,
    # Response models
    ClientResponse, EmployeeResponse as EmployeeAPIResponse, InteractionResponse, MemoryResponse,
    # Entity models
    Employee, EmployeeStatus, MemoryType,
    # Exceptions
    ClientNotFoundError, EmployeeNotFoundError, EmployeeLimitExceededError,
    InvalidConfigurationError, TenantIsolationError, MemoryAccessError
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/employees", tags=["Employee Lifecycle"])

# Dependency injection
async def get_client_manager() -> ClientManager:
    """Get client manager instance"""
    manager = ClientManager()
    await manager.initialize()
    return manager

async def get_employee_manager() -> EmployeeManager:
    """Get employee manager instance"""
    manager = EmployeeManager()
    await manager.initialize()
    return manager

async def get_memory_manager() -> EmployeeMemoryManager:
    """Get memory manager instance"""
    manager = EmployeeMemoryManager()
    await manager.initialize()
    return manager

async def get_personality_manager() -> PersonalityManager:
    """Get personality manager instance"""
    return PersonalityManager()

async def extract_tenant_context(
    request: Request,
    x_tenant_id: Optional[str] = Header(None),
    x_client_id: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None)
) -> Dict[str, str]:
    """Extract tenant context from headers"""
    
    # Try headers first
    if x_tenant_id and x_client_id and x_user_id:
        set_current_tenant(x_tenant_id)
        return {
            "tenant_id": x_tenant_id,
            "client_id": x_client_id,
            "user_id": x_user_id
        }
    
    # Try to extract from authorization header (simplified)
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        # In production, this would decode JWT and extract tenant info
        # For now, use a simple approach
        token = auth_header[7:]
        if token == "demo_token":
            demo_context = {
                "tenant_id": "demo_client_001",
                "client_id": "demo_client_001", 
                "user_id": "demo_user_001"
            }
            set_current_tenant(demo_context["tenant_id"])
            return demo_context
    
    # Default for development/testing
    default_context = {
        "tenant_id": "client_demo_001",
        "client_id": "client_demo_001",
        "user_id": "user_demo_001"
    }
    set_current_tenant(default_context["tenant_id"])
    return default_context


# Client Management Endpoints

@router.post("/clients", response_model=ClientResponse)
async def onboard_client(
    client_info: ClientInfo,
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    client_manager: ClientManager = Depends(get_client_manager)
):
    """
    Onboard a new client with initial configuration.
    
    Creates a new client account with tier-based configuration,
    employee limits, and security settings.
    """
    try:
        start_time = time.time()
        
        # Create client
        client = await client_manager.onboard_client(client_info)
        
        # Get current employee count
        employees = await client_manager.get_client_employees(client.id)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Client onboarded: {client.id} in {processing_time:.3f}s")
        
        return ClientResponse(
            id=client.id,
            name=client.name,
            industry=client.industry,
            tier=client.tier,
            max_employees=client.configuration.max_employees,
            current_employees=len(employees),
            security_level=client.configuration.security_level,
            status=client.status,
            created_at=client.created_at
        )
        
    except Exception as e:
        logger.error(f"Client onboarding failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/clients/{client_id}", response_model=ClientResponse)
async def get_client(
    client_id: str,
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    client_manager: ClientManager = Depends(get_client_manager)
):
    """Get client information and statistics."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        client = await client_manager.get_client(client_id)
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Get current employee count
        employees = await client_manager.get_client_employees(client_id)
        
        return ClientResponse(
            id=client.id,
            name=client.name,
            industry=client.industry,
            tier=client.tier,
            max_employees=client.configuration.max_employees,
            current_employees=len(employees),
            security_level=client.configuration.security_level,
            status=client.status,
            created_at=client.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get client {client_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve client")


@router.get("/clients/{client_id}/stats")
async def get_client_stats(
    client_id: str,
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    client_manager: ClientManager = Depends(get_client_manager)
):
    """Get comprehensive client statistics."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        stats = await client_manager.get_client_stats(client_id)
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get client stats {client_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve client statistics")


# Employee Management Endpoints

@router.post("/clients/{client_id}/employees", response_model=EmployeeAPIResponse)
async def create_employee(
    client_id: str,
    requirements: EmployeeRequirements,
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    employee_manager: EmployeeManager = Depends(get_employee_manager),
    personality_manager: PersonalityManager = Depends(get_personality_manager)
):
    """
    Create a new AI employee for a client based on requirements.
    
    Generates a unique AI employee with personality, model preferences,
    and capabilities tailored to the client's specifications.
    """
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        start_time = time.time()
        
        # Create employee
        employee = await employee_manager.create_employee(client_id, requirements)
        
        # Generate personality summary
        personality_summary = personality_manager.generate_personality_summary(employee.personality)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Employee created: {employee.id} for client {client_id} in {processing_time:.3f}s")
        
        return EmployeeAPIResponse(
            id=employee.id,
            client_id=employee.client_id,
            name=employee.name,
            role=employee.role,
            communication_style=employee.personality.communication_style,
            expertise_level=employee.personality.expertise_level,
            status=employee.status,
            created_at=employee.created_at,
            last_interaction_at=employee.last_interaction_at
        )
        
    except EmployeeLimitExceededError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ClientNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Employee creation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create employee")


@router.get("/clients/{client_id}/employees", response_model=List[EmployeeAPIResponse])
async def list_employees(
    client_id: str,
    status: Optional[EmployeeStatus] = Query(None, description="Filter by employee status"),
    role: Optional[str] = Query(None, description="Filter by role (partial match)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    employee_manager: EmployeeManager = Depends(get_employee_manager)
):
    """List all employees for a client with optional filtering."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        employees = await employee_manager.list_employees(
            client_id, status, role, limit, offset
        )
        
        return [
            EmployeeAPIResponse(
                id=emp.id,
                client_id=emp.client_id,
                name=emp.name,
                role=emp.role,
                communication_style=emp.personality.communication_style,
                expertise_level=emp.personality.expertise_level,
                status=emp.status,
                created_at=emp.created_at,
                last_interaction_at=emp.last_interaction_at
            )
            for emp in employees
        ]
        
    except Exception as e:
        logger.error(f"Failed to list employees for client {client_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to list employees")


@router.get("/clients/{client_id}/employees/{employee_id}", response_model=Dict[str, Any])
async def get_employee(
    client_id: str,
    employee_id: str,
    include_stats: bool = Query(False, description="Include employee statistics"),
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    employee_manager: EmployeeManager = Depends(get_employee_manager)
):
    """Get detailed employee information."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        employee = await employee_manager.get_employee(client_id, employee_id)
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        response = {
            "id": employee.id,
            "client_id": employee.client_id,
            "name": employee.name,
            "role": employee.role,
            "status": employee.status.value,
            "created_at": employee.created_at.isoformat(),
            "last_interaction_at": employee.last_interaction_at.isoformat() if employee.last_interaction_at else None,
            "personality": {
                "communication_style": employee.personality.communication_style.value,
                "formality_level": employee.personality.formality_level.value,
                "expertise_level": employee.personality.expertise_level.value,
                "response_length": employee.personality.response_length.value,
                "creativity_level": employee.personality.creativity_level,
                "empathy_level": employee.personality.empathy_level,
                "custom_traits": employee.personality.custom_traits
            },
            "model_preferences": {
                "primary_model": employee.model_preferences.primary_model,
                "fallback_models": employee.model_preferences.fallback_models,
                "temperature": employee.model_preferences.temperature,
                "max_tokens": employee.model_preferences.max_tokens,
                "specialized_models": employee.model_preferences.specialized_models
            },
            "capabilities": {
                "tools": employee.tool_access,
                "knowledge_sources": employee.knowledge_sources
            }
        }
        
        if include_stats:
            stats = await employee_manager.get_employee_stats(client_id, employee_id)
            response["statistics"] = stats
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve employee")


@router.put("/clients/{client_id}/employees/{employee_id}", response_model=Dict[str, Any])
async def update_employee(
    client_id: str,
    employee_id: str,
    updates: EmployeeUpdate,
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    employee_manager: EmployeeManager = Depends(get_employee_manager)
):
    """Update employee configuration."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        # Convert Pydantic model to dict, excluding None values
        update_dict = updates.dict(exclude_none=True)
        
        if not update_dict:
            raise HTTPException(status_code=400, detail="No updates provided")
        
        updated_employee = await employee_manager.update_employee(
            client_id, employee_id, update_dict
        )
        
        return {
            "id": updated_employee.id,
            "name": updated_employee.name,
            "role": updated_employee.role,
            "status": updated_employee.status.value,
            "updated_at": updated_employee.updated_at.isoformat(),
            "message": "Employee updated successfully"
        }
        
    except EmployeeNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidConfigurationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update employee")


@router.delete("/clients/{client_id}/employees/{employee_id}")
async def delete_employee(
    client_id: str,
    employee_id: str,
    archive_only: bool = Query(True, description="Archive instead of hard delete"),
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    employee_manager: EmployeeManager = Depends(get_employee_manager)
):
    """Delete or archive an employee."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        success = await employee_manager.delete_employee(
            client_id, employee_id, archive_memory=archive_only
        )
        
        if success:
            action = "archived" if archive_only else "deleted"
            return {"message": f"Employee {action} successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete employee")
        
    except EmployeeNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete employee")


# Employee Configuration Management Endpoints

@router.put("/clients/{client_id}/employees/{employee_id}/personality")
async def update_employee_personality(
    client_id: str,
    employee_id: str,
    personality_update: Dict[str, Any],
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    employee_manager: EmployeeManager = Depends(get_employee_manager),
    personality_manager: PersonalityManager = Depends(get_personality_manager)
):
    """Update employee personality traits and behavior configuration."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        # Validate personality update data
        valid_traits = [
            "communication_style", "formality_level", "expertise_level", 
            "response_length", "creativity_level", "empathy_level", "custom_traits"
        ]
        
        invalid_traits = [key for key in personality_update.keys() if key not in valid_traits]
        if invalid_traits:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid personality traits: {invalid_traits}"
            )
        
        # Validate trait values
        if "creativity_level" in personality_update:
            if not 0.0 <= personality_update["creativity_level"] <= 1.0:
                raise HTTPException(status_code=400, detail="creativity_level must be between 0.0 and 1.0")
        
        if "empathy_level" in personality_update:
            if not 0.0 <= personality_update["empathy_level"] <= 1.0:
                raise HTTPException(status_code=400, detail="empathy_level must be between 0.0 and 1.0")
        
        # Update personality
        updated_employee = await employee_manager.update_employee_personality(
            client_id, employee_id, personality_update
        )
        
        # Generate personality summary
        personality_summary = personality_manager.generate_personality_summary(updated_employee.personality)
        
        return {
            "employee_id": employee_id,
            "personality": {
                "communication_style": updated_employee.personality.communication_style.value,
                "formality_level": updated_employee.personality.formality_level.value,
                "expertise_level": updated_employee.personality.expertise_level.value,
                "response_length": updated_employee.personality.response_length.value,
                "creativity_level": updated_employee.personality.creativity_level,
                "empathy_level": updated_employee.personality.empathy_level,
                "custom_traits": updated_employee.personality.custom_traits
            },
            "personality_summary": personality_summary,
            "updated_at": updated_employee.updated_at.isoformat(),
            "message": "Personality updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update personality for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update employee personality")


@router.put("/clients/{client_id}/employees/{employee_id}/model-preferences")
async def update_model_preferences(
    client_id: str,
    employee_id: str,
    model_preferences: Dict[str, Any],
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    employee_manager: EmployeeManager = Depends(get_employee_manager)
):
    """Update employee model preferences and AI configuration."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        # Validate model preferences
        valid_preferences = [
            "primary_model", "fallback_models", "temperature", 
            "max_tokens", "specialized_models"
        ]
        
        invalid_preferences = [key for key in model_preferences.keys() if key not in valid_preferences]
        if invalid_preferences:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model preferences: {invalid_preferences}"
            )
        
        # Validate specific values
        if "temperature" in model_preferences:
            temp = model_preferences["temperature"]
            if not 0.0 <= temp <= 2.0:
                raise HTTPException(status_code=400, detail="temperature must be between 0.0 and 2.0")
        
        if "max_tokens" in model_preferences:
            tokens = model_preferences["max_tokens"]
            if not 1 <= tokens <= 8000:
                raise HTTPException(status_code=400, detail="max_tokens must be between 1 and 8000")
        
        # Update model preferences
        updated_employee = await employee_manager.update_employee_model_preferences(
            client_id, employee_id, model_preferences
        )
        
        return {
            "employee_id": employee_id,
            "model_preferences": {
                "primary_model": updated_employee.model_preferences.primary_model,
                "fallback_models": updated_employee.model_preferences.fallback_models,
                "temperature": updated_employee.model_preferences.temperature,
                "max_tokens": updated_employee.model_preferences.max_tokens,
                "specialized_models": updated_employee.model_preferences.specialized_models
            },
            "updated_at": updated_employee.updated_at.isoformat(),
            "message": "Model preferences updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update model preferences for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update model preferences")


@router.put("/clients/{client_id}/employees/{employee_id}/tools")
async def update_tool_access(
    client_id: str,
    employee_id: str,
    tool_config: Dict[str, Any],
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    employee_manager: EmployeeManager = Depends(get_employee_manager)
):
    """Update employee tool access and capabilities."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        # Validate tool configuration
        if "tools" not in tool_config:
            raise HTTPException(status_code=400, detail="Missing 'tools' in request body")
        
        tools = tool_config["tools"]
        if not isinstance(tools, list):
            raise HTTPException(status_code=400, detail="'tools' must be a list")
        
        # Available tools (in production, this would come from a registry)
        available_tools = [
            "web_search", "calculator", "email", "calendar", "file_manager",
            "database_query", "api_client", "code_executor", "image_generator",
            "document_analyzer", "translation", "sentiment_analysis"
        ]
        
        invalid_tools = [tool for tool in tools if tool not in available_tools]
        if invalid_tools:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid tools: {invalid_tools}. Available: {available_tools}"
            )
        
        # Update tool access
        updated_employee = await employee_manager.update_employee_tool_access(
            client_id, employee_id, tools, tool_config.get("tool_credentials", {})
        )
        
        return {
            "employee_id": employee_id,
            "tool_access": updated_employee.tool_access,
            "available_tools": available_tools,
            "updated_at": updated_employee.updated_at.isoformat(),
            "message": "Tool access updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update tool access for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update tool access")


@router.get("/clients/{client_id}/employees/{employee_id}/tools")
async def get_employee_tools(
    client_id: str,
    employee_id: str,
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    employee_manager: EmployeeManager = Depends(get_employee_manager)
):
    """Get employee's current tool access and capabilities."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        employee = await employee_manager.get_employee(client_id, employee_id)
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        # Get tool usage statistics
        tool_stats = await employee_manager.get_employee_tool_usage_stats(client_id, employee_id)
        
        return {
            "employee_id": employee_id,
            "current_tools": employee.tool_access,
            "tool_usage_stats": tool_stats,
            "last_updated": employee.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tool access for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tool access")


@router.post("/clients/{client_id}/employees/{employee_id}/knowledge-sources")
async def add_employee_knowledge_source(
    client_id: str,
    employee_id: str,
    knowledge_source: Dict[str, Any],
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    employee_manager: EmployeeManager = Depends(get_employee_manager),
    memory_manager: EmployeeMemoryManager = Depends(get_memory_manager)
):
    """Add a knowledge source to employee's knowledge base."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        # Validate knowledge source data
        required_fields = ["title", "content", "source_type"]
        missing_fields = [field for field in required_fields if field not in knowledge_source]
        if missing_fields:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required fields: {missing_fields}"
            )
        
        valid_source_types = ["document", "url", "manual", "api", "database"]
        if knowledge_source["source_type"] not in valid_source_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid source_type. Valid types: {valid_source_types}"
            )
        
        # Add knowledge source
        knowledge_id = await memory_manager.add_knowledge_source(
            client_id=client_id,
            employee_id=employee_id,
            title=knowledge_source["title"],
            content=knowledge_source["content"],
            source_type=knowledge_source["source_type"],
            source_url=knowledge_source.get("source_url"),
            keywords=knowledge_source.get("keywords", []),
            tags=knowledge_source.get("tags", [])
        )
        
        # Update employee's knowledge sources list
        await employee_manager.add_employee_knowledge_source(
            client_id, employee_id, knowledge_id, knowledge_source["title"]
        )
        
        return {
            "knowledge_id": knowledge_id,
            "employee_id": employee_id,
            "title": knowledge_source["title"],
            "source_type": knowledge_source["source_type"],
            "message": "Knowledge source added successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add knowledge source for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to add knowledge source")


@router.get("/clients/{client_id}/employees/{employee_id}/knowledge-sources")
async def list_employee_knowledge_sources(
    client_id: str,
    employee_id: str,
    source_type: Optional[str] = Query(None, description="Filter by source type"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results"),
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    memory_manager: EmployeeMemoryManager = Depends(get_memory_manager)
):
    """List employee's knowledge sources."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        # Get knowledge sources
        knowledge_sources = await memory_manager.list_employee_knowledge_sources(
            client_id, employee_id, source_type, limit
        )
        
        return {
            "employee_id": employee_id,
            "knowledge_sources": knowledge_sources,
            "total_count": len(knowledge_sources),
            "filters": {
                "source_type": source_type,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to list knowledge sources for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to list knowledge sources")


@router.delete("/clients/{client_id}/employees/{employee_id}/knowledge-sources/{knowledge_id}")
async def remove_employee_knowledge_source(
    client_id: str,
    employee_id: str,
    knowledge_id: str,
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    employee_manager: EmployeeManager = Depends(get_employee_manager),
    memory_manager: EmployeeMemoryManager = Depends(get_memory_manager)
):
    """Remove a knowledge source from employee's knowledge base."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        # Remove knowledge source
        success = await memory_manager.remove_knowledge_source(
            client_id, employee_id, knowledge_id
        )
        
        if success:
            # Update employee's knowledge sources list
            await employee_manager.remove_employee_knowledge_source(
                client_id, employee_id, knowledge_id
            )
            
            return {"message": "Knowledge source removed successfully"}
        else:
            raise HTTPException(status_code=404, detail="Knowledge source not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove knowledge source {knowledge_id} for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove knowledge source")


@router.get("/clients/{client_id}/employees/{employee_id}/configuration")
async def get_employee_configuration(
    client_id: str,
    employee_id: str,
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    employee_manager: EmployeeManager = Depends(get_employee_manager)
):
    """Get complete employee configuration including personality, models, tools, and knowledge sources."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        employee = await employee_manager.get_employee(client_id, employee_id)
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        # Get additional configuration details
        tool_stats = await employee_manager.get_employee_tool_usage_stats(client_id, employee_id)
        
        return {
            "employee_id": employee_id,
            "name": employee.name,
            "role": employee.role,
            "status": employee.status.value,
            "configuration": {
                "personality": {
                    "communication_style": employee.personality.communication_style.value,
                    "formality_level": employee.personality.formality_level.value,
                    "expertise_level": employee.personality.expertise_level.value,
                    "response_length": employee.personality.response_length.value,
                    "creativity_level": employee.personality.creativity_level,
                    "empathy_level": employee.personality.empathy_level,
                    "custom_traits": employee.personality.custom_traits
                },
                "model_preferences": {
                    "primary_model": employee.model_preferences.primary_model,
                    "fallback_models": employee.model_preferences.fallback_models,
                    "temperature": employee.model_preferences.temperature,
                    "max_tokens": employee.model_preferences.max_tokens,
                    "specialized_models": employee.model_preferences.specialized_models
                },
                "tools": {
                    "available_tools": employee.tool_access,
                    "usage_stats": tool_stats
                },
                "knowledge_sources": employee.knowledge_sources
            },
            "created_at": employee.created_at.isoformat(),
            "updated_at": employee.updated_at.isoformat(),
            "last_interaction_at": employee.last_interaction_at.isoformat() if employee.last_interaction_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get configuration for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve employee configuration")


@router.post("/clients/{client_id}/employees/{employee_id}/configuration/reset")
async def reset_employee_configuration(
    client_id: str,
    employee_id: str,
    reset_options: Dict[str, Any],
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    employee_manager: EmployeeManager = Depends(get_employee_manager)
):
    """Reset employee configuration to defaults or specified values."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        # Validate reset options
        valid_reset_types = ["personality", "model_preferences", "tools", "knowledge_sources", "all"]
        reset_type = reset_options.get("reset_type", "all")
        
        if reset_type not in valid_reset_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid reset_type. Valid types: {valid_reset_types}"
            )
        
        # Perform reset
        updated_employee = await employee_manager.reset_employee_configuration(
            client_id, employee_id, reset_type, reset_options.get("default_values", {})
        )
        
        return {
            "employee_id": employee_id,
            "reset_type": reset_type,
            "updated_at": updated_employee.updated_at.isoformat(),
            "message": f"Employee configuration reset successfully ({reset_type})"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset configuration for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset employee configuration")


# Employee Interaction Endpoints

@router.post("/clients/{client_id}/employees/{employee_id}/interact", response_model=InteractionResponse)
async def interact_with_employee(
    client_id: str,
    employee_id: str,
    interaction_request: InteractionRequest,
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    employee_manager: EmployeeManager = Depends(get_employee_manager)
):
    """
    Interact with an AI employee.
    
    Processes a message through the employee's personality, memory context,
    and model preferences to generate a personalized response.
    """
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        start_time = time.time()
        
        # Process interaction
        response = await employee_manager.interact_with_employee(
            client_id=client_id,
            employee_id=employee_id,
            message=interaction_request.message,
            session_id=interaction_request.session_id,
            context=interaction_request.context,
            include_memory=interaction_request.include_memory,
            memory_limit=interaction_request.memory_limit
        )
        
        processing_time = time.time() - start_time
        
        # Get employee name for response
        employee = await employee_manager.get_employee(client_id, employee_id)
        employee_name = employee.name if employee else "AI Employee"
        
        logger.info(f"Interaction processed for employee {employee_id} in {processing_time:.3f}s")
        
        return InteractionResponse(
            message=response.message,
            employee_id=response.employee_id,
            employee_name=employee_name,
            interaction_id=response.interaction_id,
            session_id=interaction_request.session_id,
            timestamp=response.timestamp,
            model_used=response.model_used,
            processing_time_ms=response.processing_time_ms,
            tokens_used=response.tokens_used,
            cost=response.cost,
            confidence_score=response.confidence_score
        )
        
    except EmployeeNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except TenantIsolationError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Interaction failed for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to process interaction")


# Employee Memory Endpoints

@router.get("/clients/{client_id}/employees/{employee_id}/memory", response_model=MemoryResponse)
async def get_employee_memory(
    client_id: str,
    employee_id: str,
    query: Optional[str] = Query(None, description="Semantic search query"),
    memory_types: Optional[List[MemoryType]] = Query(None, description="Filter by memory types"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    include_embeddings: bool = Query(False, description="Include embedding vectors"),
    min_importance: float = Query(0.0, ge=0.0, le=1.0, description="Minimum importance score"),
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    memory_manager: EmployeeMemoryManager = Depends(get_memory_manager)
):
    """Get employee memory/conversation history with optional semantic search."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        start_time = time.time()
        
        if query:
            # Semantic search
            memories, search_time = await memory_manager.search_employee_memory(
                client_id, employee_id, query, limit, memory_types, include_embeddings
            )
            
            # Filter by importance
            if min_importance > 0:
                memories = [m for m in memories if m.importance_score >= min_importance]
            
            query_time_ms = search_time
        else:
            # Get recent memories
            memories = await memory_manager.get_employee_context(
                client_id, employee_id, query, limit, memory_types, min_importance
            )
            query_time_ms = (time.time() - start_time) * 1000
        
        # Format memories for response
        memory_data = []
        for memory in memories:
            memory_item = {
                "id": memory.id,
                "content": memory.content,
                "response": memory.response,
                "timestamp": memory.timestamp.isoformat(),
                "memory_type": memory.memory_type.value,
                "importance_score": memory.importance_score,
                "relevance_score": memory.relevance_score,
                "context": memory.context
            }
            
            if include_embeddings and memory.embedding:
                memory_item["embedding"] = memory.embedding
            
            memory_data.append(memory_item)
        
        return MemoryResponse(
            memories=memory_data,
            total_count=len(memory_data),
            query_time_ms=query_time_ms
        )
        
    except MemoryAccessError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get memory for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve employee memory")


@router.get("/clients/{client_id}/employees/{employee_id}/knowledge")
async def search_employee_knowledge(
    client_id: str,
    employee_id: str,
    query: str = Query(..., description="Search query for knowledge base"),
    limit: int = Query(5, ge=1, le=20, description="Maximum number of results"),
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    memory_manager: EmployeeMemoryManager = Depends(get_memory_manager)
):
    """Search employee's knowledge base."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        knowledge_results = await memory_manager.search_knowledge_base(
            client_id, employee_id, query, limit
        )
        
        return {
            "query": query,
            "results": knowledge_results,
            "total_count": len(knowledge_results)
        }
        
    except Exception as e:
        logger.error(f"Knowledge search failed for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to search knowledge base")


@router.post("/clients/{client_id}/employees/{employee_id}/knowledge")
async def add_knowledge_source(
    client_id: str,
    employee_id: str,
    knowledge_data: Dict[str, Any],
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    memory_manager: EmployeeMemoryManager = Depends(get_memory_manager)
):
    """Add a knowledge source to employee's knowledge base."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        # Validate required fields
        required_fields = ["title", "content"]
        for field in required_fields:
            if field not in knowledge_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        knowledge_id = await memory_manager.add_knowledge_source(
            client_id=client_id,
            employee_id=employee_id,
            title=knowledge_data["title"],
            content=knowledge_data["content"],
            source_type=knowledge_data.get("source_type", "manual"),
            source_url=knowledge_data.get("source_url"),
            keywords=knowledge_data.get("keywords", []),
            tags=knowledge_data.get("tags", [])
        )
        
        return {
            "knowledge_id": knowledge_id,
            "message": "Knowledge source added successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add knowledge source for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to add knowledge source")


# Memory Analytics and Management Endpoints

@router.get("/clients/{client_id}/employees/{employee_id}/memory/analytics")
async def get_memory_analytics(
    client_id: str,
    employee_id: str,
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    memory_manager: EmployeeMemoryManager = Depends(get_memory_manager)
):
    """Get comprehensive memory analytics for an employee."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        analytics = await memory_manager.get_memory_analytics(
            client_id, employee_id, days
        )
        
        return {
            "employee_id": employee_id,
            "analysis_period_days": days,
            "analytics": analytics,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get memory analytics for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve memory analytics")


@router.get("/clients/{client_id}/employees/{employee_id}/memory/history")
async def get_conversation_history(
    client_id: str,
    employee_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    include_context: bool = Query(False, description="Include interaction context"),
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    memory_manager: EmployeeMemoryManager = Depends(get_memory_manager)
):
    """Get paginated conversation history for an employee."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        # Parse date filters
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format. Use ISO format.")
        
        if end_date:
            try:
                end_datetime = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format. Use ISO format.")
        
        # Get paginated history
        history_result = await memory_manager.get_conversation_history(
            client_id=client_id,
            employee_id=employee_id,
            page=page,
            page_size=page_size,
            start_date=start_datetime,
            end_date=end_datetime,
            session_id=session_id,
            include_context=include_context
        )
        
        return {
            "employee_id": employee_id,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": history_result["total_count"],
                "total_pages": (history_result["total_count"] + page_size - 1) // page_size,
                "has_next": page * page_size < history_result["total_count"],
                "has_previous": page > 1
            },
            "filters": {
                "start_date": start_date,
                "end_date": end_date,
                "session_id": session_id,
                "include_context": include_context
            },
            "conversations": history_result["conversations"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation history for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation history")


@router.get("/clients/{client_id}/employees/{employee_id}/memory/statistics")
async def get_memory_statistics(
    client_id: str,
    employee_id: str,
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    memory_manager: EmployeeMemoryManager = Depends(get_memory_manager)
):
    """Get detailed memory statistics for an employee."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        stats = await memory_manager.get_memory_statistics(client_id, employee_id)
        
        return {
            "employee_id": employee_id,
            "statistics": stats,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get memory statistics for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve memory statistics")


@router.post("/clients/{client_id}/employees/{employee_id}/memory/export")
async def export_employee_memory(
    client_id: str,
    employee_id: str,
    export_request: Dict[str, Any],
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    memory_manager: EmployeeMemoryManager = Depends(get_memory_manager)
):
    """Export employee memory data for backup or analysis."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        # Validate export request
        export_format = export_request.get("format", "json")
        if export_format not in ["json", "csv", "xml"]:
            raise HTTPException(status_code=400, detail="Invalid export format. Supported: json, csv, xml")
        
        include_embeddings = export_request.get("include_embeddings", False)
        include_analytics = export_request.get("include_analytics", True)
        date_range = export_request.get("date_range", {})
        
        # Start export process
        export_result = await memory_manager.export_memory(
            client_id=client_id,
            employee_id=employee_id,
            format=export_format,
            include_embeddings=include_embeddings,
            include_analytics=include_analytics,
            start_date=date_range.get("start_date"),
            end_date=date_range.get("end_date")
        )
        
        return {
            "export_id": export_result["export_id"],
            "status": "initiated",
            "format": export_format,
            "estimated_completion": export_result["estimated_completion"],
            "download_url": export_result.get("download_url"),
            "message": "Memory export initiated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export memory for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate memory export")


@router.get("/clients/{client_id}/employees/{employee_id}/memory/export/{export_id}")
async def get_export_status(
    client_id: str,
    employee_id: str,
    export_id: str,
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    memory_manager: EmployeeMemoryManager = Depends(get_memory_manager)
):
    """Get the status of a memory export operation."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        export_status = await memory_manager.get_export_status(
            client_id, employee_id, export_id
        )
        
        if not export_status:
            raise HTTPException(status_code=404, detail="Export not found")
        
        return export_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get export status for {export_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve export status")


@router.post("/clients/{client_id}/employees/{employee_id}/memory/backup")
async def create_memory_backup(
    client_id: str,
    employee_id: str,
    backup_request: Dict[str, Any],
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    memory_manager: EmployeeMemoryManager = Depends(get_memory_manager)
):
    """Create a backup of employee memory data."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        backup_name = backup_request.get("name", f"backup_{employee_id}_{int(time.time())}")
        include_embeddings = backup_request.get("include_embeddings", True)
        compression = backup_request.get("compression", "gzip")
        
        if compression not in ["none", "gzip", "zip"]:
            raise HTTPException(status_code=400, detail="Invalid compression format. Supported: none, gzip, zip")
        
        # Create backup
        backup_result = await memory_manager.create_memory_backup(
            client_id=client_id,
            employee_id=employee_id,
            backup_name=backup_name,
            include_embeddings=include_embeddings,
            compression=compression
        )
        
        return {
            "backup_id": backup_result["backup_id"],
            "backup_name": backup_name,
            "status": "created",
            "size_bytes": backup_result["size_bytes"],
            "created_at": backup_result["created_at"],
            "compression": compression,
            "message": "Memory backup created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create memory backup for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create memory backup")


@router.get("/clients/{client_id}/employees/{employee_id}/memory/backups")
async def list_memory_backups(
    client_id: str,
    employee_id: str,
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    memory_manager: EmployeeMemoryManager = Depends(get_memory_manager)
):
    """List all memory backups for an employee."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        backups = await memory_manager.list_memory_backups(client_id, employee_id)
        
        return {
            "employee_id": employee_id,
            "backups": backups,
            "total_count": len(backups)
        }
        
    except Exception as e:
        logger.error(f"Failed to list memory backups for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to list memory backups")


@router.post("/clients/{client_id}/employees/{employee_id}/memory/restore/{backup_id}")
async def restore_memory_backup(
    client_id: str,
    employee_id: str,
    backup_id: str,
    restore_options: Dict[str, Any],
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    memory_manager: EmployeeMemoryManager = Depends(get_memory_manager)
):
    """Restore employee memory from a backup."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        merge_mode = restore_options.get("merge_mode", "replace")
        if merge_mode not in ["replace", "merge", "append"]:
            raise HTTPException(status_code=400, detail="Invalid merge_mode. Supported: replace, merge, append")
        
        # Restore backup
        restore_result = await memory_manager.restore_memory_backup(
            client_id=client_id,
            employee_id=employee_id,
            backup_id=backup_id,
            merge_mode=merge_mode
        )
        
        return {
            "restore_id": restore_result["restore_id"],
            "status": "completed",
            "restored_items": restore_result["restored_items"],
            "merge_mode": merge_mode,
            "completed_at": restore_result["completed_at"],
            "message": "Memory backup restored successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restore memory backup {backup_id} for employee {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to restore memory backup")


# Search and Analytics Endpoints

@router.get("/clients/{client_id}/employees/search")
async def search_employees(
    client_id: str,
    query: str = Query(..., description="Search query for employees"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results"),
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    employee_manager: EmployeeManager = Depends(get_employee_manager)
):
    """Search employees by name, role, or capabilities."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        results = await employee_manager.search_employees(client_id, query, limit)
        
        return {
            "query": query,
            "results": results,
            "total_count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Employee search failed for client {client_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to search employees")


@router.get("/clients/{client_id}/employees/{employee_id}/stats")
async def get_employee_stats(
    client_id: str,
    employee_id: str,
    tenant_context: Dict[str, str] = Depends(extract_tenant_context),
    employee_manager: EmployeeManager = Depends(get_employee_manager)
):
    """Get comprehensive employee statistics."""
    try:
        # Validate tenant access
        if client_id != tenant_context["client_id"]:
            raise HTTPException(status_code=403, detail="Access denied to client")
        
        stats = await employee_manager.get_employee_stats(client_id, employee_id)
        return stats
        
    except EmployeeNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get employee stats {employee_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve employee statistics")


# System Health and Metrics Endpoints

@router.get("/health")
async def health_check():
    """Health check endpoint for the employee lifecycle API."""
    try:
        # Test basic functionality
        personality_manager = PersonalityManager()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "employee-lifecycle-api",
            "version": "1.0.0",
            "components": {
                "personality_manager": "healthy",
                "api_endpoints": "healthy"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/metrics")
async def get_api_metrics(
    employee_manager: EmployeeManager = Depends(get_employee_manager)
):
    """Get API performance metrics."""
    try:
        metrics = employee_manager.get_metrics()
        
        return {
            "employee_lifecycle_api": {
                "timestamp": datetime.now().isoformat(),
                "employee_manager_metrics": metrics
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get API metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


# Error handlers for the router
async def client_not_found_handler(request: Request, exc: ClientNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"error": "Client not found", "detail": str(exc)}
    )


async def employee_not_found_handler(request: Request, exc: EmployeeNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"error": "Employee not found", "detail": str(exc)}
    )


async def tenant_isolation_handler(request: Request, exc: TenantIsolationError):
    return JSONResponse(
        status_code=403,
        content={"error": "Access denied", "detail": "Tenant isolation violation"}
    )


async def invalid_config_handler(request: Request, exc: InvalidConfigurationError):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid configuration", "detail": str(exc)}
    )


ROUTER_EXCEPTION_HANDLERS = [
    (ClientNotFoundError, client_not_found_handler),
    (EmployeeNotFoundError, employee_not_found_handler),
    (TenantIsolationError, tenant_isolation_handler),
    (InvalidConfigurationError, invalid_config_handler),
]
