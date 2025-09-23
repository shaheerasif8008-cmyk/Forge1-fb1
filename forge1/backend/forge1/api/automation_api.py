# forge1/backend/forge1/api/automation_api.py

"""
FastAPI endpoints for automation platform connectors
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import logging

from forge1.integrations.automation_connectors import (
    AutomationPlatform,
    TriggerType,
    ActionType,
    AutomationConnectorManager
)
from forge1.auth.authentication_manager import AuthenticationManager
from forge1.core.security_manager import SecurityManager
from forge1.core.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class TriggerConfig(BaseModel):
    """Trigger configuration model"""
    type: str = Field(..., description="Trigger type")
    method: Optional[str] = Field("POST", description="HTTP method for webhook triggers")
    path: Optional[str] = Field(None, description="Webhook path")
    url: Optional[str] = Field(None, description="Webhook URL")
    secret: Optional[str] = Field(None, description="Webhook secret for signature validation")
    require_signature: Optional[bool] = Field(True, description="Require signature validation")
    schedule: Optional[str] = Field(None, description="Schedule expression for scheduled triggers")
    time: Optional[str] = Field(None, description="Time for daily triggers")
    folder: Optional[str] = Field(None, description="Email folder for email triggers")

class ActionConfig(BaseModel):
    """Action configuration model"""
    type: str = Field(..., description="Action type")
    url: Optional[str] = Field(None, description="URL for HTTP requests")
    method: Optional[str] = Field("POST", description="HTTP method")
    headers: Optional[Dict[str, str]] = Field(default_factory=dict, description="HTTP headers")
    data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Request data")
    timeout: Optional[int] = Field(30, description="Request timeout in seconds")
    to: Optional[str] = Field(None, description="Email recipient")
    subject: Optional[str] = Field(None, description="Email subject")
    body: Optional[str] = Field(None, description="Email body")
    text: Optional[str] = Field(None, description="Email text content")
    channel: Optional[str] = Field(None, description="Notification channel")
    message: Optional[str] = Field(None, description="Notification message")
    agent_type: Optional[str] = Field(None, description="Agent type for agent execution")
    task: Optional[str] = Field(None, description="Task for agent execution")
    code: Optional[str] = Field(None, description="Custom code to execute")
    continue_on_failure: Optional[bool] = Field(False, description="Continue workflow on action failure")

class WorkflowCreateRequest(BaseModel):
    """Workflow creation request model"""
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    triggers: List[TriggerConfig] = Field(..., description="Workflow triggers")
    actions: List[ActionConfig] = Field(..., description="Workflow actions")
    active: Optional[bool] = Field(True, description="Whether workflow is active")
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional settings")
    security: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Security settings")
    retry_policy: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Retry policy")

class WorkflowExecuteRequest(BaseModel):
    """Workflow execution request model"""
    input_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Input data for workflow")

class ConnectorCreateRequest(BaseModel):
    """Connector creation request model"""
    platform: str = Field(..., description="Automation platform")
    config: Dict[str, Any] = Field(..., description="Platform-specific configuration")

class WebhookRegisterRequest(BaseModel):
    """Webhook registration request model"""
    path: Optional[str] = Field(None, description="Webhook path")
    method: Optional[str] = Field("POST", description="HTTP method")
    secret: Optional[str] = Field(None, description="Webhook secret")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    require_signature: Optional[bool] = Field(True, description="Require signature validation")
    allowed_ips: Optional[List[str]] = Field(default_factory=list, description="Allowed IP addresses")
    rate_limit: Optional[Dict[str, int]] = Field(
        default_factory=lambda: {"requests": 100, "window": 3600},
        description="Rate limiting configuration"
    )

# Global connector manager instance
connector_manager: Optional[AutomationConnectorManager] = None

def get_connector_manager() -> AutomationConnectorManager:
    """Get automation connector manager instance"""
    global connector_manager
    if connector_manager is None:
        # In production, these would be injected via dependency injection
        auth_manager = AuthenticationManager()
        security_manager = SecurityManager()
        performance_monitor = PerformanceMonitor()
        
        connector_manager = AutomationConnectorManager(
            auth_manager=auth_manager,
            security_manager=security_manager,
            performance_monitor=performance_monitor
        )
    
    return connector_manager

# Create API router
router = APIRouter(prefix="/api/v1/automation", tags=["automation"])

@router.post("/connectors", response_model=Dict[str, Any])
async def create_connector(
    request: ConnectorCreateRequest,
    manager: AutomationConnectorManager = Depends(get_connector_manager)
) -> Dict[str, Any]:
    """Create automation platform connector"""
    
    try:
        # Validate platform
        try:
            platform = AutomationPlatform(request.platform)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported platform: {request.platform}"
            )
        
        result = await manager.create_connector(platform, request.config)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to create connector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/connectors", response_model=Dict[str, Any])
async def list_connectors(
    manager: AutomationConnectorManager = Depends(get_connector_manager)
) -> Dict[str, Any]:
    """List all automation connectors"""
    
    try:
        return await manager.list_connectors()
        
    except Exception as e:
        logger.error(f"Failed to list connectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/connectors/{connector_id}", response_model=Dict[str, Any])
async def get_connector(
    connector_id: str,
    manager: AutomationConnectorManager = Depends(get_connector_manager)
) -> Dict[str, Any]:
    """Get specific connector information"""
    
    try:
        connector = await manager.get_connector(connector_id)
        
        if not connector:
            raise HTTPException(status_code=404, detail="Connector not found")
        
        return connector.get_connector_info()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get connector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/connectors/{connector_id}", response_model=Dict[str, Any])
async def remove_connector(
    connector_id: str,
    manager: AutomationConnectorManager = Depends(get_connector_manager)
) -> Dict[str, Any]:
    """Remove automation connector"""
    
    try:
        result = await manager.remove_connector(connector_id)
        
        if not result["success"]:
            if "not found" in result["error"].lower():
                raise HTTPException(status_code=404, detail=result["error"])
            else:
                raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove connector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/connectors/{connector_id}/workflows", response_model=Dict[str, Any])
async def create_workflow(
    connector_id: str,
    request: WorkflowCreateRequest,
    manager: AutomationConnectorManager = Depends(get_connector_manager)
) -> Dict[str, Any]:
    """Create workflow on specific connector"""
    
    try:
        connector = await manager.get_connector(connector_id)
        
        if not connector:
            raise HTTPException(status_code=404, detail="Connector not found")
        
        # Convert Pydantic models to dictionaries
        triggers = [trigger.dict(exclude_none=True) for trigger in request.triggers]
        actions = [action.dict(exclude_none=True) for action in request.actions]
        
        # Additional kwargs from request
        kwargs = {
            "description": request.description,
            "active": request.active,
            "settings": request.settings,
            **request.security,
            **request.retry_policy
        }
        
        result = await connector.create_workflow(
            name=request.name,
            triggers=triggers,
            actions=actions,
            **{k: v for k, v in kwargs.items() if v is not None}
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/connectors/{connector_id}/workflows/{workflow_id}/execute", response_model=Dict[str, Any])
async def execute_workflow(
    connector_id: str,
    workflow_id: str,
    request: WorkflowExecuteRequest,
    background_tasks: BackgroundTasks,
    manager: AutomationConnectorManager = Depends(get_connector_manager)
) -> Dict[str, Any]:
    """Execute workflow on specific connector"""
    
    try:
        result = await manager.execute_workflow(
            connector_id=connector_id,
            workflow_id=workflow_id,
            input_data=request.input_data
        )
        
        if not result["success"]:
            if "not found" in result["error"].lower():
                raise HTTPException(status_code=404, detail=result["error"])
            else:
                raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/connectors/{connector_id}/webhooks", response_model=Dict[str, Any])
async def register_webhook(
    connector_id: str,
    request: WebhookRegisterRequest,
    manager: AutomationConnectorManager = Depends(get_connector_manager)
) -> Dict[str, Any]:
    """Register webhook endpoint on specific connector"""
    
    try:
        connector = await manager.get_connector(connector_id)
        
        if not connector:
            raise HTTPException(status_code=404, detail="Connector not found")
        
        webhook_config = request.dict(exclude_none=True)
        result = await connector.register_webhook(webhook_config)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to register webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=Dict[str, Any])
async def get_automation_metrics(
    manager: AutomationConnectorManager = Depends(get_connector_manager)
) -> Dict[str, Any]:
    """Get aggregated automation metrics"""
    
    try:
        return await manager.get_metrics()
        
    except Exception as e:
        logger.error(f"Failed to get automation metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/platforms", response_model=Dict[str, Any])
async def get_supported_platforms() -> Dict[str, Any]:
    """Get list of supported automation platforms"""
    
    platforms = []
    for platform in AutomationPlatform:
        platforms.append({
            "name": platform.value,
            "display_name": platform.value.replace("_", " ").title(),
            "description": _get_platform_description(platform)
        })
    
    return {
        "platforms": platforms,
        "total_count": len(platforms)
    }

@router.get("/trigger-types", response_model=Dict[str, Any])
async def get_trigger_types() -> Dict[str, Any]:
    """Get list of supported trigger types"""
    
    triggers = []
    for trigger_type in TriggerType:
        triggers.append({
            "name": trigger_type.value,
            "display_name": trigger_type.value.replace("_", " ").title(),
            "description": _get_trigger_description(trigger_type)
        })
    
    return {
        "trigger_types": triggers,
        "total_count": len(triggers)
    }

@router.get("/action-types", response_model=Dict[str, Any])
async def get_action_types() -> Dict[str, Any]:
    """Get list of supported action types"""
    
    actions = []
    for action_type in ActionType:
        actions.append({
            "name": action_type.value,
            "display_name": action_type.value.replace("_", " ").title(),
            "description": _get_action_description(action_type)
        })
    
    return {
        "action_types": actions,
        "total_count": len(actions)
    }

def _get_platform_description(platform: AutomationPlatform) -> str:
    """Get description for automation platform"""
    
    descriptions = {
        AutomationPlatform.N8N: "Open-source workflow automation platform with visual editor",
        AutomationPlatform.ZAPIER: "Popular cloud-based automation platform connecting apps and services",
        AutomationPlatform.MAKE: "Visual automation platform (formerly Integromat) for complex workflows",
        AutomationPlatform.POWER_AUTOMATE: "Microsoft's cloud-based automation platform",
        AutomationPlatform.IFTTT: "Simple automation platform for connecting apps and devices",
        AutomationPlatform.CUSTOM_WEBHOOK: "Custom webhook integration for enterprise automation needs"
    }
    
    return descriptions.get(platform, "Automation platform integration")

def _get_trigger_description(trigger_type: TriggerType) -> str:
    """Get description for trigger type"""
    
    descriptions = {
        TriggerType.WEBHOOK: "HTTP webhook endpoint that triggers on incoming requests",
        TriggerType.SCHEDULE: "Time-based trigger that runs on a schedule",
        TriggerType.EVENT: "Event-driven trigger that responds to system events",
        TriggerType.API_CALL: "API endpoint trigger for programmatic workflow execution",
        TriggerType.EMAIL: "Email-based trigger that responds to incoming emails",
        TriggerType.FILE_CHANGE: "File system trigger that responds to file changes",
        TriggerType.DATABASE_CHANGE: "Database trigger that responds to data changes"
    }
    
    return descriptions.get(trigger_type, "Workflow trigger mechanism")

def _get_action_description(action_type: ActionType) -> str:
    """Get description for action type"""
    
    descriptions = {
        ActionType.HTTP_REQUEST: "Make HTTP requests to external APIs and services",
        ActionType.EMAIL_SEND: "Send email notifications and messages",
        ActionType.FILE_OPERATION: "Perform file system operations like read, write, delete",
        ActionType.DATABASE_OPERATION: "Execute database queries and operations",
        ActionType.NOTIFICATION: "Send notifications to various channels (Slack, Teams, etc.)",
        ActionType.AGENT_EXECUTION: "Execute Forge 1 AI agents for complex tasks",
        ActionType.WORKFLOW_TRIGGER: "Trigger other workflows and automation chains"
    }
    
    return descriptions.get(action_type, "Workflow action operation")