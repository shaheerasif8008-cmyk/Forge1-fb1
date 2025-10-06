"""
Azure Monitor API Endpoints

Provides REST API endpoints for managing Azure Monitor integration,
querying telemetry data, and configuring monitoring settings.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from forge1.integrations.observability.azure_monitor_adapter import (
    azure_monitor_adapter, CustomEvent, CustomMetric
)
from forge1.services.azure_monitor_service import (
    azure_monitor_business_service, EmployeeInteraction, TaskExecution, ModelUsage
)
from forge1.integrations.base_adapter import ExecutionContext, TenantContext
from forge1.core.tenancy import get_current_tenant

logger = logging.getLogger(__name__)

# Pydantic models for API
class CustomEventRequest(BaseModel):
    name: str = Field(..., description="Event name")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Event properties")
    measurements: Dict[str, float] = Field(default_factory=dict, description="Event measurements")
    tenant_id: Optional[str] = Field(None, description="Tenant ID (optional, will use current tenant)")

class CustomMetricRequest(BaseModel):
    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    properties: Dict[str, str] = Field(default_factory=dict, description="Metric properties")
    tenant_id: Optional[str] = Field(None, description="Tenant ID (optional, will use current tenant)")

class EmployeeInteractionRequest(BaseModel):
    employee_id: str = Field(..., description="Employee ID")
    employee_name: str = Field(..., description="Employee name")
    interaction_type: str = Field(..., description="Type of interaction")
    user_message: str = Field(..., description="User message")
    response_message: str = Field(..., description="Employee response")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_used: str = Field(..., description="AI model used")
    tokens_used: int = Field(..., description="Number of tokens used")
    cost: float = Field(..., description="Cost of interaction")
    success: bool = Field(..., description="Whether interaction was successful")
    user_id: str = Field(..., description="User ID")

class TaskExecutionRequest(BaseModel):
    task_id: str = Field(..., description="Task ID")
    task_type: str = Field(..., description="Task type")
    description: str = Field(..., description="Task description")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    success: bool = Field(..., description="Whether task was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    employee_id: Optional[str] = Field(None, description="Employee ID")
    user_id: str = Field(..., description="User ID")

class ModelUsageRequest(BaseModel):
    model_name: str = Field(..., description="Model name")
    provider: str = Field(..., description="Model provider")
    tokens_input: int = Field(..., description="Input tokens")
    tokens_output: int = Field(..., description="Output tokens")
    cost: float = Field(..., description="Usage cost")
    latency_ms: float = Field(..., description="Model latency in milliseconds")
    user_id: str = Field(..., description="User ID")
    request_type: str = Field(..., description="Type of request")

class TelemetryQueryRequest(BaseModel):
    event_type: str = Field(..., description="Type of event to query")
    start_time: datetime = Field(..., description="Start time for query")
    end_time: datetime = Field(..., description="End time for query")
    tenant_id: Optional[str] = Field(None, description="Tenant ID filter")
    limit: int = Field(100, description="Maximum number of results")

# Create router
router = APIRouter(prefix="/api/v1/azure-monitor", tags=["Azure Monitor"])

@router.get("/health")
async def get_azure_monitor_health():
    """Get Azure Monitor adapter health status"""
    
    try:
        health_result = await azure_monitor_adapter.health_check()
        
        return {
            "status": health_result.status.value,
            "message": health_result.message,
            "details": health_result.details,
            "response_time_ms": health_result.response_time_ms,
            "timestamp": health_result.timestamp
        }
        
    except Exception as e:
        logger.error(f"Failed to get Azure Monitor health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get health status")

@router.get("/statistics")
async def get_azure_monitor_statistics():
    """Get Azure Monitor adapter and business service statistics"""
    
    try:
        adapter_stats = azure_monitor_adapter.get_statistics()
        business_stats = azure_monitor_business_service.get_statistics()
        
        return {
            "adapter_statistics": adapter_stats,
            "business_statistics": business_stats,
            "combined_metrics": {
                "total_telemetry_events": adapter_stats.get("events_sent", 0) + business_stats.get("employee_interactions", 0),
                "total_cost_tracked": business_stats.get("total_cost", 0.0),
                "monitoring_health": "healthy" if adapter_stats.get("connection_configured", False) else "degraded"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get Azure Monitor statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

@router.post("/events/custom")
async def send_custom_event(event_request: CustomEventRequest):
    """Send a custom event to Azure Monitor"""
    
    try:
        # Get tenant context
        tenant_id = event_request.tenant_id or get_current_tenant()
        if not tenant_id:
            raise HTTPException(status_code=400, detail="Tenant ID required")
        
        # Create custom event
        event = CustomEvent(
            name=event_request.name,
            properties=event_request.properties,
            measurements=event_request.measurements,
            tenant_id=tenant_id
        )
        
        # Create execution context
        tenant_context = TenantContext(tenant_id=tenant_id)
        context = ExecutionContext(
            tenant_context=tenant_context,
            request_id=f"custom_event_{int(time.time())}"
        )
        
        # Track event
        success = await azure_monitor_adapter.track_custom_event(event, context)
        
        if success:
            return {
                "message": "Custom event sent successfully",
                "event_name": event_request.name,
                "tenant_id": tenant_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to send custom event")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send custom event: {e}")
        raise HTTPException(status_code=500, detail="Failed to send custom event")

@router.post("/metrics/custom")
async def send_custom_metric(metric_request: CustomMetricRequest):
    """Send a custom metric to Azure Monitor"""
    
    try:
        # Get tenant context
        tenant_id = metric_request.tenant_id or get_current_tenant()
        if not tenant_id:
            raise HTTPException(status_code=400, detail="Tenant ID required")
        
        # Create custom metric
        metric = CustomMetric(
            name=metric_request.name,
            value=metric_request.value,
            properties=metric_request.properties,
            tenant_id=tenant_id
        )
        
        # Create execution context
        tenant_context = TenantContext(tenant_id=tenant_id)
        context = ExecutionContext(
            tenant_context=tenant_context,
            request_id=f"custom_metric_{int(time.time())}"
        )
        
        # Track metric
        success = await azure_monitor_adapter.track_custom_metric(metric, context)
        
        if success:
            return {
                "message": "Custom metric sent successfully",
                "metric_name": metric_request.name,
                "value": metric_request.value,
                "tenant_id": tenant_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to send custom metric")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send custom metric: {e}")
        raise HTTPException(status_code=500, detail="Failed to send custom metric")

@router.post("/track/employee-interaction")
async def track_employee_interaction(interaction_request: EmployeeInteractionRequest):
    """Track an employee interaction event"""
    
    try:
        # Get tenant context
        tenant_id = get_current_tenant()
        if not tenant_id:
            raise HTTPException(status_code=400, detail="Tenant context required")
        
        # Create employee interaction
        interaction = EmployeeInteraction(
            employee_id=interaction_request.employee_id,
            employee_name=interaction_request.employee_name,
            interaction_type=interaction_request.interaction_type,
            user_message=interaction_request.user_message,
            response_message=interaction_request.response_message,
            processing_time_ms=interaction_request.processing_time_ms,
            model_used=interaction_request.model_used,
            tokens_used=interaction_request.tokens_used,
            cost=interaction_request.cost,
            success=interaction_request.success,
            tenant_id=tenant_id,
            user_id=interaction_request.user_id
        )
        
        # Track interaction
        success = await azure_monitor_business_service.track_employee_interaction(interaction)
        
        if success:
            return {
                "message": "Employee interaction tracked successfully",
                "employee_id": interaction_request.employee_id,
                "tenant_id": tenant_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to track employee interaction")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to track employee interaction: {e}")
        raise HTTPException(status_code=500, detail="Failed to track employee interaction")

@router.post("/track/task-execution")
async def track_task_execution(task_request: TaskExecutionRequest):
    """Track a task execution event"""
    
    try:
        # Get tenant context
        tenant_id = get_current_tenant()
        if not tenant_id:
            raise HTTPException(status_code=400, detail="Tenant context required")
        
        # Create task execution
        task = TaskExecution(
            task_id=task_request.task_id,
            task_type=task_request.task_type,
            description=task_request.description,
            execution_time_ms=task_request.execution_time_ms,
            success=task_request.success,
            error_message=task_request.error_message,
            tenant_id=tenant_id,
            employee_id=task_request.employee_id,
            user_id=task_request.user_id
        )
        
        # Track task
        success = await azure_monitor_business_service.track_task_execution(task)
        
        if success:
            return {
                "message": "Task execution tracked successfully",
                "task_id": task_request.task_id,
                "tenant_id": tenant_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to track task execution")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to track task execution: {e}")
        raise HTTPException(status_code=500, detail="Failed to track task execution")

@router.post("/track/model-usage")
async def track_model_usage(usage_request: ModelUsageRequest):
    """Track model usage event"""
    
    try:
        # Get tenant context
        tenant_id = get_current_tenant()
        if not tenant_id:
            raise HTTPException(status_code=400, detail="Tenant context required")
        
        # Create model usage
        usage = ModelUsage(
            model_name=usage_request.model_name,
            provider=usage_request.provider,
            tokens_input=usage_request.tokens_input,
            tokens_output=usage_request.tokens_output,
            cost=usage_request.cost,
            latency_ms=usage_request.latency_ms,
            tenant_id=tenant_id,
            user_id=usage_request.user_id,
            request_type=usage_request.request_type
        )
        
        # Track usage
        success = await azure_monitor_business_service.track_model_usage(usage)
        
        if success:
            return {
                "message": "Model usage tracked successfully",
                "model_name": usage_request.model_name,
                "tenant_id": tenant_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to track model usage")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to track model usage: {e}")
        raise HTTPException(status_code=500, detail="Failed to track model usage")

@router.post("/track/policy-violation")
async def track_policy_violation(
    violation_type: str = Query(..., description="Type of policy violation"),
    resource: str = Query(..., description="Resource that was accessed"),
    severity: str = Query("medium", description="Violation severity"),
    policy_name: str = Query(..., description="Name of violated policy"),
    action_taken: str = Query("denied", description="Action taken")
):
    """Track a policy violation event"""
    
    try:
        # Get tenant context
        tenant_id = get_current_tenant()
        if not tenant_id:
            raise HTTPException(status_code=400, detail="Tenant context required")
        
        # Track violation
        success = await azure_monitor_business_service.track_policy_violation(
            violation_type=violation_type,
            resource=resource,
            tenant_id=tenant_id,
            user_id="api_user",  # Would get from auth context
            details={
                "severity": severity,
                "policy_name": policy_name,
                "action_taken": action_taken,
                "severity_score": {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}.get(severity, 0.5)
            }
        )
        
        if success:
            return {
                "message": "Policy violation tracked successfully",
                "violation_type": violation_type,
                "tenant_id": tenant_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to track policy violation")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to track policy violation: {e}")
        raise HTTPException(status_code=500, detail="Failed to track policy violation")

@router.post("/track/compliance-check")
async def track_compliance_check(
    check_type: str = Query(..., description="Type of compliance check"),
    result: str = Query(..., description="Check result"),
    regulation: str = Query("general", description="Regulation being checked"),
    data_classification: str = Query("unclassified", description="Data classification"),
    compliance_score: float = Query(1.0, description="Compliance score (0.0-1.0)")
):
    """Track a compliance check event"""
    
    try:
        # Get tenant context
        tenant_id = get_current_tenant()
        if not tenant_id:
            raise HTTPException(status_code=400, detail="Tenant context required")
        
        # Track compliance check
        success = await azure_monitor_business_service.track_compliance_check(
            check_type=check_type,
            result=result,
            tenant_id=tenant_id,
            details={
                "regulation": regulation,
                "data_classification": data_classification,
                "compliance_score": compliance_score,
                "remediation_required": compliance_score < 0.8
            }
        )
        
        if success:
            return {
                "message": "Compliance check tracked successfully",
                "check_type": check_type,
                "result": result,
                "tenant_id": tenant_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to track compliance check")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to track compliance check: {e}")
        raise HTTPException(status_code=500, detail="Failed to track compliance check")

@router.get("/configuration")
async def get_azure_monitor_configuration():
    """Get current Azure Monitor configuration"""
    
    try:
        config = azure_monitor_adapter.azure_config
        
        # Return configuration (without sensitive data)
        return {
            "connection_configured": bool(config.connection_string),
            "instrumentation_key_configured": bool(config.instrumentation_key),
            "auto_instrumentation_enabled": config.enable_auto_instrumentation,
            "custom_metrics_enabled": config.enable_custom_metrics,
            "custom_events_enabled": config.enable_custom_events,
            "dependency_tracking_enabled": config.enable_dependency_tracking,
            "exception_tracking_enabled": config.enable_exception_tracking,
            "sampling_percentage": config.sampling_percentage,
            "tenant_isolation_enabled": config.tenant_isolation_enabled,
            "custom_dimensions": config.custom_dimensions
        }
        
    except Exception as e:
        logger.error(f"Failed to get Azure Monitor configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to get configuration")

@router.post("/daily-summary/{tenant_id}")
async def send_daily_summary(tenant_id: str):
    """Send daily summary for a tenant"""
    
    try:
        success = await azure_monitor_business_service.send_daily_summary(tenant_id)
        
        if success:
            return {
                "message": "Daily summary sent successfully",
                "tenant_id": tenant_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to send daily summary")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send daily summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to send daily summary")

@router.get("/telemetry/summary")
async def get_telemetry_summary(
    hours: int = Query(24, description="Hours to look back"),
    tenant_id: Optional[str] = Query(None, description="Tenant ID filter")
):
    """Get telemetry summary for the specified period"""
    
    try:
        # Use current tenant if not specified
        if not tenant_id:
            tenant_id = get_current_tenant()
            if not tenant_id:
                raise HTTPException(status_code=400, detail="Tenant context required")
        
        # Get statistics from both adapter and business service
        adapter_stats = azure_monitor_adapter.get_statistics()
        business_stats = azure_monitor_business_service.get_statistics()
        
        # Calculate summary metrics
        summary = {
            "tenant_id": tenant_id,
            "period_hours": hours,
            "telemetry_summary": {
                "total_events": adapter_stats.get("events_sent", 0),
                "total_metrics": adapter_stats.get("metrics_sent", 0),
                "total_traces": adapter_stats.get("traces_sent", 0),
                "total_exceptions": adapter_stats.get("exceptions_logged", 0),
                "total_dependencies": adapter_stats.get("dependencies_tracked", 0)
            },
            "business_summary": {
                "employee_interactions": business_stats.get("employee_interactions", 0),
                "tasks_executed": business_stats.get("tasks_executed", 0),
                "model_calls": business_stats.get("model_calls", 0),
                "total_cost": business_stats.get("total_cost", 0.0),
                "policy_violations": business_stats.get("policy_violations", 0),
                "compliance_checks": business_stats.get("compliance_checks", 0)
            },
            "health_status": {
                "azure_monitor_healthy": adapter_stats.get("connection_configured", False),
                "telemetry_flowing": adapter_stats.get("events_sent", 0) > 0
            },
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get telemetry summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get telemetry summary")

@router.post("/test/connectivity")
async def test_azure_monitor_connectivity():
    """Test Azure Monitor connectivity and send test events"""
    
    try:
        # Get tenant context
        tenant_id = get_current_tenant()
        if not tenant_id:
            tenant_id = "test_tenant"
        
        # Create test execution context
        tenant_context = TenantContext(tenant_id=tenant_id, user_id="test_user")
        context = ExecutionContext(
            tenant_context=tenant_context,
            request_id=f"connectivity_test_{int(time.time())}"
        )
        
        # Send test event
        test_event = CustomEvent(
            name="forge1_connectivity_test",
            properties={
                "test_type": "api_connectivity",
                "tenant_id": tenant_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            measurements={
                "test_value": 1.0
            },
            tenant_id=tenant_id
        )
        
        event_success = await azure_monitor_adapter.track_custom_event(test_event, context)
        
        # Send test metric
        test_metric = CustomMetric(
            name="forge1_connectivity_test_metric",
            value=1.0,
            properties={
                "test_type": "api_connectivity",
                "tenant_id": tenant_id
            },
            tenant_id=tenant_id
        )
        
        metric_success = await azure_monitor_adapter.track_custom_metric(test_metric, context)
        
        # Test dependency tracking
        dependency_success = await azure_monitor_adapter.track_dependency(
            name="azure_monitor_test",
            dependency_type="http",
            target="azure_monitor_api",
            duration_ms=100.0,
            success=True,
            context=context
        )
        
        return {
            "message": "Azure Monitor connectivity test completed",
            "results": {
                "event_sent": event_success,
                "metric_sent": metric_success,
                "dependency_tracked": dependency_success,
                "overall_success": all([event_success, metric_success, dependency_success])
            },
            "tenant_id": tenant_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Azure Monitor connectivity test failed: {e}")
        raise HTTPException(status_code=500, detail="Connectivity test failed")

# Export the router
__all__ = ["router"]