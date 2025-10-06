"""
Analytics API Endpoints

Provides REST API endpoints for accessing Azure Monitor analytics,
business intelligence, and tenant-specific insights.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

# Import analytics service with graceful fallback
try:
    from forge1.services.analytics.azure_monitor_analytics import (
        azure_monitor_analytics_service,
        AnalyticsTimeRange
    )
    ANALYTICS_SERVICE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Analytics service not available: {e}")
    ANALYTICS_SERVICE_AVAILABLE = False
    
    # Create stub classes
    class AnalyticsTimeRange:
        LAST_HOUR = "1h"
        LAST_24_HOURS = "24h"
        LAST_7_DAYS = "7d"
        LAST_30_DAYS = "30d"
        LAST_90_DAYS = "90d"
    
    class StubAnalyticsService:
        def get_service_statistics(self):
            return {"available_queries": [], "queries_executed": 0}
        
        async def execute_analytics_query(self, *args, **kwargs):
            return None
        
        async def execute_custom_query(self, *args, **kwargs):
            return None
        
        async def get_tenant_dashboard_data(self, *args, **kwargs):
            return {"error": "Analytics service not available"}
        
        async def get_cost_insights(self, *args, **kwargs):
            return {"error": "Analytics service not available"}
        
        async def create_alert_rule(self, *args, **kwargs):
            return False
        
        def clear_cache(self):
            pass
    
    azure_monitor_analytics_service = StubAnalyticsService()
from forge1.core.tenancy import get_current_tenant
# Import middleware with graceful fallback
try:
    from forge1.middleware.azure_monitor_middleware import azure_monitor_middleware
except ImportError:
    logger.warning("Azure Monitor middleware not available for analytics API")
    
    class StubMiddleware:
        def get_statistics(self):
            return {"error": "Middleware not available"}
        
        async def send_custom_business_event(self, *args, **kwargs):
            return False
        
        async def send_tenant_activity_metric(self, *args, **kwargs):
            return False
    
    azure_monitor_middleware = StubMiddleware()

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class AnalyticsQueryRequest(BaseModel):
    """Request model for custom analytics queries"""
    query_name: str = Field(..., description="Name for the analytics query")
    kql_query: str = Field(..., description="KQL query to execute")
    time_range: Optional[AnalyticsTimeRange] = Field(
        default=AnalyticsTimeRange.LAST_24_HOURS,
        description="Time range for the query"
    )

class AnalyticsResponse(BaseModel):
    """Response model for analytics queries"""
    query_name: str
    data: List[Dict[str, Any]]
    total_records: int
    execution_time_ms: float
    timestamp: datetime
    tenant_id: Optional[str] = None

class DashboardResponse(BaseModel):
    """Response model for tenant dashboard data"""
    tenant_id: str
    time_range: str
    generated_at: str
    activity_summary: List[Dict[str, Any]]
    performance_analysis: List[Dict[str, Any]]
    error_analysis: List[Dict[str, Any]]
    usage_patterns: List[Dict[str, Any]]
    summary_metrics: Dict[str, Any]

class CostInsightsResponse(BaseModel):
    """Response model for cost insights"""
    tenant_id: str
    time_range: str
    total_cost: float
    avg_daily_cost: float
    top_cost_drivers: List[tuple]
    cost_trend: List[Dict[str, Any]]
    generated_at: str

class AlertRuleRequest(BaseModel):
    """Request model for creating alert rules"""
    alert_name: str = Field(..., description="Name for the alert rule")
    kql_query: str = Field(..., description="KQL query for the alert")
    threshold: float = Field(..., description="Alert threshold value")
    alert_type: str = Field(default="metric", description="Type of alert (metric, log, etc.)")

# Create router
router = APIRouter(prefix="/analytics", tags=["analytics"])

@router.get("/queries/available", response_model=Dict[str, List[str]])
async def get_available_queries():
    """Get list of available predefined analytics queries"""
    
    try:
        if not ANALYTICS_SERVICE_AVAILABLE:
            return {
                "available_queries": [],
                "total_count": 0,
                "message": "Analytics service not available"
            }
        
        available_queries = list(azure_monitor_analytics_service.predefined_queries.keys()) if hasattr(azure_monitor_analytics_service, 'predefined_queries') else []
        
        return {
            "available_queries": available_queries,
            "total_count": len(available_queries)
        }
        
    except Exception as e:
        logger.error(f"Failed to get available queries: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve available queries")

@router.get("/query/{query_name}", response_model=AnalyticsResponse)
async def execute_predefined_query(
    query_name: str,
    time_range: Optional[AnalyticsTimeRange] = Query(default=AnalyticsTimeRange.LAST_24_HOURS),
    tenant_id: Optional[str] = Depends(get_current_tenant)
):
    """Execute a predefined analytics query"""
    
    if not tenant_id:
        raise HTTPException(status_code=400, detail="Tenant ID is required")
    
    try:
        result = await azure_monitor_analytics_service.execute_analytics_query(
            query_name=query_name,
            tenant_id=tenant_id,
            time_range=time_range
        )
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Query '{query_name}' not found or failed to execute")
        
        return AnalyticsResponse(
            query_name=result.query_name,
            data=result.data,
            total_records=result.total_records,
            execution_time_ms=result.execution_time_ms,
            timestamp=result.timestamp,
            tenant_id=result.tenant_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute query {query_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute query: {str(e)}")

@router.post("/query/custom", response_model=AnalyticsResponse)
async def execute_custom_query(
    request: AnalyticsQueryRequest,
    tenant_id: Optional[str] = Depends(get_current_tenant)
):
    """Execute a custom analytics query"""
    
    if not tenant_id:
        raise HTTPException(status_code=400, detail="Tenant ID is required")
    
    try:
        result = await azure_monitor_analytics_service.execute_custom_query(
            kql_query=request.kql_query,
            tenant_id=tenant_id,
            query_name=request.query_name
        )
        
        if not result:
            raise HTTPException(status_code=400, detail="Failed to execute custom query")
        
        return AnalyticsResponse(
            query_name=result.query_name,
            data=result.data,
            total_records=result.total_records,
            execution_time_ms=result.execution_time_ms,
            timestamp=result.timestamp,
            tenant_id=result.tenant_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute custom query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute custom query: {str(e)}")

@router.get("/dashboard", response_model=DashboardResponse)
async def get_tenant_dashboard(
    time_range: Optional[AnalyticsTimeRange] = Query(default=AnalyticsTimeRange.LAST_24_HOURS),
    tenant_id: Optional[str] = Depends(get_current_tenant)
):
    """Get comprehensive dashboard data for the current tenant"""
    
    if not tenant_id:
        raise HTTPException(status_code=400, detail="Tenant ID is required")
    
    try:
        dashboard_data = await azure_monitor_analytics_service.get_tenant_dashboard_data(
            tenant_id=tenant_id,
            time_range=time_range
        )
        
        if "error" in dashboard_data:
            raise HTTPException(status_code=500, detail=dashboard_data["error"])
        
        return DashboardResponse(**dashboard_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dashboard data: {str(e)}")

@router.get("/cost-insights", response_model=CostInsightsResponse)
async def get_cost_insights(
    time_range: Optional[AnalyticsTimeRange] = Query(default=AnalyticsTimeRange.LAST_30_DAYS),
    tenant_id: Optional[str] = Depends(get_current_tenant)
):
    """Get cost and usage insights for the current tenant"""
    
    if not tenant_id:
        raise HTTPException(status_code=400, detail="Tenant ID is required")
    
    try:
        cost_insights = await azure_monitor_analytics_service.get_cost_insights(
            tenant_id=tenant_id,
            time_range=time_range
        )
        
        if "error" in cost_insights:
            raise HTTPException(status_code=500, detail=cost_insights["error"])
        
        return CostInsightsResponse(**cost_insights)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cost insights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve cost insights: {str(e)}")

@router.post("/alerts", response_model=Dict[str, Any])
async def create_alert_rule(
    request: AlertRuleRequest,
    tenant_id: Optional[str] = Depends(get_current_tenant)
):
    """Create an alert rule in Azure Monitor"""
    
    if not tenant_id:
        raise HTTPException(status_code=400, detail="Tenant ID is required")
    
    try:
        success = await azure_monitor_analytics_service.create_alert_rule(
            tenant_id=tenant_id,
            alert_name=request.alert_name,
            kql_query=request.kql_query,
            threshold=request.threshold,
            alert_type=request.alert_type
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create alert rule")
        
        return {
            "success": True,
            "message": f"Alert rule '{request.alert_name}' created successfully",
            "alert_name": request.alert_name,
            "tenant_id": tenant_id,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create alert rule: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create alert rule: {str(e)}")

@router.get("/statistics", response_model=Dict[str, Any])
async def get_analytics_statistics():
    """Get analytics service statistics"""
    
    try:
        service_stats = azure_monitor_analytics_service.get_service_statistics()
        middleware_stats = azure_monitor_middleware.get_statistics() if hasattr(azure_monitor_middleware, 'get_statistics') else {}
        
        return {
            "service_statistics": service_stats,
            "middleware_statistics": middleware_stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get analytics statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")

@router.delete("/cache", response_model=Dict[str, str])
async def clear_analytics_cache():
    """Clear the analytics query cache"""
    
    try:
        azure_monitor_analytics_service.clear_cache()
        
        return {
            "message": "Analytics cache cleared successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear analytics cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.post("/events/business", response_model=Dict[str, Any])
async def send_business_event(
    event_name: str,
    properties: Dict[str, Any],
    tenant_id: Optional[str] = Depends(get_current_tenant)
):
    """Send a custom business event to Azure Monitor"""
    
    if not tenant_id:
        raise HTTPException(status_code=400, detail="Tenant ID is required")
    
    try:
        # Add tenant context
        properties["tenant_id"] = tenant_id
        properties["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        success = await azure_monitor_middleware.send_custom_business_event(
            event_name=event_name,
            properties=properties
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to send business event")
        
        return {
            "success": True,
            "message": f"Business event '{event_name}' sent successfully",
            "event_name": event_name,
            "tenant_id": tenant_id,
            "sent_at": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send business event: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send business event: {str(e)}")

@router.post("/metrics/tenant-activity", response_model=Dict[str, Any])
async def send_tenant_activity_metric(
    activity_type: str,
    value: float,
    tenant_id: Optional[str] = Depends(get_current_tenant)
):
    """Send a tenant activity metric to Azure Monitor"""
    
    if not tenant_id:
        raise HTTPException(status_code=400, detail="Tenant ID is required")
    
    try:
        success = await azure_monitor_middleware.send_tenant_activity_metric(
            activity_type=activity_type,
            value=value
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to send tenant activity metric")
        
        return {
            "success": True,
            "message": f"Tenant activity metric '{activity_type}' sent successfully",
            "activity_type": activity_type,
            "value": value,
            "tenant_id": tenant_id,
            "sent_at": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send tenant activity metric: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send tenant activity metric: {str(e)}")

# Health check endpoint
@router.get("/health", response_model=Dict[str, Any])
async def analytics_health_check():
    """Health check for analytics services"""
    
    try:
        status = "healthy"
        details = {
            "analytics_service_available": ANALYTICS_SERVICE_AVAILABLE,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Try to check Azure Monitor integration health if available
        try:
            from forge1.integrations.observability.azure_monitor_adapter import azure_monitor_adapter
            health_result = await azure_monitor_adapter.health_check()
            details["azure_monitor_status"] = health_result.status.value
            details["azure_monitor_message"] = health_result.message
            details["response_time_ms"] = health_result.response_time_ms
            
            if health_result.status.value != "healthy":
                status = "degraded"
                
        except ImportError:
            details["azure_monitor_status"] = "not_available"
            details["azure_monitor_message"] = "Azure Monitor integration not available"
            status = "degraded"
        except Exception as e:
            details["azure_monitor_status"] = "error"
            details["azure_monitor_message"] = str(e)
            status = "degraded"
        
        return {
            "status": status,
            **details
        }
        
    except Exception as e:
        logger.error(f"Analytics health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }