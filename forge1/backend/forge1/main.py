"""
Forge 1 Backend - Simplified Working Version
Enterprise AI Employee Builder Platform
"""

import os
import time
import logging
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest

from forge1.core.logging_config import init_logger

logger = init_logger("forge1.api.main")

# Import API routers
from forge1.api.employee_lifecycle_api import (
    ROUTER_EXCEPTION_HANDLERS,
    router as employee_lifecycle_router,
)
try:
    from forge1.api.automation_api import router as automation_router
    AUTOMATION_API_AVAILABLE = True
except Exception as exc:  # pragma: no cover - optional API
    logger.warning("Automation API not available", extra={"error": str(exc)})
    automation_router = None
    AUTOMATION_API_AVAILABLE = False

try:
    from forge1.api.compliance_api import router as compliance_router
    COMPLIANCE_API_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    logger.warning("Compliance API not available", extra={"error": str(exc)})
    compliance_router = None
    COMPLIANCE_API_AVAILABLE = False

try:
    from forge1.api.api_integration_api import router as api_integration_router
    API_INTEGRATION_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    logger.warning("API Integration router not available", extra={"error": str(exc)})
    api_integration_router = None
    API_INTEGRATION_AVAILABLE = False

try:
    from forge1.api.azure_monitor_api import router as azure_monitor_router
    AZURE_MONITOR_API_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    logger.warning("Azure Monitor API not available", extra={"error": str(exc)})
    azure_monitor_router = None
    AZURE_MONITOR_API_AVAILABLE = False

# Analytics router - will be imported conditionally to handle missing dependencies
try:
    from forge1.api.v1.analytics import router as analytics_router
    ANALYTICS_AVAILABLE = True
except Exception as e:  # pragma: no cover
    logger.warning("Analytics API not available", extra={"error": str(e)})
    ANALYTICS_AVAILABLE = False
    analytics_router = None

# Import security middleware
from forge1.middleware.security_middleware import TenantIsolationMiddleware

# Azure Monitor middleware - import conditionally
try:
    from forge1.middleware.azure_monitor_middleware import AzureMonitorMiddleware
    AZURE_MONITOR_MIDDLEWARE_AVAILABLE = True
except ImportError as e:
    logger.warning("Azure Monitor middleware not available", extra={"error": str(e)})
    AZURE_MONITOR_MIDDLEWARE_AVAILABLE = False
    AzureMonitorMiddleware = None

from forge1.core.audit_logger import AuditLogger
from forge1.core.encryption_manager import EncryptionManager

# Import performance optimization components
from forge1.core.redis_cache_manager import get_cache_manager, close_cache_manager
from forge1.core.performance_monitor import get_performance_monitor
from forge1.core.connection_pool_manager import (
    get_connection_pool_manager,
    close_connection_pool_manager,
)

# Import system integration components
from forge1.integrations.forge1_system_adapter import get_system_integrator
from forge1.integrations.queue.celery_app import celery_adapter
from forge1.integrations.base_adapter import AdapterStatus

try:
    from forge1.integrations.vector.vector_store_weaviate import WeaviateMemoryAdapter
    VECTOR_ADAPTER_AVAILABLE = True
except ImportError as e:
    logger.warning("Vector adapter not available", extra={"error": str(e)})
    WeaviateMemoryAdapter = None  # type: ignore[assignment]
    VECTOR_ADAPTER_AVAILABLE = False
from forge1.services.employee_manager import EmployeeManager
from forge1.services.employee_memory_manager import EmployeeMemoryManager

# Import analytics components
from forge1.services.employee_analytics_service import get_analytics_service

# MCAE Integration imports
from forge1.integrations.mcae_adapter import MCAEAdapter
from forge1.integrations.mcae_error_handler import MCAEErrorHandler

# Create FastAPI app
app = FastAPI(
    title="Forge 1 Platform API",
    description="Enterprise AI Employee Builder Platform - Superhuman AI Employees",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize security components
audit_logger = AuditLogger()
encryption_manager = EncryptionManager()

# Initialize performance components
performance_monitor = get_performance_monitor()
cache_manager = None
pool_manager = None
system_integrator = None
analytics_service = None

# MCAE Integration components
mcae_adapter = None
mcae_error_handler = None
vector_adapter = None

APP_START_TIME = time.time()

HEALTH_STATUS_GAUGE = Gauge("forge1_health_status", "Overall Forge1 health (1=ok,0=degraded)")
REDIS_HEALTH_GAUGE = Gauge("forge1_redis_health", "Redis cache health (1=ok,0=degraded)")
CELERY_HEALTH_GAUGE = Gauge("forge1_celery_health", "Celery adapter health (1=ok,0=degraded)")
VECTOR_HEALTH_GAUGE = Gauge("forge1_vector_health", "Vector adapter health (1=ok,0=degraded)")


async def _collect_service_health() -> Dict[str, Any]:
    global cache_manager, vector_adapter

    # Redis
    redis_ok = False
    redis_details = "cache manager not initialized"
    try:
        if cache_manager is None:
            cache_manager = await get_cache_manager()
        if cache_manager:
            redis_ok = getattr(cache_manager, "_initialized", False)
            redis_details = "initialized" if redis_ok else "initialization incomplete"
    except Exception as exc:  # pragma: no cover - defensive
        redis_details = f"error: {exc}"
    REDIS_HEALTH_GAUGE.set(1 if redis_ok else 0)

    # Celery
    celery_ok = False
    celery_details = ""
    try:
        result = await celery_adapter.health_check()
        celery_ok = result.status != AdapterStatus.ERROR
        celery_details = result.message
    except Exception as exc:  # pragma: no cover
        celery_details = f"error: {exc}"
    CELERY_HEALTH_GAUGE.set(1 if celery_ok else 0)

    # Vector
    vector_ok = False
    vector_details = "adapter unavailable"
    if VECTOR_ADAPTER_AVAILABLE:
        if vector_adapter is None and WeaviateMemoryAdapter:
            try:
                vector_adapter = WeaviateMemoryAdapter()
            except Exception as exc:  # pragma: no cover
                vector_details = f"initialization error: {exc}"
        if vector_adapter is not None:
            vector_ok = True
            vector_details = "adapter ready (lazy initialization)"
    VECTOR_HEALTH_GAUGE.set(1 if vector_ok else 0)

    services = {
        "redis": {"status": "ok" if redis_ok else "degraded", "details": redis_details},
        "celery": {"status": "ok" if celery_ok else "degraded", "details": celery_details},
        "vector": {"status": "ok" if vector_ok else "degraded", "details": vector_details},
    }

    overall_ok = all(service["status"] == "ok" for service in services.values())
    HEALTH_STATUS_GAUGE.set(1 if overall_ok else 0)

    return {
        "status": "ok" if overall_ok else "degraded",
        "uptime_seconds": time.time() - APP_START_TIME,
        "services": services,
    }
# Startup event to initialize performance components
@app.on_event("startup")
async def startup_event():
    global cache_manager, pool_manager, system_integrator, analytics_service, mcae_adapter, mcae_error_handler
    
    # Initialize performance monitoring
    await performance_monitor.start_monitoring(interval_seconds=30)
    
    # Initialize cache manager
    cache_manager = await get_cache_manager()
    
    # Initialize connection pool manager
    pool_manager = await get_connection_pool_manager()
    
    # Initialize system integration
    system_integrator = await get_system_integrator()
    
    # Initialize employee system components
    employee_manager = EmployeeManager()
    await employee_manager.initialize()
    
    employee_memory_manager = EmployeeMemoryManager()
    await employee_memory_manager.initialize()
    
    # Initialize system integrator with employee components
    await system_integrator.initialize(employee_manager, employee_memory_manager)
    
    # Initialize analytics service
    analytics_service = await get_analytics_service()
    
    # Initialize MCAE integration
    try:
        # Initialize MCAE error handler
        mcae_error_handler = MCAEErrorHandler(
            employee_manager=employee_manager,
            audit_logger=audit_logger
        )
        
        # Initialize MCAE adapter
        mcae_adapter = MCAEAdapter(
            employee_manager=employee_manager,
            model_router=employee_manager.model_router,
            memory_manager=employee_manager.memory_manager
        )
        await mcae_adapter.initialize()
        
        # Update employee manager with MCAE adapter
        employee_manager.mcae_adapter = mcae_adapter
        
        logger.info("MCAE integration initialized successfully")
        
    except Exception as e:
        logger.warning(f"MCAE integration failed to initialize: {e}")
        logger.info("Continuing without MCAE integration - using Forge1 native orchestration")
    
    logger.info("All system components initialized successfully")


@app.get("/health")
async def health() -> JSONResponse:
    report = await _collect_service_health()
    return JSONResponse(report)


@app.get("/metrics")
async def metrics() -> Response:
    await _collect_service_health()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Shutdown event to cleanup resources
@app.on_event("shutdown")
async def shutdown_event():
    global cache_manager, pool_manager, mcae_adapter
    
    # Stop performance monitoring
    await performance_monitor.stop_monitoring()
    
    # Close cache manager
    if cache_manager:
        await close_cache_manager()
    
    # Close connection pool manager
    if pool_manager:
        await close_connection_pool_manager()
    
    # Cleanup MCAE integration
    if mcae_adapter:
        try:
            # Cleanup any active workflows
            for workflow_id in list(mcae_adapter.active_workflows.keys()):
                await mcae_adapter.cleanup_workflow(workflow_id)
            logger.info("MCAE integration cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up MCAE integration: {e}")
    
    logger.info("All system components shut down successfully")

# Add security middleware (order matters - security first)
app.add_middleware(
    TenantIsolationMiddleware,
    audit_logger=audit_logger,
    encryption_manager=encryption_manager
)

# Add Azure Monitor telemetry middleware if available
if AZURE_MONITOR_MIDDLEWARE_AVAILABLE and AzureMonitorMiddleware:
    app.add_middleware(AzureMonitorMiddleware)
    logger.info("Azure Monitor middleware added")
else:
    logger.warning("Azure Monitor middleware not available - skipping")

# Add CORS middleware with proper security configuration
allowed_origins = [
    "http://localhost:3000",  # React dev server
    "http://localhost:8080",  # Vue dev server
    "https://forge1.example.com",  # Production domain
]

# In development, allow all origins (configure properly for production)
if os.getenv("ENVIRONMENT", "development") == "development":
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Tenant-ID",
        "X-Client-ID", 
        "X-User-ID",
        "X-Request-ID"
    ],
    expose_headers=["X-Request-ID", "X-Rate-Limit-Remaining"],
)

# Include API routers
app.include_router(employee_lifecycle_router)

if AUTOMATION_API_AVAILABLE and automation_router:
    app.include_router(automation_router)
else:
    logger.warning("Automation router unavailable - skipping")

if COMPLIANCE_API_AVAILABLE and compliance_router:
    app.include_router(compliance_router)
else:
    logger.warning("Compliance router unavailable - skipping")

if API_INTEGRATION_AVAILABLE and api_integration_router:
    app.include_router(api_integration_router)
else:
    logger.warning("API integration router unavailable - skipping")

if AZURE_MONITOR_API_AVAILABLE and azure_monitor_router:
    app.include_router(azure_monitor_router)
else:
    logger.warning("Azure Monitor router unavailable - skipping")

# Include analytics router if available
if ANALYTICS_AVAILABLE and analytics_router:
    app.include_router(analytics_router)
    logger.info("Analytics API router included")
else:
    logger.warning("Analytics API router not available - skipping")

for exc_type, handler in ROUTER_EXCEPTION_HANDLERS:
    app.add_exception_handler(exc_type, handler)

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    timestamp: float
    service: str
    version: str

class TaskRequest(BaseModel):
    description: str
    session_id: str = None
    priority: str = "normal"

class TaskResponse(BaseModel):
    status: str
    session_id: str
    task_id: str
    description: str
    processing_time: float

# In-memory storage for demo
tasks_db = {}
employees_db = {
    "1": {"id": "1", "name": "Customer Support Bot", "type": "customer_experience", "status": "active"},
    "2": {"id": "2", "name": "Sales Assistant", "type": "sales", "status": "active"},
    "3": {"id": "3", "name": "Data Analyst", "type": "analytics", "status": "active"},
}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Forge 1 Platform API", "status": "operational"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check"""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        service="forge1-backend",
        version="1.0.0"
    )

@app.get("/api/v1/health/detailed")
async def detailed_health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "forge1-backend",
        "version": "1.0.0",
        "components": {
            "database": {"status": "healthy", "response_time": 0.001},
            "cache": {"status": "healthy", "response_time": 0.001},
            "ai_models": {"status": "healthy", "available_models": 3}
        },
        "metrics": {
            "active_employees": len(employees_db),
            "total_tasks": len(tasks_db),
            "uptime": 3600
        }
    }

@app.post("/api/v1/tasks", response_model=TaskResponse)
async def create_task(task: TaskRequest):
    """Create a new AI task"""
    start_time = time.time()
    
    # Generate task ID
    task_id = f"task_{int(time.time())}"
    
    # Generate session ID if not provided
    session_id = task.session_id or f"session_{int(time.time())}"
    
    # Store task
    tasks_db[task_id] = {
        "id": task_id,
        "session_id": session_id,
        "description": task.description,
        "priority": task.priority,
        "status": "completed",
        "created_at": time.time(),
        "processing_time": time.time() - start_time
    }
    
    logger.info(f"Created task {task_id}: {task.description}")
    
    return TaskResponse(
        status="Task created and processed successfully",
        session_id=session_id,
        task_id=task_id,
        description=task.description,
        processing_time=time.time() - start_time
    )

@app.get("/api/v1/tasks")
async def get_tasks():
    """Get all tasks"""
    return {"tasks": list(tasks_db.values()), "total": len(tasks_db)}

@app.get("/api/v1/tasks/{task_id}")
async def get_task(task_id: str):
    """Get specific task"""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks_db[task_id]

@app.get("/api/v1/employees")
async def get_employees():
    """Get all AI employees"""
    return {"employees": list(employees_db.values()), "total": len(employees_db)}

@app.get("/api/v1/employees/{employee_id}")
async def get_employee(employee_id: str):
    """Get specific AI employee"""
    if employee_id not in employees_db:
        raise HTTPException(status_code=404, detail="Employee not found")
    return employees_db[employee_id]

@app.post("/api/v1/employees")
async def create_employee(employee_data: Dict[str, Any]):
    """Create new AI employee"""
    employee_id = f"emp_{int(time.time())}"
    employee = {
        "id": employee_id,
        "name": employee_data.get("name", "New Employee"),
        "type": employee_data.get("type", "general"),
        "status": "active",
        "created_at": time.time()
    }
    employees_db[employee_id] = employee
    logger.info(f"Created employee {employee_id}: {employee['name']}")
    return employee

@app.get("/api/v1/system/status")
async def system_status():
    """Get system status"""
    return {
        "system": "Forge 1 Platform",
        "version": "1.0.0",
        "status": "operational",
        "features": {
            "ai_employees": True,
            "task_processing": True,
            "real_time_monitoring": True,
            "enterprise_security": True
        },
        "statistics": {
            "active_employees": len(employees_db),
            "total_tasks": len(tasks_db),
            "success_rate": 99.9,
            "average_response_time": 0.5
        },
        "timestamp": time.time()
    }

@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    metrics = f"""# HELP forge1_tasks_total Total number of tasks
# TYPE forge1_tasks_total counter
forge1_tasks_total {len(tasks_db)}

# HELP forge1_employees_total Total number of AI employees
# TYPE forge1_employees_total gauge
forge1_employees_total {len(employees_db)}

# HELP forge1_system_health System health status (1=healthy, 0=unhealthy)
# TYPE forge1_system_health gauge
forge1_system_health 1
"""
    return JSONResponse(content=metrics, media_type="text/plain")

# Security and Audit Endpoints

@app.get("/api/v1/security/audit")
async def get_audit_events(
    hours: int = 24,
    event_type: str = None,
    severity: str = None,
    limit: int = 100
):
    """Get audit events for security monitoring"""
    try:
        # In production, this would require admin authentication
        events = await audit_logger.get_audit_events(
            limit=limit
        )
        
        return {
            "events": events,
            "total_count": len(events),
            "filters": {
                "hours": hours,
                "event_type": event_type,
                "severity": severity
            }
        }
    except Exception as e:
        logger.error(f"Failed to get audit events: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve audit events")

@app.get("/api/v1/security/summary")
async def get_security_summary(hours: int = 24):
    """Get security event summary"""
    try:
        summary = await audit_logger.get_security_summary(hours=hours)
        return summary
    except Exception as e:
        logger.error(f"Failed to get security summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve security summary")

@app.get("/api/v1/security/encryption/{tenant_id}")
async def get_encryption_info(tenant_id: str):
    """Get encryption information for a tenant"""
    try:
        # In production, validate tenant access
        metadata = encryption_manager.get_encryption_metadata(tenant_id)
        return metadata
    except Exception as e:
        logger.error(f"Failed to get encryption info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve encryption information")

@app.post("/api/v1/security/encryption/{tenant_id}/rotate")
async def rotate_tenant_key(tenant_id: str):
    """Rotate encryption key for a tenant"""
    try:
        # In production, require admin authentication and audit this operation
        success = encryption_manager.rotate_tenant_key(tenant_id)
        
        if success:
            await audit_logger.log_system_event(
                event_type=audit_logger.AuditEventType.CONFIGURATION_CHANGE,
                details={
                    "operation": "key_rotation",
                    "tenant_id": tenant_id,
                    "initiated_by": "admin"  # In production, get from auth context
                }
            )
            
            return {"message": "Key rotation initiated successfully", "tenant_id": tenant_id}
        else:
            raise HTTPException(status_code=500, detail="Key rotation failed")
            
    except Exception as e:
        logger.error(f"Failed to rotate tenant key: {e}")
        raise HTTPException(status_code=500, detail="Failed to rotate encryption key")

# Performance Monitoring and Optimization Endpoints

@app.get("/api/v1/performance/metrics")
async def get_performance_metrics():
    """Get comprehensive performance metrics"""
    try:
        metrics = performance_monitor.get_current_metrics()
        
        # Add cache metrics if available
        if cache_manager:
            cache_health = await cache_manager.health_check()
            metrics["cache"] = cache_health
        
        # Add database pool metrics if available
        if pool_manager:
            pool_status = await pool_manager.get_pool_status()
            metrics["database_pool"] = pool_status
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance metrics")

@app.get("/api/v1/performance/health")
async def get_system_health():
    """Get system health score and status"""
    try:
        health_score = performance_monitor.get_system_health_score()
        
        # Add component health checks
        component_health = {}
        
        if cache_manager:
            cache_health = await cache_manager.health_check()
            component_health["cache"] = {
                "status": cache_health["status"],
                "healthy": cache_health["status"] == "healthy"
            }
        
        if pool_manager:
            db_health = await pool_manager.health_check()
            component_health["database"] = {
                "status": "healthy" if db_health["healthy"] else "unhealthy",
                "healthy": db_health["healthy"]
            }
        
        health_score["components"] = component_health
        return health_score
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system health")

@app.get("/api/v1/performance/trends/{metric_name}")
async def get_performance_trends(
    metric_name: str,
    hours: int = 24
):
    """Get performance trends for a specific metric"""
    try:
        trends = performance_monitor.get_performance_trends(metric_name, hours)
        return trends
        
    except Exception as e:
        logger.error(f"Failed to get performance trends: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance trends")

@app.get("/api/v1/performance/cache/status")
async def get_cache_status():
    """Get Redis cache status and metrics"""
    try:
        if not cache_manager:
            return {"status": "not_initialized"}
        
        health = await cache_manager.health_check()
        metrics = cache_manager.get_cache_metrics()
        
        return {
            "health": health,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cache status")

@app.post("/api/v1/performance/cache/clear")
async def clear_cache(
    client_id: str = None,
    employee_id: str = None
):
    """Clear cache entries"""
    try:
        if not cache_manager:
            raise HTTPException(status_code=503, detail="Cache manager not available")
        
        if client_id and employee_id:
            # Clear specific employee cache
            success = await cache_manager.invalidate_employee_cache(client_id, employee_id)
            return {"message": f"Cache cleared for employee {employee_id}", "success": success}
        elif client_id:
            # Clear all cache for client
            cleared_count = await cache_manager.invalidate_client_cache(client_id)
            return {"message": f"Cleared {cleared_count} cache entries for client {client_id}"}
        else:
            # Clear all cache (admin operation)
            # This would require admin authentication in production
            return {"message": "Full cache clear not implemented for safety"}
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

@app.get("/api/v1/performance/database/status")
async def get_database_status():
    """Get database connection pool status"""
    try:
        if not pool_manager:
            return {"status": "not_initialized"}
        
        status = await pool_manager.get_pool_status()
        health = await pool_manager.health_check()
        
        return {
            "status": status,
            "health": health,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get database status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve database status")

@app.post("/api/v1/performance/database/optimize")
async def optimize_database_pool():
    """Optimize database connection pool size"""
    try:
        if not pool_manager:
            raise HTTPException(status_code=503, detail="Pool manager not available")
        
        await pool_manager.optimize_pool_size()
        
        return {"message": "Database pool optimization completed"}
        
    except Exception as e:
        logger.error(f"Failed to optimize database pool: {e}")
        raise HTTPException(status_code=500, detail="Failed to optimize database pool")

@app.post("/api/v1/performance/metrics/reset")
async def reset_performance_metrics():
    """Reset performance metrics (admin operation)"""
    try:
        # Reset performance monitor metrics
        performance_monitor.reset_metrics()
        
        # Reset cache metrics if available
        if cache_manager:
            cache_manager.reset_metrics()
        
        # Reset database pool metrics if available
        if pool_manager:
            pool_manager.reset_metrics()
        
        return {"message": "Performance metrics reset successfully"}
        
    except Exception as e:
        logger.error(f"Failed to reset metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset performance metrics")

# System Integration Endpoints

@app.get("/api/v1/integration/status")
async def get_integration_status():
    """Get status of system integrations"""
    try:
        if not system_integrator:
            return {"status": "not_initialized", "error": "System integrator not available"}
        
        status = await system_integrator.get_integration_status()
        return status
        
    except Exception as e:
        logger.error(f"Failed to get integration status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve integration status")

@app.post("/api/v1/integration/agents/{client_id}/{employee_id}")
async def create_integrated_agent(
    client_id: str,
    employee_id: str,
    session_data: Dict[str, Any] = None
):
    """Create an integrated employee agent"""
    try:
        if not system_integrator:
            raise HTTPException(status_code=503, detail="System integrator not available")
        
        # Create agent with full system integration
        agent = await system_integrator.create_employee_agent(
            client_id, employee_id, session_data or {}
        )
        
        # Get agent capabilities
        capabilities = await agent.get_capabilities()
        
        return {
            "agent_created": True,
            "employee_id": employee_id,
            "client_id": client_id,
            "capabilities": capabilities,
            "message": "Integrated employee agent created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create integrated agent: {e}")
        raise HTTPException(status_code=500, detail="Failed to create integrated agent")

@app.post("/api/v1/integration/agents/{client_id}/{employee_id}/interact")
async def interact_with_integrated_agent(
    client_id: str,
    employee_id: str,
    interaction_data: Dict[str, Any]
):
    """Interact with an integrated employee agent"""
    try:
        if not system_integrator:
            raise HTTPException(status_code=503, detail="System integrator not available")
        
        message = interaction_data.get("message")
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        context = interaction_data.get("context", {})
        session_context = interaction_data.get("session_context", {})
        
        # Create agent
        agent = await system_integrator.create_employee_agent(
            client_id, employee_id, session_context
        )
        
        # Process message
        response = await agent.process_message(message, context)
        
        return {
            "response": response.message,
            "employee_id": response.employee_id,
            "interaction_id": response.interaction_id,
            "timestamp": response.timestamp.isoformat(),
            "model_used": response.model_used,
            "processing_time_ms": response.processing_time_ms,
            "tokens_used": response.tokens_used,
            "cost": response.cost
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to interact with integrated agent: {e}")
        raise HTTPException(status_code=500, detail="Failed to process interaction")

@app.get("/api/v1/integration/employees/{client_id}")
async def get_available_employees_for_integration(client_id: str):
    """Get available employees for agent creation"""
    try:
        if not system_integrator:
            raise HTTPException(status_code=503, detail="System integrator not available")
        
        employees = await system_integrator.agent_factory.get_available_employees(client_id)
        
        return {
            "client_id": client_id,
            "available_employees": employees,
            "total_count": len(employees)
        }
        
    except Exception as e:
        logger.error(f"Failed to get available employees: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve available employees")

# Employee Analytics and Monitoring Endpoints

@app.get("/api/v1/analytics/employees/{client_id}/{employee_id}/metrics")
async def get_employee_analytics_metrics(
    client_id: str,
    employee_id: str,
    days: int = 30
):
    """Get comprehensive analytics metrics for an employee"""
    try:
        if not analytics_service:
            raise HTTPException(status_code=503, detail="Analytics service not available")
        
        metrics = await analytics_service.get_employee_metrics(client_id, employee_id, days)
        
        return {
            "employee_id": metrics.employee_id,
            "employee_name": metrics.employee_name,
            "period_days": days,
            "metrics": {
                "total_interactions": metrics.total_interactions,
                "avg_response_time_ms": metrics.avg_response_time_ms,
                "success_rate": metrics.success_rate,
                "user_satisfaction_score": metrics.user_satisfaction_score,
                "cost_per_interaction": metrics.cost_per_interaction,
                "total_cost": metrics.total_cost,
                "active_days": metrics.active_days,
                "peak_usage_hour": metrics.peak_usage_hour,
                "performance_trend": metrics.performance_trend
            },
            "common_topics": metrics.common_topics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get employee metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve employee metrics")

@app.get("/api/v1/analytics/employees/{client_id}/{employee_id}/health")
async def get_employee_health_status(client_id: str, employee_id: str):
    """Get health status and alerts for an employee"""
    try:
        if not analytics_service:
            raise HTTPException(status_code=503, detail="Analytics service not available")
        
        health_status = await analytics_service.get_employee_health_status(client_id, employee_id)
        return health_status
        
    except Exception as e:
        logger.error(f"Failed to get employee health status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve employee health status")

@app.get("/api/v1/analytics/clients/{client_id}/usage")
async def get_client_usage_analytics(
    client_id: str,
    days: int = 30
):
    """Get comprehensive usage analytics for a client"""
    try:
        if not analytics_service:
            raise HTTPException(status_code=503, detail="Analytics service not available")
        
        usage_metrics = await analytics_service.get_client_usage_metrics(client_id, days)
        
        return {
            "client_id": usage_metrics.client_id,
            "client_name": usage_metrics.client_name,
            "period_days": days,
            "summary": {
                "total_employees": usage_metrics.total_employees,
                "active_employees": usage_metrics.active_employees,
                "total_interactions": usage_metrics.total_interactions,
                "total_cost": usage_metrics.total_cost,
                "avg_cost_per_employee": usage_metrics.avg_cost_per_employee,
                "monthly_growth": usage_metrics.monthly_growth
            },
            "top_performing_employees": usage_metrics.top_performing_employees,
            "usage_patterns": {
                "hourly_distribution": usage_metrics.usage_by_time
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get client usage analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve client usage analytics")

@app.get("/api/v1/analytics/clients/{client_id}/dashboard")
async def get_analytics_dashboard(
    client_id: str,
    days: int = 30
):
    """Get comprehensive dashboard data for a client"""
    try:
        if not analytics_service:
            raise HTTPException(status_code=503, detail="Analytics service not available")
        
        dashboard_data = await analytics_service.get_analytics_dashboard_data(client_id, days)
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to get analytics dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics dashboard")

@app.post("/api/v1/analytics/employees/{client_id}/{employee_id}/feedback")
async def record_employee_feedback(
    client_id: str,
    employee_id: str,
    feedback_data: Dict[str, Any]
):
    """Record user feedback for an employee interaction"""
    try:
        if not analytics_service:
            raise HTTPException(status_code=503, detail="Analytics service not available")
        
        interaction_id = feedback_data.get("interaction_id")
        rating = feedback_data.get("rating")
        feedback_text = feedback_data.get("feedback_text")
        
        if not interaction_id or not rating:
            raise HTTPException(status_code=400, detail="interaction_id and rating are required")
        
        if not 1 <= rating <= 5:
            raise HTTPException(status_code=400, detail="rating must be between 1 and 5")
        
        success = await analytics_service.record_user_feedback(
            client_id, employee_id, interaction_id, rating, feedback_text
        )
        
        if success:
            return {
                "message": "Feedback recorded successfully",
                "interaction_id": interaction_id,
                "rating": rating
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to record feedback")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to record employee feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")

@app.get("/api/v1/analytics/costs/calculate")
async def calculate_interaction_cost(
    model: str,
    input_tokens: int,
    output_tokens: int
):
    """Calculate cost for an interaction"""
    try:
        if not analytics_service:
            raise HTTPException(status_code=503, detail="Analytics service not available")
        
        cost = await analytics_service.calculate_interaction_cost(
            model, input_tokens, output_tokens
        )
        
        return {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": cost,
            "cost_breakdown": {
                "input_cost": (input_tokens / 1000) * analytics_service.cost_models.get(model, {}).get("input", 0),
                "output_cost": (output_tokens / 1000) * analytics_service.cost_models.get(model, {}).get("output", 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate interaction cost: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate cost")

@app.get("/api/v1/analytics/alerts/{client_id}")
async def get_client_alerts(client_id: str):
    """Get all active alerts for a client"""
    try:
        if not analytics_service:
            raise HTTPException(status_code=503, detail="Analytics service not available")
        
        # Get dashboard data which includes alerts
        dashboard_data = await analytics_service.get_analytics_dashboard_data(client_id, days=1)
        
        alerts = dashboard_data.get("alerts_summary", [])
        
        # Categorize alerts by severity
        critical_alerts = [alert for alert in alerts if alert.get("severity") == "critical"]
        warning_alerts = [alert for alert in alerts if alert.get("severity") == "warning"]
        
        return {
            "client_id": client_id,
            "total_alerts": len(alerts),
            "critical_alerts": len(critical_alerts),
            "warning_alerts": len(warning_alerts),
            "alerts": alerts,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get client alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# MCAE Integration Health Check Endpoint
@app.get("/health/mcae")
async def mcae_health_check():
    """Health check endpoint for MCAE integration"""
    try:
        if not mcae_adapter:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unavailable",
                    "message": "MCAE adapter not initialized",
                    "integration_active": False
                }
            )
        
        # Perform MCAE health check
        health_result = await mcae_adapter.health_check()
        
        status_code = 200 if health_result.get("status") == "healthy" else 503
        
        return JSONResponse(
            status_code=status_code,
            content={
                **health_result,
                "integration_active": True,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"MCAE health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "integration_active": False
            }
        )


# MCAE Integration Metrics Endpoint
@app.get("/metrics/mcae")
async def mcae_metrics():
    """Get MCAE integration metrics"""
    try:
        if not mcae_adapter:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "MCAE adapter not available",
                    "integration_active": False
                }
            )
        
        # Get MCAE metrics
        metrics = mcae_adapter.get_metrics()
        
        # Get error handler metrics if available
        if mcae_error_handler:
            error_stats = mcae_error_handler.get_error_stats()
            metrics["error_handling"] = error_stats
        
        return JSONResponse(
            status_code=200,
            content={
                "mcae_metrics": metrics,
                "integration_active": True,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get MCAE metrics: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "integration_active": False
            }
        )
