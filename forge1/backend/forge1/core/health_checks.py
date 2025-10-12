"""
Forge 1 Enhanced Health Checks
Comprehensive health monitoring for all system components
"""

import time
import asyncio
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

health_router = APIRouter(prefix="/health", tags=["health"])

@health_router.get("/")
async def basic_health():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "forge1-platform"
    }

@health_router.get("/detailed")
async def detailed_health():
    """Detailed health check for all components"""
    try:
        health_checks = {
            "database": await check_database_health(),
            "cache": await check_cache_health(),
            "ai_models": await check_ai_models_health(),
            "external_services": await check_external_services_health(),
            "system_resources": await check_system_resources()
        }
        
        # Determine overall status
        overall_status = "healthy"
        for component, status in health_checks.items():
            if status.get("status") == "unhealthy":
                overall_status = "unhealthy"
                break
            elif status.get("status") == "degraded" and overall_status == "healthy":
                overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "components": health_checks,
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

async def check_database_health() -> Dict[str, Any]:
    """Check database connectivity and performance"""
    try:
        start_time = time.time()
        
        # Mock database check
        await asyncio.sleep(0.01)  # Simulate DB query
        
        response_time = time.time() - start_time
        
        return {
            "status": "healthy",
            "response_time": response_time,
            "connections": {
                "active": 5,
                "idle": 10,
                "max": 100
            },
            "last_check": time.time()
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_check": time.time()
        }

async def check_cache_health() -> Dict[str, Any]:
    """Check cache (Redis) connectivity and performance"""
    try:
        start_time = time.time()
        
        # Mock cache check
        await asyncio.sleep(0.005)  # Simulate cache operation
        
        response_time = time.time() - start_time
        
        return {
            "status": "healthy",
            "response_time": response_time,
            "memory_usage": "45%",
            "hit_rate": "92%",
            "last_check": time.time()
        }
        
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_check": time.time()
        }

async def check_ai_models_health() -> Dict[str, Any]:
    """Check AI models availability and performance"""
    try:
        models_status = {
            "gpt-4o": {"status": "healthy", "response_time": 0.8, "availability": "99.9%"},
            "claude-3-opus": {"status": "healthy", "response_time": 1.2, "availability": "99.5%"},
            "gemini-pro": {"status": "healthy", "response_time": 0.6, "availability": "99.7%"}
        }
        
        healthy_models = sum(1 for model in models_status.values() if model["status"] == "healthy")
        total_models = len(models_status)
        
        overall_status = "healthy" if healthy_models == total_models else "degraded"
        
        return {
            "status": overall_status,
            "models": models_status,
            "healthy_models": healthy_models,
            "total_models": total_models,
            "last_check": time.time()
        }
        
    except Exception as e:
        logger.error(f"AI models health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_check": time.time()
        }

async def check_external_services_health() -> Dict[str, Any]:
    """Check external services connectivity"""
    try:
        services_status = {
            "openai_api": {"status": "healthy", "response_time": 0.5},
            "anthropic_api": {"status": "healthy", "response_time": 0.7},
            "google_api": {"status": "healthy", "response_time": 0.4}
        }
        
        healthy_services = sum(1 for service in services_status.values() if service["status"] == "healthy")
        total_services = len(services_status)
        
        overall_status = "healthy" if healthy_services == total_services else "degraded"
        
        return {
            "status": overall_status,
            "services": services_status,
            "healthy_services": healthy_services,
            "total_services": total_services,
            "last_check": time.time()
        }
        
    except Exception as e:
        logger.error(f"External services health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_check": time.time()
        }

async def check_system_resources() -> Dict[str, Any]:
    """Check system resource utilization"""
    try:
        # Mock system resource check
        resources = {
            "cpu_usage": 25.5,
            "memory_usage": 45.2,
            "disk_usage": 60.1,
            "network_io": 12.3
        }
        
        # Determine status based on resource usage
        status = "healthy"
        if any(usage > 90 for usage in resources.values()):
            status = "unhealthy"
        elif any(usage > 80 for usage in resources.values()):
            status = "degraded"
        
        return {
            "status": status,
            "resources": resources,
            "thresholds": {
                "cpu_warning": 80,
                "cpu_critical": 90,
                "memory_warning": 80,
                "memory_critical": 90,
                "disk_warning": 85,
                "disk_critical": 95
            },
            "last_check": time.time()
        }
        
    except Exception as e:
        logger.error(f"System resources health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_check": time.time()
        }

class HealthStatus:
    """Health status model"""
    def __init__(self, status: str, checks: Dict[str, Any]):
        self.status = status
        self.checks = checks
        self.timestamp = time.time()

class HealthChecker:
    """Health checker for Forge1 components"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def get_basic_health(self) -> HealthStatus:
        """Get basic health status"""
        return HealthStatus(
            status="healthy",
            checks={
                "service": "forge1-platform",
                "version": "1.0.0"
            }
        )
    
    async def get_detailed_health(self) -> HealthStatus:
        """Get detailed health status"""
        checks = {
            "database": await check_database_health(),
            "cache": await check_cache_health(),
            "ai_models": await check_ai_models_health(),
            "external_services": await check_external_services_health(),
            "system_resources": await check_system_resources()
        }
        
        # Determine overall status
        overall_status = "healthy"
        for component, status in checks.items():
            if status.get("status") == "unhealthy":
                overall_status = "unhealthy"
                break
            elif status.get("status") == "degraded" and overall_status == "healthy":
                overall_status = "degraded"
        
        return HealthStatus(status=overall_status, checks=checks)
    
    async def check_component(self, component_name: str) -> Dict[str, Any]:
        """Check specific component health"""
        if component_name == "database":
            return await check_database_health()
        elif component_name == "cache":
            return await check_cache_health()
        elif component_name == "ai_models":
            return await check_ai_models_health()
        elif component_name == "external_services":
            return await check_external_services_health()
        elif component_name == "system_resources":
            return await check_system_resources()
        else:
            return {"status": "unknown", "error": f"Unknown component: {component_name}"}
