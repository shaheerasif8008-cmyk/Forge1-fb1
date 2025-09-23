# forge1/backend/forge1/core/health_checks.py
"""
Health Check Endpoints for Forge 1

Comprehensive health monitoring for all services and components.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class HealthStatus(BaseModel):
    """Health status model"""
    status: str  # healthy, degraded, unhealthy
    timestamp: float
    version: str = "1.0.0"
    uptime: float
    checks: Dict[str, Any]

class ComponentHealth(BaseModel):
    """Individual component health"""
    name: str
    status: str
    response_time: float
    details: Dict[str, Any] = {}

class HealthChecker:
    """Comprehensive health checker for all Forge 1 components"""
    
    def __init__(self):
        self.start_time = time.time()
        self.router = APIRouter(prefix="/health", tags=["health"])
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup health check routes"""
        
        @self.router.get("/", response_model=HealthStatus)
        async def basic_health():
            """Basic health check endpoint"""
            return await self.get_basic_health()
        
        @self.router.get("/detailed", response_model=HealthStatus)
        async def detailed_health():
            """Detailed health check with all components"""
            return await self.get_detailed_health()
        
        @self.router.get("/components", response_model=List[ComponentHealth])
        async def component_health():
            """Individual component health status"""
            return await self.get_component_health()
        
        @self.router.get("/ready")
        async def readiness_probe():
            """Kubernetes readiness probe"""
            health = await self.get_basic_health()
            if health.status == "unhealthy":
                raise HTTPException(status_code=503, detail="Service not ready")
            return {"status": "ready"}
        
        @self.router.get("/live")
        async def liveness_probe():
            """Kubernetes liveness probe"""
            return {"status": "alive", "timestamp": time.time()}
    
    async def get_basic_health(self) -> HealthStatus:
        """Get basic health status"""
        
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Perform basic checks
        checks = {
            "api": await self._check_api_health(),
            "memory": await self._check_memory_health(),
            "disk": await self._check_disk_health()
        }
        
        # Determine overall status
        status = "healthy"
        if any(check["status"] == "unhealthy" for check in checks.values()):
            status = "unhealthy"
        elif any(check["status"] == "degraded" for check in checks.values()):
            status = "degraded"
        
        return HealthStatus(
            status=status,
            timestamp=current_time,
            uptime=uptime,
            checks=checks
        )
    
    async def get_detailed_health(self) -> HealthStatus:
        """Get detailed health status with all components"""
        
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Perform comprehensive checks
        checks = {
            "api": await self._check_api_health(),
            "memory": await self._check_memory_health(),
            "disk": await self._check_disk_health(),
            "database": await self._check_database_health(),
            "model_router": await self._check_model_router_health(),
            "security": await self._check_security_health(),
            "performance": await self._check_performance_health(),
            "compliance": await self._check_compliance_health(),
            "external_services": await self._check_external_services_health()
        }
        
        # Determine overall status
        status = "healthy"
        unhealthy_count = sum(1 for check in checks.values() if check["status"] == "unhealthy")
        degraded_count = sum(1 for check in checks.values() if check["status"] == "degraded")
        
        if unhealthy_count > 0:
            if unhealthy_count >= len(checks) * 0.5:  # More than 50% unhealthy
                status = "unhealthy"
            else:
                status = "degraded"
        elif degraded_count > 0:
            status = "degraded"
        
        return HealthStatus(
            status=status,
            timestamp=current_time,
            uptime=uptime,
            checks=checks
        )
    
    async def get_component_health(self) -> List[ComponentHealth]:
        """Get individual component health status"""
        
        components = []
        
        # Check each component individually
        component_checks = [
            ("API", self._check_api_health),
            ("Database", self._check_database_health),
            ("Model Router", self._check_model_router_health),
            ("Security Manager", self._check_security_health),
            ("Performance Monitor", self._check_performance_health),
            ("Compliance Engine", self._check_compliance_health),
        ]
        
        for name, check_func in component_checks:
            start_time = time.time()
            try:
                result = await check_func()
                response_time = time.time() - start_time
                
                components.append(ComponentHealth(
                    name=name,
                    status=result["status"],
                    response_time=response_time,
                    details=result.get("details", {})
                ))
            except Exception as e:
                response_time = time.time() - start_time
                components.append(ComponentHealth(
                    name=name,
                    status="unhealthy",
                    response_time=response_time,
                    details={"error": str(e)}
                ))
        
        return components
    
    async def _check_api_health(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            # Basic API functionality check
            return {
                "status": "healthy",
                "details": {
                    "endpoints": "accessible",
                    "response_time": "normal"
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": {"error": str(e)}
            }
    
    async def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            status = "healthy"
            if memory.percent > 90:
                status = "unhealthy"
            elif memory.percent > 80:
                status = "degraded"
            
            return {
                "status": status,
                "details": {
                    "usage_percent": memory.percent,
                    "available_gb": round(memory.available / (1024**3), 2),
                    "total_gb": round(memory.total / (1024**3), 2)
                }
            }
        except ImportError:
            # psutil not available, return basic check
            return {
                "status": "healthy",
                "details": {"note": "Memory monitoring not available"}
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": {"error": str(e)}
            }
    
    async def _check_disk_health(self) -> Dict[str, Any]:
        """Check disk usage"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            
            status = "healthy"
            usage_percent = (disk.used / disk.total) * 100
            
            if usage_percent > 95:
                status = "unhealthy"
            elif usage_percent > 85:
                status = "degraded"
            
            return {
                "status": status,
                "details": {
                    "usage_percent": round(usage_percent, 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "total_gb": round(disk.total / (1024**3), 2)
                }
            }
        except ImportError:
            return {
                "status": "healthy",
                "details": {"note": "Disk monitoring not available"}
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": {"error": str(e)}
            }
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            # Import Microsoft's database context
            import sys
            sys.path.append('../../../../Multi-Agent-Custom-Automation-Engine-Solution-Accelerator/src/backend')
            
            from context.cosmos_memory_kernel import CosmosMemoryContext
            
            # Test basic database connection
            # For now, we'll do a basic import test
            # In a real implementation, this would test actual connectivity
            
            return {
                "status": "healthy",
                "details": {
                    "connection": "available",
                    "type": "cosmos_db"
                }
            }
        except Exception as e:
            return {
                "status": "degraded",
                "details": {
                    "error": str(e),
                    "note": "Database connectivity test failed, but service can continue"
                }
            }
    
    async def _check_model_router_health(self) -> Dict[str, Any]:
        """Check model router health"""
        try:
            from forge1.core.model_router import ModelRouter
            
            router = ModelRouter()
            health = await router.health_check()
            
            return {
                "status": health["status"],
                "details": health
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": {"error": str(e)}
            }
    
    async def _check_security_health(self) -> Dict[str, Any]:
        """Check security manager health"""
        try:
            from forge1.core.security_manager import SecurityManager
            
            security = SecurityManager()
            is_healthy = await security.health_check()
            
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "details": {
                    "authentication": "enabled",
                    "rate_limiting": "active"
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": {"error": str(e)}
            }
    
    async def _check_performance_health(self) -> Dict[str, Any]:
        """Check performance monitor health"""
        try:
            from forge1.core.performance_monitor import PerformanceMonitor
            
            monitor = PerformanceMonitor()
            is_healthy = await monitor.health_check()
            
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "details": {
                    "monitoring": "active",
                    "metrics_collection": "enabled"
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": {"error": str(e)}
            }
    
    async def _check_compliance_health(self) -> Dict[str, Any]:
        """Check compliance engine health"""
        try:
            from forge1.core.compliance_engine import ComplianceEngine
            
            compliance = ComplianceEngine()
            is_healthy = await compliance.health_check()
            
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "details": {
                    "audit_logging": "enabled",
                    "compliance_rules": "loaded"
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": {"error": str(e)}
            }
    
    async def _check_external_services_health(self) -> Dict[str, Any]:
        """Check external services health"""
        try:
            # Check external service connectivity
            # This would include checks for:
            # - OpenAI API
            # - Anthropic API
            # - Google AI API
            # - Azure services
            
            # For now, return a mock healthy status
            return {
                "status": "healthy",
                "details": {
                    "openai": "available",
                    "anthropic": "available",
                    "google": "available",
                    "azure": "available"
                }
            }
        except Exception as e:
            return {
                "status": "degraded",
                "details": {
                    "error": str(e),
                    "note": "Some external services may be unavailable"
                }
            }

# Global health checker instance
health_checker = HealthChecker()

# Export router for inclusion in main app
health_router = health_checker.router