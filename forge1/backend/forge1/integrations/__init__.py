"""
Forge1 OSS Systems Integration

Main integration module that initializes and manages all OSS system integrations
including Celery, Redis, Weaviate, MemGPT, OpenTelemetry, Prometheus, Grafana,
OpenMeter, and OPA with comprehensive tenant isolation and compliance.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

# Import all adapters
from forge1.integrations.queue.celery_app import celery_adapter
from forge1.integrations.queue.redis_client import redis_adapter
from forge1.integrations.vector.weaviate_client import weaviate_adapter
from forge1.integrations.vector.vector_store_weaviate import weaviate_memory_adapter
from forge1.integrations.observability.otel_init import otel_integration
from forge1.integrations.observability.metrics import forge_metrics
from forge1.integrations.observability.azure_monitor_adapter import azure_monitor_adapter
from forge1.integrations.metering.openmeter_client import openmeter_adapter
from forge1.policy.opa_client import opa_adapter
from forge1.middleware.policy_enforcer import policy_enforcement_middleware
from forge1.services.azure_monitor_service import azure_monitor_business_service

logger = logging.getLogger(__name__)

class IntegrationManager:
    """Manages all OSS system integrations"""
    
    def __init__(self):
        self.adapters = {
            "celery": celery_adapter,
            "redis": redis_adapter,
            "weaviate": weaviate_adapter,
            "weaviate_memory": weaviate_memory_adapter,
            "otel": otel_integration,
            "azure_monitor": azure_monitor_adapter,
            "openmeter": openmeter_adapter,
            "opa": opa_adapter
        }
        
        self.initialized = False
        self.initialization_order = [
            "redis",
            "celery", 
            "weaviate",
            "weaviate_memory",
            "otel",
            "azure_monitor",
            "openmeter",
            "opa"
        ]
    
    async def initialize_all(self) -> bool:
        """Initialize all integrations in proper order"""
        
        if self.initialized:
            return True
        
        logger.info("Initializing OSS system integrations...")
        
        success_count = 0
        failed_adapters = []
        
        for adapter_name in self.initialization_order:
            adapter = self.adapters[adapter_name]
            
            try:
                logger.info(f"Initializing {adapter_name} adapter...")
                success = await adapter.initialize()
                
                if success:
                    success_count += 1
                    logger.info(f"✓ {adapter_name} adapter initialized successfully")
                else:
                    failed_adapters.append(adapter_name)
                    logger.error(f"✗ {adapter_name} adapter initialization failed")
                    
            except Exception as e:
                failed_adapters.append(adapter_name)
                logger.error(f"✗ {adapter_name} adapter initialization error: {e}")
        
        # Check if critical adapters failed
        critical_adapters = ["redis", "otel", "opa"]
        critical_failures = [name for name in failed_adapters if name in critical_adapters]
        
        if critical_failures:
            logger.error(f"Critical adapter failures: {critical_failures}")
            self.initialized = False
            return False
        
        self.initialized = True
        logger.info(f"OSS integrations initialized: {success_count}/{len(self.adapters)} successful")
        
        if failed_adapters:
            logger.warning(f"Non-critical adapter failures: {failed_adapters}")
        
        return True
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Perform health check on all integrations"""
        
        health_results = {}
        overall_healthy = True
        
        for name, adapter in self.adapters.items():
            try:
                health_result = await adapter.health_check()
                health_results[name] = {
                    "status": health_result.status.value,
                    "message": health_result.message,
                    "response_time_ms": health_result.response_time_ms,
                    "details": health_result.details
                }
                
                if health_result.status.value == "unhealthy":
                    overall_healthy = False
                    
            except Exception as e:
                health_results[name] = {
                    "status": "error",
                    "message": f"Health check failed: {str(e)}",
                    "response_time_ms": 0,
                    "details": {"error": str(e)}
                }
                overall_healthy = False
        
        return {
            "overall_status": "healthy" if overall_healthy else "unhealthy",
            "integrations": health_results,
            "initialized": self.initialized
        }
    
    async def cleanup_all(self) -> bool:
        """Clean up all integrations"""
        
        logger.info("Cleaning up OSS system integrations...")
        
        cleanup_success = True
        
        # Cleanup in reverse order
        for adapter_name in reversed(self.initialization_order):
            adapter = self.adapters[adapter_name]
            
            try:
                success = await adapter.cleanup()
                if success:
                    logger.info(f"✓ {adapter_name} adapter cleaned up successfully")
                else:
                    logger.warning(f"⚠ {adapter_name} adapter cleanup failed")
                    cleanup_success = False
                    
            except Exception as e:
                logger.error(f"✗ {adapter_name} adapter cleanup error: {e}")
                cleanup_success = False
        
        self.initialized = False
        return cleanup_success
    
    def get_adapter(self, name: str):
        """Get a specific adapter"""
        return self.adapters.get(name)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from all adapters"""
        
        metrics = {}
        
        for name, adapter in self.adapters.items():
            try:
                if hasattr(adapter, 'get_metrics'):
                    metrics[name] = adapter.get_metrics()
                elif hasattr(adapter, 'get_statistics'):
                    metrics[name] = adapter.get_statistics()
                else:
                    metrics[name] = {"status": "no_metrics_available"}
                    
            except Exception as e:
                metrics[name] = {"error": str(e)}
        
        return metrics

# Global integration manager
integration_manager = IntegrationManager()

# Convenience functions
async def initialize_integrations() -> bool:
    """Initialize all OSS system integrations"""
    return await integration_manager.initialize_all()

async def health_check_integrations() -> Dict[str, Any]:
    """Health check all integrations"""
    return await integration_manager.health_check_all()

async def cleanup_integrations() -> bool:
    """Clean up all integrations"""
    return await integration_manager.cleanup_all()

def get_integration_adapter(name: str):
    """Get a specific integration adapter"""
    return integration_manager.get_adapter(name)

def get_integration_metrics() -> Dict[str, Any]:
    """Get metrics from all integrations"""
    return integration_manager.get_metrics()

# Export key components
__all__ = [
    "integration_manager",
    "initialize_integrations",
    "health_check_integrations", 
    "cleanup_integrations",
    "get_integration_adapter",
    "get_integration_metrics",
    "celery_adapter",
    "redis_adapter",
    "weaviate_adapter",
    "weaviate_memory_adapter",
    "otel_integration",
    "forge_metrics",
    "azure_monitor_adapter",
    "azure_monitor_business_service",
    "openmeter_adapter",
    "opa_adapter",
    "policy_enforcement_middleware"
]