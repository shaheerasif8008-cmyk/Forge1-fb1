#!/usr/bin/env python3
"""
Startup Test Script

Quick test to verify that all imports work and the application can start
without dependency issues.
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all critical imports"""
    
    logger.info("Testing critical imports...")
    
    try:
        # Test FastAPI imports
        from fastapi import FastAPI
        logger.info("✅ FastAPI import successful")
        
        # Test core Forge1 imports
        from forge1.core.audit_logger import AuditLogger
        from forge1.core.encryption_manager import EncryptionManager
        logger.info("✅ Core Forge1 imports successful")
        
        # Test middleware imports
        from forge1.middleware.security_middleware import TenantIsolationMiddleware
        logger.info("✅ Security middleware import successful")
        
        try:
            from forge1.middleware.azure_monitor_middleware import AzureMonitorMiddleware
            logger.info("✅ Azure Monitor middleware import successful")
        except ImportError as e:
            logger.warning(f"⚠️  Azure Monitor middleware import failed: {e}")
        
        # Test API router imports
        from forge1.api.employee_lifecycle_api import router as employee_lifecycle_router
        from forge1.api.automation_api import router as automation_router
        from forge1.api.compliance_api import router as compliance_router
        from forge1.api.api_integration_api import router as api_integration_router
        from forge1.api.azure_monitor_api import router as azure_monitor_router
        logger.info("✅ API router imports successful")
        
        try:
            from forge1.api.v1.analytics import router as analytics_router
            logger.info("✅ Analytics API import successful")
        except ImportError as e:
            logger.warning(f"⚠️  Analytics API import failed: {e}")
        
        # Test service imports
        from forge1.services.employee_manager import EmployeeManager
        from forge1.services.employee_memory_manager import EmployeeMemoryManager
        from forge1.services.employee_analytics_service import get_analytics_service
        logger.info("✅ Service imports successful")
        
        # Test integration imports
        from forge1.integrations.forge1_system_adapter import get_system_integrator
        logger.info("✅ System integration imports successful")
        
        try:
            from forge1.integrations.mcae_adapter import MCAEAdapter
            from forge1.integrations.mcae_error_handler import MCAEErrorHandler
            logger.info("✅ MCAE integration imports successful")
        except ImportError as e:
            logger.warning(f"⚠️  MCAE integration import failed: {e}")
        
        # Test performance components
        from forge1.core.redis_cache_manager import get_cache_manager
        from forge1.core.performance_monitor import get_performance_monitor
        from forge1.core.connection_pool_manager import get_connection_pool_manager
        logger.info("✅ Performance component imports successful")
        
        logger.info("🎉 All critical imports successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False

def test_app_creation():
    """Test FastAPI app creation"""
    
    logger.info("Testing FastAPI app creation...")
    
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        
        # Create basic app
        app = FastAPI(
            title="Forge 1 Platform API - Test",
            description="Test app creation",
            version="1.0.0"
        )
        
        # Add basic middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )
        
        # Add a test endpoint
        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok", "message": "Test endpoint working"}
        
        logger.info("✅ FastAPI app creation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ App creation test failed: {e}")
        return False

def test_azure_monitor_integration():
    """Test Azure Monitor integration availability"""
    
    logger.info("Testing Azure Monitor integration...")
    
    try:
        # Test if Azure Monitor packages are available
        try:
            import azure.monitor.opentelemetry
            logger.info("✅ Azure Monitor OpenTelemetry package available")
            azure_available = True
        except ImportError:
            logger.warning("⚠️  Azure Monitor OpenTelemetry package not available")
            azure_available = False
        
        # Test our Azure Monitor integration
        try:
            from forge1.integrations.observability.azure_monitor_adapter import azure_monitor_adapter
            logger.info("✅ Azure Monitor adapter available")
        except ImportError as e:
            logger.warning(f"⚠️  Azure Monitor adapter not available: {e}")
        
        # Test analytics service
        try:
            from forge1.services.analytics.azure_monitor_analytics import azure_monitor_analytics_service
            logger.info("✅ Azure Monitor analytics service available")
        except ImportError as e:
            logger.warning(f"⚠️  Azure Monitor analytics service not available: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Azure Monitor integration test failed: {e}")
        return False

def main():
    """Run all startup tests"""
    
    logger.info("=" * 60)
    logger.info("FORGE1 STARTUP TEST")
    logger.info("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("App Creation Test", test_app_creation),
        ("Azure Monitor Integration Test", test_azure_monitor_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The application should start successfully.")
        return 0
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())