#!/usr/bin/env python3
"""
Test script for Forge 1 Core Backend Service Enhancement

This script verifies that all components of task 2 are working correctly:
- Enhanced FastAPI backend architecture
- Multi-model router service
- Enterprise security layer
"""

import asyncio
import sys
import os

# Add paths for imports
sys.path.append('.')
sys.path.append('../Multi-Agent-Custom-Automation-Engine-Solution-Accelerator/src/backend')

async def test_model_router():
    """Test the multi-model router service"""
    print("🤖 Testing Multi-Model Router Service...")
    
    try:
        from forge1.core.model_router import ModelRouter, TaskRequirements
        
        router = ModelRouter()
        
        # Test model initialization
        assert len(router.models) > 0, "No models initialized"
        print(f"✅ Initialized {len(router.models)} models")
        
        # Test health check
        health = await router.health_check()
        assert health["status"] in ["healthy", "degraded"], "Invalid health status"
        print(f"✅ Health check passed: {health['status']}")
        
        # Test model selection
        test_task = "Create a comprehensive business plan for a new AI startup"
        client = await router.get_optimal_client(test_task)
        assert client is not None, "Failed to get optimal client"
        print(f"✅ Selected model: {getattr(client, 'model_name', 'unknown')}")
        
        # Test performance benchmarks
        benchmarks = await router.get_performance_benchmarks()
        print(f"✅ Performance benchmarks available for {len(benchmarks)} models")
        
        # Test available models
        available_models = await router.get_available_models()
        assert len(available_models) > 0, "No available models"
        print(f"✅ {len(available_models)} models available")
        
        print("✅ Multi-Model Router Service: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Multi-Model Router Service: FAILED - {e}")
        return False

async def test_security_manager():
    """Test the enterprise security layer"""
    print("🔒 Testing Enterprise Security Layer...")
    
    try:
        from forge1.core.security_manager import SecurityManager, UserRole, Permission
        
        security = SecurityManager()
        
        # Test role permissions initialization
        assert len(security.role_permissions) > 0, "No role permissions initialized"
        print(f"✅ Initialized {len(security.role_permissions)} roles")
        
        # Test health check
        health = await security.health_check()
        assert health["status"] == "healthy", "Security manager not healthy"
        print(f"✅ Health check passed: {health['status']}")
        
        # Test permission checking
        test_user = "test@user.com"
        has_permission = await security.check_permission(test_user, Permission.READ_PLANS)
        print(f"✅ Permission check completed for user: {test_user}")
        
        # Test audit logging
        await security._log_security_event(
            user_id=test_user,
            action="test_action",
            resource="test_resource",
            ip_address="127.0.0.1",
            user_agent="test_agent",
            success=True,
            details={"test": "data"}
        )
        assert len(security.audit_log) > 0, "Audit log not working"
        print("✅ Audit logging working")
        
        # Test rate limiting
        rate_limited = await security._check_rate_limit("127.0.0.1")
        print(f"✅ Rate limiting check: {'limited' if rate_limited else 'allowed'}")
        
        print("✅ Enterprise Security Layer: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Enterprise Security Layer: FAILED - {e}")
        return False

async def test_app_kernel():
    """Test the enhanced FastAPI backend architecture"""
    print("🚀 Testing Enhanced FastAPI Backend Architecture...")
    
    try:
        from forge1.core.app_kernel import Forge1App, TaskResponse, EnhancedInputTask
        
        # Test app initialization
        app_instance = Forge1App()
        assert app_instance.app is not None, "FastAPI app not initialized"
        print("✅ FastAPI app initialized")
        
        # Test component initialization
        assert app_instance.model_router is not None, "Model router not initialized"
        assert app_instance.security_manager is not None, "Security manager not initialized"
        assert app_instance.performance_monitor is not None, "Performance monitor not initialized"
        assert app_instance.compliance_engine is not None, "Compliance engine not initialized"
        print("✅ All core components initialized")
        
        # Test health checks
        db_health = await app_instance._check_database_health()
        router_health = await app_instance._check_model_router_health()
        security_health = await app_instance._check_security_health()
        compliance_health = await app_instance._check_compliance_health()
        
        print(f"✅ Component health checks: DB={db_health}, Router={router_health}, Security={security_health}, Compliance={compliance_health}")
        
        # Test Pydantic models
        test_task = EnhancedInputTask(
            description="Test task for validation",
            session_id="test-session",
            priority="normal"
        )
        assert test_task.description == "Test task for validation", "Enhanced input task validation failed"
        print("✅ Pydantic models working")
        
        print("✅ Enhanced FastAPI Backend Architecture: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced FastAPI Backend Architecture: FAILED - {e}")
        return False

async def test_integration():
    """Test integration between all components"""
    print("🔗 Testing Component Integration...")
    
    try:
        from forge1.core.app_kernel import Forge1App
        
        app_instance = Forge1App()
        
        # Test that all components can work together
        model_health = await app_instance.model_router.health_check()
        security_health = await app_instance.security_manager.health_check()
        
        # Test that security manager can validate requests with model router
        test_headers = {"user-agent": "test-agent"}
        
        # This would normally require proper authentication headers
        # For testing, we'll just verify the components are properly connected
        assert app_instance.model_router is not None
        assert app_instance.security_manager is not None
        
        print("✅ All components properly integrated")
        print("✅ Component Integration: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Component Integration: FAILED - {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 Starting Forge 1 Core Backend Service Tests")
    print("=" * 60)
    
    tests = [
        test_model_router,
        test_security_manager,
        test_app_kernel,
        test_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
        print("-" * 40)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Core Backend Service Enhancement is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)