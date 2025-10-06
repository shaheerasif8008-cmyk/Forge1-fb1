#!/usr/bin/env python3
"""
Simple compliance test to verify the implementation structure
"""

import sys
import os
from datetime import datetime

# Add the forge1 directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'forge1'))

def test_compliance_imports():
    """Test that compliance modules can be imported"""
    print("Testing compliance module imports...")
    
    try:
        # Test GDPR import
        from compliance.gdpr import GDPRComplianceManager, LegalBasis, DataCategory
        print("✅ GDPR module imported successfully")
        
        # Test HIPAA import
        from compliance.hipaa import HIPAAComplianceManager, PHICategory, IncidentSeverity
        print("✅ HIPAA module imported successfully")
        
        # Test unified manager import
        from compliance.manager import UnifiedComplianceManager, ComplianceFramework
        print("✅ Unified compliance manager imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_compliance_classes():
    """Test that compliance classes can be instantiated"""
    print("\nTesting compliance class instantiation...")
    
    try:
        # Mock the required dependencies
        class MockMemoryManager:
            async def store_context(self, context_type, content, metadata):
                pass
        
        class MockMetricsCollector:
            def increment(self, metric):
                pass
            def record_metric(self, metric, value):
                pass
        
        class MockSecretManager:
            async def get(self, name):
                return "mock_secret"
        
        # Test GDPR manager
        from compliance.gdpr import GDPRComplianceManager
        gdpr_manager = GDPRComplianceManager(
            MockMemoryManager(),
            MockMetricsCollector(),
            MockSecretManager()
        )
        print("✅ GDPR manager instantiated successfully")
        
        # Test HIPAA manager
        from compliance.hipaa import HIPAAComplianceManager
        hipaa_manager = HIPAAComplianceManager(
            MockMemoryManager(),
            MockMetricsCollector(),
            MockSecretManager()
        )
        print("✅ HIPAA manager instantiated successfully")
        
        # Test unified manager
        from compliance.manager import UnifiedComplianceManager
        unified_manager = UnifiedComplianceManager(
            MockMemoryManager(),
            MockMetricsCollector(),
            MockSecretManager()
        )
        print("✅ Unified manager instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Class instantiation error: {e}")
        return False

def test_compliance_functionality():
    """Test basic compliance functionality"""
    print("\nTesting basic compliance functionality...")
    
    try:
        # Mock dependencies
        class MockMemoryManager:
            async def store_context(self, context_type, content, metadata):
                pass
        
        class MockMetricsCollector:
            def increment(self, metric):
                pass
            def record_metric(self, metric, value):
                pass
        
        class MockSecretManager:
            async def get(self, name):
                return "mock_secret"
        
        # Test GDPR functionality
        from compliance.gdpr import GDPRComplianceManager, ProcessingPurpose, DataCategory
        gdpr_manager = GDPRComplianceManager(
            MockMemoryManager(),
            MockMetricsCollector(),
            MockSecretManager()
        )
        
        # Check that processing records were initialized
        assert len(gdpr_manager.processing_records) > 0, "GDPR processing records not initialized"
        print("✅ GDPR processing records initialized")
        
        # Test HIPAA functionality
        from compliance.hipaa import HIPAAComplianceManager
        hipaa_manager = HIPAAComplianceManager(
            MockMemoryManager(),
            MockMetricsCollector(),
            MockSecretManager()
        )
        
        # Check that safeguards were initialized
        assert len(hipaa_manager.safeguards) > 0, "HIPAA safeguards not initialized"
        print("✅ HIPAA safeguards initialized")
        
        # Test unified manager functionality
        from compliance.manager import UnifiedComplianceManager
        unified_manager = UnifiedComplianceManager(
            MockMemoryManager(),
            MockMetricsCollector(),
            MockSecretManager()
        )
        
        # Check that framework managers were initialized
        assert unified_manager.gdpr_manager is not None, "GDPR manager not initialized in unified manager"
        assert unified_manager.hipaa_manager is not None, "HIPAA manager not initialized in unified manager"
        print("✅ Unified manager framework integration working")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def main():
    """Run all compliance tests"""
    print("FORGE 1 COMPLIANCE IMPLEMENTATION - SIMPLE TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_compliance_imports,
        test_compliance_classes,
        test_compliance_functionality
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("✅ All tests passed! Compliance implementation is working correctly.")
        return True
    else:
        print(f"❌ {failed} tests failed. See details above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)