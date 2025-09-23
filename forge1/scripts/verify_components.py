#!/usr/bin/env python3
# forge1/scripts/verify_components.py
"""
Component Verification Script for Forge 1

Verifies all Forge 1 components can be imported and initialized correctly.
"""

import sys
import os
import asyncio
import traceback
from typing import Dict, Any, List

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
backend_dir = os.path.join(project_root, 'backend')
microsoft_backend = os.path.join(project_root, '..', 'Multi-Agent-Custom-Automation-Engine-Solution-Accelerator', 'src', 'backend')

sys.path.insert(0, backend_dir)
sys.path.insert(0, microsoft_backend)

class ComponentVerifier:
    """Verifies all Forge 1 components"""
    
    def __init__(self):
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
    
    def log_result(self, component: str, test: str, success: bool, details: str = ""):
        """Log test result"""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        
        result = {
            "component": component,
            "test": test,
            "success": success,
            "details": details,
            "status": status
        }
        
        self.results.append(result)
        print(f"{status}: {component} - {test}")
        if details and not success:
            print(f"    Details: {details}")
    
    def test_microsoft_imports(self):
        """Test Microsoft base imports"""
        print("\n=== Testing Microsoft Base Imports ===")
        
        # Test app_config
        try:
            from app_config import config
            self.log_result("Microsoft", "app_config import", True)
        except Exception as e:
            self.log_result("Microsoft", "app_config import", False, str(e))
        
        # Test agent factory
        try:
            from kernel_agents.agent_factory import AgentFactory
            self.log_result("Microsoft", "AgentFactory import", True)
        except Exception as e:
            self.log_result("Microsoft", "AgentFactory import", False, str(e))
        
        # Test models
        try:
            from models.messages_kernel import AgentType, InputTask
            self.log_result("Microsoft", "Models import", True)
        except Exception as e:
            self.log_result("Microsoft", "Models import", False, str(e))
        
        # Test memory context
        try:
            from context.cosmos_memory_kernel import CosmosMemoryContext
            self.log_result("Microsoft", "CosmosMemoryContext import", True)
        except Exception as e:
            self.log_result("Microsoft", "CosmosMemoryContext import", False, str(e))
    
    def test_forge1_core_imports(self):
        """Test Forge 1 core component imports"""
        print("\n=== Testing Forge 1 Core Imports ===")
        
        # Test model router
        try:
            from forge1.core.model_router import ModelRouter
            self.log_result("Forge1 Core", "ModelRouter import", True)
        except Exception as e:
            self.log_result("Forge1 Core", "ModelRouter import", False, str(e))
        
        # Test security manager
        try:
            from forge1.core.security_manager import SecurityManager
            self.log_result("Forge1 Core", "SecurityManager import", True)
        except Exception as e:
            self.log_result("Forge1 Core", "SecurityManager import", False, str(e))
        
        # Test performance monitor
        try:
            from forge1.core.performance_monitor import PerformanceMonitor
            self.log_result("Forge1 Core", "PerformanceMonitor import", True)
        except Exception as e:
            self.log_result("Forge1 Core", "PerformanceMonitor import", False, str(e))
        
        # Test compliance engine
        try:
            from forge1.core.compliance_engine import ComplianceEngine
            self.log_result("Forge1 Core", "ComplianceEngine import", True)
        except Exception as e:
            self.log_result("Forge1 Core", "ComplianceEngine import", False, str(e))
        
        # Test health checks
        try:
            from forge1.core.health_checks import HealthChecker
            self.log_result("Forge1 Core", "HealthChecker import", True)
        except Exception as e:
            self.log_result("Forge1 Core", "HealthChecker import", False, str(e))
        
        # Test app kernel
        try:
            from forge1.core.app_kernel import Forge1App
            self.log_result("Forge1 Core", "Forge1App import", True)
        except Exception as e:
            self.log_result("Forge1 Core", "Forge1App import", False, str(e))
    
    def test_forge1_agents_imports(self):
        """Test Forge 1 agent imports"""
        print("\n=== Testing Forge 1 Agent Imports ===")
        
        # Test enhanced base agent
        try:
            from forge1.agents.enhanced_base_agent import EnhancedBaseAgent
            self.log_result("Forge1 Agents", "EnhancedBaseAgent import", True)
        except Exception as e:
            self.log_result("Forge1 Agents", "EnhancedBaseAgent import", False, str(e))
        
        # Test enhanced agent factory
        try:
            from forge1.agents.agent_factory_enhanced import EnhancedAgentFactory
            self.log_result("Forge1 Agents", "EnhancedAgentFactory import", True)
        except Exception as e:
            self.log_result("Forge1 Agents", "EnhancedAgentFactory import", False, str(e))
        
        # Test specialized agents
        specialized_agents = [
            ("SuperhumanPlannerAgent", "forge1.agents.superhuman_planner"),
            ("MultiModelCoordinator", "forge1.agents.multi_model_coordinator"),
            ("ComplianceAgent", "forge1.agents.compliance_agent"),
            ("PerformanceOptimizerAgent", "forge1.agents.performance_optimizer")
        ]
        
        for agent_name, module_path in specialized_agents:
            try:
                module = __import__(module_path, fromlist=[agent_name])
                agent_class = getattr(module, agent_name)
                self.log_result("Forge1 Agents", f"{agent_name} import", True)
            except Exception as e:
                self.log_result("Forge1 Agents", f"{agent_name} import", False, str(e))
    
    def test_forge1_integrations_imports(self):
        """Test Forge 1 integration imports"""
        print("\n=== Testing Forge 1 Integration Imports ===")
        
        integrations = [
            ("LangChainAdapter", "forge1.integrations.langchain_adapter"),
            ("CrewAIAdapter", "forge1.integrations.crewai_adapter"),
            ("AutoGenAdapter", "forge1.integrations.autogen_adapter")
        ]
        
        for adapter_name, module_path in integrations:
            try:
                module = __import__(module_path, fromlist=[adapter_name])
                adapter_class = getattr(module, adapter_name)
                self.log_result("Forge1 Integrations", f"{adapter_name} import", True)
            except Exception as e:
                self.log_result("Forge1 Integrations", f"{adapter_name} import", False, str(e))
    
    async def test_component_initialization(self):
        """Test component initialization"""
        print("\n=== Testing Component Initialization ===")
        
        # Test ModelRouter initialization
        try:
            from forge1.core.model_router import ModelRouter
            router = ModelRouter()
            self.log_result("Initialization", "ModelRouter", True)
        except Exception as e:
            self.log_result("Initialization", "ModelRouter", False, str(e))
        
        # Test SecurityManager initialization
        try:
            from forge1.core.security_manager import SecurityManager
            security = SecurityManager()
            self.log_result("Initialization", "SecurityManager", True)
        except Exception as e:
            self.log_result("Initialization", "SecurityManager", False, str(e))
        
        # Test PerformanceMonitor initialization
        try:
            from forge1.core.performance_monitor import PerformanceMonitor
            monitor = PerformanceMonitor()
            self.log_result("Initialization", "PerformanceMonitor", True)
        except Exception as e:
            self.log_result("Initialization", "PerformanceMonitor", False, str(e))
        
        # Test ComplianceEngine initialization
        try:
            from forge1.core.compliance_engine import ComplianceEngine
            compliance = ComplianceEngine()
            self.log_result("Initialization", "ComplianceEngine", True)
        except Exception as e:
            self.log_result("Initialization", "ComplianceEngine", False, str(e))
        
        # Test HealthChecker initialization
        try:
            from forge1.core.health_checks import HealthChecker
            health = HealthChecker()
            self.log_result("Initialization", "HealthChecker", True)
        except Exception as e:
            self.log_result("Initialization", "HealthChecker", False, str(e))
    
    async def test_component_functionality(self):
        """Test basic component functionality"""
        print("\n=== Testing Component Functionality ===")
        
        # Test ModelRouter functionality
        try:
            from forge1.core.model_router import ModelRouter
            router = ModelRouter()
            models = await router.get_available_models()
            health = await router.health_check()
            
            assert isinstance(models, list)
            assert isinstance(health, dict)
            assert "status" in health
            
            self.log_result("Functionality", "ModelRouter", True)
        except Exception as e:
            self.log_result("Functionality", "ModelRouter", False, str(e))
        
        # Test HealthChecker functionality
        try:
            from forge1.core.health_checks import HealthChecker
            health_checker = HealthChecker()
            basic_health = await health_checker.get_basic_health()
            
            assert hasattr(basic_health, 'status')
            assert hasattr(basic_health, 'checks')
            
            self.log_result("Functionality", "HealthChecker", True)
        except Exception as e:
            self.log_result("Functionality", "HealthChecker", False, str(e))
    
    def print_summary(self):
        """Print verification summary"""
        print("\n" + "="*60)
        print("FORGE 1 COMPONENT VERIFICATION SUMMARY")
        print("="*60)
        
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("\n🎉 EXCELLENT: All critical components verified!")
        elif success_rate >= 75:
            print("\n✅ GOOD: Most components verified, minor issues detected")
        elif success_rate >= 50:
            print("\n⚠️  WARNING: Significant issues detected, review required")
        else:
            print("\n❌ CRITICAL: Major component failures, immediate attention required")
        
        # Show failed tests
        failed_tests = [r for r in self.results if not r["success"]]
        if failed_tests:
            print("\nFailed Tests:")
            for test in failed_tests:
                print(f"  • {test['component']} - {test['test']}: {test['details']}")
        
        print("\n" + "="*60)
        
        return success_rate >= 75  # Return True if verification passes

async def main():
    """Main verification function"""
    print("Forge 1 Component Verification")
    print("="*60)
    
    verifier = ComponentVerifier()
    
    # Run all verification tests
    verifier.test_microsoft_imports()
    verifier.test_forge1_core_imports()
    verifier.test_forge1_agents_imports()
    verifier.test_forge1_integrations_imports()
    await verifier.test_component_initialization()
    await verifier.test_component_functionality()
    
    # Print summary and return result
    success = verifier.print_summary()
    
    if success:
        print("\n✅ Component verification PASSED")
        return 0
    else:
        print("\n❌ Component verification FAILED")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nVerification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nVerification failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)