#!/usr/bin/env python3
"""
Comprehensive test suite for Phase G - Launch, Pricing, and Scale
Tests pricing engine, capacity optimizer, and support system
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal

# Add the forge1 directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'forge1'))

def test_phase_g_imports():
    """Test that Phase G modules can be imported"""
    print("Testing Phase G module imports...")
    
    try:
        # Test billing imports
        from billing.pricing_engine import PricingEngine, PricingModel, UsageMetricType
        print("✅ Pricing engine imported successfully")
        
        # Test optimization imports
        from optimization.capacity_optimizer import CapacityOptimizer, ResourceType, OptimizationStrategy
        print("✅ Capacity optimizer imported successfully")
        
        # Test support imports
        from support.support_system import SupportSystem, SupportTier, TicketPriority
        print("✅ Support system imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_phase_g_classes():
    """Test that Phase G classes can be instantiated"""
    print("\nTesting Phase G class instantiation...")
    
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
        
        # Test pricing engine
        from billing.pricing_engine import PricingEngine
        pricing_engine = PricingEngine(
            MockMemoryManager(),
            MockMetricsCollector(),
            MockSecretManager()
        )
        print("✅ Pricing engine instantiated successfully")
        
        # Test capacity optimizer
        from optimization.capacity_optimizer import CapacityOptimizer
        capacity_optimizer = CapacityOptimizer(
            MockMemoryManager(),
            MockMetricsCollector(),
            MockSecretManager()
        )
        print("✅ Capacity optimizer instantiated successfully")
        
        # Test support system
        from support.support_system import SupportSystem
        support_system = SupportSystem(
            MockMemoryManager(),
            MockMetricsCollector(),
            MockSecretManager()
        )
        print("✅ Support system instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Class instantiation error: {e}")
        return False

async def test_pricing_functionality():
    """Test pricing engine functionality"""
    print("\nTesting pricing engine functionality...")
    
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
        
        from billing.pricing_engine import PricingEngine, PricingModel, UsageMetricType
        
        pricing_engine = PricingEngine(
            MockMemoryManager(),
            MockMetricsCollector(),
            MockSecretManager()
        )
        
        # Test usage metrics initialization
        assert len(pricing_engine.usage_metrics) > 0, "Usage metrics not initialized"
        print("✅ Usage metrics initialized")
        
        # Test pricing plans initialization
        assert len(pricing_engine.pricing_plans) > 0, "Pricing plans not initialized"
        print("✅ Pricing plans initialized")
        
        # Test usage recording
        usage_id = await pricing_engine.record_usage(
            customer_id="test_customer",
            metric_id="api_calls",
            quantity=Decimal('100'),
            user_id="test_user"
        )
        assert usage_id.startswith("usage_"), f"Invalid usage ID: {usage_id}"
        print("✅ Usage recording works")
        
        # Test subscription creation
        subscription = await pricing_engine.create_customer_subscription(
            customer_id="test_customer",
            plan_id="starter"
        )
        assert subscription["customer_id"] == "test_customer", "Subscription creation failed"
        print("✅ Subscription creation works")
        
        # Test entitlement checking
        entitlement_check = await pricing_engine.check_entitlements(
            customer_id="test_customer",
            resource_name="api_calls",
            requested_quantity=50
        )
        assert "entitled" in entitlement_check, "Entitlement check failed"
        print("✅ Entitlement checking works")
        
        # Test invoice generation
        invoice = await pricing_engine.generate_invoice(
            customer_id="test_customer",
            billing_period_start=datetime.utcnow() - timedelta(days=30),
            billing_period_end=datetime.utcnow()
        )
        assert invoice.customer_id == "test_customer", "Invoice generation failed"
        print("✅ Invoice generation works")
        
        # Test dashboard
        dashboard = pricing_engine.get_pricing_dashboard("test_customer")
        assert "current_month" in dashboard, "Customer dashboard missing data"
        print("✅ Pricing dashboard works")
        
        return True
        
    except Exception as e:
        print(f"❌ Pricing functionality test error: {e}")
        return False

async def test_capacity_optimization():
    """Test capacity optimizer functionality"""
    print("\nTesting capacity optimizer functionality...")
    
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
        
        from optimization.capacity_optimizer import CapacityOptimizer, ResourceType, OptimizationStrategy
        
        capacity_optimizer = CapacityOptimizer(
            MockMemoryManager(),
            MockMetricsCollector(),
            MockSecretManager()
        )
        
        # Test resource metrics recording
        await capacity_optimizer.record_resource_metrics(
            resource_id="test_resource",
            resource_type=ResourceType.COMPUTE,
            cpu_utilization=75.0,
            memory_utilization=60.0,
            hourly_cost=Decimal('5.00'),
            current_capacity=4,
            max_capacity=10
        )
        assert "test_resource" in capacity_optimizer.resource_metrics, "Resource metrics not recorded"
        print("✅ Resource metrics recording works")
        
        # Test model routing optimization
        model_recommendation = await capacity_optimizer.optimize_model_routing(
            task_complexity="medium",
            quality_requirement="high",
            latency_requirement="standard"
        )
        assert "recommended_model" in model_recommendation, "Model routing failed"
        print("✅ Model routing optimization works")
        
        # Test budget alert creation
        budget_alert = await capacity_optimizer.create_budget_alert(
            budget_name="test_budget",
            budget_limit=Decimal('1000'),
            current_spend=Decimal('800'),
            projected_spend=Decimal('950'),
            threshold_percentage=80.0
        )
        assert budget_alert.budget_name == "test_budget", "Budget alert creation failed"
        print("✅ Budget alert creation works")
        
        # Test cost optimization opportunities
        opportunities = await capacity_optimizer.identify_cost_optimization_opportunities(
            resource_ids=["test_resource"]
        )
        assert isinstance(opportunities, list), "Cost optimization identification failed"
        print("✅ Cost optimization identification works")
        
        # Test FinOps dashboard
        dashboard = capacity_optimizer.get_finops_dashboard()
        assert "cost_overview" in dashboard, "FinOps dashboard missing data"
        print("✅ FinOps dashboard works")
        
        return True
        
    except Exception as e:
        print(f"❌ Capacity optimization test error: {e}")
        return False

async def test_support_system():
    """Test support system functionality"""
    print("\nTesting support system functionality...")
    
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
        
        from support.support_system import SupportSystem, SupportTier, TicketPriority, IncidentSeverity
        
        support_system = SupportSystem(
            MockMemoryManager(),
            MockMetricsCollector(),
            MockSecretManager()
        )
        
        # Test support ticket creation
        ticket = await support_system.create_support_ticket(
            customer_id="test_customer",
            title="Test Issue",
            description="This is a test support ticket",
            priority=TicketPriority.MEDIUM,
            contact_email="test@example.com",
            support_tier=SupportTier.STANDARD
        )
        assert ticket.customer_id == "test_customer", "Support ticket creation failed"
        print("✅ Support ticket creation works")
        
        # Test ticket status update
        from support.support_system import TicketStatus
        updated = await support_system.update_ticket_status(
            ticket_id=ticket.ticket_id,
            new_status=TicketStatus.IN_PROGRESS,
            update_message="Working on the issue",
            updated_by="support_agent"
        )
        assert updated, "Ticket status update failed"
        print("✅ Ticket status update works")
        
        # Test incident creation
        incident = await support_system.create_incident(
            title="Test Incident",
            description="This is a test incident",
            severity=IncidentSeverity.SEV3,
            affected_services=["web_app"],
            incident_commander="incident_commander"
        )
        assert incident.title == "Test Incident", "Incident creation failed"
        print("✅ Incident creation works")
        
        # Test rollback plans initialization
        assert len(support_system.rollback_plans) > 0, "Rollback plans not initialized"
        print("✅ Rollback plans initialized")
        
        # Test rollback execution (with approval override)
        plan_id = list(support_system.rollback_plans.keys())[0]
        rollback_result = await support_system.execute_rollback(
            plan_id=plan_id,
            executed_by="ops_engineer",
            approval_override=True
        )
        assert rollback_result["success"], "Rollback execution failed"
        print("✅ Rollback execution works")
        
        # Test customer onboarding
        onboarding = await support_system.start_customer_onboarding(
            customer_id="test_customer",
            plan_type="professional",
            support_tier=SupportTier.PREMIUM
        )
        assert onboarding.customer_id == "test_customer", "Customer onboarding failed"
        print("✅ Customer onboarding works")
        
        # Test support dashboard
        dashboard = support_system.get_support_dashboard()
        assert "tickets" in dashboard, "Support dashboard missing data"
        print("✅ Support dashboard works")
        
        return True
        
    except Exception as e:
        print(f"❌ Support system test error: {e}")
        return False

async def test_integration_scenarios():
    """Test integration scenarios across Phase G components"""
    print("\nTesting Phase G integration scenarios...")
    
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
        
        # Initialize all systems
        from billing.pricing_engine import PricingEngine
        from optimization.capacity_optimizer import CapacityOptimizer, ResourceType
        from support.support_system import SupportSystem, SupportTier, TicketPriority
        
        pricing_engine = PricingEngine(MockMemoryManager(), MockMetricsCollector(), MockSecretManager())
        capacity_optimizer = CapacityOptimizer(MockMemoryManager(), MockMetricsCollector(), MockSecretManager())
        support_system = SupportSystem(MockMemoryManager(), MockMetricsCollector(), MockSecretManager())
        
        # Scenario 1: Customer lifecycle from onboarding to billing
        print("\nScenario 1: Complete Customer Lifecycle")
        
        # Start onboarding
        onboarding = await support_system.start_customer_onboarding(
            customer_id="integration_customer",
            plan_type="professional",
            support_tier=SupportTier.PREMIUM
        )
        
        # Create subscription
        subscription = await pricing_engine.create_customer_subscription(
            customer_id="integration_customer",
            plan_id="professional"
        )
        
        # Record usage
        usage_id = await pricing_engine.record_usage(
            customer_id="integration_customer",
            metric_id="api_calls",
            quantity=Decimal('5000')
        )
        
        # Record resource metrics
        await capacity_optimizer.record_resource_metrics(
            resource_id="customer_resource",
            resource_type=ResourceType.COMPUTE,
            cpu_utilization=85.0,
            memory_utilization=70.0,
            hourly_cost=Decimal('10.00'),
            current_capacity=2
        )
        
        assert all([onboarding, subscription, usage_id]), "Customer lifecycle integration failed"
        print("✅ Customer lifecycle integration works")
        
        # Scenario 2: Cost optimization triggering support action
        print("\nScenario 2: Cost Optimization and Support Integration")
        
        # Create budget alert
        budget_alert = await capacity_optimizer.create_budget_alert(
            budget_name="customer_budget",
            budget_limit=Decimal('500'),
            current_spend=Decimal('450'),
            projected_spend=Decimal('520'),
            threshold_percentage=90.0
        )
        
        # Create support ticket for budget issue
        ticket = await support_system.create_support_ticket(
            customer_id="integration_customer",
            title="Budget Alert - Cost Optimization Needed",
            description=f"Customer budget alert triggered: {budget_alert.message}",
            priority=TicketPriority.HIGH,
            contact_email="customer@example.com",
            category="billing"
        )
        
        # Get cost optimization recommendations
        opportunities = await capacity_optimizer.identify_cost_optimization_opportunities()
        
        assert all([budget_alert, ticket, isinstance(opportunities, list)]), "Cost optimization integration failed"
        print("✅ Cost optimization and support integration works")
        
        # Scenario 3: Incident response with rollback
        print("\nScenario 3: Incident Response and Rollback")
        
        # Create incident
        from support.support_system import IncidentSeverity
        incident = await support_system.create_incident(
            title="Performance Degradation",
            description="High CPU utilization causing performance issues",
            severity=IncidentSeverity.SEV2,
            affected_services=["api_server"],
            incident_commander="ops_lead"
        )
        
        # Execute rollback as part of incident response
        plan_id = list(support_system.rollback_plans.keys())[0]
        rollback_result = await support_system.execute_rollback(
            plan_id=plan_id,
            executed_by="incident_responder",
            approval_override=True
        )
        
        assert all([incident, rollback_result["success"]]), "Incident response integration failed"
        print("✅ Incident response and rollback integration works")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration scenarios test error: {e}")
        return False

async def main():
    """Run all Phase G tests"""
    print("FORGE 1 PHASE G IMPLEMENTATION TEST SUITE")
    print("=" * 80)
    
    test_functions = [
        test_phase_g_imports,
        test_phase_g_classes,
        test_pricing_functionality,
        test_capacity_optimization,
        test_support_system,
        test_integration_scenarios
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print("PHASE G TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {len(test_functions)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("✅ All Phase G tests passed! Launch, Pricing, and Scale implementation is working correctly.")
        return True
    else:
        print(f"❌ {failed} tests failed. See details above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)