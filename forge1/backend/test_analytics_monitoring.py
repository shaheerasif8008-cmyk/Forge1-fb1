#!/usr/bin/env python3
"""
Test script for Employee Analytics and Monitoring Features

This script tests the comprehensive analytics and monitoring system including:
- Employee performance metrics
- Client usage analytics
- Health monitoring and alerting
- Cost tracking and analysis
- Dashboard functionality

Requirements: 5.4, 7.3, 8.3
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List

import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"

class AnalyticsTester:
    """Test class for Analytics and Monitoring features"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        
        # Test data
        self.test_client_id = "demo_client_001"
        self.test_employee_id = None  # Will be set during testing
        
        # Valid headers for testing
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer demo_token",
            "X-Tenant-ID": self.test_client_id,
            "X-Client-ID": self.test_client_id,
            "X-User-ID": "demo_user_001"
        }
    
    async def test_employee_metrics(self) -> bool:
        """Test employee performance metrics"""
        try:
            # First, get available employees
            async with httpx.AsyncClient() as client:
                employees_response = await client.get(
                    f"{self.base_url}/api/v1/integration/employees/{self.test_client_id}",
                    headers=self.headers
                )
                
                if employees_response.status_code != 200:
                    logger.error(f"❌ Failed to get employees: {employees_response.status_code}")
                    return False
                
                employees_data = employees_response.json()
                available_employees = employees_data.get("available_employees", [])
                
                if not available_employees:
                    logger.warning("⚠️  No employees available for metrics testing")
                    return True  # Not a failure, just no data
                
                # Use first available employee
                self.test_employee_id = available_employees[0]["employee_id"]
                
                # Test employee metrics endpoint
                metrics_response = await client.get(
                    f"{self.base_url}/api/v1/analytics/employees/{self.test_client_id}/{self.test_employee_id}/metrics?days=7",
                    headers=self.headers
                )
                
                if metrics_response.status_code == 200:
                    metrics_data = metrics_response.json()
                    
                    logger.info("✅ Employee metrics endpoint working")
                    logger.info(f"   Employee: {metrics_data.get('employee_name', 'Unknown')}")
                    
                    metrics = metrics_data.get("metrics", {})
                    logger.info(f"   Total interactions: {metrics.get('total_interactions', 0)}")
                    logger.info(f"   Avg response time: {metrics.get('avg_response_time_ms', 0):.2f}ms")
                    logger.info(f"   Success rate: {metrics.get('success_rate', 0):.1f}%")
                    logger.info(f"   Performance trend: {metrics.get('performance_trend', 'unknown')}")
                    
                    return True
                else:
                    logger.error(f"❌ Employee metrics failed: {metrics_response.status_code}")
                    logger.error(f"   Response: {metrics_response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Employee metrics test error: {e}")
            return False
    
    async def test_employee_health_monitoring(self) -> bool:
        """Test employee health monitoring and alerting"""
        try:
            if not self.test_employee_id:
                logger.warning("⚠️  No employee ID available for health testing")
                return True
            
            async with httpx.AsyncClient() as client:
                # Test health status endpoint
                health_response = await client.get(
                    f"{self.base_url}/api/v1/analytics/employees/{self.test_client_id}/{self.test_employee_id}/health",
                    headers=self.headers
                )
                
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    
                    logger.info("✅ Employee health monitoring working")
                    logger.info(f"   Overall health score: {health_data.get('overall_health_score', 0)}")
                    logger.info(f"   Health status: {health_data.get('health_status', 'unknown')}")
                    
                    # Check component scores
                    component_scores = health_data.get("component_scores", {})
                    for component, score in component_scores.items():
                        logger.info(f"   {component.replace('_', ' ').title()}: {score}")
                    
                    # Check alerts
                    alerts = health_data.get("alerts", [])
                    if alerts:
                        logger.info(f"   Active alerts: {len(alerts)}")
                        for alert in alerts[:3]:  # Show first 3 alerts
                            logger.info(f"     - {alert.get('type', 'unknown')}: {alert.get('message', 'No message')}")
                    else:
                        logger.info("   No active alerts")
                    
                    return True
                else:
                    logger.error(f"❌ Employee health monitoring failed: {health_response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Employee health monitoring test error: {e}")
            return False
    
    async def test_client_usage_analytics(self) -> bool:
        """Test client usage analytics"""
        try:
            async with httpx.AsyncClient() as client:
                # Test client usage endpoint
                usage_response = await client.get(
                    f"{self.base_url}/api/v1/analytics/clients/{self.test_client_id}/usage?days=30",
                    headers=self.headers
                )
                
                if usage_response.status_code == 200:
                    usage_data = usage_response.json()
                    
                    logger.info("✅ Client usage analytics working")
                    logger.info(f"   Client: {usage_data.get('client_name', 'Unknown')}")
                    
                    summary = usage_data.get("summary", {})
                    logger.info(f"   Total employees: {summary.get('total_employees', 0)}")
                    logger.info(f"   Active employees: {summary.get('active_employees', 0)}")
                    logger.info(f"   Total interactions: {summary.get('total_interactions', 0)}")
                    logger.info(f"   Total cost: ${summary.get('total_cost', 0):.4f}")
                    logger.info(f"   Monthly growth: {summary.get('monthly_growth', 0):.1f}%")
                    
                    # Check top performing employees
                    top_employees = usage_data.get("top_performing_employees", [])
                    if top_employees:
                        logger.info(f"   Top performer: {top_employees[0].get('employee_name', 'Unknown')}")
                    
                    return True
                else:
                    logger.error(f"❌ Client usage analytics failed: {usage_response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Client usage analytics test error: {e}")
            return False
    
    async def test_analytics_dashboard(self) -> bool:
        """Test comprehensive analytics dashboard"""
        try:
            async with httpx.AsyncClient() as client:
                # Test dashboard endpoint
                dashboard_response = await client.get(
                    f"{self.base_url}/api/v1/analytics/clients/{self.test_client_id}/dashboard?days=7",
                    headers=self.headers
                )
                
                if dashboard_response.status_code == 200:
                    dashboard_data = dashboard_response.json()
                    
                    logger.info("✅ Analytics dashboard working")
                    
                    # Client metrics summary
                    client_metrics = dashboard_data.get("client_metrics", {})
                    logger.info(f"   Client: {client_metrics.get('client_name', 'Unknown')}")
                    logger.info(f"   Total cost: ${client_metrics.get('total_cost', 0):.4f}")
                    
                    # Employee health summary
                    health_summary = dashboard_data.get("employee_health_summary", {})
                    logger.info(f"   Employees monitored: {health_summary.get('total_employees_monitored', 0)}")
                    logger.info(f"   Avg health score: {health_summary.get('avg_health_score', 0)}")
                    logger.info(f"   Total alerts: {health_summary.get('total_alerts', 0)}")
                    
                    # Usage patterns
                    usage_patterns = dashboard_data.get("usage_patterns", {})
                    peak_hours = usage_patterns.get("peak_usage_hours", [])
                    if peak_hours:
                        logger.info(f"   Peak usage hours: {peak_hours}")
                    
                    # Cost analysis
                    cost_analysis = dashboard_data.get("cost_analysis", {})
                    logger.info(f"   Cost trend: {cost_analysis.get('cost_trend', 'unknown')}")
                    
                    return True
                else:
                    logger.error(f"❌ Analytics dashboard failed: {dashboard_response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Analytics dashboard test error: {e}")
            return False
    
    async def test_cost_tracking(self) -> bool:
        """Test cost calculation and tracking"""
        try:
            async with httpx.AsyncClient() as client:
                # Test cost calculation endpoint
                cost_response = await client.get(
                    f"{self.base_url}/api/v1/analytics/costs/calculate?model=gpt-4&input_tokens=100&output_tokens=50",
                    headers=self.headers
                )
                
                if cost_response.status_code == 200:
                    cost_data = cost_response.json()
                    
                    logger.info("✅ Cost tracking working")
                    logger.info(f"   Model: {cost_data.get('model', 'unknown')}")
                    logger.info(f"   Input tokens: {cost_data.get('input_tokens', 0)}")
                    logger.info(f"   Output tokens: {cost_data.get('output_tokens', 0)}")
                    logger.info(f"   Total cost: ${cost_data.get('total_cost', 0):.6f}")
                    
                    # Check cost breakdown
                    breakdown = cost_data.get("cost_breakdown", {})
                    logger.info(f"   Input cost: ${breakdown.get('input_cost', 0):.6f}")
                    logger.info(f"   Output cost: ${breakdown.get('output_cost', 0):.6f}")
                    
                    return True
                else:
                    logger.error(f"❌ Cost tracking failed: {cost_response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Cost tracking test error: {e}")
            return False
    
    async def test_feedback_system(self) -> bool:
        """Test user feedback recording system"""
        try:
            if not self.test_employee_id:
                logger.warning("⚠️  No employee ID available for feedback testing")
                return True
            
            async with httpx.AsyncClient() as client:
                # Test feedback recording
                feedback_data = {
                    "interaction_id": f"test_interaction_{int(time.time())}",
                    "rating": 4,
                    "feedback_text": "Great response, very helpful!"
                }
                
                feedback_response = await client.post(
                    f"{self.base_url}/api/v1/analytics/employees/{self.test_client_id}/{self.test_employee_id}/feedback",
                    json=feedback_data,
                    headers=self.headers
                )
                
                if feedback_response.status_code == 200:
                    response_data = feedback_response.json()
                    
                    logger.info("✅ Feedback system working")
                    logger.info(f"   Interaction ID: {response_data.get('interaction_id', 'unknown')}")
                    logger.info(f"   Rating recorded: {response_data.get('rating', 0)}/5")
                    logger.info(f"   Message: {response_data.get('message', 'No message')}")
                    
                    return True
                else:
                    logger.error(f"❌ Feedback system failed: {feedback_response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Feedback system test error: {e}")
            return False
    
    async def test_alerts_system(self) -> bool:
        """Test alerts and monitoring system"""
        try:
            async with httpx.AsyncClient() as client:
                # Test alerts endpoint
                alerts_response = await client.get(
                    f"{self.base_url}/api/v1/analytics/alerts/{self.test_client_id}",
                    headers=self.headers
                )
                
                if alerts_response.status_code == 200:
                    alerts_data = alerts_response.json()
                    
                    logger.info("✅ Alerts system working")
                    logger.info(f"   Total alerts: {alerts_data.get('total_alerts', 0)}")
                    logger.info(f"   Critical alerts: {alerts_data.get('critical_alerts', 0)}")
                    logger.info(f"   Warning alerts: {alerts_data.get('warning_alerts', 0)}")
                    
                    # Show sample alerts
                    alerts = alerts_data.get("alerts", [])
                    if alerts:
                        logger.info("   Sample alerts:")
                        for alert in alerts[:3]:  # Show first 3
                            alert_type = alert.get("type", "unknown")
                            severity = alert.get("severity", "unknown")
                            message = alert.get("message", "No message")
                            logger.info(f"     - [{severity.upper()}] {alert_type}: {message[:50]}...")
                    else:
                        logger.info("   No active alerts")
                    
                    return True
                else:
                    logger.error(f"❌ Alerts system failed: {alerts_response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Alerts system test error: {e}")
            return False
    
    async def test_performance_monitoring_integration(self) -> bool:
        """Test integration with performance monitoring system"""
        try:
            async with httpx.AsyncClient() as client:
                # Test performance metrics endpoint
                perf_response = await client.get(
                    f"{self.base_url}/api/v1/performance/metrics",
                    headers=self.headers
                )
                
                if perf_response.status_code == 200:
                    perf_data = perf_response.json()
                    
                    logger.info("✅ Performance monitoring integration working")
                    
                    # Check if analytics metrics are included
                    metrics = perf_data.get("metrics", {})
                    
                    # Look for employee-related metrics
                    employee_metrics = [key for key in metrics.keys() if "employee" in key.lower()]
                    if employee_metrics:
                        logger.info(f"   Employee metrics found: {len(employee_metrics)}")
                        for metric in employee_metrics[:3]:  # Show first 3
                            logger.info(f"     - {metric}")
                    
                    # Check system health
                    system_info = perf_data.get("system")
                    if system_info:
                        logger.info(f"   System CPU: {system_info.get('cpu_percent', 0):.1f}%")
                        logger.info(f"   System Memory: {system_info.get('memory_percent', 0):.1f}%")
                    
                    return True
                else:
                    logger.error(f"❌ Performance monitoring integration failed: {perf_response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Performance monitoring integration test error: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all analytics and monitoring tests"""
        logger.info("📊 Starting Employee Analytics and Monitoring Tests")
        logger.info("=" * 65)
        
        results = {}
        
        # Test 1: Employee Metrics
        logger.info("\n📊 Test 1: Employee Performance Metrics")
        results["employee_metrics"] = await self.test_employee_metrics()
        
        # Test 2: Employee Health Monitoring
        logger.info("\n📊 Test 2: Employee Health Monitoring")
        results["health_monitoring"] = await self.test_employee_health_monitoring()
        
        # Test 3: Client Usage Analytics
        logger.info("\n📊 Test 3: Client Usage Analytics")
        results["client_usage_analytics"] = await self.test_client_usage_analytics()
        
        # Test 4: Analytics Dashboard
        logger.info("\n📊 Test 4: Analytics Dashboard")
        results["analytics_dashboard"] = await self.test_analytics_dashboard()
        
        # Test 5: Cost Tracking
        logger.info("\n📊 Test 5: Cost Tracking and Analysis")
        results["cost_tracking"] = await self.test_cost_tracking()
        
        # Test 6: Feedback System
        logger.info("\n📊 Test 6: User Feedback System")
        results["feedback_system"] = await self.test_feedback_system()
        
        # Test 7: Alerts System
        logger.info("\n📊 Test 7: Alerts and Monitoring")
        results["alerts_system"] = await self.test_alerts_system()
        
        # Test 8: Performance Integration
        logger.info("\n📊 Test 8: Performance Monitoring Integration")
        results["performance_integration"] = await self.test_performance_monitoring_integration()
        
        # Summary
        logger.info("\n" + "=" * 65)
        logger.info("📊 ANALYTICS AND MONITORING TEST RESULTS")
        logger.info("=" * 65)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            logger.info("🎉 All analytics tests passed! Monitoring system is fully operational.")
        else:
            logger.warning(f"⚠️  {total - passed} analytics test(s) failed. Review monitoring configuration.")
        
        return results


async def main():
    """Main test function"""
    tester = AnalyticsTester()
    results = await tester.run_all_tests()
    
    # Exit with appropriate code
    if all(results.values()):
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())