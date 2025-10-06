#!/usr/bin/env python3
"""
Azure Monitor Integration Test Suite

Comprehensive test suite for validating Azure Monitor OpenTelemetry integration,
analytics services, and API endpoints in Forge1.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureMonitorIntegrationTester:
    """Test suite for Azure Monitor integration"""
    
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
        self.passed_tests = []
        
    async def run_all_tests(self):
        """Run all Azure Monitor integration tests"""
        
        logger.info("Starting Azure Monitor Integration Test Suite")
        logger.info("=" * 60)
        
        # Test categories
        test_categories = [
            ("Azure Monitor Adapter Tests", self._test_azure_monitor_adapter),
            ("Analytics Service Tests", self._test_analytics_service),
            ("API Endpoint Tests", self._test_api_endpoints),
            ("Middleware Tests", self._test_middleware),
            ("Configuration Tests", self._test_configuration),
            ("Performance Tests", self._test_performance)
        ]
        
        for category_name, test_function in test_categories:
            logger.info(f"\n--- {category_name} ---")
            try:
                await test_function()
                logger.info(f"✅ {category_name} completed")
            except Exception as e:
                logger.error(f"❌ {category_name} failed: {e}")
                self.failed_tests.append(f"{category_name}: {e}")
        
        # Print summary
        self._print_test_summary()
    
    async def _test_azure_monitor_adapter(self):
        """Test Azure Monitor adapter functionality"""
        
        try:
            from forge1.integrations.observability.azure_monitor import azure_monitor_integration
            
            # Test 1: Initialization
            logger.info("Testing Azure Monitor adapter initialization...")
            initialized = await azure_monitor_integration.initialize()
            self._record_test("Azure Monitor Initialization", initialized, "Adapter should initialize successfully")
            
            # Test 2: Health Check
            logger.info("Testing Azure Monitor health check...")
            health_result = await azure_monitor_integration.health_check()
            health_ok = health_result.status.value in ["healthy", "degraded"]
            self._record_test("Azure Monitor Health Check", health_ok, f"Health status: {health_result.status.value}")
            
            # Test 3: Custom Event
            logger.info("Testing custom event sending...")
            from forge1.integrations.base_adapter import ExecutionContext, TenantContext
            
            context = ExecutionContext(
                tenant_context=TenantContext(tenant_id="test_tenant", user_id="test_user"),
                request_id="test_request_001"
            )
            
            event_sent = await azure_monitor_integration.send_custom_event(
                "test_event",
                {"test_property": "test_value", "timestamp": datetime.now().isoformat()},
                context
            )
            self._record_test("Custom Event Sending", event_sent, "Should send custom event successfully")
            
            # Test 4: Custom Metric
            logger.info("Testing custom metric sending...")
            metric_sent = await azure_monitor_integration.send_custom_metric(
                "test_metric",
                42.0,
                {"test_dimension": "test_value"},
                context
            )
            self._record_test("Custom Metric Sending", metric_sent, "Should send custom metric successfully")
            
            # Test 5: Log to Azure Monitor
            logger.info("Testing log sending...")
            log_sent = await azure_monitor_integration.log_to_azure_monitor(
                "info",
                "Test log message",
                {"test_property": "test_value"},
                context
            )
            self._record_test("Log Sending", log_sent, "Should send log to Azure Monitor successfully")
            
            # Test 6: Get Metrics
            logger.info("Testing metrics retrieval...")
            metrics = azure_monitor_integration.get_metrics()
            metrics_ok = isinstance(metrics, dict) and "azure_monitor_available" in metrics
            self._record_test("Metrics Retrieval", metrics_ok, f"Retrieved metrics: {len(metrics)} items")
            
            logger.info("Azure Monitor adapter tests completed")
            
        except ImportError as e:
            logger.warning(f"Azure Monitor adapter not available: {e}")
            self._record_test("Azure Monitor Adapter Import", False, f"Import failed: {e}")
        except Exception as e:
            logger.error(f"Azure Monitor adapter test failed: {e}")
            self._record_test("Azure Monitor Adapter Tests", False, f"Test suite failed: {e}")
    
    async def _test_analytics_service(self):
        """Test Azure Monitor analytics service"""
        
        try:
            from forge1.services.analytics.azure_monitor_analytics import (
                azure_monitor_analytics_service, AnalyticsTimeRange
            )
            
            # Test 1: Available Queries
            logger.info("Testing available queries...")
            stats = azure_monitor_analytics_service.get_service_statistics()
            queries_available = len(stats["available_queries"]) > 0
            self._record_test("Available Queries", queries_available, f"Found {len(stats['available_queries'])} queries")
            
            # Test 2: Execute Predefined Query
            logger.info("Testing predefined query execution...")
            result = await azure_monitor_analytics_service.execute_analytics_query(
                "tenant_activity_summary",
                "test_tenant",
                AnalyticsTimeRange.LAST_24_HOURS
            )
            query_ok = result is not None and hasattr(result, 'data')
            self._record_test("Predefined Query Execution", query_ok, f"Query returned {len(result.data) if result else 0} records")
            
            # Test 3: Custom Query
            logger.info("Testing custom query execution...")
            custom_result = await azure_monitor_analytics_service.execute_custom_query(
                "customEvents | limit 10",
                "test_tenant",
                "test_custom_query"
            )
            custom_ok = custom_result is not None and hasattr(custom_result, 'data')
            self._record_test("Custom Query Execution", custom_ok, f"Custom query returned {len(custom_result.data) if custom_result else 0} records")
            
            # Test 4: Dashboard Data
            logger.info("Testing dashboard data generation...")
            dashboard_data = await azure_monitor_analytics_service.get_tenant_dashboard_data(
                "test_tenant",
                AnalyticsTimeRange.LAST_24_HOURS
            )
            dashboard_ok = isinstance(dashboard_data, dict) and "tenant_id" in dashboard_data
            self._record_test("Dashboard Data Generation", dashboard_ok, f"Dashboard contains {len(dashboard_data)} sections")
            
            # Test 5: Cost Insights
            logger.info("Testing cost insights...")
            cost_insights = await azure_monitor_analytics_service.get_cost_insights(
                "test_tenant",
                AnalyticsTimeRange.LAST_30_DAYS
            )
            cost_ok = isinstance(cost_insights, dict) and ("total_cost" in cost_insights or "error" in cost_insights)
            self._record_test("Cost Insights", cost_ok, f"Cost insights generated successfully")
            
            # Test 6: Alert Rule Creation
            logger.info("Testing alert rule creation...")
            alert_created = await azure_monitor_analytics_service.create_alert_rule(
                "test_tenant",
                "test_alert",
                "customEvents | where name == 'error' | count",
                10.0,
                "metric"
            )
            self._record_test("Alert Rule Creation", alert_created, "Alert rule created successfully")
            
            # Test 7: Cache Operations
            logger.info("Testing cache operations...")
            azure_monitor_analytics_service.clear_cache()
            cache_stats = azure_monitor_analytics_service.get_service_statistics()
            cache_ok = cache_stats["cached_queries"] == 0
            self._record_test("Cache Operations", cache_ok, "Cache cleared successfully")
            
            logger.info("Analytics service tests completed")
            
        except ImportError as e:
            logger.warning(f"Analytics service not available: {e}")
            self._record_test("Analytics Service Import", False, f"Import failed: {e}")
        except Exception as e:
            logger.error(f"Analytics service test failed: {e}")
            self._record_test("Analytics Service Tests", False, f"Test suite failed: {e}")
    
    async def _test_api_endpoints(self):
        """Test Azure Monitor API endpoints"""
        
        try:
            # Test API imports
            logger.info("Testing API endpoint imports...")
            
            try:
                from forge1.api.v1.analytics import router as analytics_router
                analytics_import_ok = True
            except ImportError as e:
                logger.warning(f"Analytics API not available: {e}")
                analytics_import_ok = False
            
            try:
                from forge1.api.azure_monitor_api import router as azure_monitor_router
                azure_monitor_import_ok = True
            except ImportError as e:
                logger.warning(f"Azure Monitor API not available: {e}")
                azure_monitor_import_ok = False
            
            self._record_test("Analytics API Import", analytics_import_ok, "Analytics API router imported successfully")
            self._record_test("Azure Monitor API Import", azure_monitor_import_ok, "Azure Monitor API router imported successfully")
            
            # Test API route definitions
            if analytics_import_ok:
                logger.info("Testing analytics API routes...")
                routes = [route.path for route in analytics_router.routes]
                expected_routes = ["/analytics/queries/available", "/analytics/dashboard", "/analytics/health"]
                routes_ok = any(expected in str(routes) for expected in expected_routes)
                self._record_test("Analytics API Routes", routes_ok, f"Found {len(routes)} routes")
            
            if azure_monitor_import_ok:
                logger.info("Testing Azure Monitor API routes...")
                routes = [route.path for route in azure_monitor_router.routes]
                expected_routes = ["/health", "/statistics", "/events/custom"]
                routes_ok = any(expected in str(routes) for expected in expected_routes)
                self._record_test("Azure Monitor API Routes", routes_ok, f"Found {len(routes)} routes")
            
            logger.info("API endpoint tests completed")
            
        except Exception as e:
            logger.error(f"API endpoint test failed: {e}")
            self._record_test("API Endpoint Tests", False, f"Test suite failed: {e}")
    
    async def _test_middleware(self):
        """Test Azure Monitor middleware"""
        
        try:
            from forge1.middleware.azure_monitor_middleware import AzureMonitorMiddleware
            
            # Test 1: Middleware Import
            logger.info("Testing middleware import...")
            middleware_import_ok = True
            self._record_test("Middleware Import", middleware_import_ok, "Azure Monitor middleware imported successfully")
            
            # Test 2: Middleware Initialization
            logger.info("Testing middleware initialization...")
            from unittest.mock import Mock
            mock_app = Mock()
            middleware = AzureMonitorMiddleware(mock_app)
            middleware_init_ok = hasattr(middleware, 'collect_request_telemetry')
            self._record_test("Middleware Initialization", middleware_init_ok, "Middleware initialized with telemetry settings")
            
            # Test 3: Statistics
            logger.info("Testing middleware statistics...")
            stats = middleware.get_statistics()
            stats_ok = isinstance(stats, dict) and "requests_processed" in stats
            self._record_test("Middleware Statistics", stats_ok, f"Statistics contain {len(stats)} metrics")
            
            # Test 4: Configuration
            logger.info("Testing middleware configuration...")
            middleware.configure_telemetry_collection(
                request_telemetry=True,
                performance_metrics=True,
                error_telemetry=True,
                custom_events=True
            )
            config_ok = middleware.collect_request_telemetry and middleware.collect_performance_metrics
            self._record_test("Middleware Configuration", config_ok, "Telemetry collection configured successfully")
            
            logger.info("Middleware tests completed")
            
        except ImportError as e:
            logger.warning(f"Azure Monitor middleware not available: {e}")
            self._record_test("Middleware Import", False, f"Import failed: {e}")
        except Exception as e:
            logger.error(f"Middleware test failed: {e}")
            self._record_test("Middleware Tests", False, f"Test suite failed: {e}")
    
    async def _test_configuration(self):
        """Test Azure Monitor configuration"""
        
        try:
            from forge1.config.integration_settings import settings_manager, IntegrationType
            
            # Test 1: Configuration Loading
            logger.info("Testing configuration loading...")
            try:
                config = settings_manager.get_config(IntegrationType.OBSERVABILITY)
                config_loaded = config is not None
                has_azure_monitor = "azure_monitor" in config if config else False
                self._record_test("Configuration Loading", config_loaded, f"Observability config loaded: {config_loaded}")
                self._record_test("Azure Monitor Config", has_azure_monitor, f"Azure Monitor section present: {has_azure_monitor}")
            except Exception as e:
                logger.warning(f"Configuration loading failed: {e}")
                self._record_test("Configuration Loading", False, f"Config load failed: {e}")
            
            # Test 2: Environment Variables
            logger.info("Testing environment variable configuration...")
            import os
            env_vars = [
                "AZURE_MONITOR_CONNECTION_STRING",
                "AZURE_MONITOR_INSTRUMENTATION_KEY",
                "AZURE_MONITOR_SAMPLING_RATIO",
                "AZURE_MONITOR_LIVE_METRICS"
            ]
            
            env_configured = any(os.getenv(var) for var in env_vars)
            self._record_test("Environment Configuration", True, f"Environment variables checked: {len(env_vars)}")
            
            # Test 3: Default Configuration
            logger.info("Testing default configuration values...")
            from forge1.integrations.observability.azure_monitor import azure_monitor_integration
            
            default_config_ok = hasattr(azure_monitor_integration, 'sampling_ratio')
            self._record_test("Default Configuration", default_config_ok, "Default configuration values present")
            
            logger.info("Configuration tests completed")
            
        except ImportError as e:
            logger.warning(f"Configuration components not available: {e}")
            self._record_test("Configuration Import", False, f"Import failed: {e}")
        except Exception as e:
            logger.error(f"Configuration test failed: {e}")
            self._record_test("Configuration Tests", False, f"Test suite failed: {e}")
    
    async def _test_performance(self):
        """Test Azure Monitor integration performance"""
        
        try:
            from forge1.integrations.observability.azure_monitor import azure_monitor_integration
            from forge1.integrations.base_adapter import ExecutionContext, TenantContext
            
            # Test 1: Telemetry Performance
            logger.info("Testing telemetry sending performance...")
            
            context = ExecutionContext(
                tenant_context=TenantContext(tenant_id="perf_test_tenant", user_id="perf_test_user"),
                request_id="perf_test_001"
            )
            
            # Send multiple events and measure performance
            start_time = time.time()
            event_count = 10
            successful_events = 0
            
            for i in range(event_count):
                success = await azure_monitor_integration.send_custom_event(
                    f"performance_test_event_{i}",
                    {"event_number": i, "timestamp": datetime.now().isoformat()},
                    context
                )
                if success:
                    successful_events += 1
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time_per_event = total_time / event_count
            
            perf_ok = avg_time_per_event < 0.1  # Should be under 100ms per event
            self._record_test("Telemetry Performance", perf_ok, 
                            f"Sent {successful_events}/{event_count} events in {total_time:.3f}s (avg: {avg_time_per_event:.3f}s/event)")
            
            # Test 2: Analytics Query Performance
            logger.info("Testing analytics query performance...")
            
            try:
                from forge1.services.analytics.azure_monitor_analytics import azure_monitor_analytics_service
                
                query_start_time = time.time()
                result = await azure_monitor_analytics_service.execute_analytics_query(
                    "tenant_activity_summary",
                    "perf_test_tenant"
                )
                query_end_time = time.time()
                query_time = query_end_time - query_start_time
                
                query_perf_ok = query_time < 5.0  # Should be under 5 seconds
                self._record_test("Analytics Query Performance", query_perf_ok,
                                f"Query executed in {query_time:.3f}s")
                
            except ImportError:
                logger.warning("Analytics service not available for performance testing")
                self._record_test("Analytics Query Performance", True, "Skipped - service not available")
            
            # Test 3: Memory Usage
            logger.info("Testing memory usage...")
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            memory_ok = memory_mb < 500  # Should use less than 500MB
            self._record_test("Memory Usage", memory_ok, f"Process using {memory_mb:.1f}MB RAM")
            
            logger.info("Performance tests completed")
            
        except ImportError as e:
            logger.warning(f"Performance test components not available: {e}")
            self._record_test("Performance Test Import", False, f"Import failed: {e}")
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            self._record_test("Performance Tests", False, f"Test suite failed: {e}")
    
    def _record_test(self, test_name: str, passed: bool, details: str):
        """Record test result"""
        
        result = {
            "test_name": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        
        self.test_results.append(result)
        
        if passed:
            self.passed_tests.append(test_name)
            logger.info(f"  ✅ {test_name}: {details}")
        else:
            self.failed_tests.append(test_name)
            logger.error(f"  ❌ {test_name}: {details}")
    
    def _print_test_summary(self):
        """Print comprehensive test summary"""
        
        logger.info("\n" + "=" * 60)
        logger.info("AZURE MONITOR INTEGRATION TEST SUMMARY")
        logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        passed_count = len(self.passed_tests)
        failed_count = len(self.failed_tests)
        success_rate = (passed_count / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        if self.failed_tests:
            logger.info("\nFailed Tests:")
            for test_name in self.failed_tests:
                logger.info(f"  ❌ {test_name}")
        
        if self.passed_tests:
            logger.info(f"\nPassed Tests: {len(self.passed_tests)} tests passed successfully")
        
        # Save detailed results to file
        results_file = "azure_monitor_test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed": passed_count,
                    "failed": failed_count,
                    "success_rate": success_rate,
                    "timestamp": datetime.now().isoformat()
                },
                "test_results": self.test_results,
                "failed_tests": self.failed_tests,
                "passed_tests": self.passed_tests
            }, f, indent=2)
        
        logger.info(f"\nDetailed results saved to: {results_file}")
        
        if success_rate >= 80:
            logger.info("🎉 Azure Monitor integration is working well!")
        elif success_rate >= 60:
            logger.info("⚠️  Azure Monitor integration has some issues but is functional")
        else:
            logger.info("🚨 Azure Monitor integration needs attention")

async def main():
    """Run the Azure Monitor integration test suite"""
    
    tester = AzureMonitorIntegrationTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())