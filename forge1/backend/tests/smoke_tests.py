#!/usr/bin/env python3
"""
Smoke Tests for Employee Lifecycle System

Quick validation tests to ensure basic system functionality
after deployment. These tests verify core endpoints and
basic system health.

Requirements: 1.5, 2.5, 3.4, 5.5, 6.5
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import click
import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class SmokeTestRunner:
    """Runs smoke tests for post-deployment validation"""
    
    def __init__(self, base_url: str, environment: str = "development"):
        self.base_url = base_url.rstrip('/')
        self.environment = environment
        self.test_results = {}
        self.start_time = time.time()
    
    def get_test_headers(self) -> Dict[str, str]:
        """Get headers for smoke test requests"""
        return {
            "Content-Type": "application/json",
            "User-Agent": f"SmokeTest/1.0 ({self.environment})",
            "X-Test-Type": "smoke"
        }
    
    async def test_health_endpoints(self) -> Dict[str, Any]:
        """Test basic health endpoints"""
        console.print("🏥 Testing Health Endpoints", style="bold blue")
        
        endpoints = [
            ('/health', 'Basic Health Check'),
            ('/ready', 'Readiness Check'),
            ('/performance/health', 'System Health'),
        ]
        
        results = {}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for endpoint, description in endpoints:
                try:
                    start_time = time.time()
                    response = await client.get(
                        f"{self.base_url}{endpoint}",
                        headers=self.get_test_headers()
                    )
                    response_time = (time.time() - start_time) * 1000
                    
                    results[endpoint] = {
                        'status': 'passed' if response.status_code == 200 else 'failed',
                        'status_code': response.status_code,
                        'response_time_ms': round(response_time, 2),
                        'description': description
                    }
                    
                    if response.status_code == 200:
                        console.print(f"✅ {description}: {response.status_code} ({response_time:.2f}ms)", style="green")
                    else:
                        console.print(f"❌ {description}: {response.status_code}", style="red")
                        
                except Exception as e:
                    results[endpoint] = {
                        'status': 'error',
                        'error': str(e),
                        'description': description
                    }
                    console.print(f"❌ {description}: {e}", style="red")
        
        return results
    
    async def test_api_documentation(self) -> Dict[str, Any]:
        """Test API documentation endpoints"""
        console.print("📚 Testing API Documentation", style="bold blue")
        
        endpoints = [
            ('/docs', 'Swagger UI'),
            ('/redoc', 'ReDoc'),
            ('/openapi.json', 'OpenAPI Spec'),
        ]
        
        results = {}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for endpoint, description in endpoints:
                try:
                    response = await client.get(
                        f"{self.base_url}{endpoint}",
                        headers=self.get_test_headers()
                    )
                    
                    results[endpoint] = {
                        'status': 'passed' if response.status_code == 200 else 'failed',
                        'status_code': response.status_code,
                        'description': description
                    }
                    
                    if response.status_code == 200:
                        console.print(f"✅ {description}: Available", style="green")
                    else:
                        console.print(f"❌ {description}: {response.status_code}", style="red")
                        
                except Exception as e:
                    results[endpoint] = {
                        'status': 'error',
                        'error': str(e),
                        'description': description
                    }
                    console.print(f"❌ {description}: {e}", style="red")
        
        return results
    
    async def test_metrics_endpoints(self) -> Dict[str, Any]:
        """Test metrics and monitoring endpoints"""
        console.print("📊 Testing Metrics Endpoints", style="bold blue")
        
        endpoints = [
            ('/metrics', 'Prometheus Metrics'),
            ('/performance/metrics', 'Performance Metrics'),
        ]
        
        results = {}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for endpoint, description in endpoints:
                try:
                    response = await client.get(
                        f"{self.base_url}{endpoint}",
                        headers=self.get_test_headers()
                    )
                    
                    results[endpoint] = {
                        'status': 'passed' if response.status_code == 200 else 'failed',
                        'status_code': response.status_code,
                        'description': description
                    }
                    
                    if response.status_code == 200:
                        console.print(f"✅ {description}: Available", style="green")
                    else:
                        console.print(f"❌ {description}: {response.status_code}", style="red")
                        
                except Exception as e:
                    results[endpoint] = {
                        'status': 'error',
                        'error': str(e),
                        'description': description
                    }
                    console.print(f"❌ {description}: {e}", style="red")
        
        return results
    
    async def test_api_endpoints_basic(self) -> Dict[str, Any]:
        """Test basic API endpoints (without authentication)"""
        console.print("🔌 Testing Basic API Endpoints", style="bold blue")
        
        # These tests expect authentication failures (401/403) which indicates the endpoints exist
        endpoints = [
            ('/api/v1/employees/clients', 'POST', 'Client Creation'),
            ('/api/v1/employees/clients/test', 'GET', 'Client Retrieval'),
        ]
        
        results = {}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for endpoint, method, description in endpoints:
                try:
                    if method == 'GET':
                        response = await client.get(
                            f"{self.base_url}{endpoint}",
                            headers=self.get_test_headers()
                        )
                    elif method == 'POST':
                        response = await client.post(
                            f"{self.base_url}{endpoint}",
                            json={"test": "data"},
                            headers=self.get_test_headers()
                        )
                    
                    # For smoke tests, we expect 401/403 (auth required) or 422 (validation error)
                    # These indicate the endpoint exists and is processing requests
                    expected_codes = [401, 403, 422]
                    
                    results[f"{method} {endpoint}"] = {
                        'status': 'passed' if response.status_code in expected_codes else 'warning',
                        'status_code': response.status_code,
                        'description': description,
                        'note': 'Authentication/validation error expected'
                    }
                    
                    if response.status_code in expected_codes:
                        console.print(f"✅ {description}: Endpoint exists ({response.status_code})", style="green")
                    elif response.status_code == 200:
                        console.print(f"⚠️  {description}: Unexpected success (no auth required?)", style="yellow")
                    else:
                        console.print(f"❌ {description}: {response.status_code}", style="red")
                        
                except Exception as e:
                    results[f"{method} {endpoint}"] = {
                        'status': 'error',
                        'error': str(e),
                        'description': description
                    }
                    console.print(f"❌ {description}: {e}", style="red")
        
        return results
    
    async def test_cors_headers(self) -> Dict[str, Any]:
        """Test CORS headers configuration"""
        console.print("🌐 Testing CORS Configuration", style="bold blue")
        
        results = {}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                # Test preflight request
                response = await client.options(
                    f"{self.base_url}/api/v1/employees/clients",
                    headers={
                        **self.get_test_headers(),
                        "Origin": "https://example.com",
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "Content-Type"
                    }
                )
                
                cors_headers = {
                    'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
                    'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
                    'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers'),
                }
                
                results['cors_preflight'] = {
                    'status': 'passed' if response.status_code in [200, 204] else 'warning',
                    'status_code': response.status_code,
                    'headers': cors_headers,
                    'description': 'CORS Preflight Request'
                }
                
                if response.status_code in [200, 204]:
                    console.print("✅ CORS: Preflight request handled", style="green")
                else:
                    console.print(f"⚠️  CORS: Preflight returned {response.status_code}", style="yellow")
                    
            except Exception as e:
                results['cors_preflight'] = {
                    'status': 'error',
                    'error': str(e),
                    'description': 'CORS Preflight Request'
                }
                console.print(f"❌ CORS: {e}", style="red")
        
        return results
    
    async def test_security_headers(self) -> Dict[str, Any]:
        """Test security headers"""
        console.print("🔒 Testing Security Headers", style="bold blue")
        
        results = {}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(
                    f"{self.base_url}/health",
                    headers=self.get_test_headers()
                )
                
                security_headers = {
                    'X-Content-Type-Options': response.headers.get('X-Content-Type-Options'),
                    'X-Frame-Options': response.headers.get('X-Frame-Options'),
                    'X-XSS-Protection': response.headers.get('X-XSS-Protection'),
                    'Strict-Transport-Security': response.headers.get('Strict-Transport-Security'),
                }
                
                # Check for presence of security headers
                headers_present = sum(1 for v in security_headers.values() if v is not None)
                
                results['security_headers'] = {
                    'status': 'passed' if headers_present >= 2 else 'warning',
                    'headers_present': headers_present,
                    'headers': security_headers,
                    'description': 'Security Headers Check'
                }
                
                if headers_present >= 2:
                    console.print(f"✅ Security Headers: {headers_present}/4 present", style="green")
                else:
                    console.print(f"⚠️  Security Headers: Only {headers_present}/4 present", style="yellow")
                    
            except Exception as e:
                results['security_headers'] = {
                    'status': 'error',
                    'error': str(e),
                    'description': 'Security Headers Check'
                }
                console.print(f"❌ Security Headers: {e}", style="red")
        
        return results
    
    async def test_response_times(self) -> Dict[str, Any]:
        """Test response times for critical endpoints"""
        console.print("⚡ Testing Response Times", style="bold blue")
        
        endpoints = [
            '/health',
            '/ready',
            '/performance/health'
        ]
        
        results = {}
        response_times = []
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for endpoint in endpoints:
                try:
                    start_time = time.time()
                    response = await client.get(
                        f"{self.base_url}{endpoint}",
                        headers=self.get_test_headers()
                    )
                    response_time = (time.time() - start_time) * 1000
                    response_times.append(response_time)
                    
                    results[endpoint] = {
                        'status': 'passed' if response_time < 2000 else 'warning',
                        'response_time_ms': round(response_time, 2),
                        'status_code': response.status_code
                    }
                    
                    if response_time < 1000:
                        console.print(f"✅ {endpoint}: {response_time:.2f}ms", style="green")
                    elif response_time < 2000:
                        console.print(f"⚠️  {endpoint}: {response_time:.2f}ms", style="yellow")
                    else:
                        console.print(f"❌ {endpoint}: {response_time:.2f}ms (slow)", style="red")
                        
                except Exception as e:
                    results[endpoint] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    console.print(f"❌ {endpoint}: {e}", style="red")
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            results['summary'] = {
                'avg_response_time_ms': round(avg_response_time, 2),
                'total_endpoints_tested': len(response_times)
            }
        
        return results
    
    async def run_all_smoke_tests(self) -> Dict[str, Any]:
        """Run all smoke tests"""
        console.print(Panel(f"🚀 Running Smoke Tests - {self.environment.upper()}", style="bold blue"))
        
        test_suites = [
            ('Health Endpoints', self.test_health_endpoints()),
            ('API Documentation', self.test_api_documentation()),
            ('Metrics Endpoints', self.test_metrics_endpoints()),
            ('Basic API Endpoints', self.test_api_endpoints_basic()),
            ('CORS Configuration', self.test_cors_headers()),
            ('Security Headers', self.test_security_headers()),
            ('Response Times', self.test_response_times()),
        ]
        
        for suite_name, test_coro in test_suites:
            console.print(f"\n--- {suite_name} ---", style="bold")
            self.test_results[suite_name] = await test_coro
        
        # Calculate overall results
        total_duration = (time.time() - self.start_time) * 1000
        
        # Count passed/failed tests
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        warning_tests = 0
        
        for suite_results in self.test_results.values():
            for test_result in suite_results.values():
                if isinstance(test_result, dict) and 'status' in test_result:
                    total_tests += 1
                    if test_result['status'] == 'passed':
                        passed_tests += 1
                    elif test_result['status'] == 'failed' or test_result['status'] == 'error':
                        failed_tests += 1
                    elif test_result['status'] == 'warning':
                        warning_tests += 1
        
        overall_status = 'passed'
        if failed_tests > 0:
            overall_status = 'failed'
        elif warning_tests > 0:
            overall_status = 'warning'
        
        summary = {
            'overall_status': overall_status,
            'environment': self.environment,
            'total_duration_ms': total_duration,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'warning_tests': warning_tests,
            'failed_tests': failed_tests,
            'timestamp': time.time(),
            'test_results': self.test_results
        }
        
        return summary
    
    def display_results(self, results: Dict[str, Any]):
        """Display smoke test results"""
        overall_status = results.get('overall_status', 'unknown')
        status_color = {
            'passed': 'green',
            'warning': 'yellow',
            'failed': 'red',
            'unknown': 'blue'
        }.get(overall_status, 'blue')
        
        console.print(Panel(
            f"Smoke Tests: {overall_status.upper()}\n"
            f"Environment: {results.get('environment', 'unknown')}\n"
            f"Duration: {results.get('total_duration_ms', 0):.2f}ms\n"
            f"Tests: {results.get('passed_tests', 0)} passed, "
            f"{results.get('warning_tests', 0)} warnings, "
            f"{results.get('failed_tests', 0)} failed",
            style=f"bold {status_color}"
        ))
        
        # Summary table
        summary_table = Table(title="Smoke Test Summary")
        summary_table.add_column("Test Suite", style="cyan")
        summary_table.add_column("Status", style="green")
        summary_table.add_column("Tests", style="blue")
        summary_table.add_column("Issues", style="red")
        
        for suite_name, suite_results in results.get('test_results', {}).items():
            suite_tests = 0
            suite_passed = 0
            suite_failed = 0
            suite_warnings = 0
            
            for test_result in suite_results.values():
                if isinstance(test_result, dict) and 'status' in test_result:
                    suite_tests += 1
                    if test_result['status'] == 'passed':
                        suite_passed += 1
                    elif test_result['status'] in ['failed', 'error']:
                        suite_failed += 1
                    elif test_result['status'] == 'warning':
                        suite_warnings += 1
            
            suite_status = 'passed'
            if suite_failed > 0:
                suite_status = 'failed'
            elif suite_warnings > 0:
                suite_status = 'warning'
            
            status_style = {
                'passed': 'green',
                'warning': 'yellow',
                'failed': 'red'
            }.get(suite_status, 'blue')
            
            summary_table.add_row(
                suite_name,
                f"[{status_style}]{suite_status}[/{status_style}]",
                f"{suite_passed}/{suite_tests}",
                str(suite_failed + suite_warnings) if (suite_failed + suite_warnings) > 0 else "-"
            )
        
        console.print(summary_table)


@click.command()
@click.option('--url', default='http://localhost:8000', help='Base URL for smoke tests')
@click.option('--environment', default='development', help='Environment name')
@click.option('--output', help='Output file for results (JSON format)')
@click.option('--timeout', default=30, help='Request timeout in seconds')
def main(url, environment, output, timeout):
    """Run smoke tests for Employee Lifecycle System"""
    
    async def run_tests():
        runner = SmokeTestRunner(url, environment)
        
        try:
            results = await runner.run_all_smoke_tests()
            
            # Display results
            runner.display_results(results)
            
            # Save results if requested
            if output:
                with open(output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                console.print(f"Results saved to: {output}", style="blue")
            
            # Exit with appropriate code
            overall_status = results.get('overall_status', 'unknown')
            if overall_status == 'passed':
                sys.exit(0)
            elif overall_status == 'warning':
                sys.exit(1)
            else:
                sys.exit(2)
                
        except Exception as e:
            console.print(f"❌ Smoke tests failed: {e}", style="red")
            sys.exit(3)
    
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()