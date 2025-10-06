#!/usr/bin/env python3
"""
Health Check Script for Employee Lifecycle System

Comprehensive health checks for deployment validation and monitoring.
Validates all system components and dependencies.

Requirements: 5.3, 5.4, 8.5
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import click
import httpx
import asyncpg
import redis
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


class HealthChecker:
    """Comprehensive health checker for Employee Lifecycle System"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.results = {}
    
    async def check_api_health(self) -> Dict[str, Any]:
        """Check API health endpoints"""
        console.print("Checking API health...", style="blue")
        
        endpoints = [
            ('/health', 'Basic Health'),
            ('/ready', 'Readiness'),
            ('/performance/health', 'System Health'),
            ('/performance/metrics', 'Metrics'),
        ]
        
        results = {}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for endpoint, description in endpoints:
                try:
                    start_time = time.time()
                    response = await client.get(f"{self.base_url}{endpoint}")
                    response_time = (time.time() - start_time) * 1000
                    
                    results[endpoint] = {
                        'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                        'status_code': response.status_code,
                        'response_time_ms': round(response_time, 2),
                        'description': description
                    }
                    
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            results[endpoint]['data'] = data
                        except:
                            results[endpoint]['data'] = response.text[:100]
                    
                except Exception as e:
                    results[endpoint] = {
                        'status': 'error',
                        'error': str(e),
                        'description': description
                    }
        
        return results
    
    async def check_database_health(self, database_url: str) -> Dict[str, Any]:
        """Check database connectivity and health"""
        console.print("Checking database health...", style="blue")
        
        try:
            start_time = time.time()
            conn = await asyncpg.connect(database_url)
            connection_time = (time.time() - start_time) * 1000
            
            # Basic connectivity test
            await conn.fetchval('SELECT 1')
            
            # Check database version
            db_version = await conn.fetchval('SELECT version()')
            
            # Check table existence
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            
            # Check connection stats
            connection_stats = await conn.fetchrow("""
                SELECT 
                    count(*) as active_connections,
                    (SELECT setting::int FROM pg_settings WHERE name='max_connections') as max_connections
                FROM pg_stat_activity
            """)
            
            # Check database size
            db_size = await conn.fetchval("""
                SELECT pg_size_pretty(pg_database_size(current_database()))
            """)
            
            await conn.close()
            
            return {
                'status': 'healthy',
                'connection_time_ms': round(connection_time, 2),
                'version': db_version.split(' ')[1] if db_version else 'unknown',
                'tables_count': len(tables),
                'active_connections': connection_stats['active_connections'],
                'max_connections': connection_stats['max_connections'],
                'database_size': db_size,
                'connection_usage': round(
                    (connection_stats['active_connections'] / connection_stats['max_connections']) * 100, 2
                )
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def check_redis_health(self, redis_url: str) -> Dict[str, Any]:
        """Check Redis connectivity and health"""
        console.print("Checking Redis health...", style="blue")
        
        try:
            start_time = time.time()
            r = redis.from_url(redis_url)
            
            # Test connection
            await asyncio.get_event_loop().run_in_executor(None, r.ping)
            connection_time = (time.time() - start_time) * 1000
            
            # Get Redis info
            info = await asyncio.get_event_loop().run_in_executor(None, r.info)
            
            # Test basic operations
            test_key = f"health_check_{int(time.time())}"
            await asyncio.get_event_loop().run_in_executor(None, r.set, test_key, "test", 60)
            test_value = await asyncio.get_event_loop().run_in_executor(None, r.get, test_key)
            await asyncio.get_event_loop().run_in_executor(None, r.delete, test_key)
            
            r.close()
            
            return {
                'status': 'healthy',
                'connection_time_ms': round(connection_time, 2),
                'version': info.get('redis_version', 'unknown'),
                'memory_used': info.get('used_memory_human', 'unknown'),
                'memory_peak': info.get('used_memory_peak_human', 'unknown'),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'operations_test': 'passed' if test_value == b'test' else 'failed'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def check_external_services(self) -> Dict[str, Any]:
        """Check external service connectivity"""
        console.print("Checking external services...", style="blue")
        
        services = {
            'openai': 'https://api.openai.com/v1/models',
            'pinecone': None,  # Would need API key to test
        }
        
        results = {}
        
        async with httpx.AsyncClient(timeout=10) as client:
            for service, url in services.items():
                if url:
                    try:
                        start_time = time.time()
                        response = await client.get(url)
                        response_time = (time.time() - start_time) * 1000
                        
                        results[service] = {
                            'status': 'reachable' if response.status_code in [200, 401] else 'unreachable',
                            'status_code': response.status_code,
                            'response_time_ms': round(response_time, 2)
                        }
                        
                    except Exception as e:
                        results[service] = {
                            'status': 'error',
                            'error': str(e)
                        }
                else:
                    results[service] = {
                        'status': 'skipped',
                        'reason': 'No test URL configured'
                    }
        
        return results
    
    async def check_performance_metrics(self) -> Dict[str, Any]:
        """Check system performance metrics"""
        console.print("Checking performance metrics...", style="blue")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/performance/metrics")
                
                if response.status_code == 200:
                    metrics = response.json()
                    
                    # Analyze key metrics
                    analysis = {
                        'status': 'healthy',
                        'response_time_ok': metrics.get('avg_response_time_ms', 0) < 2000,
                        'error_rate_ok': metrics.get('error_rate', 0) < 0.01,
                        'cpu_usage_ok': metrics.get('cpu_usage', 0) < 0.8,
                        'memory_usage_ok': metrics.get('memory_usage', 0) < 0.8,
                        'metrics': metrics
                    }
                    
                    # Overall health based on thresholds
                    if not all([
                        analysis['response_time_ok'],
                        analysis['error_rate_ok'],
                        analysis['cpu_usage_ok'],
                        analysis['memory_usage_ok']
                    ]):
                        analysis['status'] = 'degraded'
                    
                    return analysis
                else:
                    return {
                        'status': 'error',
                        'error': f'HTTP {response.status_code}'
                    }
                    
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def run_functional_tests(self) -> Dict[str, Any]:
        """Run basic functional tests"""
        console.print("Running functional tests...", style="blue")
        
        tests = {}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Test 1: Create client (if test endpoint exists)
            try:
                test_client_data = {
                    "name": "Health Check Test Client",
                    "industry": "Technology",
                    "tier": "basic"
                }
                
                response = await client.post(
                    f"{self.base_url}/api/v1/employees/clients",
                    json=test_client_data,
                    headers={
                        "Authorization": "Bearer health_check_token",
                        "X-Tenant-ID": "health_check",
                        "X-Client-ID": "health_check",
                        "X-User-ID": "health_check"
                    }
                )
                
                tests['create_client'] = {
                    'status': 'passed' if response.status_code in [200, 201, 401, 403] else 'failed',
                    'status_code': response.status_code,
                    'note': 'Authentication expected to fail in health check'
                }
                
            except Exception as e:
                tests['create_client'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # Test 2: API documentation endpoint
            try:
                response = await client.get(f"{self.base_url}/docs")
                tests['api_docs'] = {
                    'status': 'passed' if response.status_code == 200 else 'failed',
                    'status_code': response.status_code
                }
            except Exception as e:
                tests['api_docs'] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return tests
    
    def calculate_overall_health(self) -> str:
        """Calculate overall system health score"""
        critical_components = ['api_health', 'database_health', 'redis_health']
        critical_healthy = 0
        
        for component in critical_components:
            if component in self.results:
                if component == 'api_health':
                    # Check if basic health endpoint is working
                    health_status = self.results[component].get('/health', {}).get('status')
                    if health_status == 'healthy':
                        critical_healthy += 1
                else:
                    if self.results[component].get('status') == 'healthy':
                        critical_healthy += 1
        
        if critical_healthy == len(critical_components):
            return 'healthy'
        elif critical_healthy >= len(critical_components) - 1:
            return 'degraded'
        else:
            return 'unhealthy'
    
    async def run_all_checks(self, database_url: Optional[str] = None, 
                           redis_url: Optional[str] = None) -> Dict[str, Any]:
        """Run all health checks"""
        console.print(Panel("🏥 Running Health Checks", style="bold blue"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # API Health Check
            task = progress.add_task("Checking API health...", total=None)
            self.results['api_health'] = await self.check_api_health()
            progress.update(task, description="✅ API health checked")
            
            # Database Health Check
            if database_url:
                progress.update(task, description="Checking database health...")
                self.results['database_health'] = await self.check_database_health(database_url)
                progress.update(task, description="✅ Database health checked")
            
            # Redis Health Check
            if redis_url:
                progress.update(task, description="Checking Redis health...")
                self.results['redis_health'] = await self.check_redis_health(redis_url)
                progress.update(task, description="✅ Redis health checked")
            
            # External Services Check
            progress.update(task, description="Checking external services...")
            self.results['external_services'] = await self.check_external_services()
            progress.update(task, description="✅ External services checked")
            
            # Performance Metrics Check
            progress.update(task, description="Checking performance metrics...")
            self.results['performance_metrics'] = await self.check_performance_metrics()
            progress.update(task, description="✅ Performance metrics checked")
            
            # Functional Tests
            progress.update(task, description="Running functional tests...")
            self.results['functional_tests'] = await self.run_functional_tests()
            progress.update(task, description="✅ Functional tests completed")
            
            progress.update(task, description="Health checks completed")
        
        # Calculate overall health
        self.results['overall_health'] = self.calculate_overall_health()
        self.results['timestamp'] = datetime.now().isoformat()
        
        return self.results
    
    def display_results(self):
        """Display health check results in a formatted table"""
        # Overall Status
        overall_health = self.results.get('overall_health', 'unknown')
        status_color = {
            'healthy': 'green',
            'degraded': 'yellow',
            'unhealthy': 'red',
            'unknown': 'blue'
        }.get(overall_health, 'blue')
        
        console.print(Panel(
            f"Overall System Health: {overall_health.upper()}",
            style=f"bold {status_color}"
        ))
        
        # API Health Table
        if 'api_health' in self.results:
            api_table = Table(title="API Health")
            api_table.add_column("Endpoint", style="cyan")
            api_table.add_column("Status", style="green")
            api_table.add_column("Response Time", style="blue")
            api_table.add_column("Status Code", style="white")
            
            for endpoint, data in self.results['api_health'].items():
                status = data.get('status', 'unknown')
                status_style = 'green' if status == 'healthy' else 'red'
                
                api_table.add_row(
                    endpoint,
                    f"[{status_style}]{status}[/{status_style}]",
                    f"{data.get('response_time_ms', 'N/A')}ms",
                    str(data.get('status_code', 'N/A'))
                )
            
            console.print(api_table)
        
        # Component Health Table
        components_table = Table(title="Component Health")
        components_table.add_column("Component", style="cyan")
        components_table.add_column("Status", style="green")
        components_table.add_column("Details", style="white")
        
        component_mapping = {
            'database_health': 'Database',
            'redis_health': 'Redis',
            'external_services': 'External Services',
            'performance_metrics': 'Performance',
            'functional_tests': 'Functional Tests'
        }
        
        for key, name in component_mapping.items():
            if key in self.results:
                data = self.results[key]
                status = data.get('status', 'unknown')
                status_style = {
                    'healthy': 'green',
                    'degraded': 'yellow',
                    'unhealthy': 'red',
                    'error': 'red',
                    'passed': 'green',
                    'failed': 'red'
                }.get(status, 'blue')
                
                # Format details based on component
                details = []
                if key == 'database_health' and status == 'healthy':
                    details.append(f"Connections: {data.get('active_connections')}/{data.get('max_connections')}")
                    details.append(f"Size: {data.get('database_size')}")
                elif key == 'redis_health' and status == 'healthy':
                    details.append(f"Memory: {data.get('memory_used')}")
                    details.append(f"Clients: {data.get('connected_clients')}")
                elif key == 'performance_metrics' and status in ['healthy', 'degraded']:
                    metrics = data.get('metrics', {})
                    details.append(f"Avg Response: {metrics.get('avg_response_time_ms', 'N/A')}ms")
                    details.append(f"Error Rate: {metrics.get('error_rate', 'N/A')}")
                
                if data.get('error'):
                    details.append(f"Error: {data['error']}")
                
                components_table.add_row(
                    name,
                    f"[{status_style}]{status}[/{status_style}]",
                    " | ".join(details) if details else "OK"
                )
        
        console.print(components_table)
    
    def save_results(self, output_file: str):
        """Save results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        console.print(f"Results saved to: {output_file}", style="blue")


# CLI Commands
@click.command()
@click.option('--url', default='http://localhost:8000', help='Base URL for health checks')
@click.option('--database-url', help='Database URL for database health check')
@click.option('--redis-url', help='Redis URL for Redis health check')
@click.option('--timeout', default=30, help='Request timeout in seconds')
@click.option('--output', help='Output file for results (JSON format)')
@click.option('--exit-code', is_flag=True, help='Exit with non-zero code if unhealthy')
def main(url, database_url, redis_url, timeout, output, exit_code):
    """Run comprehensive health checks for Employee Lifecycle System"""
    
    async def run_checks():
        checker = HealthChecker(url, timeout)
        results = await checker.run_all_checks(database_url, redis_url)
        
        # Display results
        checker.display_results()
        
        # Save results if requested
        if output:
            checker.save_results(output)
        
        # Exit with appropriate code
        if exit_code:
            overall_health = results.get('overall_health', 'unknown')
            if overall_health == 'healthy':
                sys.exit(0)
            elif overall_health == 'degraded':
                sys.exit(1)
            else:
                sys.exit(2)
    
    asyncio.run(run_checks())


if __name__ == "__main__":
    main()