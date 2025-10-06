#!/usr/bin/env python3
"""
System Validation Script for Employee Lifecycle System

Comprehensive system validation including data integrity,
performance benchmarks, and security checks.

Requirements: 1.5, 2.5, 3.4, 5.5, 6.5
"""

import asyncio
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

import click
import asyncpg
import redis
import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


class SystemValidator:
    """Comprehensive system validator"""
    
    def __init__(self, database_url: str, redis_url: str, api_url: str):
        self.database_url = database_url
        self.redis_url = redis_url
        self.api_url = api_url.rstrip('/')
        self.validation_results = {}
        self.start_time = time.time()
    
    async def validate_database_integrity(self) -> Dict[str, Any]:
        """Validate database schema and data integrity"""
        console.print("🗄️  Validating Database Integrity", style="bold blue")
        
        results = {
            'status': 'passed',
            'checks': {},
            'errors': []
        }
        
        try:
            conn = await asyncpg.connect(self.database_url)
            
            # Check 1: Required tables exist
            console.print("Checking required tables...", style="blue")
            
            required_tables = [
                'clients', 'employees', 'interactions', 'memories',
                'knowledge_sources', 'analytics_events', 'performance_metrics',
                'audit_logs', 'schema_migrations'
            ]
            
            existing_tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            
            existing_table_names = {row['table_name'] for row in existing_tables}
            missing_tables = set(required_tables) - existing_table_names
            
            if missing_tables:
                results['status'] = 'failed'
                results['errors'].append(f"Missing tables: {missing_tables}")
                console.print(f"❌ Missing tables: {missing_tables}", style="red")
            else:
                results['checks']['required_tables'] = 'passed'
                console.print("✅ All required tables exist", style="green")
            
            # Check 2: Foreign key constraints
            console.print("Checking foreign key constraints...", style="blue")
            
            fk_constraints = await conn.fetch("""
                SELECT 
                    tc.table_name,
                    tc.constraint_name,
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                    AND tc.table_schema = 'public'
            """)
            
            if len(fk_constraints) >= 5:  # Should have several FK constraints
                results['checks']['foreign_keys'] = 'passed'
                console.print(f"✅ Foreign key constraints: {len(fk_constraints)} found", style="green")
            else:
                results['status'] = 'warning'
                results['errors'].append(f"Few foreign key constraints: {len(fk_constraints)}")
                console.print(f"⚠️  Only {len(fk_constraints)} foreign key constraints", style="yellow")
            
            # Check 3: Indexes exist
            console.print("Checking database indexes...", style="blue")
            
            indexes = await conn.fetch("""
                SELECT indexname, tablename
                FROM pg_indexes
                WHERE schemaname = 'public'
                    AND indexname LIKE 'idx_%'
            """)
            
            if len(indexes) >= 10:  # Should have many performance indexes
                results['checks']['indexes'] = 'passed'
                console.print(f"✅ Performance indexes: {len(indexes)} found", style="green")
            else:
                results['status'] = 'warning'
                results['errors'].append(f"Few performance indexes: {len(indexes)}")
                console.print(f"⚠️  Only {len(indexes)} performance indexes", style="yellow")
            
            # Check 4: RLS policies
            console.print("Checking Row Level Security policies...", style="blue")
            
            rls_policies = await conn.fetch("""
                SELECT tablename, policyname
                FROM pg_policies
                WHERE schemaname = 'public'
            """)
            
            if len(rls_policies) >= 8:  # Should have tenant isolation policies
                results['checks']['rls_policies'] = 'passed'
                console.print(f"✅ RLS policies: {len(rls_policies)} found", style="green")
            else:
                results['status'] = 'warning'
                results['errors'].append(f"Few RLS policies: {len(rls_policies)}")
                console.print(f"⚠️  Only {len(rls_policies)} RLS policies", style="yellow")
            
            # Check 5: Data consistency
            console.print("Checking data consistency...", style="blue")
            
            # Check for orphaned employees
            orphaned_employees = await conn.fetchval("""
                SELECT COUNT(*)
                FROM employees e
                LEFT JOIN clients c ON e.client_id = c.id
                WHERE c.id IS NULL
            """)
            
            # Check for orphaned interactions
            orphaned_interactions = await conn.fetchval("""
                SELECT COUNT(*)
                FROM interactions i
                LEFT JOIN employees e ON i.employee_id = e.id
                WHERE e.id IS NULL
            """)
            
            # Check for orphaned memories
            orphaned_memories = await conn.fetchval("""
                SELECT COUNT(*)
                FROM memories m
                LEFT JOIN employees e ON m.employee_id = e.id
                WHERE e.id IS NULL
            """)
            
            total_orphaned = orphaned_employees + orphaned_interactions + orphaned_memories
            
            if total_orphaned == 0:
                results['checks']['data_consistency'] = 'passed'
                console.print("✅ No orphaned records found", style="green")
            else:
                results['status'] = 'warning'
                results['errors'].append(f"Orphaned records: {total_orphaned}")
                console.print(f"⚠️  Found {total_orphaned} orphaned records", style="yellow")
            
            await conn.close()
            
        except Exception as e:
            results['status'] = 'failed'
            results['errors'].append(f"Database validation error: {e}")
            console.print(f"❌ Database validation failed: {e}", style="red")
        
        return results
    
    async def validate_redis_performance(self) -> Dict[str, Any]:
        """Validate Redis performance and configuration"""
        console.print("🔴 Validating Redis Performance", style="bold blue")
        
        results = {
            'status': 'passed',
            'checks': {},
            'errors': [],
            'metrics': {}
        }
        
        try:
            r = redis.from_url(self.redis_url)
            
            # Check 1: Basic connectivity
            console.print("Testing Redis connectivity...", style="blue")
            
            start_time = time.time()
            await asyncio.get_event_loop().run_in_executor(None, r.ping)
            ping_time = (time.time() - start_time) * 1000
            
            results['metrics']['ping_time_ms'] = round(ping_time, 2)
            
            if ping_time < 10:
                results['checks']['connectivity'] = 'passed'
                console.print(f"✅ Redis ping: {ping_time:.2f}ms", style="green")
            else:
                results['status'] = 'warning'
                results['errors'].append(f"Slow Redis ping: {ping_time:.2f}ms")
                console.print(f"⚠️  Slow Redis ping: {ping_time:.2f}ms", style="yellow")
            
            # Check 2: Memory usage
            console.print("Checking Redis memory usage...", style="blue")
            
            info = await asyncio.get_event_loop().run_in_executor(None, r.info, 'memory')
            
            used_memory = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            
            results['metrics']['used_memory_mb'] = round(used_memory / 1024 / 1024, 2)
            results['metrics']['max_memory_mb'] = round(max_memory / 1024 / 1024, 2) if max_memory > 0 else 'unlimited'
            
            if max_memory > 0:
                memory_usage_pct = (used_memory / max_memory) * 100
                results['metrics']['memory_usage_pct'] = round(memory_usage_pct, 2)
                
                if memory_usage_pct < 80:
                    results['checks']['memory_usage'] = 'passed'
                    console.print(f"✅ Memory usage: {memory_usage_pct:.1f}%", style="green")
                else:
                    results['status'] = 'warning'
                    results['errors'].append(f"High memory usage: {memory_usage_pct:.1f}%")
                    console.print(f"⚠️  High memory usage: {memory_usage_pct:.1f}%", style="yellow")
            else:
                results['checks']['memory_usage'] = 'passed'
                console.print("✅ Memory usage: No limit set", style="green")
            
            # Check 3: Performance test
            console.print("Running Redis performance test...", style="blue")
            
            test_operations = 100
            test_key_prefix = f"perf_test_{int(time.time())}"
            
            # SET operations
            set_start = time.time()
            for i in range(test_operations):
                await asyncio.get_event_loop().run_in_executor(
                    None, r.set, f"{test_key_prefix}:{i}", f"value_{i}", 60
                )
            set_duration = (time.time() - set_start) * 1000
            
            # GET operations
            get_start = time.time()
            for i in range(test_operations):
                await asyncio.get_event_loop().run_in_executor(
                    None, r.get, f"{test_key_prefix}:{i}"
                )
            get_duration = (time.time() - get_start) * 1000
            
            # Cleanup
            for i in range(test_operations):
                await asyncio.get_event_loop().run_in_executor(
                    None, r.delete, f"{test_key_prefix}:{i}"
                )
            
            avg_set_time = set_duration / test_operations
            avg_get_time = get_duration / test_operations
            
            results['metrics']['avg_set_time_ms'] = round(avg_set_time, 3)
            results['metrics']['avg_get_time_ms'] = round(avg_get_time, 3)
            
            if avg_set_time < 5 and avg_get_time < 5:
                results['checks']['performance'] = 'passed'
                console.print(f"✅ Performance: SET {avg_set_time:.3f}ms, GET {avg_get_time:.3f}ms", style="green")
            else:
                results['status'] = 'warning'
                results['errors'].append(f"Slow Redis operations: SET {avg_set_time:.3f}ms, GET {avg_get_time:.3f}ms")
                console.print(f"⚠️  Slow operations: SET {avg_set_time:.3f}ms, GET {avg_get_time:.3f}ms", style="yellow")
            
            r.close()
            
        except Exception as e:
            results['status'] = 'failed'
            results['errors'].append(f"Redis validation error: {e}")
            console.print(f"❌ Redis validation failed: {e}", style="red")
        
        return results
    
    async def validate_api_performance(self) -> Dict[str, Any]:
        """Validate API performance and functionality"""
        console.print("🌐 Validating API Performance", style="bold blue")
        
        results = {
            'status': 'passed',
            'checks': {},
            'errors': [],
            'metrics': {}
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                
                # Check 1: Health endpoint performance
                console.print("Testing health endpoint performance...", style="blue")
                
                health_times = []
                for i in range(10):
                    start_time = time.time()
                    response = await client.get(f"{self.api_url}/health")
                    response_time = (time.time() - start_time) * 1000
                    health_times.append(response_time)
                    
                    if response.status_code != 200:
                        results['status'] = 'failed'
                        results['errors'].append(f"Health endpoint failed: {response.status_code}")
                
                avg_health_time = sum(health_times) / len(health_times)
                max_health_time = max(health_times)
                min_health_time = min(health_times)
                
                results['metrics']['health_avg_ms'] = round(avg_health_time, 2)
                results['metrics']['health_max_ms'] = round(max_health_time, 2)
                results['metrics']['health_min_ms'] = round(min_health_time, 2)
                
                if avg_health_time < 100:
                    results['checks']['health_performance'] = 'passed'
                    console.print(f"✅ Health endpoint: avg {avg_health_time:.2f}ms", style="green")
                else:
                    results['status'] = 'warning'
                    results['errors'].append(f"Slow health endpoint: {avg_health_time:.2f}ms")
                    console.print(f"⚠️  Slow health endpoint: {avg_health_time:.2f}ms", style="yellow")
                
                # Check 2: Concurrent requests
                console.print("Testing concurrent request handling...", style="blue")
                
                concurrent_requests = 20
                concurrent_tasks = []
                
                for i in range(concurrent_requests):
                    task = client.get(f"{self.api_url}/health")
                    concurrent_tasks.append(task)
                
                concurrent_start = time.time()
                responses = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
                concurrent_duration = (time.time() - concurrent_start) * 1000
                
                successful_responses = sum(
                    1 for r in responses 
                    if isinstance(r, httpx.Response) and r.status_code == 200
                )
                
                results['metrics']['concurrent_requests'] = concurrent_requests
                results['metrics']['concurrent_successful'] = successful_responses
                results['metrics']['concurrent_duration_ms'] = round(concurrent_duration, 2)
                results['metrics']['concurrent_rps'] = round(concurrent_requests / (concurrent_duration / 1000), 2)
                
                if successful_responses >= concurrent_requests * 0.9:  # 90% success rate
                    results['checks']['concurrent_handling'] = 'passed'
                    console.print(f"✅ Concurrent requests: {successful_responses}/{concurrent_requests} successful", style="green")
                else:
                    results['status'] = 'warning'
                    results['errors'].append(f"Poor concurrent performance: {successful_responses}/{concurrent_requests}")
                    console.print(f"⚠️  Concurrent issues: {successful_responses}/{concurrent_requests}", style="yellow")
                
                # Check 3: API documentation availability
                console.print("Checking API documentation...", style="blue")
                
                docs_endpoints = ['/docs', '/redoc', '/openapi.json']
                docs_available = 0
                
                for endpoint in docs_endpoints:
                    try:
                        response = await client.get(f"{self.api_url}{endpoint}")
                        if response.status_code == 200:
                            docs_available += 1
                    except:
                        pass
                
                results['metrics']['docs_endpoints_available'] = docs_available
                
                if docs_available >= 2:
                    results['checks']['documentation'] = 'passed'
                    console.print(f"✅ API documentation: {docs_available}/3 endpoints available", style="green")
                else:
                    results['status'] = 'warning'
                    results['errors'].append(f"Limited API documentation: {docs_available}/3")
                    console.print(f"⚠️  Limited documentation: {docs_available}/3", style="yellow")
                
        except Exception as e:
            results['status'] = 'failed'
            results['errors'].append(f"API validation error: {e}")
            console.print(f"❌ API validation failed: {e}", style="red")
        
        return results
    
    async def validate_security_configuration(self) -> Dict[str, Any]:
        """Validate security configuration"""
        console.print("🔒 Validating Security Configuration", style="bold blue")
        
        results = {
            'status': 'passed',
            'checks': {},
            'errors': [],
            'findings': []
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                
                # Check 1: Security headers
                console.print("Checking security headers...", style="blue")
                
                response = await client.get(f"{self.api_url}/health")
                
                security_headers = {
                    'X-Content-Type-Options': response.headers.get('X-Content-Type-Options'),
                    'X-Frame-Options': response.headers.get('X-Frame-Options'),
                    'X-XSS-Protection': response.headers.get('X-XSS-Protection'),
                    'Strict-Transport-Security': response.headers.get('Strict-Transport-Security'),
                    'Content-Security-Policy': response.headers.get('Content-Security-Policy'),
                }
                
                headers_present = sum(1 for v in security_headers.values() if v is not None)
                results['findings'].append(f"Security headers present: {headers_present}/5")
                
                if headers_present >= 3:
                    results['checks']['security_headers'] = 'passed'
                    console.print(f"✅ Security headers: {headers_present}/5 present", style="green")
                else:
                    results['status'] = 'warning'
                    results['errors'].append(f"Few security headers: {headers_present}/5")
                    console.print(f"⚠️  Few security headers: {headers_present}/5", style="yellow")
                
                # Check 2: HTTPS enforcement (if applicable)
                if self.api_url.startswith('https://'):
                    console.print("Checking HTTPS configuration...", style="blue")
                    
                    # Check if HTTP redirects to HTTPS
                    http_url = self.api_url.replace('https://', 'http://')
                    try:
                        http_response = await client.get(f"{http_url}/health", follow_redirects=False)
                        if http_response.status_code in [301, 302, 307, 308]:
                            location = http_response.headers.get('location', '')
                            if location.startswith('https://'):
                                results['checks']['https_redirect'] = 'passed'
                                console.print("✅ HTTP to HTTPS redirect configured", style="green")
                            else:
                                results['status'] = 'warning'
                                results['errors'].append("HTTP redirect not to HTTPS")
                                console.print("⚠️  HTTP redirect not to HTTPS", style="yellow")
                        else:
                            results['status'] = 'warning'
                            results['errors'].append("No HTTP to HTTPS redirect")
                            console.print("⚠️  No HTTP to HTTPS redirect", style="yellow")
                    except:
                        # HTTP might not be available, which is good
                        results['checks']['https_redirect'] = 'passed'
                        console.print("✅ HTTP not accessible (HTTPS only)", style="green")
                
                # Check 3: Authentication requirements
                console.print("Checking authentication requirements...", style="blue")
                
                protected_endpoints = [
                    '/api/v1/employees/clients',
                    '/api/v1/employees/clients/test/employees'
                ]
                
                auth_protected = 0
                for endpoint in protected_endpoints:
                    try:
                        response = await client.get(f"{self.api_url}{endpoint}")
                        if response.status_code in [401, 403]:
                            auth_protected += 1
                    except:
                        pass
                
                if auth_protected >= len(protected_endpoints) * 0.8:  # 80% should be protected
                    results['checks']['authentication'] = 'passed'
                    console.print(f"✅ Authentication: {auth_protected}/{len(protected_endpoints)} endpoints protected", style="green")
                else:
                    results['status'] = 'warning'
                    results['errors'].append(f"Weak authentication: {auth_protected}/{len(protected_endpoints)}")
                    console.print(f"⚠️  Weak authentication: {auth_protected}/{len(protected_endpoints)}", style="yellow")
                
        except Exception as e:
            results['status'] = 'failed'
            results['errors'].append(f"Security validation error: {e}")
            console.print(f"❌ Security validation failed: {e}", style="red")
        
        return results
    
    async def run_complete_validation(self) -> Dict[str, Any]:
        """Run complete system validation"""
        console.print(Panel("🔍 Starting Complete System Validation", style="bold blue"))
        
        validation_suites = [
            ('Database Integrity', self.validate_database_integrity()),
            ('Redis Performance', self.validate_redis_performance()),
            ('API Performance', self.validate_api_performance()),
            ('Security Configuration', self.validate_security_configuration()),
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running validations...", total=len(validation_suites))
            
            for suite_name, validation_coro in validation_suites:
                progress.update(task, description=f"Validating {suite_name}...")
                self.validation_results[suite_name] = await validation_coro
                progress.advance(task)
        
        # Calculate overall results
        total_duration = (time.time() - self.start_time) * 1000
        
        passed_suites = sum(1 for result in self.validation_results.values() if result['status'] == 'passed')
        warning_suites = sum(1 for result in self.validation_results.values() if result['status'] == 'warning')
        failed_suites = sum(1 for result in self.validation_results.values() if result['status'] == 'failed')
        
        overall_status = 'passed'
        if failed_suites > 0:
            overall_status = 'failed'
        elif warning_suites > 0:
            overall_status = 'warning'
        
        summary = {
            'overall_status': overall_status,
            'total_duration_ms': total_duration,
            'suites_tested': len(validation_suites),
            'suites_passed': passed_suites,
            'suites_warning': warning_suites,
            'suites_failed': failed_suites,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'validation_results': self.validation_results
        }
        
        return summary
    
    def display_results(self, results: Dict[str, Any]):
        """Display validation results"""
        overall_status = results.get('overall_status', 'unknown')
        status_color = {
            'passed': 'green',
            'warning': 'yellow',
            'failed': 'red',
            'unknown': 'blue'
        }.get(overall_status, 'blue')
        
        console.print(Panel(
            f"System Validation: {overall_status.upper()}\n"
            f"Duration: {results.get('total_duration_ms', 0):.2f}ms\n"
            f"Suites: {results.get('suites_passed', 0)} passed, "
            f"{results.get('suites_warning', 0)} warnings, "
            f"{results.get('suites_failed', 0)} failed",
            style=f"bold {status_color}"
        ))
        
        # Results table
        results_table = Table(title="Validation Results")
        results_table.add_column("Validation Suite", style="cyan")
        results_table.add_column("Status", style="green")
        results_table.add_column("Checks", style="blue")
        results_table.add_column("Issues", style="red")
        
        for suite_name, suite_data in results.get('validation_results', {}).items():
            status = suite_data.get('status', 'unknown')
            status_style = {
                'passed': 'green',
                'warning': 'yellow',
                'failed': 'red'
            }.get(status, 'blue')
            
            checks_count = len(suite_data.get('checks', {}))
            errors_count = len(suite_data.get('errors', []))
            
            results_table.add_row(
                suite_name,
                f"[{status_style}]{status}[/{status_style}]",
                str(checks_count),
                str(errors_count) if errors_count > 0 else "-"
            )
        
        console.print(results_table)
        
        # Show errors for failed/warning suites
        for suite_name, suite_data in results.get('validation_results', {}).items():
            if suite_data.get('status') in ['failed', 'warning'] and suite_data.get('errors'):
                console.print(f"\n[bold]{suite_name} Issues:[/bold]")
                for error in suite_data['errors']:
                    console.print(f"  • {error}", style="red" if suite_data['status'] == 'failed' else "yellow")


@click.command()
@click.option('--database-url', required=True, help='Database URL')
@click.option('--redis-url', required=True, help='Redis URL')
@click.option('--api-url', default='http://localhost:8000', help='API URL')
@click.option('--output', help='Output file for results (JSON format)')
def main(database_url, redis_url, api_url, output):
    """Run comprehensive system validation"""
    
    async def run_validation():
        validator = SystemValidator(database_url, redis_url, api_url)
        
        try:
            results = await validator.run_complete_validation()
            
            # Display results
            validator.display_results(results)
            
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
            console.print(f"❌ System validation failed: {e}", style="red")
            sys.exit(3)
    
    asyncio.run(run_validation())


if __name__ == "__main__":
    main()