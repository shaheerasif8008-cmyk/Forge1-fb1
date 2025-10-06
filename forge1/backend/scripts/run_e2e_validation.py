#!/usr/bin/env python3
"""
End-to-End Validation Runner

Orchestrates complete end-to-end validation of the Employee Lifecycle System
including workflow validation, performance testing, and system verification.

Requirements: 1.5, 2.5, 3.4, 5.5, 6.5
"""

import asyncio
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


class E2EValidationRunner:
    """Orchestrates complete end-to-end validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        self.start_time = time.time()
        self.project_root = Path(__file__).parent.parent
    
    async def run_smoke_tests(self) -> Dict[str, Any]:
        """Run smoke tests"""
        console.print("🚬 Running Smoke Tests", style="bold blue")
        
        try:
            smoke_script = self.project_root / 'tests' / 'smoke_tests.py'
            
            cmd = [
                sys.executable, str(smoke_script),
                '--url', self.config['api_url'],
                '--environment', self.config['environment'],
                '--output', str(self.project_root / 'test_results' / 'smoke_results.json')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Load results
            results_file = self.project_root / 'test_results' / 'smoke_results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    smoke_results = json.load(f)
            else:
                smoke_results = {'overall_status': 'failed', 'error': 'No results file generated'}
            
            smoke_results['exit_code'] = result.returncode
            smoke_results['stdout'] = result.stdout
            smoke_results['stderr'] = result.stderr
            
            if result.returncode == 0:
                console.print("✅ Smoke tests passed", style="green")
            else:
                console.print(f"❌ Smoke tests failed (exit code: {result.returncode})", style="red")
            
            return smoke_results
            
        except Exception as e:
            console.print(f"❌ Smoke tests error: {e}", style="red")
            return {
                'overall_status': 'error',
                'error': str(e)
            }
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run comprehensive health checks"""
        console.print("🏥 Running Health Checks", style="bold blue")
        
        try:
            health_script = self.project_root / 'scripts' / 'health_check.py'
            
            cmd = [
                sys.executable, str(health_script),
                '--url', self.config['api_url'],
                '--timeout', str(self.config.get('timeout', 30)),
                '--output', str(self.project_root / 'test_results' / 'health_results.json'),
                '--exit-code'
            ]
            
            # Add database and Redis URLs if provided
            if self.config.get('database_url'):
                cmd.extend(['--database-url', self.config['database_url']])
            
            if self.config.get('redis_url'):
                cmd.extend(['--redis-url', self.config['redis_url']])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Load results
            results_file = self.project_root / 'test_results' / 'health_results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    health_results = json.load(f)
            else:
                health_results = {'overall_health': 'unknown', 'error': 'No results file generated'}
            
            health_results['exit_code'] = result.returncode
            
            if result.returncode == 0:
                console.print("✅ Health checks passed", style="green")
            elif result.returncode == 1:
                console.print("⚠️  Health checks passed with warnings", style="yellow")
            else:
                console.print(f"❌ Health checks failed (exit code: {result.returncode})", style="red")
            
            return health_results
            
        except Exception as e:
            console.print(f"❌ Health checks error: {e}", style="red")
            return {
                'overall_health': 'error',
                'error': str(e)
            }
    
    async def run_system_validation(self) -> Dict[str, Any]:
        """Run system validation"""
        console.print("🔍 Running System Validation", style="bold blue")
        
        try:
            validation_script = self.project_root / 'scripts' / 'validate_system.py'
            
            cmd = [
                sys.executable, str(validation_script),
                '--database-url', self.config['database_url'],
                '--redis-url', self.config['redis_url'],
                '--api-url', self.config['api_url'],
                '--output', str(self.project_root / 'test_results' / 'validation_results.json')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # Load results
            results_file = self.project_root / 'test_results' / 'validation_results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    validation_results = json.load(f)
            else:
                validation_results = {'overall_status': 'unknown', 'error': 'No results file generated'}
            
            validation_results['exit_code'] = result.returncode
            
            if result.returncode == 0:
                console.print("✅ System validation passed", style="green")
            elif result.returncode == 1:
                console.print("⚠️  System validation passed with warnings", style="yellow")
            else:
                console.print(f"❌ System validation failed (exit code: {result.returncode})", style="red")
            
            return validation_results
            
        except Exception as e:
            console.print(f"❌ System validation error: {e}", style="red")
            return {
                'overall_status': 'error',
                'error': str(e)
            }
    
    async def run_workflow_validation(self) -> Dict[str, Any]:
        """Run complete workflow validation"""
        console.print("🔄 Running Workflow Validation", style="bold blue")
        
        try:
            workflow_script = self.project_root / 'tests' / 'e2e' / 'test_complete_workflow_validation.py'
            
            cmd = [
                sys.executable, str(workflow_script),
                '--url', self.config['api_url'],
                '--timeout', str(self.config.get('timeout', 60)),
                '--output', str(self.project_root / 'test_results' / 'workflow_results.json')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)  # 20 minutes
            
            # Load results
            results_file = self.project_root / 'test_results' / 'workflow_results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    workflow_results = json.load(f)
            else:
                workflow_results = {'overall_status': 'unknown', 'error': 'No results file generated'}
            
            workflow_results['exit_code'] = result.returncode
            
            if result.returncode == 0:
                console.print("✅ Workflow validation passed", style="green")
            elif result.returncode == 1:
                console.print("⚠️  Workflow validation passed with warnings", style="yellow")
            else:
                console.print(f"❌ Workflow validation failed (exit code: {result.returncode})", style="red")
            
            return workflow_results
            
        except Exception as e:
            console.print(f"❌ Workflow validation error: {e}", style="red")
            return {
                'overall_status': 'error',
                'error': str(e)
            }
    
    async def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests"""
        console.print("⚡ Running Performance Tests", style="bold blue")
        
        try:
            # Run pytest performance tests
            cmd = [
                sys.executable, '-m', 'pytest',
                str(self.project_root / 'tests'),
                '-m', 'performance',
                '--tb=short',
                '--json-report',
                '--json-report-file', str(self.project_root / 'test_results' / 'performance_results.json')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # Load results
            results_file = self.project_root / 'test_results' / 'performance_results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    performance_results = json.load(f)
            else:
                performance_results = {'summary': {'total': 0, 'passed': 0, 'failed': 0}}
            
            performance_results['exit_code'] = result.returncode
            
            if result.returncode == 0:
                console.print("✅ Performance tests passed", style="green")
            else:
                console.print(f"❌ Performance tests failed (exit code: {result.returncode})", style="red")
            
            return performance_results
            
        except Exception as e:
            console.print(f"❌ Performance tests error: {e}", style="red")
            return {
                'summary': {'total': 0, 'passed': 0, 'failed': 1},
                'error': str(e)
            }
    
    async def run_security_tests(self) -> Dict[str, Any]:
        """Run security tests"""
        console.print("🔒 Running Security Tests", style="bold blue")
        
        try:
            # Run pytest security tests
            cmd = [
                sys.executable, '-m', 'pytest',
                str(self.project_root / 'tests'),
                '-m', 'security',
                '--tb=short',
                '--json-report',
                '--json-report-file', str(self.project_root / 'test_results' / 'security_results.json')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Load results
            results_file = self.project_root / 'test_results' / 'security_results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    security_results = json.load(f)
            else:
                security_results = {'summary': {'total': 0, 'passed': 0, 'failed': 0}}
            
            security_results['exit_code'] = result.returncode
            
            if result.returncode == 0:
                console.print("✅ Security tests passed", style="green")
            else:
                console.print(f"❌ Security tests failed (exit code: {result.returncode})", style="red")
            
            return security_results
            
        except Exception as e:
            console.print(f"❌ Security tests error: {e}", style="red")
            return {
                'summary': {'total': 0, 'passed': 0, 'failed': 1},
                'error': str(e)
            }
    
    async def run_complete_validation(self) -> Dict[str, Any]:
        """Run complete end-to-end validation"""
        console.print(Panel("🚀 Starting Complete End-to-End Validation", style="bold blue"))
        
        # Ensure test results directory exists
        test_results_dir = self.project_root / 'test_results'
        test_results_dir.mkdir(exist_ok=True)
        
        # Define validation phases
        validation_phases = [
            ('Smoke Tests', self.run_smoke_tests()),
            ('Health Checks', self.run_health_checks()),
            ('System Validation', self.run_system_validation()),
            ('Workflow Validation', self.run_workflow_validation()),
            ('Performance Tests', self.run_performance_tests()),
            ('Security Tests', self.run_security_tests()),
        ]
        
        # Run validation phases
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running validation phases...", total=len(validation_phases))
            
            for phase_name, phase_coro in validation_phases:
                progress.update(task, description=f"Running {phase_name}...")
                self.results[phase_name] = await phase_coro
                progress.advance(task)
        
        # Calculate overall results
        total_duration = (time.time() - self.start_time) * 1000
        
        # Determine overall status
        phase_statuses = []
        for phase_result in self.results.values():
            if isinstance(phase_result, dict):
                status = phase_result.get('overall_status') or phase_result.get('overall_health') or 'unknown'
                if status == 'unknown' and 'summary' in phase_result:
                    # Handle pytest results
                    summary = phase_result['summary']
                    if summary.get('failed', 0) > 0:
                        status = 'failed'
                    elif summary.get('passed', 0) > 0:
                        status = 'passed'
                phase_statuses.append(status)
        
        passed_phases = sum(1 for status in phase_statuses if status == 'passed')
        warning_phases = sum(1 for status in phase_statuses if status in ['warning', 'degraded'])
        failed_phases = sum(1 for status in phase_statuses if status in ['failed', 'error'])
        
        overall_status = 'passed'
        if failed_phases > 0:
            overall_status = 'failed'
        elif warning_phases > 0:
            overall_status = 'warning'
        
        summary = {
            'overall_status': overall_status,
            'environment': self.config['environment'],
            'total_duration_ms': total_duration,
            'phases_tested': len(validation_phases),
            'phases_passed': passed_phases,
            'phases_warning': warning_phases,
            'phases_failed': failed_phases,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'config': self.config,
            'results': self.results
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
            f"End-to-End Validation: {overall_status.upper()}\n"
            f"Environment: {results.get('environment', 'unknown')}\n"
            f"Duration: {results.get('total_duration_ms', 0):.2f}ms\n"
            f"Phases: {results.get('phases_passed', 0)} passed, "
            f"{results.get('phases_warning', 0)} warnings, "
            f"{results.get('phases_failed', 0)} failed",
            style=f"bold {status_color}"
        ))
        
        # Phase results table
        phase_table = Table(title="Validation Phase Results")
        phase_table.add_column("Phase", style="cyan")
        phase_table.add_column("Status", style="green")
        phase_table.add_column("Details", style="white")
        phase_table.add_column("Exit Code", style="blue")
        
        for phase_name, phase_data in results.get('results', {}).items():
            if isinstance(phase_data, dict):
                status = (phase_data.get('overall_status') or 
                         phase_data.get('overall_health') or 
                         'unknown')
                
                # Handle pytest results
                if status == 'unknown' and 'summary' in phase_data:
                    summary = phase_data['summary']
                    if summary.get('failed', 0) > 0:
                        status = 'failed'
                    elif summary.get('passed', 0) > 0:
                        status = 'passed'
                
                status_style = {
                    'passed': 'green',
                    'warning': 'yellow',
                    'failed': 'red',
                    'error': 'red',
                    'healthy': 'green',
                    'degraded': 'yellow',
                    'unhealthy': 'red'
                }.get(status, 'blue')
                
                # Format details
                details = []
                if 'summary' in phase_data:
                    summary = phase_data['summary']
                    details.append(f"Tests: {summary.get('passed', 0)}/{summary.get('total', 0)}")
                elif 'total_tests' in phase_data:
                    details.append(f"Tests: {phase_data.get('passed_tests', 0)}/{phase_data.get('total_tests', 0)}")
                elif 'suites_tested' in phase_data:
                    details.append(f"Suites: {phase_data.get('suites_passed', 0)}/{phase_data.get('suites_tested', 0)}")
                elif 'workflows_tested' in phase_data:
                    details.append(f"Workflows: {phase_data.get('workflows_passed', 0)}/{phase_data.get('workflows_tested', 0)}")
                
                if phase_data.get('error'):
                    details.append(f"Error: {phase_data['error'][:50]}...")
                
                exit_code = phase_data.get('exit_code', 'N/A')
                
                phase_table.add_row(
                    phase_name,
                    f"[{status_style}]{status}[/{status_style}]",
                    " | ".join(details) if details else "OK",
                    str(exit_code)
                )
        
        console.print(phase_table)
        
        # Show critical errors
        critical_errors = []
        for phase_name, phase_data in results.get('results', {}).items():
            if isinstance(phase_data, dict) and phase_data.get('error'):
                critical_errors.append(f"{phase_name}: {phase_data['error']}")
        
        if critical_errors:
            console.print("\n[bold red]Critical Errors:[/bold red]")
            for error in critical_errors:
                console.print(f"  • {error}", style="red")


def load_config(config_file: Optional[str], environment: str) -> Dict[str, Any]:
    """Load configuration for validation"""
    config = {
        'environment': environment,
        'api_url': 'http://localhost:8000',
        'timeout': 60
    }
    
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # Override with environment variables
    if os.getenv('API_URL'):
        config['api_url'] = os.getenv('API_URL')
    if os.getenv('DATABASE_URL'):
        config['database_url'] = os.getenv('DATABASE_URL')
    if os.getenv('REDIS_URL'):
        config['redis_url'] = os.getenv('REDIS_URL')
    
    return config


@click.command()
@click.option('--environment', default='development', help='Environment name')
@click.option('--config', help='Configuration file path')
@click.option('--api-url', help='API URL override')
@click.option('--database-url', help='Database URL override')
@click.option('--redis-url', help='Redis URL override')
@click.option('--output', help='Output file for results (JSON format)')
@click.option('--timeout', default=60, help='Request timeout in seconds')
def main(environment, config, api_url, database_url, redis_url, output, timeout):
    """Run complete end-to-end validation"""
    
    async def run_validation():
        # Load configuration
        validation_config = load_config(config, environment)
        
        # Apply overrides
        if api_url:
            validation_config['api_url'] = api_url
        if database_url:
            validation_config['database_url'] = database_url
        if redis_url:
            validation_config['redis_url'] = redis_url
        validation_config['timeout'] = timeout
        
        # Run validation
        runner = E2EValidationRunner(validation_config)
        
        try:
            results = await runner.run_complete_validation()
            
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
            console.print(f"❌ End-to-end validation failed: {e}", style="red")
            sys.exit(3)
    
    asyncio.run(run_validation())


if __name__ == "__main__":
    import os
    main()