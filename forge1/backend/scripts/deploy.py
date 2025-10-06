#!/usr/bin/env python3
"""
Deployment Script for Employee Lifecycle System

Handles deployment to different environments with proper validation,
health checks, and rollback capabilities.

Requirements: 5.3, 5.4, 8.5
"""

import os
import sys
import asyncio
import subprocess
import time
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import click
import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


class DeploymentManager:
    """Manages deployments for the Employee Lifecycle System"""
    
    def __init__(self, environment: str, config_path: Optional[str] = None):
        self.environment = environment
        self.config_path = config_path or f"deploy/config/{environment}.yaml"
        self.project_root = Path(__file__).parent.parent
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        config_file = self.project_root / self.config_path
        
        if not config_file.exists():
            console.print(f"❌ Configuration file not found: {config_file}", style="red")
            sys.exit(1)
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def validate_environment(self) -> bool:
        """Validate deployment environment"""
        valid_environments = ['development', 'staging', 'production']
        
        if self.environment not in valid_environments:
            console.print(f"❌ Invalid environment: {self.environment}", style="red")
            console.print(f"Valid environments: {', '.join(valid_environments)}", style="blue")
            return False
        
        return True
    
    def check_prerequisites(self) -> bool:
        """Check deployment prerequisites"""
        console.print("Checking prerequisites...", style="blue")
        
        # Check required tools
        required_tools = ['docker', 'kubectl']
        if self.config.get('deployment_type') == 'docker-compose':
            required_tools.append('docker-compose')
        
        for tool in required_tools:
            if not self.check_command_exists(tool):
                console.print(f"❌ Required tool not found: {tool}", style="red")
                return False
        
        # Check Docker daemon
        if not self.check_docker_daemon():
            console.print("❌ Docker daemon not running", style="red")
            return False
        
        # Check Kubernetes context (if using k8s)
        if self.config.get('deployment_type') == 'kubernetes':
            if not self.check_kubernetes_context():
                console.print("❌ Kubernetes context not configured", style="red")
                return False
        
        console.print("✅ Prerequisites check passed", style="green")
        return True
    
    def check_command_exists(self, command: str) -> bool:
        """Check if a command exists"""
        try:
            subprocess.run([command, '--version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def check_docker_daemon(self) -> bool:
        """Check if Docker daemon is running"""
        try:
            subprocess.run(['docker', 'info'], 
                         capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def check_kubernetes_context(self) -> bool:
        """Check Kubernetes context"""
        try:
            result = subprocess.run(['kubectl', 'config', 'current-context'], 
                                  capture_output=True, text=True, check=True)
            current_context = result.stdout.strip()
            expected_context = self.config.get('kubernetes', {}).get('context')
            
            if expected_context and current_context != expected_context:
                console.print(f"⚠️  Current context: {current_context}, expected: {expected_context}", style="yellow")
                if not click.confirm("Continue with current context?"):
                    return False
            
            return True
        except subprocess.CalledProcessError:
            return False
    
    def build_image(self) -> bool:
        """Build Docker image"""
        console.print("Building Docker image...", style="blue")
        
        image_config = self.config.get('image', {})
        image_name = image_config.get('name', 'employee-lifecycle')
        image_tag = image_config.get('tag', 'latest')
        full_image_name = f"{image_name}:{image_tag}"
        
        try:
            # Build image
            cmd = [
                'docker', 'build',
                '-t', full_image_name,
                '-f', str(self.project_root / 'Dockerfile'),
                str(self.project_root)
            ]
            
            # Add build args
            build_args = image_config.get('build_args', {})
            for key, value in build_args.items():
                cmd.extend(['--build-arg', f'{key}={value}'])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                console.print(f"❌ Image build failed: {result.stderr}", style="red")
                return False
            
            console.print(f"✅ Image built successfully: {full_image_name}", style="green")
            
            # Push image if registry is configured
            registry = image_config.get('registry')
            if registry:
                return self.push_image(full_image_name, registry)
            
            return True
            
        except Exception as e:
            console.print(f"❌ Build error: {e}", style="red")
            return False
    
    def push_image(self, image_name: str, registry: str) -> bool:
        """Push image to registry"""
        console.print(f"Pushing image to {registry}...", style="blue")
        
        try:
            # Tag for registry
            registry_image = f"{registry}/{image_name}"
            subprocess.run(['docker', 'tag', image_name, registry_image], check=True)
            
            # Push to registry
            result = subprocess.run(['docker', 'push', registry_image], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                console.print(f"❌ Image push failed: {result.stderr}", style="red")
                return False
            
            console.print(f"✅ Image pushed successfully: {registry_image}", style="green")
            return True
            
        except Exception as e:
            console.print(f"❌ Push error: {e}", style="red")
            return False
    
    def run_database_migrations(self) -> bool:
        """Run database migrations"""
        console.print("Running database migrations...", style="blue")
        
        try:
            # Get database URL from config
            db_config = self.config.get('database', {})
            db_url = db_config.get('url') or os.getenv('DATABASE_URL')
            
            if not db_url:
                console.print("❌ Database URL not configured", style="red")
                return False
            
            # Run migration script
            migration_script = self.project_root / 'scripts' / 'migrate_database.py'
            cmd = [
                sys.executable, str(migration_script),
                '--database-url', db_url,
                'up'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                console.print(f"❌ Database migration failed: {result.stderr}", style="red")
                return False
            
            console.print("✅ Database migrations completed", style="green")
            return True
            
        except Exception as e:
            console.print(f"❌ Migration error: {e}", style="red")
            return False
    
    def deploy_docker_compose(self) -> bool:
        """Deploy using Docker Compose"""
        console.print("Deploying with Docker Compose...", style="blue")
        
        try:
            compose_file = self.config.get('docker_compose', {}).get('file', 'docker-compose.yml')
            compose_path = self.project_root / compose_file
            
            if not compose_path.exists():
                console.print(f"❌ Compose file not found: {compose_path}", style="red")
                return False
            
            # Stop existing services
            subprocess.run(['docker-compose', '-f', str(compose_path), 'down'], 
                         capture_output=True)
            
            # Start services
            cmd = ['docker-compose', '-f', str(compose_path), 'up', '-d']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                console.print(f"❌ Docker Compose deployment failed: {result.stderr}", style="red")
                return False
            
            console.print("✅ Docker Compose deployment completed", style="green")
            return True
            
        except Exception as e:
            console.print(f"❌ Docker Compose deployment error: {e}", style="red")
            return False
    
    def deploy_kubernetes(self) -> bool:
        """Deploy to Kubernetes"""
        console.print("Deploying to Kubernetes...", style="blue")
        
        try:
            k8s_config = self.config.get('kubernetes', {})
            namespace = k8s_config.get('namespace', 'employee-lifecycle')
            manifests_dir = self.project_root / 'deploy' / 'k8s' / self.environment
            
            if not manifests_dir.exists():
                console.print(f"❌ Kubernetes manifests not found: {manifests_dir}", style="red")
                return False
            
            # Create namespace if it doesn't exist
            subprocess.run(['kubectl', 'create', 'namespace', namespace], 
                         capture_output=True)
            
            # Apply manifests
            manifest_files = sorted(manifests_dir.glob('*.yaml'))
            
            for manifest_file in manifest_files:
                console.print(f"Applying {manifest_file.name}...", style="blue")
                
                cmd = ['kubectl', 'apply', '-f', str(manifest_file), '-n', namespace]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    console.print(f"❌ Failed to apply {manifest_file.name}: {result.stderr}", style="red")
                    return False
            
            # Wait for deployment to be ready
            deployment_name = k8s_config.get('deployment_name', 'employee-api')
            if not self.wait_for_kubernetes_deployment(deployment_name, namespace):
                return False
            
            console.print("✅ Kubernetes deployment completed", style="green")
            return True
            
        except Exception as e:
            console.print(f"❌ Kubernetes deployment error: {e}", style="red")
            return False
    
    def wait_for_kubernetes_deployment(self, deployment_name: str, namespace: str, timeout: int = 300) -> bool:
        """Wait for Kubernetes deployment to be ready"""
        console.print(f"Waiting for deployment {deployment_name} to be ready...", style="blue")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = subprocess.run([
                    'kubectl', 'rollout', 'status', 
                    f'deployment/{deployment_name}',
                    '-n', namespace,
                    '--timeout=30s'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    console.print(f"✅ Deployment {deployment_name} is ready", style="green")
                    return True
                
            except Exception:
                pass
            
            time.sleep(10)
        
        console.print(f"❌ Deployment {deployment_name} failed to become ready within {timeout}s", style="red")
        return False
    
    async def run_health_checks(self) -> bool:
        """Run health checks after deployment"""
        console.print("Running health checks...", style="blue")
        
        health_config = self.config.get('health_checks', {})
        base_url = health_config.get('base_url', 'http://localhost:8000')
        endpoints = health_config.get('endpoints', ['/health'])
        timeout = health_config.get('timeout', 30)
        retry_count = health_config.get('retry_count', 5)
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for endpoint in endpoints:
                url = f"{base_url}{endpoint}"
                console.print(f"Checking {url}...", style="blue")
                
                for attempt in range(retry_count):
                    try:
                        response = await client.get(url)
                        
                        if response.status_code == 200:
                            console.print(f"✅ Health check passed: {endpoint}", style="green")
                            break
                        else:
                            console.print(f"⚠️  Health check returned {response.status_code}: {endpoint}", style="yellow")
                    
                    except Exception as e:
                        if attempt == retry_count - 1:
                            console.print(f"❌ Health check failed: {endpoint} - {e}", style="red")
                            return False
                        else:
                            console.print(f"Attempt {attempt + 1}/{retry_count} failed, retrying...", style="yellow")
                            await asyncio.sleep(5)
        
        console.print("✅ All health checks passed", style="green")
        return True
    
    def run_smoke_tests(self) -> bool:
        """Run smoke tests after deployment"""
        console.print("Running smoke tests...", style="blue")
        
        try:
            smoke_tests_script = self.project_root / 'tests' / 'smoke_tests.py'
            
            if not smoke_tests_script.exists():
                console.print("⚠️  No smoke tests found, skipping", style="yellow")
                return True
            
            # Run smoke tests
            cmd = [sys.executable, str(smoke_tests_script), '--environment', self.environment]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                console.print(f"❌ Smoke tests failed: {result.stderr}", style="red")
                return False
            
            console.print("✅ Smoke tests passed", style="green")
            return True
            
        except Exception as e:
            console.print(f"❌ Smoke tests error: {e}", style="red")
            return False
    
    def rollback_deployment(self) -> bool:
        """Rollback deployment"""
        console.print("Rolling back deployment...", style="yellow")
        
        deployment_type = self.config.get('deployment_type', 'kubernetes')
        
        try:
            if deployment_type == 'kubernetes':
                k8s_config = self.config.get('kubernetes', {})
                namespace = k8s_config.get('namespace', 'employee-lifecycle')
                deployment_name = k8s_config.get('deployment_name', 'employee-api')
                
                # Rollback to previous revision
                cmd = ['kubectl', 'rollout', 'undo', f'deployment/{deployment_name}', '-n', namespace]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    console.print(f"❌ Rollback failed: {result.stderr}", style="red")
                    return False
                
                # Wait for rollback to complete
                return self.wait_for_kubernetes_deployment(deployment_name, namespace)
            
            elif deployment_type == 'docker-compose':
                # For Docker Compose, we would need to restore previous images
                console.print("⚠️  Docker Compose rollback not implemented", style="yellow")
                return False
            
        except Exception as e:
            console.print(f"❌ Rollback error: {e}", style="red")
            return False
    
    def create_deployment_record(self, success: bool) -> None:
        """Create deployment record"""
        record = {
            'environment': self.environment,
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'config': self.config,
            'git_commit': self.get_git_commit(),
            'deployed_by': os.getenv('USER', 'unknown')
        }
        
        # Save deployment record
        records_dir = self.project_root / 'deploy' / 'records'
        records_dir.mkdir(exist_ok=True)
        
        record_file = records_dir / f"{self.environment}_{int(time.time())}.json"
        with open(record_file, 'w') as f:
            json.dump(record, f, indent=2)
        
        console.print(f"Deployment record saved: {record_file}", style="blue")
    
    def get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    async def deploy(self) -> bool:
        """Run complete deployment process"""
        console.print(Panel(f"🚀 Deploying Employee Lifecycle System to {self.environment}", 
                          style="bold blue"))
        
        # Validation
        if not self.validate_environment():
            return False
        
        if not self.check_prerequisites():
            return False
        
        # Deployment steps
        steps = [
            ("Building image", self.build_image),
            ("Running database migrations", self.run_database_migrations),
        ]
        
        # Add deployment method
        deployment_type = self.config.get('deployment_type', 'kubernetes')
        if deployment_type == 'kubernetes':
            steps.append(("Deploying to Kubernetes", self.deploy_kubernetes))
        elif deployment_type == 'docker-compose':
            steps.append(("Deploying with Docker Compose", self.deploy_docker_compose))
        
        # Add post-deployment steps
        steps.extend([
            ("Running health checks", lambda: asyncio.run(self.run_health_checks())),
            ("Running smoke tests", self.run_smoke_tests),
        ])
        
        # Execute deployment steps
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Deploying...", total=len(steps))
            
            for step_name, step_func in steps:
                progress.update(task, description=step_name)
                
                if not step_func():
                    console.print(f"❌ Deployment failed at step: {step_name}", style="red")
                    
                    # Attempt rollback in production
                    if self.environment == 'production':
                        console.print("Attempting rollback...", style="yellow")
                        self.rollback_deployment()
                    
                    self.create_deployment_record(False)
                    return False
                
                progress.advance(task)
        
        console.print(Panel("✅ Deployment completed successfully!", style="bold green"))
        self.create_deployment_record(True)
        return True


# CLI Commands
@click.group()
def cli():
    """Deployment tool for Employee Lifecycle System"""
    pass


@cli.command()
@click.argument('environment')
@click.option('--config', help='Configuration file path')
@click.option('--skip-tests', is_flag=True, help='Skip smoke tests')
def deploy(environment, config, skip_tests):
    """Deploy to specified environment"""
    async def run():
        manager = DeploymentManager(environment, config)
        
        if skip_tests:
            # Remove smoke tests from config
            manager.config.setdefault('health_checks', {})['skip_smoke_tests'] = True
        
        success = await manager.deploy()
        sys.exit(0 if success else 1)
    
    asyncio.run(run())


@cli.command()
@click.argument('environment')
@click.option('--config', help='Configuration file path')
def rollback(environment, config):
    """Rollback deployment"""
    manager = DeploymentManager(environment, config)
    success = manager.rollback_deployment()
    sys.exit(0 if success else 1)


@cli.command()
@click.argument('environment')
@click.option('--config', help='Configuration file path')
def health_check(environment, config):
    """Run health checks"""
    async def run():
        manager = DeploymentManager(environment, config)
        success = await manager.run_health_checks()
        sys.exit(0 if success else 1)
    
    asyncio.run(run())


@cli.command()
@click.argument('environment')
def status(environment):
    """Show deployment status"""
    # Show recent deployment records
    records_dir = Path(__file__).parent.parent / 'deploy' / 'records'
    
    if not records_dir.exists():
        console.print("No deployment records found", style="yellow")
        return
    
    # Find records for environment
    record_files = sorted([
        f for f in records_dir.glob(f"{environment}_*.json")
    ], reverse=True)
    
    if not record_files:
        console.print(f"No deployment records found for {environment}", style="yellow")
        return
    
    # Show recent deployments
    table = Table(title=f"Recent Deployments - {environment}")
    table.add_column("Timestamp", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Git Commit", style="blue")
    table.add_column("Deployed By", style="white")
    
    for record_file in record_files[:10]:  # Show last 10
        with open(record_file, 'r') as f:
            record = json.load(f)
        
        status = "✅ Success" if record['success'] else "❌ Failed"
        timestamp = datetime.fromisoformat(record['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
        git_commit = record.get('git_commit', 'Unknown')[:8]
        deployed_by = record.get('deployed_by', 'Unknown')
        
        table.add_row(timestamp, status, git_commit, deployed_by)
    
    console.print(table)


if __name__ == "__main__":
    cli()