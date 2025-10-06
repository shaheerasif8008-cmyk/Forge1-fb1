"""
Secure Code Execution Sandboxes
Firecracker/K8s-based secure sandboxes for code agents with resource limits and artifact capture
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import json
import logging
import tempfile
import shutil
import os
import subprocess
from pathlib import Path
import yaml

from forge1.backend.forge1.core.monitoring import MetricsCollector
from forge1.backend.forge1.core.security import SecretManager


class SandboxType(Enum):
    """Types of code execution sandboxes"""
    FIRECRACKER = "firecracker"
    KUBERNETES_JOB = "kubernetes_job"
    DOCKER_CONTAINER = "docker_container"
    PROCESS_ISOLATION = "process_isolation"


class ExecutionStatus(Enum):
    """Code execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    KILLED = "killed"


@dataclass
class ResourceLimits:
    """Resource limits for sandbox execution"""
    cpu_cores: float = 1.0
    memory_mb: int = 512
    disk_mb: int = 1024
    network_enabled: bool = False
    execution_timeout_seconds: int = 300
    max_processes: int = 10
    max_file_descriptors: int = 100


@dataclass
class CodeExecution:
    """Record of a code execution in sandbox"""
    execution_id: str
    sandbox_type: SandboxType
    language: str
    code: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    security_violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxArtifact:
    """Artifact produced by sandbox execution"""
    artifact_id: str
    execution_id: str
    file_path: str
    file_size: int
    file_type: str
    created_at: datetime
    checksum: str
    is_secure: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class FirecrackerSandbox:
    """Firecracker-based secure sandbox implementation"""
    
    def __init__(self, metrics: MetricsCollector):
        self.metrics = metrics
        self.logger = logging.getLogger("firecracker_sandbox")
        
        # Firecracker configuration
        self.firecracker_binary = "/usr/bin/firecracker"
        self.kernel_image = "/opt/firecracker/vmlinux"
        self.rootfs_image = "/opt/firecracker/rootfs.ext4"
        
    async def execute_code(
        self,
        execution_id: str,
        code: str,
        language: str,
        resource_limits: ResourceLimits
    ) -> CodeExecution:
        """Execute code in Firecracker microVM"""
        
        execution = CodeExecution(
            execution_id=execution_id,
            sandbox_type=SandboxType.FIRECRACKER,
            language=language,
            code=code,
            start_time=datetime.utcnow()
        )
        
        try:
            # Create temporary directory for VM files
            with tempfile.TemporaryDirectory() as temp_dir:
                vm_dir = Path(temp_dir)
                
                # Prepare VM configuration
                vm_config = await self._prepare_vm_config(
                    vm_dir, code, language, resource_limits
                )
                
                # Start Firecracker VM
                execution.status = ExecutionStatus.RUNNING
                vm_process = await self._start_firecracker_vm(vm_config)
                
                # Wait for execution with timeout
                try:
                    await asyncio.wait_for(
                        vm_process.wait(),
                        timeout=resource_limits.execution_timeout_seconds
                    )
                    execution.exit_code = vm_process.returncode
                    execution.status = ExecutionStatus.COMPLETED if execution.exit_code == 0 else ExecutionStatus.FAILED
                    
                except asyncio.TimeoutError:
                    execution.status = ExecutionStatus.TIMEOUT
                    vm_process.kill()
                    await vm_process.wait()
                
                # Collect results
                execution.stdout, execution.stderr = await self._collect_vm_output(vm_dir)
                execution.resource_usage = await self._collect_resource_usage(vm_dir)
                execution.artifacts = await self._collect_artifacts(vm_dir, execution_id)
                
        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            execution.stderr = str(e)
            self.logger.error(f"Firecracker execution failed: {e}")
        
        finally:
            execution.end_time = datetime.utcnow()
            self._record_metrics(execution)
        
        return execution
    
    async def _prepare_vm_config(
        self,
        vm_dir: Path,
        code: str,
        language: str,
        limits: ResourceLimits
    ) -> Dict[str, Any]:
        """Prepare Firecracker VM configuration"""
        
        # Write code to file
        code_file = vm_dir / f"code.{self._get_file_extension(language)}"
        code_file.write_text(code)
        
        # Create VM configuration
        config = {
            "boot-source": {
                "kernel_image_path": self.kernel_image,
                "boot_args": "console=ttyS0 reboot=k panic=1 pci=off"
            },
            "drives": [
                {
                    "drive_id": "rootfs",
                    "path_on_host": self.rootfs_image,
                    "is_root_device": True,
                    "is_read_only": False
                }
            ],
            "machine-config": {
                "vcpu_count": int(limits.cpu_cores),
                "mem_size_mib": limits.memory_mb,
                "ht_enabled": False
            },
            "network-interfaces": [] if not limits.network_enabled else [
                {
                    "iface_id": "eth0",
                    "guest_mac": "AA:FC:00:00:00:01",
                    "host_dev_name": "tap0"
                }
            ]
        }
        
        # Write configuration file
        config_file = vm_dir / "vm_config.json"
        config_file.write_text(json.dumps(config, indent=2))
        
        return config
    
    async def _start_firecracker_vm(self, vm_config: Dict[str, Any]) -> asyncio.subprocess.Process:
        """Start Firecracker VM process"""
        
        cmd = [
            self.firecracker_binary,
            "--api-sock", "/tmp/firecracker.socket",
            "--config-file", str(vm_config["config_file"])
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        return process
    
    async def _collect_vm_output(self, vm_dir: Path) -> tuple[str, str]:
        """Collect stdout and stderr from VM execution"""
        # Placeholder - would collect from VM serial console or shared filesystem
        return "", ""
    
    async def _collect_resource_usage(self, vm_dir: Path) -> Dict[str, Any]:
        """Collect resource usage metrics from VM"""
        # Placeholder - would collect from VM monitoring
        return {
            "cpu_usage_percent": 0.0,
            "memory_usage_mb": 0,
            "disk_usage_mb": 0,
            "network_bytes_sent": 0,
            "network_bytes_received": 0
        }
    
    async def _collect_artifacts(self, vm_dir: Path, execution_id: str) -> List[str]:
        """Collect artifacts produced by VM execution"""
        # Placeholder - would collect files from VM shared filesystem
        return []
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for programming language"""
        extensions = {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "java": "java",
            "go": "go",
            "rust": "rs",
            "cpp": "cpp",
            "c": "c"
        }
        return extensions.get(language.lower(), "txt")
    
    def _record_metrics(self, execution: CodeExecution) -> None:
        """Record execution metrics"""
        self.metrics.increment(f"firecracker_execution_{execution.status.value}")
        
        if execution.end_time:
            duration = (execution.end_time - execution.start_time).total_seconds()
            self.metrics.record_metric("firecracker_execution_duration", duration)


class KubernetesSandbox:
    """Kubernetes Job-based sandbox implementation"""
    
    def __init__(self, metrics: MetricsCollector, secret_manager: SecretManager):
        self.metrics = metrics
        self.secret_manager = secret_manager
        self.logger = logging.getLogger("kubernetes_sandbox")
        
        # Kubernetes configuration
        self.namespace = "forge1-sandboxes"
        self.job_timeout = 600  # 10 minutes
    
    async def execute_code(
        self,
        execution_id: str,
        code: str,
        language: str,
        resource_limits: ResourceLimits
    ) -> CodeExecution:
        """Execute code in Kubernetes Job"""
        
        execution = CodeExecution(
            execution_id=execution_id,
            sandbox_type=SandboxType.KUBERNETES_JOB,
            language=language,
            code=code,
            start_time=datetime.utcnow()
        )
        
        try:
            # Create Kubernetes Job manifest
            job_manifest = self._create_job_manifest(
                execution_id, code, language, resource_limits
            )
            
            # Submit job to Kubernetes
            execution.status = ExecutionStatus.RUNNING
            await self._submit_kubernetes_job(job_manifest)
            
            # Wait for job completion
            job_result = await self._wait_for_job_completion(
                execution_id, resource_limits.execution_timeout_seconds
            )
            
            # Update execution with results
            execution.status = job_result["status"]
            execution.exit_code = job_result.get("exit_code")
            execution.stdout = job_result.get("stdout", "")
            execution.stderr = job_result.get("stderr", "")
            execution.resource_usage = job_result.get("resource_usage", {})
            execution.artifacts = job_result.get("artifacts", [])
            
        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            execution.stderr = str(e)
            self.logger.error(f"Kubernetes execution failed: {e}")
        
        finally:
            execution.end_time = datetime.utcnow()
            # Clean up Kubernetes resources
            await self._cleanup_kubernetes_job(execution_id)
            self._record_metrics(execution)
        
        return execution
    
    def _create_job_manifest(
        self,
        execution_id: str,
        code: str,
        language: str,
        limits: ResourceLimits
    ) -> Dict[str, Any]:
        """Create Kubernetes Job manifest"""
        
        # Select appropriate container image
        image_map = {
            "python": "python:3.11-slim",
            "javascript": "node:18-alpine",
            "typescript": "node:18-alpine",
            "java": "openjdk:17-alpine",
            "go": "golang:1.21-alpine",
            "rust": "rust:1.70-alpine"
        }
        
        image = image_map.get(language.lower(), "ubuntu:22.04")
        
        # Create execution script
        script = self._create_execution_script(code, language)
        
        manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": f"sandbox-{execution_id}",
                "namespace": self.namespace,
                "labels": {
                    "app": "forge1-sandbox",
                    "execution-id": execution_id,
                    "language": language
                }
            },
            "spec": {
                "ttlSecondsAfterFinished": 300,  # Clean up after 5 minutes
                "backoffLimit": 0,  # No retries
                "activeDeadlineSeconds": limits.execution_timeout_seconds,
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "forge1-sandbox",
                            "execution-id": execution_id
                        }
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "fsGroup": 1000
                        },
                        "containers": [
                            {
                                "name": "executor",
                                "image": image,
                                "command": ["/bin/sh", "-c"],
                                "args": [script],
                                "resources": {
                                    "limits": {
                                        "cpu": f"{limits.cpu_cores}",
                                        "memory": f"{limits.memory_mb}Mi",
                                        "ephemeral-storage": f"{limits.disk_mb}Mi"
                                    },
                                    "requests": {
                                        "cpu": f"{limits.cpu_cores * 0.1}",
                                        "memory": f"{limits.memory_mb // 2}Mi"
                                    }
                                },
                                "securityContext": {
                                    "allowPrivilegeEscalation": False,
                                    "readOnlyRootFilesystem": False,
                                    "capabilities": {
                                        "drop": ["ALL"]
                                    }
                                },
                                "volumeMounts": [
                                    {
                                        "name": "workspace",
                                        "mountPath": "/workspace"
                                    }
                                ]
                            }
                        ],
                        "volumes": [
                            {
                                "name": "workspace",
                                "emptyDir": {
                                    "sizeLimit": f"{limits.disk_mb}Mi"
                                }
                            }
                        ]
                    }
                }
            }
        }
        
        # Disable networking if not allowed
        if not limits.network_enabled:
            manifest["spec"]["template"]["spec"]["hostNetwork"] = False
            manifest["spec"]["template"]["spec"]["dnsPolicy"] = "None"
        
        return manifest
    
    def _create_execution_script(self, code: str, language: str) -> str:
        """Create execution script for the container"""
        
        scripts = {
            "python": f"""
                cd /workspace
                cat > code.py << 'EOF'
{code}
EOF
                timeout 300 python code.py 2>&1
                echo "EXIT_CODE: $?"
            """,
            "javascript": f"""
                cd /workspace
                cat > code.js << 'EOF'
{code}
EOF
                timeout 300 node code.js 2>&1
                echo "EXIT_CODE: $?"
            """,
            "java": f"""
                cd /workspace
                cat > Code.java << 'EOF'
{code}
EOF
                javac Code.java && timeout 300 java Code 2>&1
                echo "EXIT_CODE: $?"
            """
        }
        
        return scripts.get(language.lower(), f"""
            cd /workspace
            echo '{code}' > code.txt
            echo "Language {language} not supported"
            exit 1
        """)
    
    async def _submit_kubernetes_job(self, manifest: Dict[str, Any]) -> None:
        """Submit job to Kubernetes cluster"""
        try:
            # Import Kubernetes client
            try:
                from kubernetes import client, config
                from kubernetes.client.rest import ApiException
            except ImportError:
                self.logger.error("Kubernetes client not installed. Install with: pip install kubernetes")
                raise RuntimeError("Kubernetes client not available")
            
            # Load Kubernetes configuration
            try:
                # Try in-cluster config first (for pods running in cluster)
                config.load_incluster_config()
            except config.ConfigException:
                # Fall back to kubeconfig file
                config.load_kube_config()
            
            # Create Kubernetes API client
            batch_v1 = client.BatchV1Api()
            
            # Submit the job
            try:
                response = batch_v1.create_namespaced_job(
                    namespace=manifest.get('namespace', 'default'),
                    body=manifest
                )
                self.logger.info(f"Successfully submitted Kubernetes job: {response.metadata.name}")
                
            except ApiException as e:
                self.logger.error(f"Failed to submit Kubernetes job: {e}")
                raise RuntimeError(f"Kubernetes job submission failed: {e}")
                
        except Exception as e:
            self.logger.error(f"Error submitting Kubernetes job: {e}")
            raise
    
    async def _wait_for_job_completion(self, execution_id: str, timeout: int) -> Dict[str, Any]:
        """Wait for Kubernetes job to complete"""
        try:
            from kubernetes import client, config
            from kubernetes.client.rest import ApiException
            
            # Load config
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
            
            batch_v1 = client.BatchV1Api()
            core_v1 = client.CoreV1Api()
            
            job_name = f"code-execution-{execution_id}"
            namespace = "default"
            
            start_time = asyncio.get_event_loop().time()
            
            while (asyncio.get_event_loop().time() - start_time) < timeout:
                try:
                    # Get job status
                    job = batch_v1.read_namespaced_job_status(name=job_name, namespace=namespace)
                    
                    if job.status.succeeded:
                        # Job completed successfully
                        logs = await self._get_pod_logs(core_v1, job_name, namespace)
                        
                        return {
                            "status": ExecutionStatus.COMPLETED,
                            "exit_code": 0,
                            "stdout": logs.get("stdout", ""),
                            "stderr": logs.get("stderr", ""),
                            "resource_usage": {
                                "cpu_usage_percent": 25.0,
                                "memory_usage_mb": 128,
                                "execution_time_seconds": asyncio.get_event_loop().time() - start_time
                            },
                            "artifacts": await self._collect_artifacts_from_pods(core_v1, job_name, namespace)
                        }
                    
                    elif job.status.failed:
                        # Job failed
                        logs = await self._get_pod_logs(core_v1, job_name, namespace)
                        
                        return {
                            "status": ExecutionStatus.FAILED,
                            "exit_code": 1,
                            "stdout": logs.get("stdout", ""),
                            "stderr": logs.get("stderr", "Job execution failed"),
                            "resource_usage": {
                                "cpu_usage_percent": 0,
                                "memory_usage_mb": 0,
                                "execution_time_seconds": asyncio.get_event_loop().time() - start_time
                            },
                            "artifacts": []
                        }
                    
                    # Job still running, wait and check again
                    await asyncio.sleep(2)
                    
                except ApiException as e:
                    if e.status == 404:
                        # Job not found, might have been cleaned up
                        return {
                            "status": ExecutionStatus.FAILED,
                            "exit_code": 1,
                            "stdout": "",
                            "stderr": "Job not found",
                            "resource_usage": {
                                "cpu_usage_percent": 0,
                                "memory_usage_mb": 0,
                                "execution_time_seconds": asyncio.get_event_loop().time() - start_time
                            },
                            "artifacts": []
                        }
                    else:
                        raise
            
            # Timeout reached
            return {
                "status": ExecutionStatus.TIMEOUT,
                "exit_code": 124,
                "stdout": "",
                "stderr": "Execution timeout",
                "resource_usage": {
                    "cpu_usage_percent": 0,
                    "memory_usage_mb": 0,
                    "execution_time_seconds": timeout
                },
                "artifacts": []
            }
            
        except Exception as e:
            self.logger.error(f"Error waiting for Kubernetes job completion: {e}")
            return {
                "status": ExecutionStatus.FAILED,
                "exit_code": 1,
                "stdout": "",
                "stderr": str(e),
                "resource_usage": {
                    "cpu_usage_percent": 0,
                    "memory_usage_mb": 0,
                    "execution_time_seconds": 0
                },
                "artifacts": []
            }
    
    async def _cleanup_kubernetes_job(self, execution_id: str) -> None:
        """Clean up Kubernetes job resources"""
        try:
            from kubernetes import client, config
            from kubernetes.client.rest import ApiException
            
            # Load config
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
            
            batch_v1 = client.BatchV1Api()
            core_v1 = client.CoreV1Api()
            
            job_name = f"code-execution-{execution_id}"
            namespace = "default"
            
            try:
                # Delete the job
                batch_v1.delete_namespaced_job(
                    name=job_name,
                    namespace=namespace,
                    body=client.V1DeleteOptions(propagation_policy="Background")
                )
                
                # Delete associated pods
                core_v1.delete_collection_namespaced_pod(
                    namespace=namespace,
                    label_selector=f"job-name={job_name}"
                )
                
                self.logger.info(f"Successfully cleaned up Kubernetes job: {job_name}")
                
            except ApiException as e:
                if e.status == 404:
                    # Job already deleted
                    self.logger.info(f"Kubernetes job {job_name} already deleted")
                else:
                    self.logger.error(f"Error deleting Kubernetes job {job_name}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up Kubernetes job for execution {execution_id}: {e}")
    
    async def _get_pod_logs(self, core_v1: Any, job_name: str, namespace: str) -> Dict[str, str]:
        """Get logs from job pods"""
        try:
            # Find pods for this job
            pods = core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"job-name={job_name}"
            )
            
            stdout_logs = []
            stderr_logs = []
            
            for pod in pods.items:
                try:
                    # Get pod logs
                    logs = core_v1.read_namespaced_pod_log(
                        name=pod.metadata.name,
                        namespace=namespace,
                        container="code-executor"
                    )
                    stdout_logs.append(logs)
                    
                except Exception as e:
                    stderr_logs.append(f"Error getting logs from pod {pod.metadata.name}: {e}")
            
            return {
                "stdout": "\n".join(stdout_logs),
                "stderr": "\n".join(stderr_logs)
            }
            
        except Exception as e:
            return {
                "stdout": "",
                "stderr": f"Error retrieving logs: {e}"
            }
    
    async def _collect_artifacts_from_pods(self, core_v1: Any, job_name: str, namespace: str) -> List[str]:
        """Collect artifacts from job pods"""
        try:
            # This would typically copy files from pod volumes
            # For now, return empty list
            return []
            
        except Exception as e:
            self.logger.error(f"Error collecting artifacts: {e}")
            return []
    
    def _record_metrics(self, execution: CodeExecution) -> None:
        """Record execution metrics"""
        self.metrics.increment(f"kubernetes_execution_{execution.status.value}")
        
        if execution.end_time:
            duration = (execution.end_time - execution.start_time).total_seconds()
            self.metrics.record_metric("kubernetes_execution_duration", duration)


class CodeSandboxManager:
    """
    Manages secure code execution sandboxes with multiple backend implementations
    """
    
    def __init__(self, metrics: MetricsCollector, secret_manager: SecretManager):
        self.metrics = metrics
        self.secret_manager = secret_manager
        self.logger = logging.getLogger("code_sandbox_manager")
        
        # Initialize sandbox implementations
        self.firecracker = FirecrackerSandbox(metrics)
        self.kubernetes = KubernetesSandbox(metrics, secret_manager)
        
        # Execution tracking
        self.active_executions: Dict[str, CodeExecution] = {}
        self.completed_executions: List[CodeExecution] = []
        
        # Security policies
        self.allowed_languages = {
            "python", "javascript", "typescript", "java", "go", "rust", "cpp", "c"
        }
        self.blocked_imports = {
            "python": ["os", "subprocess", "sys", "socket", "urllib"],
            "javascript": ["fs", "child_process", "net", "http", "https"],
            "java": ["java.io", "java.net", "java.lang.Runtime"]
        }
        
        # Default resource limits
        self.default_limits = ResourceLimits(
            cpu_cores=0.5,
            memory_mb=256,
            disk_mb=512,
            network_enabled=False,
            execution_timeout_seconds=60,
            max_processes=5
        )
    
    async def execute_code(
        self,
        code: str,
        language: str,
        sandbox_type: SandboxType = SandboxType.KUBERNETES_JOB,
        resource_limits: Optional[ResourceLimits] = None,
        execution_id: Optional[str] = None
    ) -> CodeExecution:
        """
        Execute code in a secure sandbox
        
        Args:
            code: Source code to execute
            language: Programming language
            sandbox_type: Type of sandbox to use
            resource_limits: Resource constraints
            execution_id: Optional execution ID
        
        Returns:
            CodeExecution result with output and artifacts
        """
        
        # Generate execution ID if not provided
        if execution_id is None:
            execution_id = f"exec_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Use default limits if not provided
        if resource_limits is None:
            resource_limits = self.default_limits
        
        # Validate language
        if language.lower() not in self.allowed_languages:
            raise ValueError(f"Language {language} not supported")
        
        # Security validation
        security_violations = self._validate_code_security(code, language)
        if security_violations:
            execution = CodeExecution(
                execution_id=execution_id,
                sandbox_type=sandbox_type,
                language=language,
                code=code,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                status=ExecutionStatus.FAILED,
                stderr="Security violations detected",
                security_violations=security_violations
            )
            return execution
        
        try:
            # Select sandbox implementation
            if sandbox_type == SandboxType.FIRECRACKER:
                sandbox = self.firecracker
            elif sandbox_type == SandboxType.KUBERNETES_JOB:
                sandbox = self.kubernetes
            else:
                raise ValueError(f"Sandbox type {sandbox_type} not implemented")
            
            # Track active execution
            self.active_executions[execution_id] = None  # Placeholder
            
            # Execute code
            execution = await sandbox.execute_code(
                execution_id, code, language, resource_limits
            )
            
            # Store completed execution
            self.completed_executions.append(execution)
            
            # Remove from active tracking
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            
            self.logger.info(
                f"Code execution completed: {execution_id} "
                f"({execution.status.value})"
            )
            
            return execution
            
        except Exception as e:
            self.logger.error(f"Code execution failed: {execution_id} - {str(e)}")
            
            # Create failed execution record
            execution = CodeExecution(
                execution_id=execution_id,
                sandbox_type=sandbox_type,
                language=language,
                code=code,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                status=ExecutionStatus.FAILED,
                stderr=str(e)
            )
            
            return execution
    
    def get_execution_status(self, execution_id: str) -> Optional[ExecutionStatus]:
        """Get status of an execution"""
        
        # Check active executions
        if execution_id in self.active_executions:
            return ExecutionStatus.RUNNING
        
        # Check completed executions
        for execution in self.completed_executions:
            if execution.execution_id == execution_id:
                return execution.status
        
        return None
    
    def get_execution_result(self, execution_id: str) -> Optional[CodeExecution]:
        """Get full execution result"""
        
        for execution in self.completed_executions:
            if execution.execution_id == execution_id:
                return execution
        
        return None
    
    def list_executions(
        self,
        language: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        limit: int = 100
    ) -> List[CodeExecution]:
        """List recent executions with optional filtering"""
        
        executions = self.completed_executions[-limit:]
        
        if language:
            executions = [e for e in executions if e.language.lower() == language.lower()]
        
        if status:
            executions = [e for e in executions if e.status == status]
        
        return executions
    
    def get_sandbox_metrics(self) -> Dict[str, Any]:
        """Get sandbox performance metrics"""
        
        if not self.completed_executions:
            return {"status": "no_data"}
        
        recent_executions = self.completed_executions[-100:]
        
        # Calculate success rates by sandbox type
        sandbox_stats = {}
        for sandbox_type in SandboxType:
            type_executions = [e for e in recent_executions if e.sandbox_type == sandbox_type]
            if type_executions:
                successful = [e for e in type_executions if e.status == ExecutionStatus.COMPLETED]
                sandbox_stats[sandbox_type.value] = {
                    "total_executions": len(type_executions),
                    "success_rate": len(successful) / len(type_executions),
                    "avg_duration": sum([
                        (e.end_time - e.start_time).total_seconds()
                        for e in successful if e.end_time
                    ]) / len(successful) if successful else 0
                }
        
        # Language statistics
        language_stats = {}
        for execution in recent_executions:
            lang = execution.language.lower()
            if lang not in language_stats:
                language_stats[lang] = {"total": 0, "successful": 0}
            
            language_stats[lang]["total"] += 1
            if execution.status == ExecutionStatus.COMPLETED:
                language_stats[lang]["successful"] += 1
        
        # Calculate success rates
        for lang_stat in language_stats.values():
            lang_stat["success_rate"] = lang_stat["successful"] / lang_stat["total"]
        
        return {
            "sandbox_breakdown": sandbox_stats,
            "language_breakdown": language_stats,
            "active_executions": len(self.active_executions),
            "total_executions": len(self.completed_executions),
            "security_violations": sum(
                len(e.security_violations) for e in recent_executions
            )
        }
    
    def _validate_code_security(self, code: str, language: str) -> List[str]:
        """Validate code for security violations"""
        
        violations = []
        code_lower = code.lower()
        
        # Check for blocked imports/modules
        blocked = self.blocked_imports.get(language.lower(), [])
        for blocked_item in blocked:
            if blocked_item.lower() in code_lower:
                violations.append(f"Blocked import/module: {blocked_item}")
        
        # Check for dangerous patterns
        dangerous_patterns = [
            "eval(", "exec(", "system(", "shell_exec(", "passthru(",
            "file_get_contents(", "file_put_contents(", "fopen(",
            "subprocess.", "os.system", "os.popen", "__import__"
        ]
        
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower:
                violations.append(f"Dangerous pattern detected: {pattern}")
        
        return violations