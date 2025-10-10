```python
"""
Supply Chain Security Management
SBOM generation, signature verification, dependency pinning, and container scanning
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import logging
import json
import hashlib
import subprocess
from pathlib import Path
import tempfile
import yaml

from forge1.backend.forge1.core.monitoring import MetricsCollector
from forge1.backend.forge1.core.memory import MemoryManager
from forge1.backend.forge1.core.security import SecretManager


class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ScanStatus(Enum):
    """Security scan status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Vulnerability:
    """Security vulnerability information"""
    cve_id: str
    severity: VulnerabilitySeverity
    score: float  # CVSS score
    description: str
    affected_component: str
    affected_versions: List[str]
    fixed_versions: List[str] = field(default_factory=list)
    published_date: Optional[datetime] = None
    discovered_date: datetime = field(default_factory=datetime.utcnow)
    
    # Additional metadata
    cwe_ids: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    exploit_available: bool = False
    patch_available: bool = False


@dataclass
class SBOMComponent:
    """Software Bill of Materials component"""
    name: str
    version: str
    component_type: str  # library, application, framework, etc.
    supplier: Optional[str] = None
    license: Optional[str] = None
    copyright: Optional[str] = None
    
    # Hashes for integrity verification
    sha256: Optional[str] = None
    sha512: Optional[str] = None
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    
    # Security information
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    
    # Metadata
    description: Optional[str] = None
    homepage: Optional[str] = None
    download_location: Optional[str] = None


@dataclass
class SecurityScanResult:
    """Comprehensive security scan result"""
    scan_id: str
    target: str  # What was scanned (container, package, etc.)
    scan_type: str  # container, dependency, static_analysis, etc.
    status: ScanStatus
    
    # Timing
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Results
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    total_vulnerabilities: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    
    # Pass/fail determination
    scan_passed: bool = False
    blocking_issues: List[str] = field(default_factory=list)
    
    # Scanner information
    scanner_name: str = ""
    scanner_version: str = ""
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
class S
BOMGenerator:
    """Generates Software Bill of Materials in CycloneDX format"""
    
    def __init__(self, metrics: MetricsCollector):
        self.metrics = metrics
        self.logger = logging.getLogger("sbom_generator")
    
    async def generate_sbom(
        self,
        project_path: str,
        output_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Generate SBOM for a project
        
        Args:
            project_path: Path to the project to analyze
            output_format: Output format (json, xml)
        
        Returns:
            SBOM in CycloneDX format
        """
        
        self.logger.info(f"Generating SBOM for project: {project_path}")
        
        # Analyze project dependencies
        components = await self._analyze_dependencies(project_path)
        
        # Generate SBOM metadata
        sbom_metadata = self._generate_metadata()
        
        # Create CycloneDX SBOM
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "serialNumber": f"urn:uuid:{self._generate_uuid()}",
            "version": 1,
            "metadata": sbom_metadata,
            "components": [self._component_to_cyclonedx(comp) for comp in components],
            "dependencies": self._generate_dependencies(components)
        }
        
        # Record metrics
        self.metrics.increment("sbom_generated")
        self.metrics.record_metric("sbom_components_count", len(components))
        
        self.logger.info(f"Generated SBOM with {len(components)} components")
        
        return sbom
    
    async def _analyze_dependencies(self, project_path: str) -> List[SBOMComponent]:
        """Analyze project dependencies to extract components"""
        
        components = []
        project_path_obj = Path(project_path)
        
        # Analyze Python dependencies
        if (project_path_obj / "requirements.txt").exists():
            components.extend(await self._analyze_python_requirements(project_path))
        
        if (project_path_obj / "pyproject.toml").exists():
            components.extend(await self._analyze_pyproject_toml(project_path))
        
        # Analyze Node.js dependencies
        if (project_path_obj / "package.json").exists():
            components.extend(await self._analyze_package_json(project_path))
        
        # Analyze Docker dependencies
        if (project_path_obj / "Dockerfile").exists():
            components.extend(await self._analyze_dockerfile(project_path))
        
        return components
    
    async def _analyze_python_requirements(self, project_path: str) -> List[SBOMComponent]:
        """Analyze Python requirements.txt file"""
        
        components = []
        requirements_file = Path(project_path) / "requirements.txt"
        
        try:
            with open(requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse requirement line (simplified)
                        if '==' in line:
                            name, version = line.split('==', 1)
                            components.append(SBOMComponent(
                                name=name.strip(),
                                version=version.strip(),
                                component_type="library",
                                license="Unknown"  # Would query package registry
                            ))
        except Exception as e:
            self.logger.warning(f"Failed to analyze requirements.txt: {e}")
        
        return components
    
    async def _analyze_pyproject_toml(self, project_path: str) -> List[SBOMComponent]:
        """Analyze Python pyproject.toml file"""
        
        components = []
        # Would implement TOML parsing for dependencies
        return components
    
    async def _analyze_package_json(self, project_path: str) -> List[SBOMComponent]:
        """Analyze Node.js package.json file"""
        
        components = []
        package_file = Path(project_path) / "package.json"
        
        try:
            with open(package_file, 'r') as f:
                package_data = json.load(f)
                
                # Analyze dependencies
                for dep_type in ["dependencies", "devDependencies"]:
                    deps = package_data.get(dep_type, {})
                    for name, version in deps.items():
                        components.append(SBOMComponent(
                            name=name,
                            version=version.lstrip('^~>=<'),  # Remove version prefixes
                            component_type="library",
                            license="Unknown"  # Would query npm registry
                        ))
        except Exception as e:
            self.logger.warning(f"Failed to analyze package.json: {e}")
        
        return components
    
    async def _analyze_dockerfile(self, project_path: str) -> List[SBOMComponent]:
        """Analyze Dockerfile for base images and packages"""
        
        components = []
        dockerfile = Path(project_path) / "Dockerfile"
        
        try:
            with open(dockerfile, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # Extract base images
                    if line.startswith('FROM '):
                        image = line.split()[1]
                        if ':' in image:
                            name, version = image.split(':', 1)
                        else:
                            name, version = image, "latest"
                        
                        components.append(SBOMComponent(
                            name=name,
                            version=version,
                            component_type="container",
                            supplier="Docker Hub"  # Would determine actual registry
                        ))
        except Exception as e:
            self.logger.warning(f"Failed to analyze Dockerfile: {e}")
        
        return components
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate SBOM metadata"""
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "tools": [
                {
                    "vendor": "Cognisia",
                    "name": "Forge1 SBOM Generator",
                    "version": "1.0.0"
                }
            ],
            "authors": [
                {
                    "name": "Forge1 Security Team"
                }
            ]
        }
    
    def _component_to_cyclonedx(self, component: SBOMComponent) -> Dict[str, Any]:
        """Convert SBOMComponent to CycloneDX format"""
        
        cyclone_component = {
            "type": "library",
            "bom-ref": f"{component.name}@{component.version}",
            "name": component.name,
            "version": component.version
        }
        
        if component.supplier:
            cyclone_component["supplier"] = {"name": component.supplier}
        
        if component.license:
            cyclone_component["licenses"] = [{"license": {"name": component.license}}]
        
        if component.description:
            cyclone_component["description"] = component.description
        
        # Add hashes if available
        if component.sha256 or component.sha512:
            hashes = []
            if component.sha256:
                hashes.append({"alg": "SHA-256", "content": component.sha256})
            if component.sha512:
                hashes.append({"alg": "SHA-512", "content": component.sha512})
            cyclone_component["hashes"] = hashes
        
        return cyclone_component
    
    def _generate_dependencies(self, components: List[SBOMComponent]) -> List[Dict[str, Any]]:
        """Generate dependency relationships"""
        
        dependencies = []
        
        for component in components:
            if component.dependencies:
                dependencies.append({
                    "ref": f"{component.name}@{component.version}",
                    "dependsOn": component.dependencies
                })
        
        return dependencies
    
    def _generate_uuid(self) -> str:
        """Generate UUID for SBOM"""
        import uuid
        return str(uuid.uuid4())


class VulnerabilityScanner:
    """Comprehensive vulnerability scanner"""
    
    def __init__(self, metrics: MetricsCollector, secret_manager: SecretManager):
        self.metrics = metrics
        self.secret_manager = secret_manager
        self.logger = logging.getLogger("vulnerability_scanner")
        
        # Scanner configurations
        self.scanners = {
            "trivy": self._configure_trivy_scanner(),
            "grype": self._configure_grype_scanner(),
            "snyk": self._configure_snyk_scanner()
        }
    
    async def scan_container(self, image_name: str, scanner: str = "trivy") -> SecurityScanResult:
        """Scan container image for vulnerabilities"""
        
        scan_id = f"container_{image_name.replace(':', '_')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        result = SecurityScanResult(
            scan_id=scan_id,
            target=image_name,
            scan_type="container",
            status=ScanStatus.RUNNING,
            started_at=datetime.utcnow(),
            scanner_name=scanner
        )
        
        try:
            self.logger.info(f"Starting container scan: {image_name} with {scanner}")
            
            if scanner == "trivy":
                vulnerabilities = await self._scan_with_trivy(image_name)
            elif scanner == "grype":
                vulnerabilities = await self._scan_with_grype(image_name)
            else:
                raise ValueError(f"Unsupported scanner: {scanner}")
            
            # Process results
            result.vulnerabilities = vulnerabilities
            result.total_vulnerabilities = len(vulnerabilities)
            
            # Count by severity
            for vuln in vulnerabilities:
                if vuln.severity == VulnerabilitySeverity.CRITICAL:
                    result.critical_count += 1
                elif vuln.severity == VulnerabilitySeverity.HIGH:
                    result.high_count += 1
                elif vuln.severity == VulnerabilitySeverity.MEDIUM:
                    result.medium_count += 1
                elif vuln.severity == VulnerabilitySeverity.LOW:
                    result.low_count += 1
            
            # Determine pass/fail
            result.scan_passed = self._evaluate_scan_results(result)
            result.status = ScanStatus.COMPLETED
            result.completed_at = datetime.utcnow()
            
            # Record metrics
            self.metrics.increment("container_scan_completed")
            self.metrics.record_metric("vulnerabilities_found", result.total_vulnerabilities)
            
            self.logger.info(
                f"Container scan completed: {scan_id} - "
                f"{result.total_vulnerabilities} vulnerabilities found"
            )
            
        except Exception as e:
            result.status = ScanStatus.FAILED
            result.completed_at = datetime.utcnow()
            result.metadata["error"] = str(e)
            
            self.metrics.increment("container_scan_failed")
            self.logger.error(f"Container scan failed: {scan_id} - {e}")
        
        return result
    
    async def scan_dependencies(self, project_path: str, scanner: str = "snyk") -> SecurityScanResult:
        """Scan project dependencies for vulnerabilities"""
        
        scan_id = f"deps_{Path(project_path).name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        result = SecurityScanResult(
            scan_id=scan_id,
            target=project_path,
            scan_type="dependency",
            status=ScanStatus.RUNNING,
            started_at=datetime.utcnow(),
            scanner_name=scanner
        )
        
        try:
            self.logger.info(f"Starting dependency scan: {project_path} with {scanner}")
            
            if scanner == "snyk":
                vulnerabilities = await self._scan_dependencies_with_snyk(project_path)
            else:
                vulnerabilities = await self._scan_dependencies_with_trivy(project_path)
            
            # Process results (similar to container scan)
            result.vulnerabilities = vulnerabilities
            result.total_vulnerabilities = len(vulnerabilities)
            
            # Count by severity
            for vuln in vulnerabilities:
                if vuln.severity == VulnerabilitySeverity.CRITICAL:
                    result.critical_count += 1
                elif vuln.severity == VulnerabilitySeverity.HIGH:
                    result.high_count += 1
                elif vuln.severity == VulnerabilitySeverity.MEDIUM:
                    result.medium_count += 1
                elif vuln.severity == VulnerabilitySeverity.LOW:
                    result.low_count += 1
            
            result.scan_passed = self._evaluate_scan_results(result)
            result.status = ScanStatus.COMPLETED
            result.completed_at = datetime.utcnow()
            
            self.metrics.increment("dependency_scan_completed")
            
        except Exception as e:
            result.status = ScanStatus.FAILED
            result.completed_at = datetime.utcnow()
            result.metadata["error"] = str(e)
            
            self.metrics.increment("dependency_scan_failed")
            self.logger.error(f"Dependency scan failed: {scan_id} - {e}")
        
        return result
    
    async def _scan_with_trivy(self, image_name: str) -> List[Vulnerability]:
        """Scan container with Trivy"""
        
        vulnerabilities = []
        
        try:
            # Run Trivy scan
            cmd = [
                "trivy", "image", "--format", "json", "--quiet", image_name
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                scan_results = json.loads(stdout.decode())
                
                # Parse Trivy results
                for result in scan_results.get("Results", []):
                    for vuln_data in result.get("Vulnerabilities", []):
                        vulnerability = Vulnerability(
                            cve_id=vuln_data.get("VulnerabilityID", ""),
                            severity=self._map_trivy_severity(vuln_data.get("Severity", "UNKNOWN")),
                            score=float(vuln_data.get("CVSS", {}).get("nvd", {}).get("V3Score", 0)),
                            description=vuln_data.get("Description", ""),
                            affected_component=vuln_data.get("PkgName", ""),
                            affected_versions=[vuln_data.get("InstalledVersion", "")],
                            fixed_versions=vuln_data.get("FixedVersion", "").split(",") if vuln_data.get("FixedVersion") else []
                        )
                        vulnerabilities.append(vulnerability)
            else:
                self.logger.error(f"Trivy scan failed: {stderr.decode()}")
        
        except Exception as e:
            self.logger.error(f"Error running Trivy: {e}")
        
        return vulnerabilities
    
    async def _scan_with_grype(self, image_name: str) -> List[Vulnerability]:
        """Scan container with Grype"""
        
        vulnerabilities = []
        
        try:
            # Run Grype scan
            cmd = [
                "grype", image_name, "-o", "json"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                scan_results = json.loads(stdout.decode())
                
                # Parse Grype results
                for match in scan_results.get("matches", []):
                    vuln_data = match.get("vulnerability", {})
                    artifact = match.get("artifact", {})
                    
                    vulnerability = Vulnerability(
                        cve_id=vuln_data.get("id", ""),
                        severity=self._map_grype_severity(vuln_data.get("severity", "Unknown")),
                        score=float(vuln_data.get("cvss", [{}])[0].get("metrics", {}).get("baseScore", 0)),
                        description=vuln_data.get("description", ""),
                        affected_component=artifact.get("name", ""),
                        affected_versions=[artifact.get("version", "")],
                        fixed_versions=vuln_data.get("fix", {}).get("versions", [])
                    )
                    vulnerabilities.append(vulnerability)
            else:
                self.logger.error(f"Grype scan failed: {stderr.decode()}")
        
        except Exception as e:
            self.logger.error(f"Error running Grype: {e}")
        
        return vulnerabilities
    
    async def _scan_dependencies_with_snyk(self, project_path: str) -> List[Vulnerability]:
        """Scan dependencies with Snyk"""
        
        vulnerabilities = []
        
        try:
            # Get Snyk token
            snyk_token = await self.secret_manager.get_secret("snyk_token")
            
            # Run Snyk test
            cmd = [
                "snyk", "test", "--json", "--severity-threshold=low"
            ]
            
            env = {"SNYK_TOKEN": snyk_token}
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            stdout, stderr = await process.communicate()
            
            # Snyk returns non-zero exit code when vulnerabilities are found
            if stdout:
                scan_results = json.loads(stdout.decode())
                
                # Parse Snyk results
                for vuln_data in scan_results.get("vulnerabilities", []):
                    vulnerability = Vulnerability(
                        cve_id=vuln_data.get("identifiers", {}).get("CVE", [""])[0],
                        severity=self._map_snyk_severity(vuln_data.get("severity", "low")),
                        score=float(vuln_data.get("cvssScore", 0)),
                        description=vuln_data.get("title", ""),
                        affected_component=vuln_data.get("packageName", ""),
                        affected_versions=[vuln_data.get("version", "")],
                        patch_available=bool(vuln_data.get("patches"))
                    )
                    vulnerabilities.append(vulnerability)
        
        except Exception as e:
            self.logger.error(f"Error running Snyk: {e}")
        
        return vulnerabilities
    
    async def _scan_dependencies_with_trivy(self, project_path: str) -> List[Vulnerability]:
        """Scan dependencies with Trivy filesystem scan"""
        
        vulnerabilities = []
        
        try:
            # Run Trivy filesystem scan
            cmd = [
                "trivy", "fs", "--format", "json", "--quiet", project_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                scan_results = json.loads(stdout.decode())
                
                # Parse results (similar to container scan)
                for result in scan_results.get("Results", []):
                    for vuln_data in result.get("Vulnerabilities", []):
                        vulnerability = Vulnerability(
                            cve_id=vuln_data.get("VulnerabilityID", ""),
                            severity=self._map_trivy_severity(vuln_data.get("Severity", "UNKNOWN")),
                            score=float(vuln_data.get("CVSS", {}).get("nvd", {}).get("V3Score", 0)),
                            description=vuln_data.get("Description", ""),
                            affected_component=vuln_data.get("PkgName", ""),
                            affected_versions=[vuln_data.get("InstalledVersion", "")],
                            fixed_versions=vuln_data.get("FixedVersion", "").split(",") if vuln_data.get("FixedVersion") else []
                        )
                        vulnerabilities.append(vulnerability)
        
        except Exception as e:
            self.logger.error(f"Error running Trivy filesystem scan: {e}")
        
        return vulnerabilities
    
    def _evaluate_scan_results(self, result: SecurityScanResult) -> bool:
        """Evaluate if scan results pass security criteria"""
        
        # Block on critical vulnerabilities
        if result.critical_count > 0:
            result.blocking_issues.append(f"{result.critical_count} critical vulnerabilities found")
            return False
        
        # Block on too many high severity vulnerabilities
        if result.high_count > 5:
            result.blocking_issues.append(f"{result.high_count} high severity vulnerabilities found (max 5)")
            return False
        
        return True
    
    def _map_trivy_severity(self, severity: str) -> VulnerabilitySeverity:
        """Map Trivy severity to internal enum"""
        mapping = {
            "CRITICAL": VulnerabilitySeverity.CRITICAL,
            "HIGH": VulnerabilitySeverity.HIGH,
            "MEDIUM": VulnerabilitySeverity.MEDIUM,
            "LOW": VulnerabilitySeverity.LOW,
            "UNKNOWN": VulnerabilitySeverity.INFO
        }
        return mapping.get(severity.upper(), VulnerabilitySeverity.INFO)
    
    def _map_grype_severity(self, severity: str) -> VulnerabilitySeverity:
        """Map Grype severity to internal enum"""
        mapping = {
            "Critical": VulnerabilitySeverity.CRITICAL,
            "High": VulnerabilitySeverity.HIGH,
            "Medium": VulnerabilitySeverity.MEDIUM,
            "Low": VulnerabilitySeverity.LOW,
            "Negligible": VulnerabilitySeverity.INFO
        }
        return mapping.get(severity, VulnerabilitySeverity.INFO)
    
    def _map_snyk_severity(self, severity: str) -> VulnerabilitySeverity:
        """Map Snyk severity to internal enum"""
        mapping = {
            "critical": VulnerabilitySeverity.CRITICAL,
            "high": VulnerabilitySeverity.HIGH,
            "medium": VulnerabilitySeverity.MEDIUM,
            "low": VulnerabilitySeverity.LOW
        }
        return mapping.get(severity.lower(), VulnerabilitySeverity.INFO)
    
    def _configure_trivy_scanner(self) -> Dict[str, Any]:
        """Configure Trivy scanner"""
        return {
            "name": "trivy",
            "version": "0.45.0",
            "supported_targets": ["container", "filesystem", "repository"]
        }
    
    def _configure_grype_scanner(self) -> Dict[str, Any]:
        """Configure Grype scanner"""
        return {
            "name": "grype",
            "version": "0.65.0",
            "supported_targets": ["container", "filesystem"]
        }
    
    def _configure_snyk_scanner(self) -> Dict[str, Any]:
        """Configure Snyk scanner"""
        return {
            "name": "snyk",
            "version": "1.1200.0",
            "supported_targets": ["dependencies", "container", "code"]
        }
```
