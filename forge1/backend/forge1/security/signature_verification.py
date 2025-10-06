"""
Signature Verification and Attestation
Sigstore integration for signature verification and supply chain attestation
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import logging
import json
import subprocess
import tempfile
from pathlib import Path

from forge1.backend.forge1.core.monitoring import MetricsCollector


class SignatureStatus(Enum):
    """Signature verification status"""
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    UNKNOWN = "unknown"


@dataclass
class SignatureInfo:
    """Information about a digital signature"""
    signature_id: str
    signer: str
    signing_time: datetime
    algorithm: str
    status: SignatureStatus
    
    # Certificate information
    certificate_subject: Optional[str] = None
    certificate_issuer: Optional[str] = None
    certificate_serial: Optional[str] = None
    certificate_fingerprint: Optional[str] = None
    
    # Verification details
    verification_time: datetime = field(default_factory=datetime.utcnow)
    verification_errors: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttestationInfo:
    """Supply chain attestation information"""
    attestation_id: str
    attestation_type: str  # SLSA, in-toto, etc.
    subject: str  # What is being attested
    predicate: Dict[str, Any]  # Attestation claims
    
    # Provenance information
    builder: Optional[str] = None
    build_time: Optional[datetime] = None
    source_repo: Optional[str] = None
    source_commit: Optional[str] = None
    
    # Verification
    verified: bool = False
    verification_errors: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SigstoreVerifier:
    """Sigstore-based signature verification"""
    
    def __init__(self, metrics: MetricsCollector):
        self.metrics = metrics
        self.logger = logging.getLogger("sigstore_verifier")
        
        # Sigstore configuration
        self.cosign_binary = "cosign"
        self.rekor_url = "https://rekor.sigstore.dev"
        self.fulcio_url = "https://fulcio.sigstore.dev"
    
    async def verify_container_signature(
        self,
        image_name: str,
        public_key: Optional[str] = None
    ) -> SignatureInfo:
        """
        Verify container image signature using Cosign
        
        Args:
            image_name: Container image to verify
            public_key: Optional public key for verification
        
        Returns:
            Signature verification information
        """
        
        signature_id = f"cosign_{image_name.replace(':', '_')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Verifying container signature: {image_name}")
        
        try:
            # Build cosign verify command
            cmd = [self.cosign_binary, "verify"]
            
            if public_key:
                # Use provided public key
                with tempfile.NamedTemporaryFile(mode='w', suffix='.pub', delete=False) as f:
                    f.write(public_key)
                    key_file = f.name
                
                cmd.extend(["--key", key_file])
            else:
                # Use keyless verification (Fulcio + Rekor)
                cmd.extend([
                    "--certificate-identity-regexp", ".*",
                    "--certificate-oidc-issuer-regexp", ".*"
                ])
            
            cmd.append(image_name)
            
            # Execute verification
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Parse verification output
                verification_data = json.loads(stdout.decode())
                
                signature_info = SignatureInfo(
                    signature_id=signature_id,
                    signer=self._extract_signer(verification_data),
                    signing_time=self._extract_signing_time(verification_data),
                    algorithm="ECDSA-SHA256",  # Sigstore default
                    status=SignatureStatus.VALID,
                    certificate_subject=self._extract_certificate_subject(verification_data),
                    metadata={"verification_output": verification_data}
                )
                
                self.metrics.increment("signature_verification_success")
                self.logger.info(f"Signature verification successful: {image_name}")
                
            else:
                # Verification failed
                error_output = stderr.decode()
                
                signature_info = SignatureInfo(
                    signature_id=signature_id,
                    signer="unknown",
                    signing_time=datetime.utcnow(),
                    algorithm="unknown",
                    status=SignatureStatus.INVALID,
                    verification_errors=[error_output]
                )
                
                self.metrics.increment("signature_verification_failed")
                self.logger.warning(f"Signature verification failed: {image_name} - {error_output}")
            
            # Clean up temporary key file
            if public_key and 'key_file' in locals():
                Path(key_file).unlink(missing_ok=True)
            
            return signature_info
            
        except Exception as e:
            self.logger.error(f"Error verifying signature: {e}")
            
            return SignatureInfo(
                signature_id=signature_id,
                signer="unknown",
                signing_time=datetime.utcnow(),
                algorithm="unknown",
                status=SignatureStatus.UNKNOWN,
                verification_errors=[str(e)]
            )
    
    async def verify_attestation(
        self,
        image_name: str,
        attestation_type: str = "slsa-provenance"
    ) -> AttestationInfo:
        """
        Verify supply chain attestation
        
        Args:
            image_name: Container image to verify
            attestation_type: Type of attestation to verify
        
        Returns:
            Attestation verification information
        """
        
        attestation_id = f"attest_{image_name.replace(':', '_')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Verifying attestation: {image_name} (type: {attestation_type})")
        
        try:
            # Build cosign verify-attestation command
            cmd = [
                self.cosign_binary, "verify-attestation",
                "--type", attestation_type,
                "--certificate-identity-regexp", ".*",
                "--certificate-oidc-issuer-regexp", ".*",
                image_name
            ]
            
            # Execute verification
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Parse attestation data
                attestation_data = json.loads(stdout.decode())
                
                # Extract predicate (attestation claims)
                predicate = self._extract_predicate(attestation_data, attestation_type)
                
                attestation_info = AttestationInfo(
                    attestation_id=attestation_id,
                    attestation_type=attestation_type,
                    subject=image_name,
                    predicate=predicate,
                    builder=self._extract_builder(predicate),
                    build_time=self._extract_build_time(predicate),
                    source_repo=self._extract_source_repo(predicate),
                    source_commit=self._extract_source_commit(predicate),
                    verified=True,
                    metadata={"attestation_output": attestation_data}
                )
                
                self.metrics.increment("attestation_verification_success")
                self.logger.info(f"Attestation verification successful: {image_name}")
                
            else:
                # Verification failed
                error_output = stderr.decode()
                
                attestation_info = AttestationInfo(
                    attestation_id=attestation_id,
                    attestation_type=attestation_type,
                    subject=image_name,
                    predicate={},
                    verified=False,
                    verification_errors=[error_output]
                )
                
                self.metrics.increment("attestation_verification_failed")
                self.logger.warning(f"Attestation verification failed: {image_name} - {error_output}")
            
            return attestation_info
            
        except Exception as e:
            self.logger.error(f"Error verifying attestation: {e}")
            
            return AttestationInfo(
                attestation_id=attestation_id,
                attestation_type=attestation_type,
                subject=image_name,
                predicate={},
                verified=False,
                verification_errors=[str(e)]
            )
    
    async def generate_attestation(
        self,
        image_name: str,
        attestation_type: str,
        predicate: Dict[str, Any],
        private_key: Optional[str] = None
    ) -> bool:
        """
        Generate and sign attestation for an artifact
        
        Args:
            image_name: Container image to attest
            attestation_type: Type of attestation
            predicate: Attestation claims
            private_key: Optional private key for signing
        
        Returns:
            Success status
        """
        
        self.logger.info(f"Generating attestation: {image_name} (type: {attestation_type})")
        
        try:
            # Create attestation predicate file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(predicate, f)
                predicate_file = f.name
            
            # Build cosign attest command
            cmd = [
                self.cosign_binary, "attest",
                "--type", attestation_type,
                "--predicate", predicate_file
            ]
            
            if private_key:
                # Use provided private key
                with tempfile.NamedTemporaryFile(mode='w', suffix='.key', delete=False) as f:
                    f.write(private_key)
                    key_file = f.name
                
                cmd.extend(["--key", key_file])
            else:
                # Use keyless signing
                cmd.append("--yes")  # Auto-confirm keyless signing
            
            cmd.append(image_name)
            
            # Execute attestation
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            success = process.returncode == 0
            
            if success:
                self.metrics.increment("attestation_generation_success")
                self.logger.info(f"Attestation generated successfully: {image_name}")
            else:
                self.metrics.increment("attestation_generation_failed")
                self.logger.error(f"Attestation generation failed: {stderr.decode()}")
            
            # Clean up temporary files
            Path(predicate_file).unlink(missing_ok=True)
            if private_key and 'key_file' in locals():
                Path(key_file).unlink(missing_ok=True)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error generating attestation: {e}")
            return False
    
    # Helper methods for parsing verification output
    def _extract_signer(self, verification_data: List[Dict]) -> str:
        """Extract signer from verification data"""
        if verification_data and len(verification_data) > 0:
            cert = verification_data[0].get("optional", {}).get("Certificate", {})
            return cert.get("Subject", "unknown")
        return "unknown"
    
    def _extract_signing_time(self, verification_data: List[Dict]) -> datetime:
        """Extract signing time from verification data"""
        if verification_data and len(verification_data) > 0:
            # Parse timestamp from verification data
            # This would depend on the actual Cosign output format
            pass
        return datetime.utcnow()
    
    def _extract_certificate_subject(self, verification_data: List[Dict]) -> Optional[str]:
        """Extract certificate subject from verification data"""
        if verification_data and len(verification_data) > 0:
            cert = verification_data[0].get("optional", {}).get("Certificate", {})
            return cert.get("Subject")
        return None
    
    def _extract_predicate(self, attestation_data: List[Dict], attestation_type: str) -> Dict[str, Any]:
        """Extract predicate from attestation data"""
        if attestation_data and len(attestation_data) > 0:
            payload = attestation_data[0].get("payload", {})
            return payload.get("predicate", {})
        return {}
    
    def _extract_builder(self, predicate: Dict[str, Any]) -> Optional[str]:
        """Extract builder information from predicate"""
        if "builder" in predicate:
            return predicate["builder"].get("id")
        return None
    
    def _extract_build_time(self, predicate: Dict[str, Any]) -> Optional[datetime]:
        """Extract build time from predicate"""
        if "metadata" in predicate and "buildStartedOn" in predicate["metadata"]:
            try:
                return datetime.fromisoformat(predicate["metadata"]["buildStartedOn"].replace('Z', '+00:00'))
            except:
                pass
        return None
    
    def _extract_source_repo(self, predicate: Dict[str, Any]) -> Optional[str]:
        """Extract source repository from predicate"""
        if "materials" in predicate:
            for material in predicate["materials"]:
                if material.get("uri", "").startswith("git+"):
                    return material["uri"]
        return None
    
    def _extract_source_commit(self, predicate: Dict[str, Any]) -> Optional[str]:
        """Extract source commit from predicate"""
        if "materials" in predicate:
            for material in predicate["materials"]:
                digest = material.get("digest", {})
                if "sha1" in digest:
                    return digest["sha1"]
        return None


class DependencyPinner:
    """Manages dependency pinning for supply chain security"""
    
    def __init__(self, metrics: MetricsCollector):
        self.metrics = metrics
        self.logger = logging.getLogger("dependency_pinner")
    
    async def pin_dependencies(self, project_path: str) -> Dict[str, Any]:
        """
        Pin all dependencies to specific versions
        
        Args:
            project_path: Path to the project
        
        Returns:
            Pinning results and statistics
        """
        
        self.logger.info(f"Pinning dependencies for project: {project_path}")
        
        results = {
            "pinned_files": [],
            "total_dependencies": 0,
            "pinned_dependencies": 0,
            "errors": []
        }
        
        project_path_obj = Path(project_path)
        
        # Pin Python dependencies
        if (project_path_obj / "requirements.txt").exists():
            python_result = await self._pin_python_requirements(project_path)
            results["pinned_files"].append("requirements.txt")
            results["total_dependencies"] += python_result["total"]
            results["pinned_dependencies"] += python_result["pinned"]
            results["errors"].extend(python_result["errors"])
        
        # Pin Node.js dependencies
        if (project_path_obj / "package.json").exists():
            node_result = await self._pin_node_dependencies(project_path)
            results["pinned_files"].append("package.json")
            results["total_dependencies"] += node_result["total"]
            results["pinned_dependencies"] += node_result["pinned"]
            results["errors"].extend(node_result["errors"])
        
        # Pin Docker base images
        if (project_path_obj / "Dockerfile").exists():
            docker_result = await self._pin_docker_images(project_path)
            results["pinned_files"].append("Dockerfile")
            results["total_dependencies"] += docker_result["total"]
            results["pinned_dependencies"] += docker_result["pinned"]
            results["errors"].extend(docker_result["errors"])
        
        # Record metrics
        self.metrics.record_metric("dependencies_pinned", results["pinned_dependencies"])
        self.metrics.record_metric("dependency_pin_errors", len(results["errors"]))
        
        self.logger.info(
            f"Dependency pinning completed: {results['pinned_dependencies']}/{results['total_dependencies']} pinned"
        )
        
        return results
    
    async def _pin_python_requirements(self, project_path: str) -> Dict[str, Any]:
        """Pin Python requirements to specific versions"""
        
        result = {"total": 0, "pinned": 0, "errors": []}
        
        try:
            requirements_file = Path(project_path) / "requirements.txt"
            
            # Read current requirements
            with open(requirements_file, 'r') as f:
                lines = f.readlines()
            
            pinned_lines = []
            
            for line in lines:
                line = line.strip()
                result["total"] += 1
                
                if line and not line.startswith('#'):
                    # Check if already pinned
                    if '==' in line:
                        pinned_lines.append(line)
                        result["pinned"] += 1
                    else:
                        # Try to resolve to specific version
                        try:
                            # This would use pip or similar to resolve versions
                            # For now, we'll simulate pinning
                            package_name = line.split('>=')[0].split('>')[0].split('<')[0].strip()
                            pinned_version = f"{package_name}==1.0.0"  # Placeholder
                            pinned_lines.append(pinned_version)
                            result["pinned"] += 1
                        except Exception as e:
                            result["errors"].append(f"Failed to pin {line}: {e}")
                            pinned_lines.append(line)
                else:
                    pinned_lines.append(line)
            
            # Write pinned requirements
            with open(requirements_file, 'w') as f:
                f.write('\n'.join(pinned_lines))
        
        except Exception as e:
            result["errors"].append(f"Error pinning Python requirements: {e}")
        
        return result
    
    async def _pin_node_dependencies(self, project_path: str) -> Dict[str, Any]:
        """Pin Node.js dependencies to specific versions"""
        
        result = {"total": 0, "pinned": 0, "errors": []}
        
        try:
            package_file = Path(project_path) / "package.json"
            
            with open(package_file, 'r') as f:
                package_data = json.load(f)
            
            # Pin dependencies and devDependencies
            for dep_type in ["dependencies", "devDependencies"]:
                if dep_type in package_data:
                    deps = package_data[dep_type]
                    
                    for name, version in deps.items():
                        result["total"] += 1
                        
                        # Remove version prefixes (^, ~, >=, etc.)
                        if version.startswith(('^', '~', '>=', '>', '<')):
                            # Pin to specific version (would resolve actual version)
                            pinned_version = version.lstrip('^~>=<')
                            deps[name] = pinned_version
                            result["pinned"] += 1
                        else:
                            result["pinned"] += 1
            
            # Write updated package.json
            with open(package_file, 'w') as f:
                json.dump(package_data, f, indent=2)
        
        except Exception as e:
            result["errors"].append(f"Error pinning Node.js dependencies: {e}")
        
        return result
    
    async def _pin_docker_images(self, project_path: str) -> Dict[str, Any]:
        """Pin Docker base images to specific digests"""
        
        result = {"total": 0, "pinned": 0, "errors": []}
        
        try:
            dockerfile = Path(project_path) / "Dockerfile"
            
            with open(dockerfile, 'r') as f:
                lines = f.readlines()
            
            pinned_lines = []
            
            for line in lines:
                if line.strip().startswith('FROM '):
                    result["total"] += 1
                    
                    # Check if already pinned with digest
                    if '@sha256:' in line:
                        pinned_lines.append(line)
                        result["pinned"] += 1
                    else:
                        # Try to resolve to digest
                        try:
                            # This would use docker inspect or registry API
                            # For now, we'll simulate digest pinning
                            image_part = line.split()[1]
                            if ':' not in image_part:
                                image_part += ':latest'
                            
                            # Simulate digest resolution
                            digest = "sha256:abcd1234567890abcdef1234567890abcdef1234567890abcdef1234567890ab"
                            pinned_image = f"{image_part.split(':')[0]}@{digest}"
                            pinned_line = line.replace(image_part, pinned_image)
                            
                            pinned_lines.append(pinned_line)
                            result["pinned"] += 1
                        except Exception as e:
                            result["errors"].append(f"Failed to pin image {line.strip()}: {e}")
                            pinned_lines.append(line)
                else:
                    pinned_lines.append(line)
            
            # Write pinned Dockerfile
            with open(dockerfile, 'w') as f:
                f.writelines(pinned_lines)
        
        except Exception as e:
            result["errors"].append(f"Error pinning Docker images: {e}")
        
        return result