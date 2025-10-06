"""
Marketplace Registry
Manages marketplace items, versions, dependencies, and installations
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
import json
import hashlib
from collections import defaultdict

from forge1.backend.forge1.core.monitoring import MetricsCollector
from forge1.backend.forge1.core.memory import MemoryManager
from forge1.backend.forge1.core.security import SecretManager
from forge1.backend.forge1.marketplace.models import (
    MarketplaceItem, AgentTemplate, ToolConnector, WorkflowTemplate,
    Installation, MarketplaceReview, PublishingRequest, MarketplaceMetrics,
    TenantMarketplaceConfig, Version, Dependency, SBOM, SecurityScan,
    MarketplaceItemType, PublishingStatus, InstallationStatus, CompatibilityLevel
)


class DependencyResolver:
    """Resolves and validates marketplace item dependencies"""
    
    def __init__(self):
        self.logger = logging.getLogger("dependency_resolver")
    
    def resolve_dependencies(
        self,
        item: MarketplaceItem,
        available_items: Dict[str, MarketplaceItem],
        installed_items: Dict[str, Installation]
    ) -> Tuple[List[MarketplaceItem], List[str]]:
        """
        Resolve dependencies for a marketplace item
        
        Returns:
            Tuple of (resolved_dependencies, conflicts)
        """
        
        resolved = []
        conflicts = []
        visited = set()
        
        def resolve_recursive(current_item: MarketplaceItem, path: List[str]):
            if current_item.item_id in visited:
                return
            
            if current_item.item_id in path:
                conflicts.append(f"Circular dependency detected: {' -> '.join(path + [current_item.item_id])}")
                return
            
            visited.add(current_item.item_id)
            new_path = path + [current_item.item_id]
            
            for dependency in current_item.dependencies:
                # Find matching item
                matching_item = None
                for available_item in available_items.values():
                    if (available_item.name == dependency.name and 
                        available_item.item_type == dependency.item_type):
                        if self._version_satisfies_constraint(
                            available_item.version, dependency.version_constraint
                        ):
                            matching_item = available_item
                            break
                
                if not matching_item:
                    if dependency.required:
                        conflicts.append(
                            f"Required dependency not found: {dependency.name} "
                            f"({dependency.version_constraint})"
                        )
                    continue
                
                # Check if already installed with compatible version
                if dependency.name in installed_items:
                    installed = installed_items[dependency.name]
                    if not self._version_satisfies_constraint(
                        Version.from_string(installed.item_version),
                        dependency.version_constraint
                    ):
                        conflicts.append(
                            f"Version conflict: {dependency.name} "
                            f"installed {installed.item_version}, "
                            f"required {dependency.version_constraint}"
                        )
                        continue
                
                resolved.append(matching_item)
                resolve_recursive(matching_item, new_path)
        
        resolve_recursive(item, [])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_resolved = []
        for item in resolved:
            if item.item_id not in seen:
                seen.add(item.item_id)
                unique_resolved.append(item)
        
        return unique_resolved, conflicts
    
    def _version_satisfies_constraint(self, version: Version, constraint: str) -> bool:
        """Check if version satisfies constraint (simplified implementation)"""
        # Simplified constraint checking - would use proper semver library
        if constraint.startswith(">="):
            min_version = Version.from_string(constraint[2:])
            return version >= min_version
        elif constraint.startswith("<="):
            max_version = Version.from_string(constraint[2:])
            return version <= max_version
        elif constraint.startswith("=="):
            exact_version = Version.from_string(constraint[2:])
            return version == exact_version
        else:
            # Default to exact match
            return str(version) == constraint


class SecurityValidator:
    """Validates security aspects of marketplace items"""
    
    def __init__(self, secret_manager: SecretManager):
        self.secret_manager = secret_manager
        self.logger = logging.getLogger("security_validator")
    
    async def scan_item(self, item: MarketplaceItem) -> SecurityScan:
        """Perform security scan on marketplace item"""
        
        scan_id = f"scan_{item.item_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Simulate security scanning (would integrate with actual security tools)
        await asyncio.sleep(1)  # Simulate scan time
        
        # Analyze item for security issues
        vulnerabilities = self._analyze_vulnerabilities(item)
        
        scan = SecurityScan(
            scan_id=scan_id,
            scan_date=datetime.utcnow(),
            scanner_version="1.0.0",
            vulnerabilities_found=len(vulnerabilities),
            critical_issues=len([v for v in vulnerabilities if v["severity"] == "critical"]),
            high_issues=len([v for v in vulnerabilities if v["severity"] == "high"]),
            medium_issues=len([v for v in vulnerabilities if v["severity"] == "medium"]),
            low_issues=len([v for v in vulnerabilities if v["severity"] == "low"]),
            scan_passed=len([v for v in vulnerabilities if v["severity"] in ["critical", "high"]]) == 0,
            detailed_report={"vulnerabilities": vulnerabilities}
        )
        
        self.logger.info(
            f"Security scan completed for {item.item_id}: "
            f"{scan.vulnerabilities_found} issues found, "
            f"passed: {scan.scan_passed}"
        )
        
        return scan
    
    def generate_sbom(self, item: MarketplaceItem) -> SBOM:
        """Generate Software Bill of Materials for item"""
        
        sbom_id = f"sbom_{item.item_id}_{item.version}"
        
        # Extract components from item
        components = self._extract_components(item)
        dependencies = self._extract_sbom_dependencies(item)
        licenses = self._extract_licenses(item)
        
        return SBOM(
            sbom_id=sbom_id,
            format_version="1.4",
            components=components,
            dependencies=dependencies,
            licenses=licenses
        )
    
    def _analyze_vulnerabilities(self, item: MarketplaceItem) -> List[Dict[str, Any]]:
        """Analyze item for security vulnerabilities"""
        
        vulnerabilities = []
        
        # Check for common security issues
        if hasattr(item, 'template_definition'):
            template_def = getattr(item, 'template_definition', {})
            
            # Check for hardcoded secrets
            template_str = json.dumps(template_def)
            if any(keyword in template_str.lower() for keyword in 
                   ['password', 'secret', 'key', 'token']):
                vulnerabilities.append({
                    "id": "HARDCODED_SECRETS",
                    "severity": "high",
                    "description": "Potential hardcoded secrets detected",
                    "recommendation": "Use secure secret management"
                })
            
            # Check for unsafe operations
            if any(op in template_str.lower() for op in 
                   ['eval', 'exec', 'system', 'shell']):
                vulnerabilities.append({
                    "id": "UNSAFE_OPERATIONS",
                    "severity": "critical",
                    "description": "Unsafe operations detected",
                    "recommendation": "Remove unsafe code execution"
                })
        
        # Check dependencies for known vulnerabilities
        for dependency in item.dependencies:
            if dependency.name in ["vulnerable-package", "insecure-lib"]:  # Mock vulnerable packages
                vulnerabilities.append({
                    "id": "VULNERABLE_DEPENDENCY",
                    "severity": "medium",
                    "description": f"Dependency {dependency.name} has known vulnerabilities",
                    "recommendation": "Update to secure version"
                })
        
        return vulnerabilities
    
    def _extract_components(self, item: MarketplaceItem) -> List[Dict[str, Any]]:
        """Extract components for SBOM"""
        
        components = [{
            "type": "application",
            "bom-ref": item.item_id,
            "name": item.name,
            "version": str(item.version),
            "description": item.description,
            "licenses": [{"license": {"name": item.license}}]
        }]
        
        # Add dependency components
        for dependency in item.dependencies:
            components.append({
                "type": "library",
                "bom-ref": dependency.name,
                "name": dependency.name,
                "version": dependency.version_constraint,
                "description": dependency.description or ""
            })
        
        return components
    
    def _extract_sbom_dependencies(self, item: MarketplaceItem) -> List[Dict[str, Any]]:
        """Extract dependencies for SBOM"""
        
        dependencies = []
        
        for dependency in item.dependencies:
            dependencies.append({
                "ref": item.item_id,
                "dependsOn": [dependency.name]
            })
        
        return dependencies
    
    def _extract_licenses(self, item: MarketplaceItem) -> List[str]:
        """Extract license information"""
        
        licenses = [item.license]
        
        # Add dependency licenses (would be extracted from actual packages)
        for dependency in item.dependencies:
            licenses.append("MIT")  # Mock license
        
        return list(set(licenses))


class MarketplaceRegistry:
    """
    Central registry for marketplace items with versioning, dependencies, and security
    """
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        metrics_collector: MetricsCollector,
        secret_manager: SecretManager
    ):
        self.memory_manager = memory_manager
        self.metrics = metrics_collector
        self.secret_manager = secret_manager
        self.logger = logging.getLogger("marketplace_registry")
        
        # Initialize components
        self.dependency_resolver = DependencyResolver()
        self.security_validator = SecurityValidator(secret_manager)
        
        # In-memory storage (would be backed by database in production)
        self.items: Dict[str, MarketplaceItem] = {}
        self.installations: Dict[str, Dict[str, Installation]] = defaultdict(dict)  # tenant_id -> item_id -> installation
        self.reviews: Dict[str, List[MarketplaceReview]] = defaultdict(list)  # item_id -> reviews
        self.publishing_requests: Dict[str, PublishingRequest] = {}
        self.tenant_configs: Dict[str, TenantMarketplaceConfig] = {}
        
        # Caching
        self.search_cache: Dict[str, Tuple[List[MarketplaceItem], datetime]] = {}
        self.cache_ttl = timedelta(minutes=15)
    
    async def register_item(
        self,
        item: MarketplaceItem,
        publisher_id: str,
        auto_publish: bool = False
    ) -> str:
        """
        Register a new marketplace item
        
        Args:
            item: Marketplace item to register
            publisher_id: ID of the publisher
            auto_publish: Whether to auto-publish (for internal items)
        
        Returns:
            Item ID
        """
        
        # Validate item
        validation_errors = await self._validate_item(item)
        if validation_errors:
            raise ValueError(f"Item validation failed: {validation_errors}")
        
        # Set publisher information
        item.publisher_id = publisher_id
        
        # Perform security scan
        security_scan = await self.security_validator.scan_item(item)
        item.security_scan = security_scan
        
        # Generate SBOM
        sbom = self.security_validator.generate_sbom(item)
        item.sbom = sbom
        
        # Store item
        self.items[item.item_id] = item
        
        # Auto-publish if requested and security scan passed
        if auto_publish and security_scan.scan_passed:
            item.publishing_status = PublishingStatus.PUBLISHED
            item.published_at = datetime.utcnow()
        
        # Store in persistent memory
        await self._store_item(item)
        
        # Record metrics
        self.metrics.increment("marketplace_item_registered")
        self.metrics.increment(f"marketplace_item_registered_{item.item_type.value}")
        
        self.logger.info(f"Registered marketplace item: {item.item_id} ({item.name})")
        
        return item.item_id
    
    async def install_item(
        self,
        item_id: str,
        tenant_id: str,
        user_id: str,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Installation:
        """
        Install a marketplace item for a tenant
        
        Args:
            item_id: ID of item to install
            tenant_id: Target tenant ID
            user_id: User performing installation
            custom_config: Custom configuration overrides
        
        Returns:
            Installation record
        """
        
        # Get item
        item = self.items.get(item_id)
        if not item:
            raise ValueError(f"Item not found: {item_id}")
        
        if item.publishing_status != PublishingStatus.PUBLISHED:
            raise ValueError(f"Item not published: {item_id}")
        
        # Check tenant permissions
        tenant_config = await self._get_tenant_config(tenant_id)
        if not self._check_install_permissions(item, tenant_config):
            raise ValueError(f"Installation not permitted for tenant: {tenant_id}")
        
        # Resolve dependencies
        installed_items = self.installations[tenant_id]
        resolved_deps, conflicts = self.dependency_resolver.resolve_dependencies(
            item, self.items, installed_items
        )
        
        if conflicts:
            raise ValueError(f"Dependency conflicts: {conflicts}")
        
        # Create installation record
        installation_id = f"install_{item_id}_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        installation = Installation(
            installation_id=installation_id,
            item_id=item_id,
            item_version=str(item.version),
            tenant_id=tenant_id,
            installed_by=user_id,
            installation_status=InstallationStatus.INSTALLING,
            installation_config=item.default_configuration.copy(),
            custom_configuration=custom_config or {}
        )
        
        try:
            # Install dependencies first
            for dep_item in resolved_deps:
                if dep_item.item_id not in installed_items:
                    await self.install_item(dep_item.item_id, tenant_id, user_id)
            
            # Perform installation
            await self._perform_installation(installation, item)
            
            # Update installation status
            installation.installation_status = InstallationStatus.INSTALLED
            installation.installed_at = datetime.utcnow()
            installation.health_status = "healthy"
            
            # Store installation
            self.installations[tenant_id][item_id] = installation
            await self._store_installation(installation)
            
            # Update item metrics
            item.download_count += 1
            await self._store_item(item)
            
            # Record metrics
            self.metrics.increment("marketplace_item_installed")
            self.metrics.increment(f"marketplace_item_installed_{item.item_type.value}")
            
            self.logger.info(f"Installed item {item_id} for tenant {tenant_id}")
            
            return installation
            
        except Exception as e:
            # Mark installation as failed
            installation.installation_status = InstallationStatus.FAILED
            installation.health_status = f"failed: {str(e)}"
            
            self.logger.error(f"Installation failed for {item_id}: {e}")
            raise e
    
    async def uninstall_item(
        self,
        item_id: str,
        tenant_id: str,
        user_id: str,
        force: bool = False
    ) -> bool:
        """
        Uninstall a marketplace item
        
        Args:
            item_id: ID of item to uninstall
            tenant_id: Tenant ID
            user_id: User performing uninstallation
            force: Force uninstall even with dependents
        
        Returns:
            Success status
        """
        
        # Check if item is installed
        if item_id not in self.installations[tenant_id]:
            raise ValueError(f"Item not installed: {item_id}")
        
        installation = self.installations[tenant_id][item_id]
        
        # Check for dependent items
        dependents = self._find_dependents(item_id, tenant_id)
        if dependents and not force:
            raise ValueError(f"Cannot uninstall: item has dependents: {dependents}")
        
        try:
            # Update status
            installation.installation_status = InstallationStatus.UNINSTALLING
            
            # Perform uninstallation
            await self._perform_uninstallation(installation)
            
            # Remove from installations
            del self.installations[tenant_id][item_id]
            
            # Record metrics
            self.metrics.increment("marketplace_item_uninstalled")
            
            self.logger.info(f"Uninstalled item {item_id} for tenant {tenant_id}")
            
            return True
            
        except Exception as e:
            installation.installation_status = InstallationStatus.FAILED
            self.logger.error(f"Uninstallation failed for {item_id}: {e}")
            return False
    
    async def update_item(
        self,
        item_id: str,
        tenant_id: str,
        target_version: Optional[str] = None
    ) -> Installation:
        """Update an installed item to a newer version"""
        
        # Get current installation
        if item_id not in self.installations[tenant_id]:
            raise ValueError(f"Item not installed: {item_id}")
        
        current_installation = self.installations[tenant_id][item_id]
        
        # Find target version
        if target_version:
            # Specific version requested
            target_item = None
            for item in self.items.values():
                if item.item_id == item_id and str(item.version) == target_version:
                    target_item = item
                    break
        else:
            # Find latest version
            target_item = self._find_latest_version(item_id)
        
        if not target_item:
            raise ValueError(f"Target version not found: {target_version or 'latest'}")
        
        # Check if update is needed
        if str(target_item.version) == current_installation.item_version:
            return current_installation
        
        # Store previous version for rollback
        current_installation.previous_version = current_installation.item_version
        current_installation.rollback_available = True
        
        # Update installation
        current_installation.installation_status = InstallationStatus.UPDATING
        current_installation.item_version = str(target_item.version)
        
        try:
            # Perform update
            await self._perform_update(current_installation, target_item)
            
            current_installation.installation_status = InstallationStatus.INSTALLED
            current_installation.updated_at = datetime.utcnow()
            
            # Store updated installation
            await self._store_installation(current_installation)
            
            self.logger.info(f"Updated item {item_id} to version {target_item.version}")
            
            return current_installation
            
        except Exception as e:
            current_installation.installation_status = InstallationStatus.FAILED
            self.logger.error(f"Update failed for {item_id}: {e}")
            raise e
    
    async def rollback_item(
        self,
        item_id: str,
        tenant_id: str
    ) -> Installation:
        """Rollback an item to its previous version"""
        
        installation = self.installations[tenant_id].get(item_id)
        if not installation or not installation.rollback_available:
            raise ValueError(f"Rollback not available for item: {item_id}")
        
        previous_version = installation.previous_version
        if not previous_version:
            raise ValueError(f"No previous version available for rollback: {item_id}")
        
        # Find previous version item
        previous_item = None
        for item in self.items.values():
            if item.item_id == item_id and str(item.version) == previous_version:
                previous_item = item
                break
        
        if not previous_item:
            raise ValueError(f"Previous version not found: {previous_version}")
        
        # Perform rollback
        installation.installation_status = InstallationStatus.UPDATING
        
        try:
            await self._perform_rollback(installation, previous_item)
            
            installation.item_version = previous_version
            installation.previous_version = None
            installation.rollback_available = False
            installation.installation_status = InstallationStatus.INSTALLED
            installation.updated_at = datetime.utcnow()
            
            await self._store_installation(installation)
            
            self.logger.info(f"Rolled back item {item_id} to version {previous_version}")
            
            return installation
            
        except Exception as e:
            installation.installation_status = InstallationStatus.FAILED
            self.logger.error(f"Rollback failed for {item_id}: {e}")
            raise e
    
    def search_items(
        self,
        query: Optional[str] = None,
        item_type: Optional[MarketplaceItemType] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        publisher_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[MarketplaceItem]:
        """Search marketplace items with filters"""
        
        # Check cache
        cache_key = f"{query}_{item_type}_{category}_{tags}_{publisher_id}_{limit}_{offset}"
        if cache_key in self.search_cache:
            cached_results, cached_time = self.search_cache[cache_key]
            if datetime.utcnow() - cached_time < self.cache_ttl:
                return cached_results
        
        # Filter items
        results = []
        
        for item in self.items.values():
            # Only include published items
            if item.publishing_status != PublishingStatus.PUBLISHED:
                continue
            
            # Check tenant permissions
            if tenant_id:
                tenant_config = self.tenant_configs.get(tenant_id)
                if tenant_config and not self._check_view_permissions(item, tenant_config):
                    continue
            
            # Apply filters
            if item_type and item.item_type != item_type:
                continue
            
            if category and item.category != category:
                continue
            
            if publisher_id and item.publisher_id != publisher_id:
                continue
            
            if tags and not any(tag in item.tags for tag in tags):
                continue
            
            if query:
                query_lower = query.lower()
                if not any(query_lower in field.lower() for field in [
                    item.name, item.display_name, item.description
                ]):
                    continue
            
            results.append(item)
        
        # Sort by relevance (simplified - would use proper search scoring)
        results.sort(key=lambda x: (x.rating, x.download_count), reverse=True)
        
        # Apply pagination
        paginated_results = results[offset:offset + limit]
        
        # Cache results
        self.search_cache[cache_key] = (paginated_results, datetime.utcnow())
        
        return paginated_results
    
    def get_marketplace_metrics(self) -> MarketplaceMetrics:
        """Get comprehensive marketplace metrics"""
        
        # Calculate metrics
        total_items = len([item for item in self.items.values() 
                          if item.publishing_status == PublishingStatus.PUBLISHED])
        total_downloads = sum(item.download_count for item in self.items.values())
        
        # Item type breakdown
        items_by_type = defaultdict(int)
        downloads_by_type = defaultdict(int)
        
        for item in self.items.values():
            if item.publishing_status == PublishingStatus.PUBLISHED:
                items_by_type[item.item_type.value] += 1
                downloads_by_type[item.item_type.value] += item.download_count
        
        # Most popular items
        published_items = [item for item in self.items.values() 
                          if item.publishing_status == PublishingStatus.PUBLISHED]
        
        most_downloaded = sorted(published_items, key=lambda x: x.download_count, reverse=True)[:10]
        highest_rated = sorted(published_items, key=lambda x: x.rating, reverse=True)[:10]
        
        # Publisher metrics
        publisher_stats = defaultdict(lambda: {"items": 0, "downloads": 0})
        for item in published_items:
            publisher_stats[item.publisher_id]["items"] += 1
            publisher_stats[item.publisher_id]["downloads"] += item.download_count
        
        top_publishers = sorted(
            [{"publisher_id": pid, **stats} for pid, stats in publisher_stats.items()],
            key=lambda x: x["downloads"],
            reverse=True
        )[:10]
        
        # Quality metrics
        ratings = [item.rating for item in published_items if item.rating > 0]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
        
        scanned_items = [item for item in published_items if item.security_scan]
        security_scan_pass_rate = (
            len([item for item in scanned_items if item.security_scan.scan_passed]) / 
            len(scanned_items) if scanned_items else 0.0
        )
        
        return MarketplaceMetrics(
            total_items=total_items,
            total_downloads=total_downloads,
            total_publishers=len(publisher_stats),
            total_tenants=len(self.installations),
            items_by_type=dict(items_by_type),
            downloads_by_type=dict(downloads_by_type),
            most_downloaded=[
                {"item_id": item.item_id, "name": item.name, "downloads": item.download_count}
                for item in most_downloaded
            ],
            highest_rated=[
                {"item_id": item.item_id, "name": item.name, "rating": item.rating}
                for item in highest_rated
            ],
            top_publishers=top_publishers,
            avg_rating=avg_rating,
            security_scan_pass_rate=security_scan_pass_rate
        )
    
    # Private methods
    async def _validate_item(self, item: MarketplaceItem) -> List[str]:
        """Validate marketplace item"""
        
        errors = []
        
        # Basic validation
        if not item.name or len(item.name) < 3:
            errors.append("Name must be at least 3 characters")
        
        if not item.description or len(item.description) < 10:
            errors.append("Description must be at least 10 characters")
        
        if item.item_id in self.items:
            existing_item = self.items[item.item_id]
            if existing_item.version >= item.version:
                errors.append("Version must be higher than existing version")
        
        # Type-specific validation
        if isinstance(item, AgentTemplate):
            if not item.capabilities:
                errors.append("Agent template must have capabilities")
        
        elif isinstance(item, ToolConnector):
            if not item.supported_protocols:
                errors.append("Tool connector must specify supported protocols")
        
        return errors
    
    async def _perform_installation(self, installation: Installation, item: MarketplaceItem) -> None:
        """Perform actual installation of marketplace item"""
        
        # Simulate installation process
        await asyncio.sleep(1)
        
        # Create installation artifacts
        installation.installation_path = f"/marketplace/{installation.tenant_id}/{item.item_id}"
        installation.artifacts = [
            f"{installation.installation_path}/config.json",
            f"{installation.installation_path}/metadata.json"
        ]
        
        self.logger.info(f"Installation completed for {item.item_id}")
    
    async def _perform_uninstallation(self, installation: Installation) -> None:
        """Perform actual uninstallation"""
        
        # Simulate uninstallation
        await asyncio.sleep(0.5)
        
        # Clean up artifacts
        installation.artifacts = []
        
        self.logger.info(f"Uninstallation completed for {installation.item_id}")
    
    async def _perform_update(self, installation: Installation, target_item: MarketplaceItem) -> None:
        """Perform item update"""
        
        # Simulate update process
        await asyncio.sleep(1)
        
        # Update configuration if needed
        installation.installation_config.update(target_item.default_configuration)
        
        self.logger.info(f"Update completed for {installation.item_id}")
    
    async def _perform_rollback(self, installation: Installation, previous_item: MarketplaceItem) -> None:
        """Perform item rollback"""
        
        # Simulate rollback process
        await asyncio.sleep(1)
        
        # Restore previous configuration
        installation.installation_config = previous_item.default_configuration.copy()
        
        self.logger.info(f"Rollback completed for {installation.item_id}")
    
    def _find_dependents(self, item_id: str, tenant_id: str) -> List[str]:
        """Find items that depend on the given item"""
        
        dependents = []
        
        for installation in self.installations[tenant_id].values():
            item = self.items.get(installation.item_id)
            if item:
                for dependency in item.dependencies:
                    if dependency.name == item_id:
                        dependents.append(installation.item_id)
                        break
        
        return dependents
    
    def _find_latest_version(self, item_id: str) -> Optional[MarketplaceItem]:
        """Find the latest version of an item"""
        
        versions = []
        for item in self.items.values():
            if (item.item_id == item_id and 
                item.publishing_status == PublishingStatus.PUBLISHED):
                versions.append(item)
        
        if not versions:
            return None
        
        return max(versions, key=lambda x: x.version)
    
    async def _get_tenant_config(self, tenant_id: str) -> TenantMarketplaceConfig:
        """Get or create tenant marketplace configuration"""
        
        if tenant_id not in self.tenant_configs:
            self.tenant_configs[tenant_id] = TenantMarketplaceConfig(tenant_id=tenant_id)
        
        return self.tenant_configs[tenant_id]
    
    def _check_install_permissions(self, item: MarketplaceItem, config: TenantMarketplaceConfig) -> bool:
        """Check if tenant can install the item"""
        
        # Check item type restrictions
        if item.item_type not in config.allowed_item_types:
            return False
        
        # Check publisher restrictions
        if item.publisher_id in config.blocked_publishers:
            return False
        
        # Check category restrictions
        if config.allowed_categories and item.category not in config.allowed_categories:
            return False
        
        # Check budget constraints
        if config.monthly_budget and config.current_spend + item.price > config.monthly_budget:
            return False
        
        return True
    
    def _check_view_permissions(self, item: MarketplaceItem, config: TenantMarketplaceConfig) -> bool:
        """Check if tenant can view the item"""
        
        # Similar to install permissions but more permissive
        if item.publisher_id in config.blocked_publishers:
            return False
        
        return True
    
    async def _store_item(self, item: MarketplaceItem) -> None:
        """Store item in persistent memory"""
        
        await self.memory_manager.store_context(
            context_type="marketplace_item",
            content=item.__dict__,
            metadata={
                "item_id": item.item_id,
                "item_type": item.item_type.value,
                "publisher_id": item.publisher_id
            }
        )
    
    async def _store_installation(self, installation: Installation) -> None:
        """Store installation in persistent memory"""
        
        await self.memory_manager.store_context(
            context_type="marketplace_installation",
            content=installation.__dict__,
            metadata={
                "installation_id": installation.installation_id,
                "item_id": installation.item_id,
                "tenant_id": installation.tenant_id
            }
        )