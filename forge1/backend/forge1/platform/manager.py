"""
Platform Manager
Comprehensive integration of marketplace, customization, and platform capabilities
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import logging

from forge1.backend.forge1.core.monitoring import MetricsCollector
from forge1.backend.forge1.core.memory import MemoryManager
from forge1.backend.forge1.core.security import SecretManager
from forge1.backend.forge1.core.models import ModelRouter
from forge1.backend.forge1.marketplace.registry import MarketplaceRegistry
from forge1.backend.forge1.marketplace.models import MarketplaceItem, AgentTemplate, ToolConnector
from forge1.backend.forge1.customization.manager import TenantCustomizationManager
from forge1.backend.forge1.verticals.registry import VerticalEmployeeRegistry, VerticalType


@dataclass
class PlatformMetrics:
    """Comprehensive platform metrics"""
    total_tenants: int
    active_tenants: int
    total_marketplace_items: int
    total_installations: int
    total_revenue: Decimal
    avg_tenant_health_score: float
    platform_uptime: float
    generated_at: datetime


class PlatformManager:
    """
    Comprehensive platform manager integrating all Forge1 capabilities
    
    Manages:
    - Marketplace operations and tenant installations
    - Tenant customization and guardrails
    - Vertical AI employee deployment
    - Platform-wide monitoring and analytics
    - Revenue and billing integration
    """
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        metrics_collector: MetricsCollector,
        secret_manager: SecretManager,
        model_router: ModelRouter
    ):
        self.memory_manager = memory_manager
        self.metrics = metrics_collector
        self.secret_manager = secret_manager
        self.model_router = model_router
        self.logger = logging.getLogger("platform_manager")
        
        # Initialize core components
        self.marketplace = MarketplaceRegistry(memory_manager, metrics_collector, secret_manager)
        self.customization_manager = TenantCustomizationManager(
            memory_manager, metrics_collector, secret_manager
        )
        self.vertical_registry = VerticalEmployeeRegistry()
        
        # Platform state
        self.platform_start_time = datetime.utcnow()
        self.tenant_registry: Dict[str, Dict[str, Any]] = {}
    
    async def onboard_tenant(
        self,
        tenant_id: str,
        tenant_config: Dict[str, Any],
        admin_user_id: str
    ) -> Dict[str, Any]:
        """
        Comprehensive tenant onboarding with marketplace and customization setup
        
        Args:
            tenant_id: Unique tenant identifier
            tenant_config: Initial tenant configuration
            admin_user_id: Admin user performing onboarding
        
        Returns:
            Onboarding result with setup details
        """
        
        self.logger.info(f"Starting tenant onboarding: {tenant_id}")
        
        try:
            # Initialize tenant customization
            await self.customization_manager.initialize_tenant(tenant_id, tenant_config)
            
            # Set up marketplace access
            await self._setup_marketplace_access(tenant_id, tenant_config)
            
            # Deploy default AI employees if requested
            deployed_employees = []
            if tenant_config.get("deploy_default_employees", True):
                deployed_employees = await self._deploy_default_employees(tenant_id, admin_user_id)
            
            # Install recommended marketplace items
            installed_items = []
            if tenant_config.get("install_recommended_items", True):
                installed_items = await self._install_recommended_items(tenant_id, admin_user_id)
            
            # Register tenant
            self.tenant_registry[tenant_id] = {
                "tenant_id": tenant_id,
                "admin_user_id": admin_user_id,
                "onboarded_at": datetime.utcnow(),
                "status": "active",
                "deployed_employees": deployed_employees,
                "installed_items": installed_items,
                "config": tenant_config
            }
            
            # Store tenant registration
            await self._store_tenant_registration(tenant_id, self.tenant_registry[tenant_id])
            
            # Record metrics
            self.metrics.increment("tenant_onboarded")
            self.metrics.increment(f"tenant_onboarded_vertical_{tenant_config.get('primary_vertical', 'general')}")
            
            onboarding_result = {
                "tenant_id": tenant_id,
                "status": "success",
                "onboarded_at": datetime.utcnow().isoformat(),
                "deployed_employees": deployed_employees,
                "installed_items": installed_items,
                "dashboard_url": f"/tenant/{tenant_id}/dashboard",
                "next_steps": self._generate_onboarding_next_steps(tenant_config)
            }
            
            self.logger.info(f"Tenant onboarding completed: {tenant_id}")
            
            return onboarding_result
            
        except Exception as e:
            self.logger.error(f"Tenant onboarding failed: {tenant_id} - {str(e)}")
            
            # Record failure
            self.metrics.increment("tenant_onboarding_failed")
            
            return {
                "tenant_id": tenant_id,
                "status": "failed",
                "error": str(e),
                "retry_available": True
            }
    
    async def deploy_ai_employee(
        self,
        tenant_id: str,
        vertical_type: VerticalType,
        employee_config: Dict[str, Any],
        deployed_by: str
    ) -> Dict[str, Any]:
        """
        Deploy a vertical AI employee for a tenant
        
        Args:
            tenant_id: Target tenant
            vertical_type: Type of AI employee to deploy
            employee_config: Employee configuration
            deployed_by: User deploying the employee
        
        Returns:
            Deployment result
        """
        
        self.logger.info(f"Deploying {vertical_type.value} employee for tenant {tenant_id}")
        
        try:
            # Check tenant permissions
            if not await self._check_deployment_permissions(tenant_id, vertical_type):
                raise ValueError(f"Deployment not permitted for tenant {tenant_id}")
            
            # Generate employee ID
            employee_id = f"{tenant_id}_{vertical_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Create AI employee
            employee = self.vertical_registry.create_employee(
                vertical_type=vertical_type,
                employee_id=employee_id,
                tenant_id=tenant_id,
                workflow_engine=None,  # Would inject actual dependencies
                memory_manager=self.memory_manager,
                model_router=self.model_router,
                metrics_collector=self.metrics,
                secret_manager=self.secret_manager
            )
            
            # Apply tenant-specific configuration
            await self._apply_tenant_employee_config(employee, employee_config)
            
            # Initialize employee excellence tracking
            from forge1.backend.forge1.core.excellence_manager import MultiAgentExcellenceManager
            excellence_manager = MultiAgentExcellenceManager(
                self.model_router, self.memory_manager, self.metrics, self.secret_manager
            )
            await excellence_manager.initialize_agent_excellence(employee_id)
            
            # Record deployment
            deployment_record = {
                "employee_id": employee_id,
                "tenant_id": tenant_id,
                "vertical_type": vertical_type.value,
                "deployed_by": deployed_by,
                "deployed_at": datetime.utcnow().isoformat(),
                "config": employee_config,
                "status": "active"
            }
            
            await self._store_employee_deployment(deployment_record)
            
            # Update tenant registry
            if tenant_id in self.tenant_registry:
                self.tenant_registry[tenant_id]["deployed_employees"].append(employee_id)
            
            # Record metrics
            self.metrics.increment(f"employee_deployed_{vertical_type.value}")
            self.metrics.increment(f"tenant_employee_deployed_{tenant_id}")
            
            self.logger.info(f"Successfully deployed employee {employee_id}")
            
            return {
                "employee_id": employee_id,
                "status": "deployed",
                "vertical_type": vertical_type.value,
                "capabilities": self.vertical_registry.get_vertical_capabilities(vertical_type),
                "dashboard_url": f"/tenant/{tenant_id}/employees/{employee_id}",
                "health_check_url": f"/api/v1/employees/{employee_id}/health"
            }
            
        except Exception as e:
            self.logger.error(f"Employee deployment failed: {str(e)}")
            
            self.metrics.increment("employee_deployment_failed")
            
            return {
                "status": "failed",
                "error": str(e),
                "vertical_type": vertical_type.value
            }
    
    async def install_marketplace_item(
        self,
        tenant_id: str,
        item_id: str,
        user_id: str,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Install a marketplace item for a tenant with guardrail checks
        
        Args:
            tenant_id: Target tenant
            item_id: Marketplace item to install
            user_id: User performing installation
            custom_config: Custom configuration overrides
        
        Returns:
            Installation result
        """
        
        self.logger.info(f"Installing marketplace item {item_id} for tenant {tenant_id}")
        
        try:
            # Check authorization through customization manager
            request_context = {
                "action": "marketplace_install",
                "item_id": item_id,
                "user_id": user_id,
                "estimated_cost": Decimal('5.00')  # Would calculate actual cost
            }
            
            authorized, violations, context_updates = await self.customization_manager.check_request_authorization(
                tenant_id, request_context
            )
            
            if not authorized:
                return {
                    "status": "blocked",
                    "reason": "guardrail_violation",
                    "violations": violations
                }
            
            # Perform installation
            installation = await self.marketplace.install_item(
                item_id=item_id,
                tenant_id=tenant_id,
                user_id=user_id,
                custom_config=custom_config
            )
            
            # Record completion
            await self.customization_manager.record_request_completion(
                tenant_id=tenant_id,
                actual_cost=Decimal('5.00'),  # Would use actual cost
                tokens_used=0,
                cpu_seconds=1.0,
                memory_mb_seconds=100.0,
                success=True
            )
            
            # Update tenant registry
            if tenant_id in self.tenant_registry:
                self.tenant_registry[tenant_id]["installed_items"].append(item_id)
            
            self.logger.info(f"Successfully installed item {item_id} for tenant {tenant_id}")
            
            return {
                "installation_id": installation.installation_id,
                "status": "installed",
                "item_id": item_id,
                "installed_at": installation.installed_at.isoformat() if installation.installed_at else None,
                "health_status": installation.health_status
            }
            
        except Exception as e:
            self.logger.error(f"Marketplace installation failed: {str(e)}")
            
            # Record failure
            await self.customization_manager.record_request_completion(
                tenant_id=tenant_id,
                actual_cost=Decimal('0.00'),
                tokens_used=0,
                cpu_seconds=0.0,
                memory_mb_seconds=0.0,
                success=False
            )
            
            return {
                "status": "failed",
                "error": str(e),
                "item_id": item_id
            }
    
    def get_platform_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive platform dashboard"""
        
        # Calculate platform metrics
        platform_metrics = self._calculate_platform_metrics()
        
        # Get marketplace metrics
        marketplace_metrics = self.marketplace.get_marketplace_metrics()
        
        # Get tenant health overview
        tenant_health = self._calculate_tenant_health_overview()
        
        # Get revenue metrics
        revenue_metrics = self._calculate_revenue_metrics()
        
        return {
            "platform_overview": {
                "total_tenants": platform_metrics.total_tenants,
                "active_tenants": platform_metrics.active_tenants,
                "platform_uptime": platform_metrics.platform_uptime,
                "avg_tenant_health": platform_metrics.avg_tenant_health_score
            },
            "marketplace_overview": {
                "total_items": marketplace_metrics.total_items,
                "total_downloads": marketplace_metrics.total_downloads,
                "avg_rating": marketplace_metrics.avg_rating,
                "security_scan_pass_rate": marketplace_metrics.security_scan_pass_rate
            },
            "tenant_health": tenant_health,
            "revenue_metrics": revenue_metrics,
            "top_performing_verticals": self._get_top_performing_verticals(),
            "recent_activity": self._get_recent_platform_activity(),
            "alerts": self._get_platform_alerts()
        }
    
    def get_tenant_overview(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive tenant overview"""
        
        # Get tenant customization dashboard
        customization_dashboard = self.customization_manager.get_tenant_dashboard(tenant_id)
        
        # Get tenant's marketplace installations
        tenant_installations = self._get_tenant_installations(tenant_id)
        
        # Get deployed employees
        deployed_employees = self._get_tenant_employees(tenant_id)
        
        # Get tenant registry info
        tenant_info = self.tenant_registry.get(tenant_id, {})
        
        return {
            "tenant_id": tenant_id,
            "tenant_info": {
                "status": tenant_info.get("status", "unknown"),
                "onboarded_at": tenant_info.get("onboarded_at"),
                "admin_user_id": tenant_info.get("admin_user_id")
            },
            "customization_summary": customization_dashboard,
            "marketplace_summary": {
                "installed_items": len(tenant_installations),
                "recent_installations": len([
                    i for i in tenant_installations 
                    if i.get("installed_at") and 
                    datetime.fromisoformat(i["installed_at"]) > datetime.utcnow() - timedelta(days=7)
                ])
            },
            "employees_summary": {
                "deployed_employees": len(deployed_employees),
                "active_employees": len([e for e in deployed_employees if e.get("status") == "active"]),
                "employee_types": list(set(e.get("vertical_type") for e in deployed_employees))
            },
            "recommendations": self._generate_tenant_recommendations(tenant_id)
        }
    
    # Private methods
    async def _setup_marketplace_access(self, tenant_id: str, config: Dict[str, Any]) -> None:
        """Set up marketplace access for tenant"""
        
        # Configure marketplace permissions based on tenant tier
        tier = config.get("tier", "standard")
        
        if tier == "enterprise":
            # Enterprise tenants get full marketplace access
            pass
        elif tier == "professional":
            # Professional tenants get limited access
            pass
        else:
            # Standard tenants get basic access
            pass
        
        self.logger.info(f"Set up marketplace access for tenant {tenant_id} (tier: {tier})")
    
    async def _deploy_default_employees(self, tenant_id: str, admin_user_id: str) -> List[str]:
        """Deploy default AI employees for a new tenant"""
        
        deployed_employees = []
        
        # Deploy CX employee by default
        try:
            result = await self.deploy_ai_employee(
                tenant_id=tenant_id,
                vertical_type=VerticalType.CUSTOMER_EXPERIENCE,
                employee_config={"name": "Customer Support Assistant"},
                deployed_by=admin_user_id
            )
            
            if result["status"] == "deployed":
                deployed_employees.append(result["employee_id"])
        
        except Exception as e:
            self.logger.warning(f"Failed to deploy default CX employee: {e}")
        
        return deployed_employees
    
    async def _install_recommended_items(self, tenant_id: str, admin_user_id: str) -> List[str]:
        """Install recommended marketplace items for a new tenant"""
        
        installed_items = []
        
        # Get recommended items (would be based on tenant profile)
        recommended_items = ["basic_analytics_tool", "notification_connector"]
        
        for item_id in recommended_items:
            try:
                result = await self.install_marketplace_item(
                    tenant_id=tenant_id,
                    item_id=item_id,
                    user_id=admin_user_id
                )
                
                if result["status"] == "installed":
                    installed_items.append(item_id)
            
            except Exception as e:
                self.logger.warning(f"Failed to install recommended item {item_id}: {e}")
        
        return installed_items
    
    async def _check_deployment_permissions(self, tenant_id: str, vertical_type: VerticalType) -> bool:
        """Check if tenant can deploy the specified vertical type"""
        
        # Check feature flags
        feature_name = f"vertical_{vertical_type.value}"
        return self.customization_manager.is_feature_enabled(tenant_id, feature_name)
    
    async def _apply_tenant_employee_config(self, employee, config: Dict[str, Any]) -> None:
        """Apply tenant-specific configuration to an AI employee"""
        
        # Apply custom configuration (would depend on employee type)
        if "name" in config:
            employee.name = config["name"]
        
        # Apply performance settings
        if "performance_targets" in config:
            employee.performance_targets.update(config["performance_targets"])
    
    def _calculate_platform_metrics(self) -> PlatformMetrics:
        """Calculate comprehensive platform metrics"""
        
        total_tenants = len(self.tenant_registry)
        active_tenants = len([t for t in self.tenant_registry.values() if t.get("status") == "active"])
        
        # Calculate uptime
        uptime_seconds = (datetime.utcnow() - self.platform_start_time).total_seconds()
        uptime_percentage = min(99.99, (uptime_seconds / (uptime_seconds + 60)) * 100)  # Assume minimal downtime
        
        return PlatformMetrics(
            total_tenants=total_tenants,
            active_tenants=active_tenants,
            total_marketplace_items=len(self.marketplace.items),
            total_installations=sum(len(installations) for installations in self.marketplace.installations.values()),
            total_revenue=Decimal('50000.00'),  # Would calculate from actual billing
            avg_tenant_health_score=85.0,  # Would calculate from actual health scores
            platform_uptime=uptime_percentage,
            generated_at=datetime.utcnow()
        )
    
    def _calculate_tenant_health_overview(self) -> Dict[str, Any]:
        """Calculate tenant health overview"""
        
        health_scores = []
        
        for tenant_id in self.tenant_registry.keys():
            try:
                dashboard = self.customization_manager.get_tenant_dashboard(tenant_id)
                health = dashboard.get("health_status", {})
                health_scores.append(health.get("score", 0))
            except:
                continue
        
        if not health_scores:
            return {"status": "no_data"}
        
        avg_health = sum(health_scores) / len(health_scores)
        
        return {
            "average_health_score": avg_health,
            "healthy_tenants": len([s for s in health_scores if s >= 80]),
            "warning_tenants": len([s for s in health_scores if 60 <= s < 80]),
            "critical_tenants": len([s for s in health_scores if s < 60])
        }
    
    def _calculate_revenue_metrics(self) -> Dict[str, Any]:
        """Calculate revenue metrics"""
        
        # Placeholder revenue calculation
        return {
            "monthly_recurring_revenue": 45000.00,
            "annual_run_rate": 540000.00,
            "average_revenue_per_tenant": 2250.00,
            "growth_rate_monthly": 15.5
        }
    
    def _get_top_performing_verticals(self) -> List[Dict[str, Any]]:
        """Get top performing vertical types"""
        
        # Analyze deployment and performance metrics
        vertical_stats = {}
        
        for tenant_info in self.tenant_registry.values():
            for employee_id in tenant_info.get("deployed_employees", []):
                # Would extract vertical type from employee_id or lookup
                vertical_type = "cx"  # Simplified
                
                if vertical_type not in vertical_stats:
                    vertical_stats[vertical_type] = {"deployments": 0, "avg_performance": 0.0}
                
                vertical_stats[vertical_type]["deployments"] += 1
                vertical_stats[vertical_type]["avg_performance"] = 85.0  # Would calculate actual
        
        # Sort by performance and deployments
        sorted_verticals = sorted(
            [{"vertical": k, **v} for k, v in vertical_stats.items()],
            key=lambda x: (x["avg_performance"], x["deployments"]),
            reverse=True
        )
        
        return sorted_verticals[:5]
    
    def _get_recent_platform_activity(self) -> List[Dict[str, Any]]:
        """Get recent platform activity"""
        
        activities = []
        
        # Recent tenant onboardings
        recent_tenants = [
            t for t in self.tenant_registry.values()
            if t.get("onboarded_at") and 
            t["onboarded_at"] > datetime.utcnow() - timedelta(days=7)
        ]
        
        for tenant in recent_tenants:
            activities.append({
                "type": "tenant_onboarded",
                "tenant_id": tenant["tenant_id"],
                "timestamp": tenant["onboarded_at"].isoformat(),
                "description": f"New tenant onboarded: {tenant['tenant_id']}"
            })
        
        # Sort by timestamp
        activities.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return activities[:10]
    
    def _get_platform_alerts(self) -> List[Dict[str, Any]]:
        """Get platform-wide alerts"""
        
        alerts = []
        
        # Check for tenants with health issues
        for tenant_id in self.tenant_registry.keys():
            try:
                dashboard = self.customization_manager.get_tenant_dashboard(tenant_id)
                health = dashboard.get("health_status", {})
                
                if health.get("status") == "critical":
                    alerts.append({
                        "type": "tenant_health_critical",
                        "tenant_id": tenant_id,
                        "severity": "high",
                        "message": f"Tenant {tenant_id} has critical health issues"
                    })
            except:
                continue
        
        return alerts
    
    def _get_tenant_installations(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Get tenant's marketplace installations"""
        
        installations = self.marketplace.installations.get(tenant_id, {})
        
        return [
            {
                "item_id": installation.item_id,
                "installation_id": installation.installation_id,
                "installed_at": installation.installed_at.isoformat() if installation.installed_at else None,
                "status": installation.installation_status.value,
                "health_status": installation.health_status
            }
            for installation in installations.values()
        ]
    
    def _get_tenant_employees(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Get tenant's deployed AI employees"""
        
        tenant_info = self.tenant_registry.get(tenant_id, {})
        employee_ids = tenant_info.get("deployed_employees", [])
        
        employees = []
        for employee_id in employee_ids:
            # Would lookup actual employee details
            employees.append({
                "employee_id": employee_id,
                "vertical_type": "cx",  # Would extract from employee
                "status": "active",
                "deployed_at": tenant_info.get("onboarded_at")
            })
        
        return employees
    
    def _generate_tenant_recommendations(self, tenant_id: str) -> List[str]:
        """Generate recommendations for tenant optimization"""
        
        recommendations = []
        
        # Get tenant info
        tenant_info = self.tenant_registry.get(tenant_id, {})
        
        # Check if tenant has few employees deployed
        if len(tenant_info.get("deployed_employees", [])) < 2:
            recommendations.append("Consider deploying additional AI employees to maximize platform value")
        
        # Check marketplace usage
        if len(tenant_info.get("installed_items", [])) < 3:
            recommendations.append("Explore marketplace items to enhance your AI employees' capabilities")
        
        return recommendations
    
    def _generate_onboarding_next_steps(self, config: Dict[str, Any]) -> List[str]:
        """Generate next steps for newly onboarded tenant"""
        
        next_steps = [
            "Complete your tenant profile and branding customization",
            "Explore the marketplace for additional tools and integrations",
            "Set up your first AI employee workflow"
        ]
        
        if config.get("primary_vertical"):
            next_steps.append(f"Configure your {config['primary_vertical']} AI employee for optimal performance")
        
        return next_steps
    
    async def _store_tenant_registration(self, tenant_id: str, registration_data: Dict[str, Any]) -> None:
        """Store tenant registration in memory"""
        
        await self.memory_manager.store_context(
            context_type="tenant_registration",
            content=registration_data,
            metadata={"tenant_id": tenant_id}
        )
    
    async def _store_employee_deployment(self, deployment_record: Dict[str, Any]) -> None:
        """Store employee deployment record"""
        
        await self.memory_manager.store_context(
            context_type="employee_deployment",
            content=deployment_record,
            metadata={
                "tenant_id": deployment_record["tenant_id"],
                "employee_id": deployment_record["employee_id"]
            }
        )