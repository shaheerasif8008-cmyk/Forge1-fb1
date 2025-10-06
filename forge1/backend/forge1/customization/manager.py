"""
Tenant Customization Manager
Comprehensive management of tenant customizations, guardrails, and platform configuration
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import logging
import json

from forge1.backend.forge1.core.monitoring import MetricsCollector
from forge1.backend.forge1.core.memory import MemoryManager
from forge1.backend.forge1.core.security import SecretManager
from forge1.backend.forge1.customization.guardrails import (
    GuardrailEngine, GuardrailRule, TenantQuota, TenantCustomization,
    GuardrailType, ViolationSeverity, ActionType
)


@dataclass
class TenantBudgetAlert:
    """Budget alert configuration for tenants"""
    alert_id: str
    tenant_id: str
    alert_type: str  # cost, quota, usage
    threshold_percentage: float  # 0.0 to 1.0
    notification_channels: List[str]
    enabled: bool = True
    last_triggered: Optional[datetime] = None


class TenantCustomizationManager:
    """
    Comprehensive manager for tenant customizations, guardrails, and platform configuration
    
    Manages:
    - Tenant-specific guardrails and policies
    - Cost caps and execution quotas
    - Budget monitoring and alerts
    - Custom branding and UI configuration
    - Feature toggles and permissions
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
        self.logger = logging.getLogger("tenant_customization_manager")
        
        # Initialize guardrail engine
        self.guardrail_engine = GuardrailEngine(memory_manager, metrics_collector)
        
        # Budget and alert management
        self.budget_alerts: Dict[str, List[TenantBudgetAlert]] = {}
        
        # Feature flags and permissions
        self.tenant_features: Dict[str, Dict[str, bool]] = {}
        
        # Custom configurations
        self.tenant_configs: Dict[str, Dict[str, Any]] = {}
    
    async def initialize_tenant(
        self,
        tenant_id: str,
        initial_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize a new tenant with default configuration"""
        
        self.logger.info(f"Initializing tenant: {tenant_id}")
        
        # Create default quota
        default_quota = TenantQuota(
            tenant_id=tenant_id,
            max_requests_per_hour=1000,
            max_requests_per_day=10000,
            max_concurrent_requests=50,
            max_cost_per_hour=Decimal('100.00'),
            max_cost_per_day=Decimal('1000.00'),
            max_cost_per_month=Decimal('10000.00'),
            max_tokens_per_hour=100000,
            max_tokens_per_day=1000000
        )
        
        await self.guardrail_engine.update_tenant_quota(tenant_id, default_quota)
        
        # Create default customization
        default_customization = TenantCustomization(
            tenant_id=tenant_id,
            brand_name="Forge1",
            enabled_features=["ai_employees", "marketplace", "analytics"],
            default_model_preferences={
                "fast": "gpt-4o-mini",
                "balanced": "gpt-4o",
                "reasoning": "claude-3.5-sonnet"
            }
        )
        
        await self.guardrail_engine.update_tenant_customization(tenant_id, default_customization)
        
        # Apply initial configuration if provided
        if initial_config:
            await self.update_tenant_configuration(tenant_id, initial_config)
        
        # Set up default budget alerts
        await self._setup_default_budget_alerts(tenant_id)
        
        # Record metrics
        self.metrics.increment("tenant_initialized")
        
        self.logger.info(f"Tenant initialization completed: {tenant_id}")
    
    async def update_tenant_quota(
        self,
        tenant_id: str,
        quota_updates: Dict[str, Any],
        updated_by: str
    ) -> TenantQuota:
        """Update tenant resource quotas"""
        
        # Get current quota
        current_quota = self.guardrail_engine.tenant_quotas.get(tenant_id)
        if not current_quota:
            current_quota = TenantQuota(tenant_id=tenant_id)
        
        # Apply updates
        for key, value in quota_updates.items():
            if hasattr(current_quota, key):
                if key.startswith("max_cost"):
                    setattr(current_quota, key, Decimal(str(value)))
                else:
                    setattr(current_quota, key, value)
        
        # Update quota
        await self.guardrail_engine.update_tenant_quota(tenant_id, current_quota)
        
        # Log change
        self.logger.info(f"Updated quota for tenant {tenant_id} by {updated_by}: {quota_updates}")
        
        # Record metrics
        self.metrics.increment(f"tenant_quota_updated_{tenant_id}")
        
        return current_quota
    
    async def add_custom_guardrail(
        self,
        tenant_id: str,
        rule_config: Dict[str, Any],
        created_by: str
    ) -> str:
        """Add a custom guardrail rule for a tenant"""
        
        # Create guardrail rule
        rule = GuardrailRule(
            rule_id=f"custom_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            name=rule_config["name"],
            description=rule_config["description"],
            rule_type=GuardrailType(rule_config["rule_type"]),
            pattern=rule_config.get("pattern", ""),
            action=ActionType(rule_config["action"]),
            severity=ViolationSeverity(rule_config["severity"]),
            threshold=Decimal(str(rule_config["threshold"])) if rule_config.get("threshold") else None,
            time_window=timedelta(seconds=rule_config["time_window_seconds"]) if rule_config.get("time_window_seconds") else None,
            custom_config=rule_config.get("custom_config", {})
        )
        
        # Add rule
        rule_id = await self.guardrail_engine.add_guardrail_rule(tenant_id, rule, created_by)
        
        self.logger.info(f"Added custom guardrail {rule_id} for tenant {tenant_id}")
        
        return rule_id
    
    async def update_tenant_branding(
        self,
        tenant_id: str,
        branding_config: Dict[str, Any]
    ) -> None:
        """Update tenant branding and UI customization"""
        
        # Get current customization
        customization = self.guardrail_engine.tenant_customizations.get(tenant_id)
        if not customization:
            customization = TenantCustomization(tenant_id=tenant_id)
        
        # Update branding fields
        if "brand_name" in branding_config:
            customization.brand_name = branding_config["brand_name"]
        
        if "brand_logo_url" in branding_config:
            customization.brand_logo_url = branding_config["brand_logo_url"]
        
        if "primary_color" in branding_config:
            customization.primary_color = branding_config["primary_color"]
        
        if "secondary_color" in branding_config:
            customization.secondary_color = branding_config["secondary_color"]
        
        if "custom_css" in branding_config:
            customization.custom_css = branding_config["custom_css"]
        
        # Update customization
        await self.guardrail_engine.update_tenant_customization(tenant_id, customization)
        
        self.logger.info(f"Updated branding for tenant {tenant_id}")
    
    async def configure_feature_toggles(
        self,
        tenant_id: str,
        feature_config: Dict[str, bool]
    ) -> None:
        """Configure feature toggles for a tenant"""
        
        # Get current customization
        customization = self.guardrail_engine.tenant_customizations.get(tenant_id)
        if not customization:
            customization = TenantCustomization(tenant_id=tenant_id)
        
        # Update feature lists
        enabled_features = []
        disabled_features = []
        
        for feature, enabled in feature_config.items():
            if enabled:
                enabled_features.append(feature)
            else:
                disabled_features.append(feature)
        
        customization.enabled_features = enabled_features
        customization.disabled_features = disabled_features
        
        # Update customization
        await self.guardrail_engine.update_tenant_customization(tenant_id, customization)
        
        # Store feature flags
        self.tenant_features[tenant_id] = feature_config
        
        self.logger.info(f"Updated feature toggles for tenant {tenant_id}: {feature_config}")
    
    async def setup_budget_alerts(
        self,
        tenant_id: str,
        alert_configs: List[Dict[str, Any]]
    ) -> List[str]:
        """Set up budget alerts for a tenant"""
        
        alert_ids = []
        
        for config in alert_configs:
            alert_id = f"alert_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
            
            alert = TenantBudgetAlert(
                alert_id=alert_id,
                tenant_id=tenant_id,
                alert_type=config["alert_type"],
                threshold_percentage=config["threshold_percentage"],
                notification_channels=config["notification_channels"],
                enabled=config.get("enabled", True)
            )
            
            # Add to tenant alerts
            if tenant_id not in self.budget_alerts:
                self.budget_alerts[tenant_id] = []
            
            self.budget_alerts[tenant_id].append(alert)
            alert_ids.append(alert_id)
            
            # Store alert
            await self._store_budget_alert(alert)
        
        self.logger.info(f"Set up {len(alert_ids)} budget alerts for tenant {tenant_id}")
        
        return alert_ids
    
    async def check_request_authorization(
        self,
        tenant_id: str,
        request_context: Dict[str, Any]
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Check if a request is authorized based on guardrails and quotas
        
        Returns:
            Tuple of (authorized, violation_messages, context_updates)
        """
        
        # Extract request details
        content = request_context.get("content", "")
        estimated_cost = Decimal(str(request_context.get("estimated_cost", 0)))
        
        # Check guardrails
        content_allowed, content_violations = await self.guardrail_engine.check_guardrails(
            tenant_id=tenant_id,
            content=content,
            context=request_context,
            agent_id=request_context.get("agent_id"),
            user_id=request_context.get("user_id")
        )
        
        # Check cost guardrails
        cost_allowed, cost_violations = await self.guardrail_engine.check_cost_guardrails(
            tenant_id=tenant_id,
            estimated_cost=estimated_cost,
            context=request_context
        )
        
        # Combine results
        authorized = content_allowed and cost_allowed
        all_violations = content_violations + cost_violations
        
        violation_messages = [v.description for v in all_violations]
        
        # Check budget alerts
        await self._check_budget_alerts(tenant_id, request_context)
        
        # Prepare context updates
        context_updates = {
            "guardrail_checks_passed": authorized,
            "violation_count": len(all_violations),
            "cost_check_passed": cost_allowed
        }
        
        return authorized, violation_messages, context_updates
    
    async def record_request_completion(
        self,
        tenant_id: str,
        actual_cost: Decimal,
        tokens_used: int,
        cpu_seconds: float,
        memory_mb_seconds: float,
        success: bool
    ) -> None:
        """Record completion of a request for usage tracking"""
        
        # Record usage in guardrail engine
        await self.guardrail_engine.record_usage(
            tenant_id=tenant_id,
            cost=actual_cost,
            tokens=tokens_used,
            cpu_seconds=cpu_seconds,
            memory_mb_seconds=memory_mb_seconds
        )
        
        # Update concurrent request count (decrement)
        quota = self.guardrail_engine.tenant_quotas.get(tenant_id)
        if quota and quota.current_concurrent_requests > 0:
            quota.current_concurrent_requests -= 1
        
        # Record success/failure metrics
        if success:
            self.metrics.increment(f"tenant_request_success_{tenant_id}")
        else:
            self.metrics.increment(f"tenant_request_failure_{tenant_id}")
        
        # Check if budget alerts should be triggered
        await self._check_budget_alerts(tenant_id, {
            "actual_cost": actual_cost,
            "tokens_used": tokens_used
        })
    
    def get_tenant_dashboard(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive tenant dashboard data"""
        
        # Get usage report
        usage_report = self.guardrail_engine.get_tenant_usage_report(tenant_id)
        
        # Get guardrail report
        guardrail_report = self.guardrail_engine.get_guardrail_report(tenant_id)
        
        # Get customization settings
        customization = self.guardrail_engine.tenant_customizations.get(tenant_id)
        
        # Get feature flags
        features = self.tenant_features.get(tenant_id, {})
        
        # Get budget alerts
        alerts = self.budget_alerts.get(tenant_id, [])
        active_alerts = [a for a in alerts if a.enabled]
        
        return {
            "tenant_id": tenant_id,
            "usage_summary": usage_report,
            "guardrail_summary": guardrail_report,
            "customization": {
                "brand_name": customization.brand_name if customization else "Forge1",
                "primary_color": customization.primary_color if customization else "#1f2937",
                "enabled_features": customization.enabled_features if customization else [],
                "custom_integrations": len(customization.custom_integrations) if customization else 0
            },
            "feature_flags": features,
            "budget_alerts": {
                "total_alerts": len(alerts),
                "active_alerts": len(active_alerts),
                "recent_triggers": len([
                    a for a in alerts 
                    if a.last_triggered and a.last_triggered > datetime.utcnow() - timedelta(hours=24)
                ])
            },
            "health_status": self._calculate_tenant_health(tenant_id, usage_report, guardrail_report),
            "recommendations": self._generate_tenant_recommendations(tenant_id, usage_report, guardrail_report)
        }
    
    def is_feature_enabled(self, tenant_id: str, feature_name: str) -> bool:
        """Check if a feature is enabled for a tenant"""
        
        # Check customization settings
        customization = self.guardrail_engine.tenant_customizations.get(tenant_id)
        if customization:
            if feature_name in customization.disabled_features:
                return False
            if feature_name in customization.enabled_features:
                return True
        
        # Check feature flags
        features = self.tenant_features.get(tenant_id, {})
        return features.get(feature_name, True)  # Default to enabled
    
    async def update_tenant_configuration(
        self,
        tenant_id: str,
        config_updates: Dict[str, Any]
    ) -> None:
        """Update comprehensive tenant configuration"""
        
        # Store configuration
        if tenant_id not in self.tenant_configs:
            self.tenant_configs[tenant_id] = {}
        
        self.tenant_configs[tenant_id].update(config_updates)
        
        # Apply specific configuration types
        if "quota" in config_updates:
            await self.update_tenant_quota(tenant_id, config_updates["quota"], "system")
        
        if "branding" in config_updates:
            await self.update_tenant_branding(tenant_id, config_updates["branding"])
        
        if "features" in config_updates:
            await self.configure_feature_toggles(tenant_id, config_updates["features"])
        
        if "budget_alerts" in config_updates:
            await self.setup_budget_alerts(tenant_id, config_updates["budget_alerts"])
        
        # Store in memory
        await self.memory_manager.store_context(
            context_type="tenant_configuration",
            content=self.tenant_configs[tenant_id],
            metadata={"tenant_id": tenant_id}
        )
        
        self.logger.info(f"Updated configuration for tenant {tenant_id}")
    
    # Private methods
    async def _setup_default_budget_alerts(self, tenant_id: str) -> None:
        """Set up default budget alerts for a new tenant"""
        
        default_alerts = [
            {
                "alert_type": "cost",
                "threshold_percentage": 0.8,  # 80% of budget
                "notification_channels": ["email", "dashboard"]
            },
            {
                "alert_type": "cost",
                "threshold_percentage": 0.95,  # 95% of budget
                "notification_channels": ["email", "dashboard", "slack"]
            },
            {
                "alert_type": "quota",
                "threshold_percentage": 0.9,  # 90% of quota
                "notification_channels": ["dashboard"]
            }
        ]
        
        await self.setup_budget_alerts(tenant_id, default_alerts)
    
    async def _check_budget_alerts(self, tenant_id: str, context: Dict[str, Any]) -> None:
        """Check if budget alerts should be triggered"""
        
        alerts = self.budget_alerts.get(tenant_id, [])
        usage_report = self.guardrail_engine.get_tenant_usage_report(tenant_id)
        
        if usage_report.get("status") == "no_data":
            return
        
        utilization = usage_report.get("utilization_percentages", {})
        
        for alert in alerts:
            if not alert.enabled:
                continue
            
            # Check if alert should trigger
            should_trigger = False
            
            if alert.alert_type == "cost":
                cost_utilization = max(
                    utilization.get("cost_hour", 0),
                    utilization.get("cost_day", 0),
                    utilization.get("cost_month", 0)
                )
                should_trigger = cost_utilization >= alert.threshold_percentage * 100
            
            elif alert.alert_type == "quota":
                quota_utilization = max(
                    utilization.get("requests_hour", 0),
                    utilization.get("requests_day", 0),
                    utilization.get("tokens_hour", 0),
                    utilization.get("tokens_day", 0)
                )
                should_trigger = quota_utilization >= alert.threshold_percentage * 100
            
            # Trigger alert if needed
            if should_trigger:
                await self._trigger_budget_alert(alert, utilization)
    
    async def _trigger_budget_alert(self, alert: TenantBudgetAlert, utilization: Dict[str, float]) -> None:
        """Trigger a budget alert"""
        
        # Check if alert was recently triggered (avoid spam)
        if (alert.last_triggered and 
            datetime.utcnow() - alert.last_triggered < timedelta(hours=1)):
            return
        
        alert.last_triggered = datetime.utcnow()
        
        # Log alert
        self.logger.warning(
            f"Budget alert triggered: {alert.alert_id} for tenant {alert.tenant_id} "
            f"(type: {alert.alert_type}, threshold: {alert.threshold_percentage:.1%})"
        )
        
        # Record metrics
        self.metrics.increment(f"budget_alert_triggered_{alert.tenant_id}")
        self.metrics.increment(f"budget_alert_type_{alert.alert_type}")
        
        # Store alert event
        await self.memory_manager.store_context(
            context_type="budget_alert_event",
            content={
                "alert_id": alert.alert_id,
                "tenant_id": alert.tenant_id,
                "alert_type": alert.alert_type,
                "threshold_percentage": alert.threshold_percentage,
                "current_utilization": utilization,
                "triggered_at": alert.last_triggered.isoformat()
            },
            metadata={
                "tenant_id": alert.tenant_id,
                "alert_type": alert.alert_type
            }
        )
    
    def _calculate_tenant_health(
        self,
        tenant_id: str,
        usage_report: Dict[str, Any],
        guardrail_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall tenant health score"""
        
        health_score = 100.0
        issues = []
        
        # Check usage patterns
        if usage_report.get("status") != "no_data":
            utilization = usage_report.get("utilization_percentages", {})
            
            # Penalize high utilization
            for metric, percentage in utilization.items():
                if percentage > 90:
                    health_score -= 20
                    issues.append(f"High {metric} utilization: {percentage:.1f}%")
                elif percentage > 75:
                    health_score -= 10
        
        # Check guardrail violations
        recent_violations = guardrail_report.get("recent_violations", 0)
        if recent_violations > 10:
            health_score -= 30
            issues.append(f"High violation count: {recent_violations}")
        elif recent_violations > 5:
            health_score -= 15
        
        # Determine health status
        if health_score >= 80:
            status = "healthy"
        elif health_score >= 60:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "status": status,
            "score": max(0, health_score),
            "issues": issues
        }
    
    def _generate_tenant_recommendations(
        self,
        tenant_id: str,
        usage_report: Dict[str, Any],
        guardrail_report: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for tenant optimization"""
        
        recommendations = []
        
        # Usage-based recommendations
        if usage_report.get("status") != "no_data":
            utilization = usage_report.get("utilization_percentages", {})
            
            # High cost utilization
            if utilization.get("cost_day", 0) > 80:
                recommendations.append("Consider optimizing model usage to reduce costs")
            
            # High request utilization
            if utilization.get("requests_hour", 0) > 80:
                recommendations.append("Consider increasing request quotas or optimizing request patterns")
        
        # Guardrail-based recommendations
        recent_violations = guardrail_report.get("recent_violations", 0)
        if recent_violations > 5:
            recommendations.append("Review and adjust guardrail rules to reduce violations")
        
        # Feature recommendations
        customization = self.guardrail_engine.tenant_customizations.get(tenant_id)
        if customization and len(customization.enabled_features) < 3:
            recommendations.append("Explore additional platform features to maximize value")
        
        return recommendations
    
    async def _store_budget_alert(self, alert: TenantBudgetAlert) -> None:
        """Store budget alert configuration"""
        
        await self.memory_manager.store_context(
            context_type="budget_alert_config",
            content=alert.__dict__,
            metadata={
                "tenant_id": alert.tenant_id,
                "alert_id": alert.alert_id
            }
        )