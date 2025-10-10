```python
"""
Tenant Customization & Guardrails System
Policy and prompt guardrails per tenant with cost caps, execution quotas, and budget management
"""

from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import logging
import json
import re
from collections import defaultdict, deque

from forge1.backend.forge1.core.monitoring import MetricsCollector
from forge1.backend.forge1.core.memory import MemoryManager


class GuardrailType(Enum):
    """Types of guardrails"""
    POLICY = "policy"
    PROMPT = "prompt"
    COST = "cost"
    QUOTA = "quota"
    CONTENT = "content"
    SECURITY = "security"


class ViolationSeverity(Enum):
    """Severity levels for guardrail violations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(Enum):
    """Actions that can be taken when guardrails are violated"""
    WARN = "warn"
    BLOCK = "block"
    THROTTLE = "throttle"
    ESCALATE = "escalate"
    AUDIT = "audit"


@dataclass
class GuardrailRule:
    """Individual guardrail rule"""
    rule_id: str
    name: str
    description: str
    rule_type: GuardrailType
    pattern: str  # Regex pattern or policy expression
    action: ActionType
    severity: ViolationSeverity
    enabled: bool = True
    
    # Rule parameters
    threshold: Optional[Union[int, float, Decimal]] = None
    time_window: Optional[timedelta] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    
    # Custom configuration
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailViolation:
    """Record of a guardrail violation"""
    violation_id: str
    rule_id: str
    tenant_id: str
    agent_id: Optional[str]
    user_id: Optional[str]
    
    # Violation details
    violation_type: GuardrailType
    severity: ViolationSeverity
    action_taken: ActionType
    description: str
    
    # Context
    input_content: str = ""
    output_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    
    # Resolution
    resolved: bool = False
    resolution_notes: str = ""


@dataclass
class TenantQuota:
    """Tenant resource quota configuration"""
    tenant_id: str
    
    # Execution quotas
    max_requests_per_hour: int = 1000
    max_requests_per_day: int = 10000
    max_concurrent_requests: int = 50
    
    # Resource quotas
    max_cpu_hours_per_day: float = 24.0
    max_memory_gb_hours_per_day: float = 100.0
    max_storage_gb: float = 10.0
    
    # Cost quotas
    max_cost_per_hour: Decimal = Decimal('100.00')
    max_cost_per_day: Decimal = Decimal('1000.00')
    max_cost_per_month: Decimal = Decimal('10000.00')
    
    # Model usage quotas
    max_tokens_per_hour: int = 100000
    max_tokens_per_day: int = 1000000
    
    # Current usage (tracked in real-time)
    current_requests_hour: int = 0
    current_requests_day: int = 0
    current_concurrent_requests: int = 0
    current_cost_hour: Decimal = Decimal('0.00')
    current_cost_day: Decimal = Decimal('0.00')
    current_cost_month: Decimal = Decimal('0.00')
    current_tokens_hour: int = 0
    current_tokens_day: int = 0
    
    # Reset timestamps
    last_hour_reset: datetime = field(default_factory=datetime.utcnow)
    last_day_reset: datetime = field(default_factory=datetime.utcnow)
    last_month_reset: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TenantCustomization:
    """Tenant-specific customization settings"""
    tenant_id: str
    
    # Branding and UI
    brand_name: str = "Forge1"
    brand_logo_url: Optional[str] = None
    primary_color: str = "#1f2937"
    secondary_color: str = "#3b82f6"
    custom_css: str = ""
    
    # Feature toggles
    enabled_features: List[str] = field(default_factory=list)
    disabled_features: List[str] = field(default_factory=list)
    
    # Default configurations
    default_model_preferences: Dict[str, str] = field(default_factory=dict)
    default_agent_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Custom prompts and templates
    custom_prompt_templates: Dict[str, str] = field(default_factory=dict)
    system_prompts: Dict[str, str] = field(default_factory=dict)
    
    # Workflow customizations
    custom_workflows: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    workflow_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Integration settings
    custom_integrations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    integration_overrides: Dict[str, Any] = field(default_factory=dict)
    
    # Notification preferences
    notification_channels: List[str] = field(default_factory=list)
    notification_settings: Dict[str, bool] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
class G
uardrailEngine:
    """
    Comprehensive guardrail engine for tenant customization and policy enforcement
    """
    
    def __init__(self, memory_manager: MemoryManager, metrics_collector: MetricsCollector):
        self.memory_manager = memory_manager
        self.metrics = metrics_collector
        self.logger = logging.getLogger("guardrail_engine")
        
        # Guardrail storage
        self.tenant_rules: Dict[str, List[GuardrailRule]] = defaultdict(list)
        self.tenant_quotas: Dict[str, TenantQuota] = {}
        self.tenant_customizations: Dict[str, TenantCustomization] = {}
        
        # Violation tracking
        self.violations: Dict[str, List[GuardrailViolation]] = defaultdict(list)
        self.violation_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Usage tracking
        self.usage_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Built-in rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self) -> None:
        """Initialize default guardrail rules"""
        
        self.default_rules = [
            GuardrailRule(
                rule_id="default_pii_detection",
                name="PII Detection",
                description="Detect and block personally identifiable information",
                rule_type=GuardrailType.CONTENT,
                pattern=r'\b(?:\d{3}-\d{2}-\d{4}|\d{4}\s\d{4}\s\d{4}\s\d{4}|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)',
                action=ActionType.WARN,
                severity=ViolationSeverity.HIGH
            ),
            GuardrailRule(
                rule_id="default_cost_limit",
                name="Cost Limit",
                description="Prevent excessive costs per request",
                rule_type=GuardrailType.COST,
                pattern="",
                action=ActionType.BLOCK,
                severity=ViolationSeverity.CRITICAL,
                threshold=Decimal('10.00')
            ),
            GuardrailRule(
                rule_id="default_prompt_injection",
                name="Prompt Injection Detection",
                description="Detect potential prompt injection attacks",
                rule_type=GuardrailType.SECURITY,
                pattern=r'(?i)(ignore\s+previous|forget\s+instructions|system\s*:|admin\s*:|root\s*:)',
                action=ActionType.BLOCK,
                severity=ViolationSeverity.CRITICAL
            )
        ]
    
    async def add_guardrail_rule(
        self,
        tenant_id: str,
        rule: GuardrailRule,
        created_by: str
    ) -> str:
        """Add a new guardrail rule for a tenant"""
        
        rule.created_by = created_by
        rule.created_at = datetime.utcnow()
        rule.updated_at = datetime.utcnow()
        
        # Validate rule
        validation_errors = self._validate_rule(rule)
        if validation_errors:
            raise ValueError(f"Rule validation failed: {validation_errors}")
        
        # Add to tenant rules
        self.tenant_rules[tenant_id].append(rule)
        
        # Store in memory
        await self._store_guardrail_rule(tenant_id, rule)
        
        # Record metrics
        self.metrics.increment(f"guardrail_rule_added_{tenant_id}")
        self.metrics.increment(f"guardrail_rule_type_{rule.rule_type.value}")
        
        self.logger.info(f"Added guardrail rule {rule.rule_id} for tenant {tenant_id}")
        
        return rule.rule_id
    
    async def update_tenant_quota(self, tenant_id: str, quota: TenantQuota) -> None:
        """Update tenant resource quota"""
        
        quota.tenant_id = tenant_id
        self.tenant_quotas[tenant_id] = quota
        
        # Store in memory
        await self._store_tenant_quota(tenant_id, quota)
        
        self.logger.info(f"Updated quota for tenant {tenant_id}")
    
    async def update_tenant_customization(
        self,
        tenant_id: str,
        customization: TenantCustomization
    ) -> None:
        """Update tenant customization settings"""
        
        customization.tenant_id = tenant_id
        customization.updated_at = datetime.utcnow()
        
        self.tenant_customizations[tenant_id] = customization
        
        # Store in memory
        await self._store_tenant_customization(tenant_id, customization)
        
        self.logger.info(f"Updated customization for tenant {tenant_id}")
    
    async def check_guardrails(
        self,
        tenant_id: str,
        content: str,
        context: Dict[str, Any],
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Tuple[bool, List[GuardrailViolation]]:
        """
        Check content against all applicable guardrails
        
        Returns:
            Tuple of (allowed, violations)
        """
        
        violations = []
        allowed = True
        
        # Get all applicable rules
        rules = self._get_applicable_rules(tenant_id)
        
        # Check each rule
        for rule in rules:
            if not rule.enabled:
                continue
            
            violation = await self._check_rule(
                rule, tenant_id, content, context, agent_id, user_id
            )
            
            if violation:
                violations.append(violation)
                
                # Record violation
                await self._record_violation(violation)
                
                # Determine if request should be blocked
                if rule.action in [ActionType.BLOCK]:
                    allowed = False
                elif rule.action == ActionType.THROTTLE:
                    # Check if throttling threshold exceeded
                    if self._should_throttle(tenant_id, rule):
                        allowed = False
        
        # Check quotas
        quota_violations = await self._check_quotas(tenant_id, context)
        violations.extend(quota_violations)
        
        if quota_violations:
            # Block if any quota violations
            allowed = False
        
        return allowed, violations
    
    async def check_cost_guardrails(
        self,
        tenant_id: str,
        estimated_cost: Decimal,
        context: Dict[str, Any]
    ) -> Tuple[bool, List[GuardrailViolation]]:
        """Check cost-specific guardrails"""
        
        violations = []
        allowed = True
        
        # Get quota
        quota = self.tenant_quotas.get(tenant_id)
        if not quota:
            quota = TenantQuota(tenant_id=tenant_id)
            self.tenant_quotas[tenant_id] = quota
        
        # Reset quotas if needed
        await self._reset_quotas_if_needed(quota)
        
        # Check cost limits
        if quota.current_cost_hour + estimated_cost > quota.max_cost_per_hour:
            violation = GuardrailViolation(
                violation_id=f"cost_hour_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                rule_id="cost_limit_hour",
                tenant_id=tenant_id,
                agent_id=context.get("agent_id"),
                user_id=context.get("user_id"),
                violation_type=GuardrailType.COST,
                severity=ViolationSeverity.HIGH,
                action_taken=ActionType.BLOCK,
                description=f"Hourly cost limit exceeded: {quota.current_cost_hour + estimated_cost} > {quota.max_cost_per_hour}",
                metadata={"estimated_cost": str(estimated_cost), "current_cost": str(quota.current_cost_hour)}
            )
            violations.append(violation)
            allowed = False
        
        if quota.current_cost_day + estimated_cost > quota.max_cost_per_day:
            violation = GuardrailViolation(
                violation_id=f"cost_day_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                rule_id="cost_limit_day",
                tenant_id=tenant_id,
                agent_id=context.get("agent_id"),
                user_id=context.get("user_id"),
                violation_type=GuardrailType.COST,
                severity=ViolationSeverity.CRITICAL,
                action_taken=ActionType.BLOCK,
                description=f"Daily cost limit exceeded: {quota.current_cost_day + estimated_cost} > {quota.max_cost_per_day}",
                metadata={"estimated_cost": str(estimated_cost), "current_cost": str(quota.current_cost_day)}
            )
            violations.append(violation)
            allowed = False
        
        return allowed, violations
    
    async def record_usage(
        self,
        tenant_id: str,
        cost: Decimal,
        tokens: int,
        cpu_seconds: float,
        memory_mb_seconds: float
    ) -> None:
        """Record resource usage for a tenant"""
        
        # Get or create quota
        quota = self.tenant_quotas.get(tenant_id)
        if not quota:
            quota = TenantQuota(tenant_id=tenant_id)
            self.tenant_quotas[tenant_id] = quota
        
        # Reset quotas if needed
        await self._reset_quotas_if_needed(quota)
        
        # Update usage
        quota.current_requests_hour += 1
        quota.current_requests_day += 1
        quota.current_concurrent_requests += 1  # Would be decremented when request completes
        
        quota.current_cost_hour += cost
        quota.current_cost_day += cost
        quota.current_cost_month += cost
        
        quota.current_tokens_hour += tokens
        quota.current_tokens_day += tokens
        
        # Record usage event
        usage_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "cost": str(cost),
            "tokens": tokens,
            "cpu_seconds": cpu_seconds,
            "memory_mb_seconds": memory_mb_seconds
        }
        
        self.usage_windows[tenant_id].append(usage_event)
        
        # Store updated quota
        await self._store_tenant_quota(tenant_id, quota)
        
        # Record metrics
        self.metrics.record_metric(f"tenant_cost_{tenant_id}", float(cost))
        self.metrics.record_metric(f"tenant_tokens_{tenant_id}", tokens)
    
    def get_tenant_usage_report(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive usage report for tenant"""
        
        quota = self.tenant_quotas.get(tenant_id)
        if not quota:
            return {"status": "no_data", "tenant_id": tenant_id}
        
        # Calculate utilization percentages
        utilization = {
            "requests_hour": (quota.current_requests_hour / quota.max_requests_per_hour) * 100,
            "requests_day": (quota.current_requests_day / quota.max_requests_per_day) * 100,
            "cost_hour": (float(quota.current_cost_hour) / float(quota.max_cost_per_hour)) * 100,
            "cost_day": (float(quota.current_cost_day) / float(quota.max_cost_per_day)) * 100,
            "cost_month": (float(quota.current_cost_month) / float(quota.max_cost_per_month)) * 100,
            "tokens_hour": (quota.current_tokens_hour / quota.max_tokens_per_hour) * 100,
            "tokens_day": (quota.current_tokens_day / quota.max_tokens_per_day) * 100
        }
        
        # Get recent violations
        recent_violations = [
            v for v in self.violations[tenant_id]
            if v.occurred_at > datetime.utcnow() - timedelta(hours=24)
        ]
        
        # Usage trends
        usage_events = list(self.usage_windows[tenant_id])
        hourly_costs = defaultdict(float)
        
        for event in usage_events:
            hour = datetime.fromisoformat(event["timestamp"]).replace(minute=0, second=0, microsecond=0)
            hourly_costs[hour.isoformat()] += float(event["cost"])
        
        return {
            "tenant_id": tenant_id,
            "current_usage": {
                "requests_hour": quota.current_requests_hour,
                "requests_day": quota.current_requests_day,
                "concurrent_requests": quota.current_concurrent_requests,
                "cost_hour": float(quota.current_cost_hour),
                "cost_day": float(quota.current_cost_day),
                "cost_month": float(quota.current_cost_month),
                "tokens_hour": quota.current_tokens_hour,
                "tokens_day": quota.current_tokens_day
            },
            "quota_limits": {
                "max_requests_hour": quota.max_requests_per_hour,
                "max_requests_day": quota.max_requests_per_day,
                "max_cost_hour": float(quota.max_cost_per_hour),
                "max_cost_day": float(quota.max_cost_per_day),
                "max_cost_month": float(quota.max_cost_per_month),
                "max_tokens_hour": quota.max_tokens_per_hour,
                "max_tokens_day": quota.max_tokens_per_day
            },
            "utilization_percentages": utilization,
            "recent_violations": len(recent_violations),
            "violation_breakdown": {
                severity.value: len([v for v in recent_violations if v.severity == severity])
                for severity in ViolationSeverity
            },
            "hourly_cost_trend": dict(hourly_costs),
            "alerts": self._generate_usage_alerts(quota, utilization)
        }
    
    def get_guardrail_report(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive guardrail report for tenant"""
        
        # Get tenant rules
        tenant_rules = self.tenant_rules.get(tenant_id, [])
        all_rules = tenant_rules + self.default_rules
        
        # Get violations
        violations = self.violations.get(tenant_id, [])
        recent_violations = [
            v for v in violations
            if v.occurred_at > datetime.utcnow() - timedelta(days=7)
        ]
        
        # Rule effectiveness
        rule_stats = {}
        for rule in all_rules:
            rule_violations = [v for v in violations if v.rule_id == rule.rule_id]
            rule_stats[rule.rule_id] = {
                "name": rule.name,
                "type": rule.rule_type.value,
                "enabled": rule.enabled,
                "total_violations": len(rule_violations),
                "recent_violations": len([v for v in rule_violations if v.occurred_at > datetime.utcnow() - timedelta(days=7)])
            }
        
        return {
            "tenant_id": tenant_id,
            "total_rules": len(all_rules),
            "active_rules": len([r for r in all_rules if r.enabled]),
            "custom_rules": len(tenant_rules),
            "total_violations": len(violations),
            "recent_violations": len(recent_violations),
            "violation_trends": {
                "by_severity": {
                    severity.value: len([v for v in recent_violations if v.severity == severity])
                    for severity in ViolationSeverity
                },
                "by_type": {
                    rule_type.value: len([v for v in recent_violations if v.violation_type == rule_type])
                    for rule_type in GuardrailType
                }
            },
            "rule_effectiveness": rule_stats,
            "recommendations": self._generate_guardrail_recommendations(tenant_id, violations, all_rules)
        }
    
    # Private methods
    def _get_applicable_rules(self, tenant_id: str) -> List[GuardrailRule]:
        """Get all applicable rules for a tenant"""
        
        # Combine default rules with tenant-specific rules
        tenant_rules = self.tenant_rules.get(tenant_id, [])
        return self.default_rules + tenant_rules
    
    async def _check_rule(
        self,
        rule: GuardrailRule,
        tenant_id: str,
        content: str,
        context: Dict[str, Any],
        agent_id: Optional[str],
        user_id: Optional[str]
    ) -> Optional[GuardrailViolation]:
        """Check a specific rule against content"""
        
        violation_detected = False
        violation_details = ""
        
        if rule.rule_type == GuardrailType.CONTENT:
            # Check content patterns
            if rule.pattern and re.search(rule.pattern, content, re.IGNORECASE):
                violation_detected = True
                violation_details = f"Content pattern matched: {rule.pattern}"
        
        elif rule.rule_type == GuardrailType.PROMPT:
            # Check prompt injection patterns
            if rule.pattern and re.search(rule.pattern, content, re.IGNORECASE):
                violation_detected = True
                violation_details = f"Prompt pattern matched: {rule.pattern}"
        
        elif rule.rule_type == GuardrailType.SECURITY:
            # Check security patterns
            if rule.pattern and re.search(rule.pattern, content, re.IGNORECASE):
                violation_detected = True
                violation_details = f"Security pattern matched: {rule.pattern}"
        
        elif rule.rule_type == GuardrailType.COST:
            # Check cost thresholds
            estimated_cost = context.get("estimated_cost", 0)
            if rule.threshold and estimated_cost > rule.threshold:
                violation_detected = True
                violation_details = f"Cost threshold exceeded: {estimated_cost} > {rule.threshold}"
        
        if violation_detected:
            violation_id = f"{rule.rule_id}_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
            
            return GuardrailViolation(
                violation_id=violation_id,
                rule_id=rule.rule_id,
                tenant_id=tenant_id,
                agent_id=agent_id,
                user_id=user_id,
                violation_type=rule.rule_type,
                severity=rule.severity,
                action_taken=rule.action,
                description=violation_details,
                input_content=content[:1000],  # Truncate for storage
                metadata=context
            )
        
        return None
    
    async def _check_quotas(
        self,
        tenant_id: str,
        context: Dict[str, Any]
    ) -> List[GuardrailViolation]:
        """Check quota violations"""
        
        violations = []
        
        quota = self.tenant_quotas.get(tenant_id)
        if not quota:
            return violations
        
        # Reset quotas if needed
        await self._reset_quotas_if_needed(quota)
        
        # Check request quotas
        if quota.current_requests_hour >= quota.max_requests_per_hour:
            violations.append(GuardrailViolation(
                violation_id=f"quota_requests_hour_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                rule_id="quota_requests_hour",
                tenant_id=tenant_id,
                agent_id=context.get("agent_id"),
                user_id=context.get("user_id"),
                violation_type=GuardrailType.QUOTA,
                severity=ViolationSeverity.HIGH,
                action_taken=ActionType.BLOCK,
                description=f"Hourly request quota exceeded: {quota.current_requests_hour} >= {quota.max_requests_per_hour}"
            ))
        
        if quota.current_concurrent_requests >= quota.max_concurrent_requests:
            violations.append(GuardrailViolation(
                violation_id=f"quota_concurrent_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                rule_id="quota_concurrent_requests",
                tenant_id=tenant_id,
                agent_id=context.get("agent_id"),
                user_id=context.get("user_id"),
                violation_type=GuardrailType.QUOTA,
                severity=ViolationSeverity.MEDIUM,
                action_taken=ActionType.THROTTLE,
                description=f"Concurrent request quota exceeded: {quota.current_concurrent_requests} >= {quota.max_concurrent_requests}"
            ))
        
        return violations
    
    async def _reset_quotas_if_needed(self, quota: TenantQuota) -> None:
        """Reset quota counters if time windows have passed"""
        
        now = datetime.utcnow()
        
        # Reset hourly quotas
        if now - quota.last_hour_reset >= timedelta(hours=1):
            quota.current_requests_hour = 0
            quota.current_cost_hour = Decimal('0.00')
            quota.current_tokens_hour = 0
            quota.last_hour_reset = now.replace(minute=0, second=0, microsecond=0)
        
        # Reset daily quotas
        if now - quota.last_day_reset >= timedelta(days=1):
            quota.current_requests_day = 0
            quota.current_cost_day = Decimal('0.00')
            quota.current_tokens_day = 0
            quota.last_day_reset = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Reset monthly quotas
        if now.month != quota.last_month_reset.month or now.year != quota.last_month_reset.year:
            quota.current_cost_month = Decimal('0.00')
            quota.last_month_reset = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    def _should_throttle(self, tenant_id: str, rule: GuardrailRule) -> bool:
        """Determine if request should be throttled based on violation frequency"""
        
        if not rule.time_window:
            return False
        
        # Count recent violations for this rule
        recent_violations = [
            v for v in self.violations[tenant_id]
            if (v.rule_id == rule.rule_id and 
                v.occurred_at > datetime.utcnow() - rule.time_window)
        ]
        
        # Throttle if threshold exceeded
        return len(recent_violations) >= (rule.threshold or 5)
    
    async def _record_violation(self, violation: GuardrailViolation) -> None:
        """Record a guardrail violation"""
        
        # Add to violations list
        self.violations[violation.tenant_id].append(violation)
        
        # Update violation counts
        self.violation_counts[violation.tenant_id][violation.rule_id] += 1
        
        # Store in memory
        await self.memory_manager.store_context(
            context_type="guardrail_violation",
            content=violation.__dict__,
            metadata={
                "tenant_id": violation.tenant_id,
                "rule_id": violation.rule_id,
                "severity": violation.severity.value
            }
        )
        
        # Record metrics
        self.metrics.increment(f"guardrail_violation_{violation.tenant_id}")
        self.metrics.increment(f"guardrail_violation_severity_{violation.severity.value}")
        self.metrics.increment(f"guardrail_violation_type_{violation.violation_type.value}")
        
        self.logger.warning(
            f"Guardrail violation: {violation.rule_id} for tenant {violation.tenant_id} "
            f"(severity: {violation.severity.value})"
        )
    
    def _validate_rule(self, rule: GuardrailRule) -> List[str]:
        """Validate a guardrail rule"""
        
        errors = []
        
        # Basic validation
        if not rule.name or len(rule.name) < 3:
            errors.append("Rule name must be at least 3 characters")
        
        if not rule.description:
            errors.append("Rule description is required")
        
        # Pattern validation for regex rules
        if rule.rule_type in [GuardrailType.CONTENT, GuardrailType.PROMPT, GuardrailType.SECURITY]:
            if not rule.pattern:
                errors.append("Pattern is required for content/prompt/security rules")
            else:
                try:
                    re.compile(rule.pattern)
                except re.error as e:
                    errors.append(f"Invalid regex pattern: {e}")
        
        # Threshold validation for cost/quota rules
        if rule.rule_type in [GuardrailType.COST, GuardrailType.QUOTA]:
            if rule.threshold is None:
                errors.append("Threshold is required for cost/quota rules")
        
        return errors
    
    def _generate_usage_alerts(self, quota: TenantQuota, utilization: Dict[str, float]) -> List[str]:
        """Generate usage alerts based on utilization"""
        
        alerts = []
        
        # High utilization alerts
        for metric, percentage in utilization.items():
            if percentage > 90:
                alerts.append(f"Critical: {metric} at {percentage:.1f}% of quota")
            elif percentage > 75:
                alerts.append(f"Warning: {metric} at {percentage:.1f}% of quota")
        
        return alerts
    
    def _generate_guardrail_recommendations(
        self,
        tenant_id: str,
        violations: List[GuardrailViolation],
        rules: List[GuardrailRule]
    ) -> List[str]:
        """Generate recommendations for guardrail optimization"""
        
        recommendations = []
        
        # Analyze violation patterns
        violation_counts = defaultdict(int)
        for violation in violations:
            violation_counts[violation.rule_id] += 1
        
        # Recommend rule adjustments
        for rule_id, count in violation_counts.items():
            if count > 100:  # High violation count
                recommendations.append(f"Consider adjusting rule {rule_id} - high violation count ({count})")
        
        # Recommend new rules based on patterns
        security_violations = [v for v in violations if v.violation_type == GuardrailType.SECURITY]
        if len(security_violations) > 10:
            recommendations.append("Consider adding additional security rules - multiple security violations detected")
        
        return recommendations
    
    async def _store_guardrail_rule(self, tenant_id: str, rule: GuardrailRule) -> None:
        """Store guardrail rule in memory"""
        
        await self.memory_manager.store_context(
            context_type="guardrail_rule",
            content=rule.__dict__,
            metadata={
                "tenant_id": tenant_id,
                "rule_id": rule.rule_id,
                "rule_type": rule.rule_type.value
            }
        )
    
    async def _store_tenant_quota(self, tenant_id: str, quota: TenantQuota) -> None:
        """Store tenant quota in memory"""
        
        await self.memory_manager.store_context(
            context_type="tenant_quota",
            content=quota.__dict__,
            metadata={
                "tenant_id": tenant_id
            }
        )
    
    async def _store_tenant_customization(self, tenant_id: str, customization: TenantCustomization) -> None:
        """Store tenant customization in memory"""
        
        await self.memory_manager.store_context(
            context_type="tenant_customization",
            content=customization.__dict__,
            metadata={
                "tenant_id": tenant_id
            }
        )
```
