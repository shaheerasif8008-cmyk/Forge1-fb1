"""
Forge 1 Pricing Engine
Comprehensive pricing, billing, and entitlements system with usage metering and cost optimization
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import logging
import json
from collections import defaultdict

# Mock dependencies for standalone operation
class MetricsCollector:
    def increment(self, metric): pass
    def record_metric(self, metric, value): pass

class MemoryManager:
    async def store_context(self, context_type, content, metadata): pass

class SecretManager:
    async def get(self, name): return "mock_secret"


class PricingModel(Enum):
    """Pricing model types"""
    USAGE_BASED = "usage_based"
    SUBSCRIPTION = "subscription"
    HYBRID = "hybrid"
    ENTERPRISE = "enterprise"
    FREEMIUM = "freemium"


class BillingCycle(Enum):
    """Billing cycle options"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    USAGE = "usage"  # Pay-as-you-go


class UsageMetricType(Enum):
    """Types of usage metrics"""
    API_CALLS = "api_calls"
    COMPUTE_HOURS = "compute_hours"
    STORAGE_GB = "storage_gb"
    BANDWIDTH_GB = "bandwidth_gb"
    ACTIVE_USERS = "active_users"
    TRANSACTIONS = "transactions"
    AI_TOKENS = "ai_tokens"
    AGENT_HOURS = "agent_hours"
    CUSTOM_METRIC = "custom_metric"


class PlanTier(Enum):
    """Subscription plan tiers"""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class EntitlementType(Enum):
    """Types of entitlements"""
    FEATURE_ACCESS = "feature_access"
    USAGE_LIMIT = "usage_limit"
    RATE_LIMIT = "rate_limit"
    STORAGE_LIMIT = "storage_limit"
    USER_LIMIT = "user_limit"
    API_LIMIT = "api_limit"


@dataclass
class UsageMetric:
    """Usage metric definition and tracking"""
    metric_id: str
    metric_type: UsageMetricType
    name: str
    description: str
    
    # Pricing
    unit_price: Decimal
    currency: str = "USD"
    
    # Limits and tiers
    free_tier_limit: int = 0
    overage_price: Optional[Decimal] = None
    
    # Aggregation
    aggregation_period: str = "monthly"  # daily, weekly, monthly
    reset_cycle: str = "monthly"
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True


@dataclass
class PricingPlan:
    """Subscription pricing plan"""
    plan_id: str
    name: str
    description: str
    tier: PlanTier
    pricing_model: PricingModel
    
    # Pricing
    base_price: Decimal
    currency: str = "USD"
    billing_cycle: BillingCycle = BillingCycle.MONTHLY
    
    # Entitlements
    included_usage: Dict[str, int] = field(default_factory=dict)  # metric_id -> limit
    feature_access: List[str] = field(default_factory=list)
    
    # Limits
    max_users: Optional[int] = None
    max_storage_gb: Optional[int] = None
    max_api_calls_per_month: Optional[int] = None
    
    # Trial and discounts
    trial_days: int = 0
    discount_percentage: Decimal = Decimal('0')
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True


@dataclass
class CustomerEntitlement:
    """Customer entitlement and usage tracking"""
    entitlement_id: str
    customer_id: str
    plan_id: str
    
    # Entitlement details
    entitlement_type: EntitlementType
    resource_name: str
    limit_value: int
    
    # Billing period
    period_start: datetime
    period_end: datetime
    
    # Usage tracking
    current_usage: int = 0
    
    # Status
    active: bool = True
    overage_allowed: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UsageRecord:
    """Individual usage record"""
    usage_id: str
    customer_id: str
    metric_id: str
    
    # Usage details
    quantity: Decimal
    unit_price: Decimal
    total_cost: Decimal
    
    # Context
    resource_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Timing
    usage_timestamp: datetime = field(default_factory=datetime.utcnow)
    billing_period: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False


@dataclass
class Invoice:
    """Customer invoice"""
    invoice_id: str
    customer_id: str
    
    # Invoice details
    invoice_number: str
    billing_period_start: datetime
    billing_period_end: datetime
    
    # Amounts
    subtotal: Decimal
    tax_amount: Decimal
    discount_amount: Decimal
    total_amount: Decimal
    currency: str = "USD"
    
    # Line items
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    
    # Status
    status: str = "draft"  # draft, sent, paid, overdue, cancelled
    due_date: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=30))
    paid_date: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class PricingEngine:
    """
    Comprehensive pricing and billing engine
    
    Features:
    - Usage metering and tracking
    - Plan-based entitlements
    - Overage calculation and billing
    - Invoice generation
    - Cost optimization recommendations
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
        self.logger = logging.getLogger("pricing_engine")
        
        # Pricing configuration
        self.usage_metrics: Dict[str, UsageMetric] = {}
        self.pricing_plans: Dict[str, PricingPlan] = {}
        
        # Customer data
        self.customer_entitlements: Dict[str, List[CustomerEntitlement]] = defaultdict(list)
        self.usage_records: List[UsageRecord] = []
        self.invoices: Dict[str, Invoice] = {}
        
        # Configuration
        self.tax_rate = Decimal('0.08')  # 8% default tax rate
        self.overage_grace_period_days = 3
        
        # Initialize default pricing structure
        self._initialize_default_pricing()
    
    def _initialize_default_pricing(self) -> None:
        """Initialize default pricing metrics and plans"""
        
        # Define usage metrics
        default_metrics = [
            UsageMetric(
                metric_id="api_calls",
                metric_type=UsageMetricType.API_CALLS,
                name="API Calls",
                description="Number of API calls made to Forge 1 services",
                unit_price=Decimal('0.001'),  # $0.001 per call
                free_tier_limit=10000,
                overage_price=Decimal('0.0015')
            ),
            UsageMetric(
                metric_id="ai_tokens",
                metric_type=UsageMetricType.AI_TOKENS,
                name="AI Tokens",
                description="Number of AI model tokens processed",
                unit_price=Decimal('0.00002'),  # $0.00002 per token
                free_tier_limit=1000000,
                overage_price=Decimal('0.00003')
            ),
            UsageMetric(
                metric_id="agent_hours",
                metric_type=UsageMetricType.AGENT_HOURS,
                name="Agent Hours",
                description="Hours of AI agent execution time",
                unit_price=Decimal('2.50'),  # $2.50 per hour
                free_tier_limit=10,
                overage_price=Decimal('3.00')
            ),
            UsageMetric(
                metric_id="storage_gb",
                metric_type=UsageMetricType.STORAGE_GB,
                name="Storage",
                description="Gigabytes of data storage used",
                unit_price=Decimal('0.10'),  # $0.10 per GB per month
                free_tier_limit=10,
                overage_price=Decimal('0.15')
            ),
            UsageMetric(
                metric_id="active_users",
                metric_type=UsageMetricType.ACTIVE_USERS,
                name="Active Users",
                description="Number of active users per month",
                unit_price=Decimal('25.00'),  # $25 per user per month
                free_tier_limit=5,
                overage_price=Decimal('30.00')
            )
        ]
        
        for metric in default_metrics:
            self.usage_metrics[metric.metric_id] = metric
        
        # Define pricing plans
        default_plans = [
            PricingPlan(
                plan_id="free",
                name="Free Plan",
                description="Perfect for getting started with AI employees",
                tier=PlanTier.FREE,
                pricing_model=PricingModel.FREEMIUM,
                base_price=Decimal('0'),
                included_usage={
                    "api_calls": 10000,
                    "ai_tokens": 1000000,
                    "agent_hours": 10,
                    "storage_gb": 10,
                    "active_users": 5
                },
                feature_access=["basic_agents", "standard_support"],
                trial_days=0
            ),
            PricingPlan(
                plan_id="starter",
                name="Starter Plan",
                description="For small teams getting serious about AI automation",
                tier=PlanTier.STARTER,
                pricing_model=PricingModel.HYBRID,
                base_price=Decimal('99'),
                billing_cycle=BillingCycle.MONTHLY,
                included_usage={
                    "api_calls": 100000,
                    "ai_tokens": 10000000,
                    "agent_hours": 100,
                    "storage_gb": 100,
                    "active_users": 25
                },
                feature_access=["basic_agents", "advanced_agents", "priority_support", "analytics"],
                max_users=25,
                trial_days=14
            ),
            PricingPlan(
                plan_id="professional",
                name="Professional Plan",
                description="For growing businesses scaling AI operations",
                tier=PlanTier.PROFESSIONAL,
                pricing_model=PricingModel.HYBRID,
                base_price=Decimal('499'),
                billing_cycle=BillingCycle.MONTHLY,
                included_usage={
                    "api_calls": 1000000,
                    "ai_tokens": 100000000,
                    "agent_hours": 500,
                    "storage_gb": 500,
                    "active_users": 100
                },
                feature_access=["all_agents", "custom_workflows", "priority_support", "advanced_analytics", "compliance_tools"],
                max_users=100,
                trial_days=30
            ),
            PricingPlan(
                plan_id="enterprise",
                name="Enterprise Plan",
                description="For large organizations with complex AI needs",
                tier=PlanTier.ENTERPRISE,
                pricing_model=PricingModel.ENTERPRISE,
                base_price=Decimal('2499'),
                billing_cycle=BillingCycle.MONTHLY,
                included_usage={
                    "api_calls": 10000000,
                    "ai_tokens": 1000000000,
                    "agent_hours": 2500,
                    "storage_gb": 2500,
                    "active_users": 500
                },
                feature_access=["all_features", "custom_development", "dedicated_support", "sla_guarantees"],
                max_users=None,  # Unlimited
                trial_days=60
            )
        ]
        
        for plan in default_plans:
            self.pricing_plans[plan.plan_id] = plan
        
        self.logger.info(f"Initialized {len(default_metrics)} usage metrics and {len(default_plans)} pricing plans")
    
    async def record_usage(
        self,
        customer_id: str,
        metric_id: str,
        quantity: Decimal,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record usage for a customer"""
        
        if metric_id not in self.usage_metrics:
            raise ValueError(f"Unknown usage metric: {metric_id}")
        
        metric = self.usage_metrics[metric_id]
        usage_id = f"usage_{customer_id}_{metric_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Calculate cost
        unit_price = metric.unit_price
        total_cost = quantity * unit_price
        
        # Create usage record
        usage_record = UsageRecord(
            usage_id=usage_id,
            customer_id=customer_id,
            metric_id=metric_id,
            quantity=quantity,
            unit_price=unit_price,
            total_cost=total_cost,
            resource_id=resource_id,
            user_id=user_id,
            session_id=session_id,
            billing_period=self._get_current_billing_period(),
            metadata=metadata or {}
        )
        
        # Store usage record
        self.usage_records.append(usage_record)
        
        # Update customer entitlements
        await self._update_customer_usage(customer_id, metric_id, quantity)
        
        # Store in memory
        await self._store_usage_record(usage_record)
        
        # Record metrics
        self.metrics.increment(f"usage_recorded_{metric_id}")
        self.metrics.record_metric(f"usage_quantity_{metric_id}", float(quantity))
        self.metrics.record_metric(f"usage_cost_{metric_id}", float(total_cost))
        
        self.logger.info(f"Recorded usage {usage_id}: {quantity} {metric_id} for customer {customer_id}")
        
        return usage_id
    
    async def check_entitlements(
        self,
        customer_id: str,
        resource_name: str,
        requested_quantity: int = 1
    ) -> Dict[str, Any]:
        """Check if customer has entitlement for requested resource usage"""
        
        entitlements = self.customer_entitlements.get(customer_id, [])
        
        for entitlement in entitlements:
            if entitlement.resource_name == resource_name and entitlement.active:
                remaining = entitlement.limit_value - entitlement.current_usage
                
                result = {
                    "entitled": remaining >= requested_quantity,
                    "remaining": remaining,
                    "limit": entitlement.limit_value,
                    "current_usage": entitlement.current_usage,
                    "overage_allowed": entitlement.overage_allowed,
                    "entitlement_id": entitlement.entitlement_id
                }
                
                if not result["entitled"] and entitlement.overage_allowed:
                    overage_quantity = requested_quantity - remaining
                    overage_cost = await self._calculate_overage_cost(resource_name, overage_quantity)
                    result["overage_cost"] = overage_cost
                
                return result
        
        # No entitlement found - check if it's a paid resource
        return {
            "entitled": False,
            "remaining": 0,
            "limit": 0,
            "current_usage": 0,
            "overage_allowed": True,
            "pay_per_use": True
        }
    
    async def create_customer_subscription(
        self,
        customer_id: str,
        plan_id: str,
        billing_cycle: Optional[BillingCycle] = None,
        trial_override_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a new customer subscription"""
        
        if plan_id not in self.pricing_plans:
            raise ValueError(f"Unknown pricing plan: {plan_id}")
        
        plan = self.pricing_plans[plan_id]
        
        # Create entitlements based on plan
        subscription_start = datetime.utcnow()
        trial_days = trial_override_days if trial_override_days is not None else plan.trial_days
        
        if trial_days > 0:
            trial_end = subscription_start + timedelta(days=trial_days)
            billing_start = trial_end
        else:
            billing_start = subscription_start
        
        # Calculate billing period end
        cycle = billing_cycle or plan.billing_cycle
        if cycle == BillingCycle.MONTHLY:
            period_end = billing_start + timedelta(days=30)
        elif cycle == BillingCycle.QUARTERLY:
            period_end = billing_start + timedelta(days=90)
        elif cycle == BillingCycle.ANNUALLY:
            period_end = billing_start + timedelta(days=365)
        else:
            period_end = billing_start + timedelta(days=30)  # Default to monthly
        
        # Create entitlements for included usage
        entitlements = []
        for metric_id, limit in plan.included_usage.items():
            entitlement_id = f"ent_{customer_id}_{metric_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            entitlement = CustomerEntitlement(
                entitlement_id=entitlement_id,
                customer_id=customer_id,
                plan_id=plan_id,
                entitlement_type=EntitlementType.USAGE_LIMIT,
                resource_name=metric_id,
                limit_value=limit,
                period_start=billing_start,
                period_end=period_end
            )
            
            entitlements.append(entitlement)
            self.customer_entitlements[customer_id].append(entitlement)
        
        # Store subscription
        subscription_data = {
            "customer_id": customer_id,
            "plan_id": plan_id,
            "billing_cycle": cycle.value,
            "subscription_start": subscription_start,
            "trial_days": trial_days,
            "billing_start": billing_start,
            "period_end": period_end,
            "entitlements": [e.entitlement_id for e in entitlements],
            "status": "trial" if trial_days > 0 else "active"
        }
        
        await self._store_subscription(subscription_data)
        
        # Record metrics
        self.metrics.increment("subscription_created")
        self.metrics.increment(f"subscription_plan_{plan_id}")
        
        self.logger.info(f"Created subscription for customer {customer_id} on plan {plan_id}")
        
        return subscription_data
    
    async def generate_invoice(
        self,
        customer_id: str,
        billing_period_start: datetime,
        billing_period_end: datetime
    ) -> Invoice:
        """Generate invoice for customer billing period"""
        
        invoice_id = f"inv_{customer_id}_{billing_period_start.strftime('%Y%m%d')}"
        invoice_number = f"INV-{datetime.utcnow().strftime('%Y%m%d')}-{customer_id[-6:].upper()}"
        
        # Get usage records for billing period
        period_usage = [
            record for record in self.usage_records
            if (record.customer_id == customer_id and
                billing_period_start <= record.usage_timestamp <= billing_period_end)
        ]
        
        # Calculate line items
        line_items = []
        subtotal = Decimal('0')
        
        # Group usage by metric
        usage_by_metric = defaultdict(list)
        for record in period_usage:
            usage_by_metric[record.metric_id].append(record)
        
        # Calculate charges for each metric
        for metric_id, records in usage_by_metric.items():
            metric = self.usage_metrics[metric_id]
            total_quantity = sum(record.quantity for record in records)
            total_cost = sum(record.total_cost for record in records)
            
            line_item = {
                "description": f"{metric.name} ({total_quantity} units)",
                "quantity": float(total_quantity),
                "unit_price": float(metric.unit_price),
                "total": float(total_cost),
                "metric_id": metric_id
            }
            
            line_items.append(line_item)
            subtotal += total_cost
        
        # Add subscription base charges
        customer_entitlements = self.customer_entitlements.get(customer_id, [])
        for entitlement in customer_entitlements:
            if entitlement.period_start <= billing_period_start <= entitlement.period_end:
                plan = self.pricing_plans.get(entitlement.plan_id)
                if plan and plan.base_price > 0:
                    line_item = {
                        "description": f"{plan.name} - Base Subscription",
                        "quantity": 1,
                        "unit_price": float(plan.base_price),
                        "total": float(plan.base_price),
                        "plan_id": plan.plan_id
                    }
                    line_items.append(line_item)
                    subtotal += plan.base_price
                    break  # Only add base charge once
        
        # Calculate tax and total
        tax_amount = subtotal * self.tax_rate
        total_amount = subtotal + tax_amount
        
        # Create invoice
        invoice = Invoice(
            invoice_id=invoice_id,
            customer_id=customer_id,
            invoice_number=invoice_number,
            billing_period_start=billing_period_start,
            billing_period_end=billing_period_end,
            subtotal=subtotal,
            tax_amount=tax_amount,
            discount_amount=Decimal('0'),
            total_amount=total_amount,
            line_items=line_items,
            status="sent"
        )
        
        # Store invoice
        self.invoices[invoice_id] = invoice
        await self._store_invoice(invoice)
        
        # Mark usage records as processed
        for record in period_usage:
            record.processed = True
        
        # Record metrics
        self.metrics.increment("invoice_generated")
        self.metrics.record_metric("invoice_amount", float(total_amount))
        
        self.logger.info(f"Generated invoice {invoice_number} for customer {customer_id}: ${total_amount}")
        
        return invoice
    
    async def get_usage_analytics(
        self,
        customer_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get usage analytics for customer"""
        
        # Filter usage records
        customer_usage = [
            record for record in self.usage_records
            if (record.customer_id == customer_id and
                start_date <= record.usage_timestamp <= end_date)
        ]
        
        # Aggregate by metric
        analytics = {
            "customer_id": customer_id,
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "total_cost": Decimal('0'),
            "total_usage_events": len(customer_usage),
            "metrics": {}
        }
        
        usage_by_metric = defaultdict(list)
        for record in customer_usage:
            usage_by_metric[record.metric_id].append(record)
            analytics["total_cost"] += record.total_cost
        
        for metric_id, records in usage_by_metric.items():
            metric = self.usage_metrics[metric_id]
            total_quantity = sum(record.quantity for record in records)
            total_cost = sum(record.total_cost for record in records)
            
            analytics["metrics"][metric_id] = {
                "name": metric.name,
                "total_quantity": float(total_quantity),
                "total_cost": float(total_cost),
                "average_unit_price": float(total_cost / total_quantity) if total_quantity > 0 else 0,
                "usage_events": len(records)
            }
        
        # Get entitlement status
        entitlements = self.customer_entitlements.get(customer_id, [])
        analytics["entitlements"] = []
        
        for entitlement in entitlements:
            if entitlement.active:
                utilization = (entitlement.current_usage / entitlement.limit_value * 100) if entitlement.limit_value > 0 else 0
                
                analytics["entitlements"].append({
                    "resource_name": entitlement.resource_name,
                    "limit": entitlement.limit_value,
                    "current_usage": entitlement.current_usage,
                    "utilization_percentage": utilization,
                    "period_start": entitlement.period_start.isoformat(),
                    "period_end": entitlement.period_end.isoformat()
                })
        
        return analytics
    
    async def get_cost_optimization_recommendations(
        self,
        customer_id: str
    ) -> List[Dict[str, Any]]:
        """Get cost optimization recommendations for customer"""
        
        recommendations = []
        
        # Analyze usage patterns
        recent_usage = [
            record for record in self.usage_records
            if (record.customer_id == customer_id and
                record.usage_timestamp > datetime.utcnow() - timedelta(days=30))
        ]
        
        if not recent_usage:
            return recommendations
        
        # Check for underutilized entitlements
        entitlements = self.customer_entitlements.get(customer_id, [])
        for entitlement in entitlements:
            if entitlement.active:
                utilization = (entitlement.current_usage / entitlement.limit_value) if entitlement.limit_value > 0 else 0
                
                if utilization < 0.5:  # Less than 50% utilization
                    # Find a lower plan
                    current_plan = self.pricing_plans.get(entitlement.plan_id)
                    if current_plan:
                        lower_plans = [
                            plan for plan in self.pricing_plans.values()
                            if (plan.base_price < current_plan.base_price and
                                plan.included_usage.get(entitlement.resource_name, 0) >= entitlement.current_usage * 2)
                        ]
                        
                        if lower_plans:
                            best_plan = min(lower_plans, key=lambda p: p.base_price)
                            potential_savings = current_plan.base_price - best_plan.base_price
                            
                            recommendations.append({
                                "type": "plan_downgrade",
                                "title": "Consider Downgrading Plan",
                                "description": f"You're only using {utilization:.1%} of your {entitlement.resource_name} allocation",
                                "current_plan": current_plan.name,
                                "recommended_plan": best_plan.name,
                                "potential_monthly_savings": float(potential_savings),
                                "confidence": "high" if utilization < 0.3 else "medium"
                            })
        
        # Check for high overage usage
        usage_by_metric = defaultdict(Decimal)
        for record in recent_usage:
            usage_by_metric[record.metric_id] += record.quantity
        
        for metric_id, total_usage in usage_by_metric.items():
            # Find entitlement for this metric
            metric_entitlement = None
            for entitlement in entitlements:
                if entitlement.resource_name == metric_id:
                    metric_entitlement = entitlement
                    break
            
            if metric_entitlement and total_usage > metric_entitlement.limit_value * 1.5:
                # High overage usage - recommend plan upgrade
                current_plan = self.pricing_plans.get(metric_entitlement.plan_id)
                if current_plan:
                    higher_plans = [
                        plan for plan in self.pricing_plans.values()
                        if (plan.base_price > current_plan.base_price and
                            plan.included_usage.get(metric_id, 0) >= total_usage)
                    ]
                    
                    if higher_plans:
                        best_plan = min(higher_plans, key=lambda p: p.base_price)
                        
                        # Calculate overage costs vs upgrade cost
                        overage_quantity = total_usage - metric_entitlement.limit_value
                        metric = self.usage_metrics[metric_id]
                        overage_cost = overage_quantity * (metric.overage_price or metric.unit_price)
                        upgrade_cost = best_plan.base_price - current_plan.base_price
                        
                        if overage_cost > upgrade_cost:
                            recommendations.append({
                                "type": "plan_upgrade",
                                "title": "Consider Upgrading Plan",
                                "description": f"High overage usage for {metric.name}",
                                "current_plan": current_plan.name,
                                "recommended_plan": best_plan.name,
                                "monthly_overage_cost": float(overage_cost),
                                "upgrade_cost": float(upgrade_cost),
                                "potential_monthly_savings": float(overage_cost - upgrade_cost),
                                "confidence": "high"
                            })
        
        # Check for unused features
        current_plans = set(e.plan_id for e in entitlements if e.active)
        for plan_id in current_plans:
            plan = self.pricing_plans.get(plan_id)
            if plan and len(plan.feature_access) > 5:  # Plans with many features
                recommendations.append({
                    "type": "feature_optimization",
                    "title": "Review Feature Usage",
                    "description": f"Your {plan.name} includes {len(plan.feature_access)} features - ensure you're using them effectively",
                    "features": plan.feature_access,
                    "confidence": "low"
                })
        
        return recommendations
    
    def get_pricing_dashboard(self, customer_id: Optional[str] = None) -> Dict[str, Any]:
        """Get pricing and billing dashboard data"""
        
        now = datetime.utcnow()
        
        if customer_id:
            # Customer-specific dashboard
            customer_usage = [r for r in self.usage_records if r.customer_id == customer_id]
            customer_entitlements = self.customer_entitlements.get(customer_id, [])
            customer_invoices = [i for i in self.invoices.values() if i.customer_id == customer_id]
            
            # Calculate current month usage
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            current_month_usage = [
                r for r in customer_usage
                if r.usage_timestamp >= month_start
            ]
            
            current_month_cost = sum(r.total_cost for r in current_month_usage)
            
            return {
                "customer_id": customer_id,
                "current_month": {
                    "usage_events": len(current_month_usage),
                    "total_cost": float(current_month_cost),
                    "period_start": month_start.isoformat(),
                    "period_end": now.isoformat()
                },
                "entitlements": [
                    {
                        "resource_name": e.resource_name,
                        "limit": e.limit_value,
                        "current_usage": e.current_usage,
                        "utilization": (e.current_usage / e.limit_value * 100) if e.limit_value > 0 else 0,
                        "active": e.active
                    }
                    for e in customer_entitlements if e.active
                ],
                "recent_invoices": [
                    {
                        "invoice_number": i.invoice_number,
                        "total_amount": float(i.total_amount),
                        "status": i.status,
                        "due_date": i.due_date.isoformat()
                    }
                    for i in sorted(customer_invoices, key=lambda x: x.created_at, reverse=True)[:5]
                ]
            }
        
        else:
            # Platform-wide dashboard
            total_customers = len(set(r.customer_id for r in self.usage_records))
            total_revenue = sum(i.total_amount for i in self.invoices.values() if i.status == "paid")
            
            # Plan distribution
            plan_distribution = defaultdict(int)
            for entitlements in self.customer_entitlements.values():
                for entitlement in entitlements:
                    if entitlement.active:
                        plan_distribution[entitlement.plan_id] += 1
                        break  # Count each customer once
            
            return {
                "overview": {
                    "total_customers": total_customers,
                    "total_revenue": float(total_revenue),
                    "active_plans": len(self.pricing_plans),
                    "usage_metrics": len(self.usage_metrics)
                },
                "plan_distribution": dict(plan_distribution),
                "recent_activity": {
                    "usage_events_today": len([
                        r for r in self.usage_records
                        if r.usage_timestamp.date() == now.date()
                    ]),
                    "invoices_generated_this_month": len([
                        i for i in self.invoices.values()
                        if i.created_at.month == now.month and i.created_at.year == now.year
                    ])
                }
            }
    
    # Private helper methods
    def _get_current_billing_period(self) -> str:
        """Get current billing period identifier"""
        now = datetime.utcnow()
        return f"{now.year}-{now.month:02d}"
    
    async def _update_customer_usage(
        self,
        customer_id: str,
        metric_id: str,
        quantity: Decimal
    ) -> None:
        """Update customer usage against entitlements"""
        
        entitlements = self.customer_entitlements.get(customer_id, [])
        
        for entitlement in entitlements:
            if (entitlement.resource_name == metric_id and
                entitlement.active and
                entitlement.period_start <= datetime.utcnow() <= entitlement.period_end):
                
                entitlement.current_usage += int(quantity)
                entitlement.updated_at = datetime.utcnow()
                break
    
    async def _calculate_overage_cost(
        self,
        resource_name: str,
        overage_quantity: int
    ) -> Decimal:
        """Calculate cost for overage usage"""
        
        if resource_name in self.usage_metrics:
            metric = self.usage_metrics[resource_name]
            overage_price = metric.overage_price or metric.unit_price
            return Decimal(overage_quantity) * overage_price
        
        return Decimal('0')
    
    # Storage methods
    async def _store_usage_record(self, usage_record: UsageRecord) -> None:
        """Store usage record in memory"""
        
        await self.memory_manager.store_context(
            context_type="billing_usage_record",
            content=usage_record.__dict__,
            metadata={
                "customer_id": usage_record.customer_id,
                "metric_id": usage_record.metric_id,
                "quantity": float(usage_record.quantity),
                "cost": float(usage_record.total_cost)
            }
        )
    
    async def _store_subscription(self, subscription_data: Dict[str, Any]) -> None:
        """Store subscription data in memory"""
        
        await self.memory_manager.store_context(
            context_type="billing_subscription",
            content=subscription_data,
            metadata={
                "customer_id": subscription_data["customer_id"],
                "plan_id": subscription_data["plan_id"],
                "status": subscription_data["status"]
            }
        )
    
    async def _store_invoice(self, invoice: Invoice) -> None:
        """Store invoice in memory"""
        
        await self.memory_manager.store_context(
            context_type="billing_invoice",
            content=invoice.__dict__,
            metadata={
                "invoice_id": invoice.invoice_id,
                "customer_id": invoice.customer_id,
                "total_amount": float(invoice.total_amount),
                "status": invoice.status
            }
        )