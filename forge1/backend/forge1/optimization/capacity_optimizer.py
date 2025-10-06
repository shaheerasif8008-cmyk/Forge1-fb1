"""
Forge 1 Capacity and Cost Optimization Engine
Predictive scaling, model cost routing, budget alerts, and FinOps dashboards
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import logging
import json
import math
from collections import defaultdict, deque

# Mock dependencies for standalone operation
class MetricsCollector:
    def increment(self, metric): pass
    def record_metric(self, metric, value): pass

class MemoryManager:
    async def store_context(self, context_type, content, metadata): pass

class SecretManager:
    async def get(self, name): return "mock_secret"


class ResourceType(Enum):
    """Types of resources to optimize"""
    COMPUTE = "compute"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    AI_MODEL = "ai_model"
    DATABASE = "database"
    CACHE = "cache"


class ScalingDirection(Enum):
    """Scaling direction"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class OptimizationStrategy(Enum):
    """Optimization strategies"""
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    BALANCED = "balanced"
    PREDICTIVE = "predictive"


class AlertSeverity(Enum):
    """Budget alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""
    resource_id: str
    resource_type: ResourceType
    
    # Utilization metrics
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    storage_utilization: float = 0.0
    network_utilization: float = 0.0
    
    # Performance metrics
    response_time_ms: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0
    
    # Cost metrics
    hourly_cost: Decimal = Decimal('0')
    daily_cost: Decimal = Decimal('0')
    monthly_cost: Decimal = Decimal('0')
    
    # Capacity metrics
    current_capacity: int = 0
    max_capacity: int = 0
    reserved_capacity: int = 0
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ScalingRecommendation:
    """Scaling recommendation"""
    recommendation_id: str
    resource_id: str
    resource_type: ResourceType
    
    # Recommendation details
    direction: ScalingDirection
    current_capacity: int
    recommended_capacity: int
    confidence_score: float
    
    # Justification
    reason: str
    expected_cost_impact: Decimal
    expected_performance_impact: str
    
    # Timing
    recommended_at: datetime = field(default_factory=datetime.utcnow)
    execute_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Metadata
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetAlert:
    """Budget monitoring alert"""
    alert_id: str
    budget_name: str
    severity: AlertSeverity
    
    # Budget details
    budget_limit: Decimal
    current_spend: Decimal
    projected_spend: Decimal
    threshold_percentage: float
    
    # Alert details
    message: str
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    resolved: bool = False
    
    # Actions
    recommended_actions: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostOptimizationOpportunity:
    """Cost optimization opportunity"""
    opportunity_id: str
    title: str
    description: str
    
    # Impact
    potential_monthly_savings: Decimal
    implementation_effort: str  # low, medium, high
    risk_level: str  # low, medium, high
    
    # Details
    affected_resources: List[str] = field(default_factory=list)
    implementation_steps: List[str] = field(default_factory=list)
    
    # Timeline
    identified_at: datetime = field(default_factory=datetime.utcnow)
    estimated_implementation_time: str = ""
    
    # Status
    status: str = "identified"  # identified, planned, in_progress, completed, dismissed
    
    # Metadata
    category: str = ""  # rightsizing, reserved_instances, spot_instances, etc.
    confidence: float = 0.0


class CapacityOptimizer:
    """
    Comprehensive capacity and cost optimization engine
    
    Features:
    - Predictive scaling based on usage patterns
    - Model cost routing and optimization
    - Budget monitoring and alerting
    - FinOps dashboards and recommendations
    - Automated cost optimization
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
        self.logger = logging.getLogger("capacity_optimizer")
        
        # Resource tracking
        self.resource_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1440))  # 24 hours of minute data
        self.scaling_recommendations: Dict[str, ScalingRecommendation] = {}
        
        # Budget monitoring
        self.budgets: Dict[str, Dict[str, Any]] = {}
        self.budget_alerts: Dict[str, BudgetAlert] = {}
        
        # Cost optimization
        self.optimization_opportunities: Dict[str, CostOptimizationOpportunity] = {}
        
        # Configuration
        self.optimization_strategy = OptimizationStrategy.BALANCED
        self.scaling_cooldown_minutes = 15
        self.prediction_window_hours = 24
        
        # Model cost routing
        self.model_costs = {
            "gpt-4o": {"input": Decimal('0.005'), "output": Decimal('0.015')},  # per 1K tokens
            "gpt-4o-mini": {"input": Decimal('0.00015'), "output": Decimal('0.0006')},
            "claude-3.5-sonnet": {"input": Decimal('0.003'), "output": Decimal('0.015')},
            "gemini-1.5-pro": {"input": Decimal('0.00125'), "output": Decimal('0.005')},
            "llama-3.1-70b": {"input": Decimal('0.0009'), "output": Decimal('0.0009')}
        }
        
        self.logger.info("Initialized Capacity Optimizer")
    
    async def record_resource_metrics(
        self,
        resource_id: str,
        resource_type: ResourceType,
        cpu_utilization: float = 0.0,
        memory_utilization: float = 0.0,
        storage_utilization: float = 0.0,
        response_time_ms: float = 0.0,
        throughput_rps: float = 0.0,
        error_rate: float = 0.0,
        hourly_cost: Decimal = Decimal('0'),
        current_capacity: int = 0,
        max_capacity: int = 0
    ) -> None:
        """Record resource utilization metrics"""
        
        metrics = ResourceMetrics(
            resource_id=resource_id,
            resource_type=resource_type,
            cpu_utilization=cpu_utilization,
            memory_utilization=memory_utilization,
            storage_utilization=storage_utilization,
            response_time_ms=response_time_ms,
            throughput_rps=throughput_rps,
            error_rate=error_rate,
            hourly_cost=hourly_cost,
            daily_cost=hourly_cost * 24,
            monthly_cost=hourly_cost * 24 * 30,
            current_capacity=current_capacity,
            max_capacity=max_capacity
        )
        
        # Store metrics in time series
        self.resource_metrics[resource_id].append(metrics)
        
        # Store in memory
        await self._store_resource_metrics(metrics)
        
        # Record platform metrics
        self.metrics.record_metric(f"resource_cpu_utilization_{resource_id}", cpu_utilization)
        self.metrics.record_metric(f"resource_memory_utilization_{resource_id}", memory_utilization)
        self.metrics.record_metric(f"resource_cost_hourly_{resource_id}", float(hourly_cost))
        
        # Check for scaling opportunities
        await self._analyze_scaling_opportunity(resource_id)
        
        self.logger.debug(f"Recorded metrics for resource {resource_id}: CPU {cpu_utilization:.1f}%, Cost ${hourly_cost}/hr")
    
    async def generate_scaling_recommendation(
        self,
        resource_id: str,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    ) -> Optional[ScalingRecommendation]:
        """Generate scaling recommendation for a resource"""
        
        if resource_id not in self.resource_metrics:
            return None
        
        metrics_history = list(self.resource_metrics[resource_id])
        if len(metrics_history) < 10:  # Need at least 10 data points
            return None
        
        latest_metrics = metrics_history[-1]
        
        # Analyze utilization trends
        recent_metrics = metrics_history[-30:]  # Last 30 minutes
        avg_cpu = sum(m.cpu_utilization for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_utilization for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics)
        
        # Calculate trend
        if len(recent_metrics) >= 10:
            early_avg = sum(m.cpu_utilization for m in recent_metrics[:10]) / 10
            late_avg = sum(m.cpu_utilization for m in recent_metrics[-10:]) / 10
            cpu_trend = late_avg - early_avg
        else:
            cpu_trend = 0
        
        # Determine scaling recommendation
        recommendation = None
        
        if strategy == OptimizationStrategy.COST_OPTIMIZED:
            # Prioritize cost savings
            if avg_cpu < 30 and avg_memory < 40 and avg_response_time < 100:
                # Scale down for cost savings
                new_capacity = max(1, int(latest_metrics.current_capacity * 0.7))
                cost_savings = (latest_metrics.current_capacity - new_capacity) * latest_metrics.hourly_cost / latest_metrics.current_capacity
                
                recommendation = ScalingRecommendation(
                    recommendation_id=f"scale_{resource_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    resource_id=resource_id,
                    resource_type=latest_metrics.resource_type,
                    direction=ScalingDirection.DOWN,
                    current_capacity=latest_metrics.current_capacity,
                    recommended_capacity=new_capacity,
                    confidence_score=0.8,
                    reason="Low utilization detected - cost optimization opportunity",
                    expected_cost_impact=-cost_savings,
                    expected_performance_impact="Minimal impact expected",
                    strategy=strategy
                )
        
        elif strategy == OptimizationStrategy.PERFORMANCE_OPTIMIZED:
            # Prioritize performance
            if avg_cpu > 70 or avg_memory > 80 or avg_response_time > 500 or cpu_trend > 10:
                # Scale up for performance
                new_capacity = min(latest_metrics.max_capacity, int(latest_metrics.current_capacity * 1.5))
                cost_increase = (new_capacity - latest_metrics.current_capacity) * latest_metrics.hourly_cost / latest_metrics.current_capacity
                
                recommendation = ScalingRecommendation(
                    recommendation_id=f"scale_{resource_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    resource_id=resource_id,
                    resource_type=latest_metrics.resource_type,
                    direction=ScalingDirection.UP,
                    current_capacity=latest_metrics.current_capacity,
                    recommended_capacity=new_capacity,
                    confidence_score=0.9,
                    reason="High utilization or performance degradation detected",
                    expected_cost_impact=cost_increase,
                    expected_performance_impact="Significant performance improvement expected",
                    strategy=strategy
                )
        
        else:  # BALANCED or PREDICTIVE
            # Balance cost and performance
            if avg_cpu > 80 or avg_memory > 85 or (avg_response_time > 300 and cpu_trend > 5):
                # Scale up
                new_capacity = min(latest_metrics.max_capacity, int(latest_metrics.current_capacity * 1.3))
                cost_increase = (new_capacity - latest_metrics.current_capacity) * latest_metrics.hourly_cost / latest_metrics.current_capacity
                
                recommendation = ScalingRecommendation(
                    recommendation_id=f"scale_{resource_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    resource_id=resource_id,
                    resource_type=latest_metrics.resource_type,
                    direction=ScalingDirection.UP,
                    current_capacity=latest_metrics.current_capacity,
                    recommended_capacity=new_capacity,
                    confidence_score=0.85,
                    reason="Balanced scaling up for performance and efficiency",
                    expected_cost_impact=cost_increase,
                    expected_performance_impact="Improved performance with controlled cost increase",
                    strategy=strategy
                )
            
            elif avg_cpu < 25 and avg_memory < 30 and avg_response_time < 100 and cpu_trend < -5:
                # Scale down
                new_capacity = max(1, int(latest_metrics.current_capacity * 0.8))
                cost_savings = (latest_metrics.current_capacity - new_capacity) * latest_metrics.hourly_cost / latest_metrics.current_capacity
                
                recommendation = ScalingRecommendation(
                    recommendation_id=f"scale_{resource_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    resource_id=resource_id,
                    resource_type=latest_metrics.resource_type,
                    direction=ScalingDirection.DOWN,
                    current_capacity=latest_metrics.current_capacity,
                    recommended_capacity=new_capacity,
                    confidence_score=0.75,
                    reason="Balanced scaling down for cost optimization",
                    expected_cost_impact=-cost_savings,
                    expected_performance_impact="Minimal performance impact with cost savings",
                    strategy=strategy
                )
        
        if recommendation:
            # Set execution timing
            recommendation.execute_at = datetime.utcnow() + timedelta(minutes=5)  # 5-minute delay
            recommendation.expires_at = datetime.utcnow() + timedelta(hours=1)  # 1-hour expiry
            
            # Store recommendation
            self.scaling_recommendations[recommendation.recommendation_id] = recommendation
            
            # Store in memory
            await self._store_scaling_recommendation(recommendation)
            
            # Record metrics
            self.metrics.increment(f"scaling_recommendation_{recommendation.direction.value}")
            
            self.logger.info(f"Generated scaling recommendation for {resource_id}: {recommendation.direction.value} to {recommendation.recommended_capacity}")
        
        return recommendation
    
    async def optimize_model_routing(
        self,
        task_complexity: str,  # simple, medium, complex
        quality_requirement: str,  # basic, high, premium
        latency_requirement: str,  # relaxed, standard, fast
        budget_constraint: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """Optimize AI model selection based on requirements and cost"""
        
        # Define model capabilities and characteristics
        model_profiles = {
            "gpt-4o-mini": {
                "complexity_score": 7,
                "quality_score": 8,
                "latency_score": 9,
                "cost_score": 10,
                "suitable_for": ["simple", "medium"]
            },
            "gpt-4o": {
                "complexity_score": 10,
                "quality_score": 10,
                "latency_score": 7,
                "cost_score": 3,
                "suitable_for": ["medium", "complex"]
            },
            "claude-3.5-sonnet": {
                "complexity_score": 9,
                "quality_score": 9,
                "latency_score": 8,
                "cost_score": 5,
                "suitable_for": ["medium", "complex"]
            },
            "gemini-1.5-pro": {
                "complexity_score": 8,
                "quality_score": 8,
                "latency_score": 8,
                "cost_score": 7,
                "suitable_for": ["simple", "medium", "complex"]
            },
            "llama-3.1-70b": {
                "complexity_score": 8,
                "quality_score": 7,
                "latency_score": 6,
                "cost_score": 8,
                "suitable_for": ["simple", "medium"]
            }
        }
        
        # Score models based on requirements
        model_scores = {}
        
        for model_name, profile in model_profiles.items():
            score = 0
            
            # Task complexity matching
            if task_complexity in profile["suitable_for"]:
                score += profile["complexity_score"] * 0.3
            else:
                score += profile["complexity_score"] * 0.1  # Penalty for mismatch
            
            # Quality requirement
            quality_weight = {"basic": 0.2, "high": 0.3, "premium": 0.4}[quality_requirement]
            score += profile["quality_score"] * quality_weight
            
            # Latency requirement
            latency_weight = {"relaxed": 0.1, "standard": 0.2, "fast": 0.3}[latency_requirement]
            score += profile["latency_score"] * latency_weight
            
            # Cost consideration
            cost_weight = 0.3 if budget_constraint else 0.2
            score += profile["cost_score"] * cost_weight
            
            model_scores[model_name] = score
        
        # Select best model
        best_model = max(model_scores.items(), key=lambda x: x[1])
        
        # Calculate estimated costs for different token counts
        estimated_costs = {}
        for tokens in [1000, 5000, 10000, 50000]:
            token_multiplier = Decimal(str(tokens / 1000))
            input_cost = self.model_costs[best_model[0]]["input"] * token_multiplier
            output_cost = self.model_costs[best_model[0]]["output"] * token_multiplier
            estimated_costs[f"{tokens}_tokens"] = {
                "input_cost": float(input_cost),
                "output_cost": float(output_cost),
                "total_cost": float(input_cost + output_cost)
            }
        
        # Generate alternatives
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        alternatives = []
        
        for model_name, score in sorted_models[1:4]:  # Top 3 alternatives
            profile = model_profiles[model_name]
            alternatives.append({
                "model": model_name,
                "score": score,
                "strengths": self._get_model_strengths(profile),
                "estimated_cost_1k_tokens": float(
                    self.model_costs[model_name]["input"] + self.model_costs[model_name]["output"]
                )
            })
        
        recommendation = {
            "recommended_model": best_model[0],
            "confidence_score": best_model[1] / 10,  # Normalize to 0-1
            "reasoning": self._generate_model_selection_reasoning(
                best_model[0], model_profiles[best_model[0]], 
                task_complexity, quality_requirement, latency_requirement
            ),
            "estimated_costs": estimated_costs,
            "alternatives": alternatives,
            "optimization_factors": {
                "task_complexity": task_complexity,
                "quality_requirement": quality_requirement,
                "latency_requirement": latency_requirement,
                "budget_constraint": float(budget_constraint) if budget_constraint else None
            }
        }
        
        # Record metrics
        self.metrics.increment(f"model_routing_recommendation_{best_model[0]}")
        self.metrics.record_metric("model_routing_confidence", recommendation["confidence_score"])
        
        self.logger.info(f"Optimized model routing: {best_model[0]} (score: {best_model[1]:.2f})")
        
        return recommendation
    
    async def create_budget_alert(
        self,
        budget_name: str,
        budget_limit: Decimal,
        current_spend: Decimal,
        projected_spend: Decimal,
        threshold_percentage: float
    ) -> BudgetAlert:
        """Create budget monitoring alert"""
        
        alert_id = f"budget_alert_{budget_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine severity
        spend_percentage = (current_spend / budget_limit) * 100
        projected_percentage = (projected_spend / budget_limit) * 100
        
        if projected_percentage >= 100:
            severity = AlertSeverity.EMERGENCY
        elif projected_percentage >= 90:
            severity = AlertSeverity.CRITICAL
        elif spend_percentage >= threshold_percentage:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO
        
        # Generate message
        if severity == AlertSeverity.EMERGENCY:
            message = f"EMERGENCY: Budget '{budget_name}' projected to exceed limit by {projected_percentage - 100:.1f}%"
        elif severity == AlertSeverity.CRITICAL:
            message = f"CRITICAL: Budget '{budget_name}' projected to reach {projected_percentage:.1f}% of limit"
        elif severity == AlertSeverity.WARNING:
            message = f"WARNING: Budget '{budget_name}' has reached {spend_percentage:.1f}% of limit"
        else:
            message = f"INFO: Budget '{budget_name}' tracking at {spend_percentage:.1f}% of limit"
        
        # Generate recommended actions
        recommended_actions = []
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            recommended_actions.extend([
                "Review and optimize high-cost resources immediately",
                "Consider scaling down non-critical services",
                "Implement cost controls and spending limits",
                "Schedule emergency budget review meeting"
            ])
        elif severity == AlertSeverity.WARNING:
            recommended_actions.extend([
                "Review current spending trends",
                "Identify cost optimization opportunities",
                "Consider adjusting resource allocation"
            ])
        
        alert = BudgetAlert(
            alert_id=alert_id,
            budget_name=budget_name,
            severity=severity,
            budget_limit=budget_limit,
            current_spend=current_spend,
            projected_spend=projected_spend,
            threshold_percentage=threshold_percentage,
            message=message,
            recommended_actions=recommended_actions
        )
        
        # Store alert
        self.budget_alerts[alert_id] = alert
        
        # Store in memory
        await self._store_budget_alert(alert)
        
        # Record metrics
        self.metrics.increment(f"budget_alert_{severity.value}")
        self.metrics.record_metric("budget_utilization_percentage", float(spend_percentage))
        
        self.logger.warning(f"Budget alert created: {message}")
        
        return alert
    
    async def identify_cost_optimization_opportunities(
        self,
        resource_ids: Optional[List[str]] = None
    ) -> List[CostOptimizationOpportunity]:
        """Identify cost optimization opportunities"""
        
        opportunities = []
        
        # Analyze resource utilization
        resources_to_analyze = resource_ids or list(self.resource_metrics.keys())
        
        for resource_id in resources_to_analyze:
            if resource_id not in self.resource_metrics:
                continue
            
            metrics_history = list(self.resource_metrics[resource_id])
            if len(metrics_history) < 100:  # Need sufficient data
                continue
            
            # Analyze last 24 hours
            recent_metrics = metrics_history[-1440:] if len(metrics_history) >= 1440 else metrics_history
            
            avg_cpu = sum(m.cpu_utilization for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_utilization for m in recent_metrics) / len(recent_metrics)
            avg_cost = Decimal(str(sum(m.hourly_cost for m in recent_metrics) / len(recent_metrics)))
            
            # Rightsizing opportunity
            if avg_cpu < 20 and avg_memory < 30:
                potential_savings = avg_cost * Decimal('0.3') * Decimal('24') * Decimal('30')  # 30% savings per month
                
                opportunity = CostOptimizationOpportunity(
                    opportunity_id=f"rightsize_{resource_id}_{datetime.utcnow().strftime('%Y%m%d')}",
                    title=f"Rightsize Resource {resource_id}",
                    description=f"Resource is underutilized (CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.1f}%)",
                    potential_monthly_savings=potential_savings,
                    implementation_effort="low",
                    risk_level="low",
                    affected_resources=[resource_id],
                    implementation_steps=[
                        "Analyze peak usage patterns",
                        "Reduce resource allocation by 30-50%",
                        "Monitor performance after change",
                        "Adjust further if needed"
                    ],
                    category="rightsizing",
                    confidence=0.8,
                    estimated_implementation_time="1-2 hours"
                )
                
                opportunities.append(opportunity)
            
            # Scheduling opportunity (for batch workloads)
            peak_hours = [m for m in recent_metrics if 9 <= m.timestamp.hour <= 17]
            off_peak_hours = [m for m in recent_metrics if m.timestamp.hour < 9 or m.timestamp.hour > 17]
            
            if len(peak_hours) > 0 and len(off_peak_hours) > 0:
                peak_avg_cpu = sum(m.cpu_utilization for m in peak_hours) / len(peak_hours)
                off_peak_avg_cpu = sum(m.cpu_utilization for m in off_peak_hours) / len(off_peak_hours)
                
                if off_peak_avg_cpu > 50 and peak_avg_cpu < 30:
                    # Workload could be shifted to off-peak hours for cost savings
                    potential_savings = avg_cost * Decimal('0.4') * Decimal('8') * Decimal('30')  # 40% savings on 8 hours/day
                    
                    opportunity = CostOptimizationOpportunity(
                        opportunity_id=f"schedule_{resource_id}_{datetime.utcnow().strftime('%Y%m%d')}",
                        title=f"Optimize Scheduling for {resource_id}",
                        description="Workload can be shifted to off-peak hours for cost savings",
                        potential_monthly_savings=potential_savings,
                        implementation_effort="medium",
                        risk_level="low",
                        affected_resources=[resource_id],
                        implementation_steps=[
                            "Identify batch workloads suitable for scheduling",
                            "Implement job scheduling system",
                            "Shift non-critical workloads to off-peak hours",
                            "Use spot instances for scheduled workloads"
                        ],
                        category="scheduling",
                        confidence=0.7,
                        estimated_implementation_time="1-2 days"
                    )
                    
                    opportunities.append(opportunity)
        
        # Storage optimization
        storage_resources = [
            resource_id for resource_id in resources_to_analyze
            if resource_id in self.resource_metrics and 
            any(m.resource_type == ResourceType.STORAGE for m in self.resource_metrics[resource_id])
        ]
        
        if storage_resources:
            total_storage_cost = sum(
                list(self.resource_metrics[rid])[-1].monthly_cost
                for rid in storage_resources
            )
            
            if total_storage_cost > Decimal('1000'):  # Significant storage costs
                potential_savings = total_storage_cost * Decimal('0.25')  # 25% savings
                
                opportunity = CostOptimizationOpportunity(
                    opportunity_id=f"storage_optimization_{datetime.utcnow().strftime('%Y%m%d')}",
                    title="Storage Optimization",
                    description="Optimize storage costs through tiering and lifecycle policies",
                    potential_monthly_savings=potential_savings,
                    implementation_effort="medium",
                    risk_level="low",
                    affected_resources=storage_resources,
                    implementation_steps=[
                        "Implement storage tiering (hot/warm/cold)",
                        "Set up automated lifecycle policies",
                        "Compress and deduplicate data",
                        "Archive old data to cheaper storage tiers"
                    ],
                    category="storage_optimization",
                    confidence=0.75,
                    estimated_implementation_time="3-5 days"
                )
                
                opportunities.append(opportunity)
        
        # Store opportunities
        for opportunity in opportunities:
            self.optimization_opportunities[opportunity.opportunity_id] = opportunity
            await self._store_optimization_opportunity(opportunity)
        
        # Record metrics
        self.metrics.record_metric("cost_optimization_opportunities", len(opportunities))
        total_potential_savings = sum(op.potential_monthly_savings for op in opportunities)
        self.metrics.record_metric("potential_monthly_savings", float(total_potential_savings))
        
        self.logger.info(f"Identified {len(opportunities)} cost optimization opportunities with potential savings of ${total_potential_savings}/month")
        
        return opportunities
    
    def get_finops_dashboard(self) -> Dict[str, Any]:
        """Get FinOps dashboard data"""
        
        now = datetime.utcnow()
        
        # Calculate current costs
        current_hourly_cost = Decimal('0')
        total_resources = 0
        
        for resource_id, metrics_deque in self.resource_metrics.items():
            if metrics_deque:
                latest_metrics = metrics_deque[-1]
                current_hourly_cost += latest_metrics.hourly_cost
                total_resources += 1
        
        current_daily_cost = current_hourly_cost * 24
        current_monthly_cost = current_daily_cost * 30
        
        # Get recent scaling recommendations
        recent_recommendations = [
            rec for rec in self.scaling_recommendations.values()
            if rec.recommended_at > now - timedelta(hours=24)
        ]
        
        # Get active budget alerts
        active_alerts = [
            alert for alert in self.budget_alerts.values()
            if not alert.resolved
        ]
        
        # Get optimization opportunities
        active_opportunities = [
            opp for opp in self.optimization_opportunities.values()
            if opp.status in ["identified", "planned"]
        ]
        
        total_potential_savings = sum(opp.potential_monthly_savings for opp in active_opportunities)
        
        # Calculate utilization statistics
        utilization_stats = self._calculate_utilization_statistics()
        
        return {
            "cost_overview": {
                "current_hourly_cost": float(current_hourly_cost),
                "current_daily_cost": float(current_daily_cost),
                "projected_monthly_cost": float(current_monthly_cost),
                "total_resources": total_resources
            },
            "optimization_summary": {
                "active_opportunities": len(active_opportunities),
                "potential_monthly_savings": float(total_potential_savings),
                "recent_recommendations": len(recent_recommendations),
                "implemented_optimizations": len([
                    opp for opp in self.optimization_opportunities.values()
                    if opp.status == "completed"
                ])
            },
            "budget_monitoring": {
                "active_alerts": len(active_alerts),
                "critical_alerts": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                "emergency_alerts": len([a for a in active_alerts if a.severity == AlertSeverity.EMERGENCY])
            },
            "resource_utilization": utilization_stats,
            "scaling_activity": {
                "recommendations_today": len([
                    rec for rec in recent_recommendations
                    if rec.recommended_at.date() == now.date()
                ]),
                "scale_up_recommendations": len([
                    rec for rec in recent_recommendations
                    if rec.direction == ScalingDirection.UP
                ]),
                "scale_down_recommendations": len([
                    rec for rec in recent_recommendations
                    if rec.direction == ScalingDirection.DOWN
                ])
            },
            "top_cost_drivers": self._get_top_cost_drivers(),
            "efficiency_metrics": {
                "average_cpu_utilization": utilization_stats.get("avg_cpu_utilization", 0),
                "average_memory_utilization": utilization_stats.get("avg_memory_utilization", 0),
                "cost_per_transaction": self._calculate_cost_per_transaction(),
                "resource_efficiency_score": self._calculate_efficiency_score()
            }
        }
    
    # Private helper methods
    async def _analyze_scaling_opportunity(self, resource_id: str) -> None:
        """Analyze if resource needs scaling"""
        
        # Check if we have enough data and if cooldown period has passed
        if resource_id not in self.resource_metrics:
            return
        
        metrics_history = list(self.resource_metrics[resource_id])
        if len(metrics_history) < 5:
            return
        
        # Check for existing recent recommendations
        recent_recommendations = [
            rec for rec in self.scaling_recommendations.values()
            if (rec.resource_id == resource_id and
                rec.recommended_at > datetime.utcnow() - timedelta(minutes=self.scaling_cooldown_minutes))
        ]
        
        if recent_recommendations:
            return  # Still in cooldown period
        
        # Generate recommendation if needed
        recommendation = await self.generate_scaling_recommendation(resource_id, self.optimization_strategy)
        
        if recommendation and recommendation.confidence_score > 0.7:
            self.logger.info(f"High-confidence scaling opportunity identified for {resource_id}")
    
    def _get_model_strengths(self, profile: Dict[str, Any]) -> List[str]:
        """Get model strengths based on profile"""
        
        strengths = []
        
        if profile["cost_score"] >= 8:
            strengths.append("Cost-effective")
        if profile["quality_score"] >= 9:
            strengths.append("High quality output")
        if profile["latency_score"] >= 8:
            strengths.append("Fast response times")
        if profile["complexity_score"] >= 9:
            strengths.append("Handles complex tasks")
        
        return strengths
    
    def _generate_model_selection_reasoning(
        self,
        model_name: str,
        profile: Dict[str, Any],
        task_complexity: str,
        quality_requirement: str,
        latency_requirement: str
    ) -> str:
        """Generate reasoning for model selection"""
        
        reasons = []
        
        if task_complexity in profile["suitable_for"]:
            reasons.append(f"Well-suited for {task_complexity} tasks")
        
        if quality_requirement == "premium" and profile["quality_score"] >= 9:
            reasons.append("Meets premium quality requirements")
        elif quality_requirement == "high" and profile["quality_score"] >= 8:
            reasons.append("Meets high quality requirements")
        
        if latency_requirement == "fast" and profile["latency_score"] >= 8:
            reasons.append("Provides fast response times")
        
        if profile["cost_score"] >= 7:
            reasons.append("Cost-effective option")
        
        return "; ".join(reasons) if reasons else f"Best overall match for requirements"
    
    def _calculate_utilization_statistics(self) -> Dict[str, float]:
        """Calculate resource utilization statistics"""
        
        if not self.resource_metrics:
            return {}
        
        all_cpu_values = []
        all_memory_values = []
        all_storage_values = []
        
        for metrics_deque in self.resource_metrics.values():
            if metrics_deque:
                latest_metrics = metrics_deque[-1]
                all_cpu_values.append(latest_metrics.cpu_utilization)
                all_memory_values.append(latest_metrics.memory_utilization)
                all_storage_values.append(latest_metrics.storage_utilization)
        
        return {
            "avg_cpu_utilization": sum(all_cpu_values) / len(all_cpu_values) if all_cpu_values else 0,
            "avg_memory_utilization": sum(all_memory_values) / len(all_memory_values) if all_memory_values else 0,
            "avg_storage_utilization": sum(all_storage_values) / len(all_storage_values) if all_storage_values else 0,
            "max_cpu_utilization": max(all_cpu_values) if all_cpu_values else 0,
            "max_memory_utilization": max(all_memory_values) if all_memory_values else 0,
            "resources_over_80_percent_cpu": len([v for v in all_cpu_values if v > 80]),
            "resources_under_20_percent_cpu": len([v for v in all_cpu_values if v < 20])
        }
    
    def _get_top_cost_drivers(self) -> List[Dict[str, Any]]:
        """Get top cost-driving resources"""
        
        resource_costs = []
        
        for resource_id, metrics_deque in self.resource_metrics.items():
            if metrics_deque:
                latest_metrics = metrics_deque[-1]
                resource_costs.append({
                    "resource_id": resource_id,
                    "resource_type": latest_metrics.resource_type.value,
                    "hourly_cost": float(latest_metrics.hourly_cost),
                    "monthly_cost": float(latest_metrics.monthly_cost),
                    "cpu_utilization": latest_metrics.cpu_utilization,
                    "memory_utilization": latest_metrics.memory_utilization
                })
        
        # Sort by monthly cost and return top 10
        resource_costs.sort(key=lambda x: x["monthly_cost"], reverse=True)
        return resource_costs[:10]
    
    def _calculate_cost_per_transaction(self) -> float:
        """Calculate cost per transaction (mock implementation)"""
        # In a real implementation, this would calculate based on actual transaction volume
        total_hourly_cost = sum(
            list(metrics_deque)[-1].hourly_cost
            for metrics_deque in self.resource_metrics.values()
            if metrics_deque
        )
        
        # Mock transaction volume
        estimated_transactions_per_hour = 10000
        
        return float(total_hourly_cost / estimated_transactions_per_hour) if estimated_transactions_per_hour > 0 else 0
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate overall resource efficiency score"""
        
        utilization_stats = self._calculate_utilization_statistics()
        
        if not utilization_stats:
            return 0.0
        
        # Ideal utilization ranges
        ideal_cpu_range = (60, 80)  # 60-80% CPU utilization
        ideal_memory_range = (70, 85)  # 70-85% memory utilization
        
        cpu_util = utilization_stats.get("avg_cpu_utilization", 0)
        memory_util = utilization_stats.get("avg_memory_utilization", 0)
        
        # Calculate efficiency scores
        cpu_efficiency = 1.0
        if cpu_util < ideal_cpu_range[0]:
            cpu_efficiency = cpu_util / ideal_cpu_range[0]
        elif cpu_util > ideal_cpu_range[1]:
            cpu_efficiency = ideal_cpu_range[1] / cpu_util
        
        memory_efficiency = 1.0
        if memory_util < ideal_memory_range[0]:
            memory_efficiency = memory_util / ideal_memory_range[0]
        elif memory_util > ideal_memory_range[1]:
            memory_efficiency = ideal_memory_range[1] / memory_util
        
        # Combined efficiency score (0-100)
        overall_efficiency = (cpu_efficiency + memory_efficiency) / 2 * 100
        
        return min(100.0, max(0.0, overall_efficiency))
    
    # Storage methods
    async def _store_resource_metrics(self, metrics: ResourceMetrics) -> None:
        """Store resource metrics in memory"""
        
        await self.memory_manager.store_context(
            context_type="capacity_resource_metrics",
            content=metrics.__dict__,
            metadata={
                "resource_id": metrics.resource_id,
                "resource_type": metrics.resource_type.value,
                "timestamp": metrics.timestamp.isoformat(),
                "hourly_cost": float(metrics.hourly_cost)
            }
        )
    
    async def _store_scaling_recommendation(self, recommendation: ScalingRecommendation) -> None:
        """Store scaling recommendation in memory"""
        
        await self.memory_manager.store_context(
            context_type="capacity_scaling_recommendation",
            content=recommendation.__dict__,
            metadata={
                "recommendation_id": recommendation.recommendation_id,
                "resource_id": recommendation.resource_id,
                "direction": recommendation.direction.value,
                "confidence_score": recommendation.confidence_score
            }
        )
    
    async def _store_budget_alert(self, alert: BudgetAlert) -> None:
        """Store budget alert in memory"""
        
        await self.memory_manager.store_context(
            context_type="capacity_budget_alert",
            content=alert.__dict__,
            metadata={
                "alert_id": alert.alert_id,
                "budget_name": alert.budget_name,
                "severity": alert.severity.value,
                "current_spend": float(alert.current_spend)
            }
        )
    
    async def _store_optimization_opportunity(self, opportunity: CostOptimizationOpportunity) -> None:
        """Store optimization opportunity in memory"""
        
        await self.memory_manager.store_context(
            context_type="capacity_optimization_opportunity",
            content=opportunity.__dict__,
            metadata={
                "opportunity_id": opportunity.opportunity_id,
                "category": opportunity.category,
                "potential_savings": float(opportunity.potential_monthly_savings),
                "status": opportunity.status
            }
        )