# forge1/backend/forge1/services/employee_analytics_service.py
"""
Employee Analytics and Monitoring Service

Comprehensive analytics system for employee interactions, performance metrics,
client usage tracking, health monitoring, and cost analysis.

Requirements: 5.4, 7.3, 8.3
"""

import asyncio
import logging
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal

from forge1.core.database_config import DatabaseManager, get_database_manager
from forge1.core.tenancy import get_current_tenant
from forge1.core.performance_monitor import get_performance_monitor
from forge1.services.employee_manager import EmployeeManager
from forge1.services.employee_memory_manager import EmployeeMemoryManager
from forge1.models.employee_models import EmployeeStatus, MemoryType

logger = logging.getLogger(__name__)


@dataclass
class EmployeeMetrics:
    """Employee performance metrics"""
    employee_id: str
    employee_name: str
    total_interactions: int
    avg_response_time_ms: float
    success_rate: float
    user_satisfaction_score: float
    cost_per_interaction: float
    total_cost: float
    active_days: int
    peak_usage_hour: int
    common_topics: List[Dict[str, Any]]
    performance_trend: str


@dataclass
class ClientUsageMetrics:
    """Client usage analytics"""
    client_id: str
    client_name: str
    total_employees: int
    active_employees: int
    total_interactions: int
    total_cost: float
    avg_cost_per_employee: float
    top_performing_employees: List[Dict[str, Any]]
    usage_by_time: Dict[str, int]
    monthly_growth: float


class EmployeeAnalyticsService:
    """
    Comprehensive analytics service for employee performance and usage tracking.
    
    Features:
    - Real-time interaction analytics
    - Performance metrics calculation
    - Cost tracking and optimization
    - Health monitoring and alerting
    - Usage pattern analysis
    - Predictive analytics
    """
    
    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        employee_manager: Optional[EmployeeManager] = None,
        memory_manager: Optional[EmployeeMemoryManager] = None
    ):
        self.db_manager = db_manager
        self.employee_manager = employee_manager
        self.memory_manager = memory_manager
        self.performance_monitor = get_performance_monitor()
        
        # Analytics cache
        self.analytics_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Cost tracking
        self.cost_models = {
            "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015}
        }
        
        # Health thresholds
        self.health_thresholds = {
            "response_time_ms": 3000,
            "success_rate": 0.95,
            "user_satisfaction": 4.0,
            "error_rate": 0.05,
            "cost_per_interaction": 0.10
        }
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize the analytics service"""
        if self._initialized:
            return
        
        if not self.db_manager:
            self.db_manager = await get_database_manager()
        
        if not self.employee_manager:
            self.employee_manager = EmployeeManager(self.db_manager)
            await self.employee_manager.initialize()
        
        if not self.memory_manager:
            self.memory_manager = EmployeeMemoryManager(self.db_manager)
            await self.memory_manager.initialize()
        
        # Create analytics tables if they don't exist
        await self._create_analytics_tables()
        
        self._initialized = True
        logger.info("Employee Analytics Service initialized")
    
    async def get_employee_metrics(
        self,
        client_id: str,
        employee_id: str,
        days: int = 30
    ) -> EmployeeMetrics:
        """Get comprehensive metrics for a specific employee"""
        try:
            # Check cache first
            cache_key = f"employee_metrics:{client_id}:{employee_id}:{days}"
            if cache_key in self.analytics_cache:
                cached_data, cached_time = self.analytics_cache[cache_key]
                if (datetime.now(timezone.utc) - cached_time).seconds < self.cache_ttl:
                    return cached_data
            
            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            # Get employee info
            employee = await self.employee_manager.get_employee(client_id, employee_id)
            if not employee:
                raise ValueError(f"Employee {employee_id} not found")
            
            # Get interaction statistics
            async with self.db_manager.get_connection() as conn:
                # Basic interaction stats
                interaction_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_interactions,
                        AVG(processing_time_ms) as avg_response_time,
                        COUNT(CASE WHEN context->>'success' = 'true' THEN 1 END) as successful_interactions,
                        SUM(tokens_used) as total_tokens,
                        SUM(cost) as total_cost,
                        COUNT(DISTINCT DATE(timestamp)) as active_days,
                        EXTRACT(HOUR FROM timestamp) as peak_hour
                    FROM employee_interactions 
                    WHERE client_id = $1 AND employee_id = $2 
                    AND timestamp >= $3 AND timestamp <= $4
                """, client_id, employee_id, start_date, end_date)
                
                # User satisfaction scores (from feedback)
                satisfaction_stats = await conn.fetchrow("""
                    SELECT AVG(rating) as avg_satisfaction
                    FROM employee_feedback 
                    WHERE client_id = $1 AND employee_id = $2 
                    AND created_at >= $3 AND created_at <= $4
                """, client_id, employee_id, start_date, end_date)
                
                # Common topics analysis
                topic_stats = await conn.fetch("""
                    SELECT 
                        CASE 
                            WHEN LOWER(message) LIKE '%help%' OR LOWER(message) LIKE '%support%' THEN 'support'
                            WHEN LOWER(message) LIKE '%question%' OR LOWER(message) LIKE '%how%' THEN 'questions'
                            WHEN LOWER(message) LIKE '%problem%' OR LOWER(message) LIKE '%issue%' THEN 'problems'
                            WHEN LOWER(message) LIKE '%information%' OR LOWER(message) LIKE '%info%' THEN 'information'
                            ELSE 'general'
                        END as topic,
                        COUNT(*) as count,
                        AVG(processing_time_ms) as avg_response_time
                    FROM employee_interactions 
                    WHERE client_id = $1 AND employee_id = $2 
                    AND timestamp >= $3 AND timestamp <= $4
                    GROUP BY topic
                    ORDER BY count DESC
                    LIMIT 5
                """, client_id, employee_id, start_date, end_date)
                
                # Performance trend (comparing first half vs second half of period)
                trend_stats = await conn.fetch("""
                    SELECT 
                        CASE 
                            WHEN timestamp < $3 + INTERVAL '%s days' THEN 'first_half'
                            ELSE 'second_half'
                        END as period,
                        AVG(processing_time_ms) as avg_response_time,
                        COUNT(*) as interaction_count
                    FROM employee_interactions 
                    WHERE client_id = $1 AND employee_id = $2 
                    AND timestamp >= $3 AND timestamp <= $4
                    GROUP BY period
                """ % (days // 2), client_id, employee_id, start_date, end_date)
            
            # Calculate metrics
            total_interactions = interaction_stats["total_interactions"] or 0
            successful_interactions = interaction_stats["successful_interactions"] or 0
            success_rate = (successful_interactions / max(total_interactions, 1)) * 100
            
            avg_response_time = float(interaction_stats["avg_response_time"] or 0)
            total_cost = float(interaction_stats["total_cost"] or 0)
            cost_per_interaction = total_cost / max(total_interactions, 1)
            
            user_satisfaction = float(satisfaction_stats["avg_satisfaction"] or 0) if satisfaction_stats else 0
            active_days = interaction_stats["active_days"] or 0
            
            # Determine performance trend
            performance_trend = "stable"
            if len(trend_stats) == 2:
                first_half_time = next((s["avg_response_time"] for s in trend_stats if s["period"] == "first_half"), 0)
                second_half_time = next((s["avg_response_time"] for s in trend_stats if s["period"] == "second_half"), 0)
                
                if first_half_time and second_half_time:
                    change_percent = ((second_half_time - first_half_time) / first_half_time) * 100
                    if change_percent > 10:
                        performance_trend = "declining"
                    elif change_percent < -10:
                        performance_trend = "improving"
            
            # Format common topics
            common_topics = [
                {
                    "topic": row["topic"],
                    "count": row["count"],
                    "avg_response_time": float(row["avg_response_time"] or 0)
                }
                for row in topic_stats
            ]
            
            # Create metrics object
            metrics = EmployeeMetrics(
                employee_id=employee_id,
                employee_name=employee.name,
                total_interactions=total_interactions,
                avg_response_time_ms=avg_response_time,
                success_rate=success_rate,
                user_satisfaction_score=user_satisfaction,
                cost_per_interaction=cost_per_interaction,
                total_cost=total_cost,
                active_days=active_days,
                peak_usage_hour=int(interaction_stats["peak_hour"] or 12),
                common_topics=common_topics,
                performance_trend=performance_trend
            )
            
            # Cache the result
            self.analytics_cache[cache_key] = (metrics, datetime.now(timezone.utc))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get employee metrics: {e}")
            raise
    
    async def get_client_usage_metrics(
        self,
        client_id: str,
        days: int = 30
    ) -> ClientUsageMetrics:
        """Get comprehensive usage metrics for a client"""
        try:
            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            async with self.db_manager.get_connection() as conn:
                # Client basic info
                client_info = await conn.fetchrow("""
                    SELECT name FROM clients WHERE id = $1
                """, client_id)
                
                # Employee counts
                employee_counts = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_employees,
                        COUNT(CASE WHEN status = 'active' THEN 1 END) as active_employees
                    FROM employees 
                    WHERE client_id = $1
                """, client_id)
                
                # Interaction and cost totals
                usage_totals = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_interactions,
                        SUM(cost) as total_cost
                    FROM employee_interactions 
                    WHERE client_id = $1 
                    AND timestamp >= $2 AND timestamp <= $3
                """, client_id, start_date, end_date)
                
                # Top performing employees
                top_employees = await conn.fetch("""
                    SELECT 
                        e.id,
                        e.name,
                        COUNT(ei.id) as interaction_count,
                        AVG(ei.processing_time_ms) as avg_response_time,
                        SUM(ei.cost) as total_cost
                    FROM employees e
                    LEFT JOIN employee_interactions ei ON e.id = ei.employee_id
                    WHERE e.client_id = $1 
                    AND (ei.timestamp IS NULL OR (ei.timestamp >= $2 AND ei.timestamp <= $3))
                    GROUP BY e.id, e.name
                    ORDER BY interaction_count DESC
                    LIMIT 5
                """, client_id, start_date, end_date)
                
                # Usage by hour of day
                hourly_usage = await conn.fetch("""
                    SELECT 
                        EXTRACT(HOUR FROM timestamp) as hour,
                        COUNT(*) as interaction_count
                    FROM employee_interactions 
                    WHERE client_id = $1 
                    AND timestamp >= $2 AND timestamp <= $3
                    GROUP BY hour
                    ORDER BY hour
                """, client_id, start_date, end_date)
                
                # Monthly growth calculation
                previous_month_start = start_date - timedelta(days=days)
                previous_month_interactions = await conn.fetchval("""
                    SELECT COUNT(*) FROM employee_interactions 
                    WHERE client_id = $1 
                    AND timestamp >= $2 AND timestamp < $3
                """, client_id, previous_month_start, start_date)
            
            # Calculate metrics
            total_employees = employee_counts["total_employees"] or 0
            active_employees = employee_counts["active_employees"] or 0
            total_interactions = usage_totals["total_interactions"] or 0
            total_cost = float(usage_totals["total_cost"] or 0)
            
            avg_cost_per_employee = total_cost / max(active_employees, 1)
            
            # Calculate monthly growth
            current_interactions = total_interactions
            previous_interactions = previous_month_interactions or 0
            monthly_growth = 0.0
            
            if previous_interactions > 0:
                monthly_growth = ((current_interactions - previous_interactions) / previous_interactions) * 100
            
            # Format top employees
            top_performing_employees = [
                {
                    "employee_id": row["id"],
                    "employee_name": row["name"],
                    "interaction_count": row["interaction_count"],
                    "avg_response_time": float(row["avg_response_time"] or 0),
                    "total_cost": float(row["total_cost"] or 0)
                }
                for row in top_employees
            ]
            
            # Format usage by time
            usage_by_time = {
                str(int(row["hour"])): row["interaction_count"]
                for row in hourly_usage
            }
            
            # Create metrics object
            metrics = ClientUsageMetrics(
                client_id=client_id,
                client_name=client_info["name"] if client_info else "Unknown Client",
                total_employees=total_employees,
                active_employees=active_employees,
                total_interactions=total_interactions,
                total_cost=total_cost,
                avg_cost_per_employee=avg_cost_per_employee,
                top_performing_employees=top_performing_employees,
                usage_by_time=usage_by_time,
                monthly_growth=monthly_growth
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get client usage metrics: {e}")
            raise
    
    async def get_employee_health_status(
        self,
        client_id: str,
        employee_id: str
    ) -> Dict[str, Any]:
        """Get health status for an employee with alerts"""
        try:
            # Get recent metrics (last 24 hours)
            metrics = await self.get_employee_metrics(client_id, employee_id, days=1)
            
            # Calculate health scores
            health_scores = {}
            alerts = []
            
            # Response time health
            response_time_score = max(0, 100 - (metrics.avg_response_time_ms / self.health_thresholds["response_time_ms"]) * 100)
            health_scores["response_time"] = min(100, response_time_score)
            
            if metrics.avg_response_time_ms > self.health_thresholds["response_time_ms"]:
                alerts.append({
                    "type": "performance",
                    "severity": "warning",
                    "message": f"Average response time ({metrics.avg_response_time_ms:.0f}ms) exceeds threshold",
                    "threshold": self.health_thresholds["response_time_ms"]
                })
            
            # Success rate health
            health_scores["success_rate"] = metrics.success_rate
            
            if metrics.success_rate < self.health_thresholds["success_rate"] * 100:
                alerts.append({
                    "type": "reliability",
                    "severity": "critical" if metrics.success_rate < 90 else "warning",
                    "message": f"Success rate ({metrics.success_rate:.1f}%) below threshold",
                    "threshold": self.health_thresholds["success_rate"] * 100
                })
            
            # User satisfaction health
            satisfaction_score = (metrics.user_satisfaction_score / 5.0) * 100
            health_scores["user_satisfaction"] = satisfaction_score
            
            if metrics.user_satisfaction_score < self.health_thresholds["user_satisfaction"]:
                alerts.append({
                    "type": "satisfaction",
                    "severity": "warning",
                    "message": f"User satisfaction ({metrics.user_satisfaction_score:.1f}/5.0) below threshold",
                    "threshold": self.health_thresholds["user_satisfaction"]
                })
            
            # Cost efficiency health
            cost_score = max(0, 100 - (metrics.cost_per_interaction / self.health_thresholds["cost_per_interaction"]) * 100)
            health_scores["cost_efficiency"] = min(100, cost_score)
            
            if metrics.cost_per_interaction > self.health_thresholds["cost_per_interaction"]:
                alerts.append({
                    "type": "cost",
                    "severity": "warning",
                    "message": f"Cost per interaction (${metrics.cost_per_interaction:.3f}) exceeds threshold",
                    "threshold": self.health_thresholds["cost_per_interaction"]
                })
            
            # Overall health score
            overall_health = statistics.mean(health_scores.values())
            
            # Determine health status
            if overall_health >= 90:
                health_status = "excellent"
            elif overall_health >= 75:
                health_status = "good"
            elif overall_health >= 60:
                health_status = "fair"
            elif overall_health >= 40:
                health_status = "poor"
            else:
                health_status = "critical"
            
            return {
                "employee_id": employee_id,
                "overall_health_score": round(overall_health, 1),
                "health_status": health_status,
                "component_scores": {k: round(v, 1) for k, v in health_scores.items()},
                "alerts": alerts,
                "metrics_summary": {
                    "total_interactions": metrics.total_interactions,
                    "avg_response_time_ms": metrics.avg_response_time_ms,
                    "success_rate": metrics.success_rate,
                    "user_satisfaction": metrics.user_satisfaction_score,
                    "cost_per_interaction": metrics.cost_per_interaction
                },
                "performance_trend": metrics.performance_trend,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get employee health status: {e}")
            raise
    
    async def calculate_interaction_cost(
        self,
        model_used: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost for an interaction"""
        try:
            if model_used not in self.cost_models:
                # Default cost if model not found
                return 0.001
            
            model_costs = self.cost_models[model_used]
            
            # Calculate cost per 1K tokens
            input_cost = (input_tokens / 1000) * model_costs["input"]
            output_cost = (output_tokens / 1000) * model_costs["output"]
            
            total_cost = input_cost + output_cost
            
            return round(total_cost, 6)
            
        except Exception as e:
            logger.error(f"Failed to calculate interaction cost: {e}")
            return 0.001
    
    async def record_user_feedback(
        self,
        client_id: str,
        employee_id: str,
        interaction_id: str,
        rating: int,
        feedback_text: Optional[str] = None
    ) -> bool:
        """Record user feedback for an interaction"""
        try:
            async with self.db_manager.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO employee_feedback (
                        client_id, employee_id, interaction_id, rating, 
                        feedback_text, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (client_id, employee_id, interaction_id)
                    DO UPDATE SET 
                        rating = $4,
                        feedback_text = $5,
                        updated_at = $6
                """, client_id, employee_id, interaction_id, rating, 
                    feedback_text, datetime.now(timezone.utc))
            
            logger.info(f"Recorded feedback for interaction {interaction_id}: {rating}/5")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record user feedback: {e}")
            return False
    
    async def get_analytics_dashboard_data(
        self,
        client_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive dashboard data for a client"""
        try:
            # Get client usage metrics
            client_metrics = await self.get_client_usage_metrics(client_id, days)
            
            # Get health status for all active employees
            employees = await self.employee_manager.list_employees(
                client_id, status=EmployeeStatus.ACTIVE
            )
            
            employee_health = []
            for employee in employees[:10]:  # Limit to top 10 for performance
                try:
                    health = await self.get_employee_health_status(client_id, employee.id)
                    employee_health.append(health)
                except Exception as e:
                    logger.warning(f"Failed to get health for employee {employee.id}: {e}")
            
            # Calculate summary statistics
            total_alerts = sum(len(emp["alerts"]) for emp in employee_health)
            avg_health_score = statistics.mean([emp["overall_health_score"] for emp in employee_health]) if employee_health else 0
            
            # Get cost breakdown by employee
            cost_breakdown = [
                {
                    "employee_name": emp["employee_name"],
                    "total_cost": emp["total_cost"],
                    "interaction_count": emp["interaction_count"]
                }
                for emp in client_metrics.top_performing_employees
            ]
            
            return {
                "client_metrics": {
                    "client_id": client_metrics.client_id,
                    "client_name": client_metrics.client_name,
                    "total_employees": client_metrics.total_employees,
                    "active_employees": client_metrics.active_employees,
                    "total_interactions": client_metrics.total_interactions,
                    "total_cost": client_metrics.total_cost,
                    "avg_cost_per_employee": client_metrics.avg_cost_per_employee,
                    "monthly_growth": client_metrics.monthly_growth
                },
                "employee_health_summary": {
                    "total_employees_monitored": len(employee_health),
                    "avg_health_score": round(avg_health_score, 1),
                    "total_alerts": total_alerts,
                    "employees_with_issues": len([emp for emp in employee_health if emp["alerts"]])
                },
                "top_performing_employees": client_metrics.top_performing_employees,
                "usage_patterns": {
                    "hourly_distribution": client_metrics.usage_by_time,
                    "peak_usage_hours": self._get_peak_hours(client_metrics.usage_by_time)
                },
                "cost_analysis": {
                    "total_cost": client_metrics.total_cost,
                    "cost_breakdown": cost_breakdown,
                    "cost_trend": "stable"  # Could be calculated from historical data
                },
                "alerts_summary": [
                    alert for emp in employee_health for alert in emp["alerts"]
                ][:10],  # Top 10 alerts
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            raise
    
    def _get_peak_hours(self, hourly_usage: Dict[str, int]) -> List[int]:
        """Get peak usage hours from hourly distribution"""
        if not hourly_usage:
            return []
        
        # Sort hours by usage count
        sorted_hours = sorted(hourly_usage.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 3 peak hours
        return [int(hour) for hour, _ in sorted_hours[:3]]
    
    async def _create_analytics_tables(self):
        """Create analytics-specific database tables"""
        try:
            async with self.db_manager.get_connection() as conn:
                # Employee feedback table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS employee_feedback (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        client_id VARCHAR(255) NOT NULL,
                        employee_id VARCHAR(255) NOT NULL,
                        interaction_id VARCHAR(255) NOT NULL,
                        rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                        feedback_text TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        UNIQUE(client_id, employee_id, interaction_id)
                    )
                """)
                
                # Analytics cache table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS analytics_cache (
                        cache_key VARCHAR(255) PRIMARY KEY,
                        cache_data JSONB NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        expires_at TIMESTAMP WITH TIME ZONE NOT NULL
                    )
                """)
                
                # Create indexes
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_employee_feedback_employee 
                    ON employee_feedback(client_id, employee_id);
                    
                    CREATE INDEX IF NOT EXISTS idx_employee_feedback_rating 
                    ON employee_feedback(rating);
                    
                    CREATE INDEX IF NOT EXISTS idx_analytics_cache_expires 
                    ON analytics_cache(expires_at);
                """)
                
        except Exception as e:
            logger.error(f"Failed to create analytics tables: {e}")


# Global analytics service instance
_analytics_service = None


async def get_analytics_service() -> EmployeeAnalyticsService:
    """Get global analytics service instance"""
    global _analytics_service
    
    if _analytics_service is None:
        _analytics_service = EmployeeAnalyticsService()
        await _analytics_service.initialize()
    
    return _analytics_service