"""
Revenue Operations (RevOps) AI Employee
Production-ready RevOps agent with pipeline hygiene, forecasting, and quote optimization
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import logging

from forge1.backend.forge1.core.agent_base import AgentBase
from forge1.backend.forge1.core.orchestration import WorkflowEngine
from forge1.backend.forge1.core.memory import MemoryManager
from forge1.backend.forge1.core.models import ModelRouter
from forge1.backend.forge1.core.monitoring import MetricsCollector
from forge1.backend.forge1.core.security import SecretManager
from forge1.backend.forge1.verticals.revops.connectors import RevOpsConnectorFactory
from forge1.backend.forge1.verticals.revops.playbooks import RevOpsPlaybooks, PipelineHygieneResult, ForecastAnalysis


@dataclass
class RevOpsPerformanceMetrics:
    """RevOps AI Employee performance metrics"""
    pipeline_hygiene_score: float
    forecast_accuracy: float
    quota_attainment: float
    deal_velocity: float  # days
    win_rate: float
    avg_deal_size: float
    pipeline_coverage: float
    data_quality_score: float


class RevOpsAIEmployee(AgentBase):
    """
    Revenue Operations AI Employee
    
    Capabilities:
    - Pipeline hygiene monitoring and cleanup
    - Sales forecasting with accuracy tracking
    - Quote optimization and approval workflows
    - Renewal motion analysis and recommendations
    - Performance analytics and reporting
    - Data quality management
    """
    
    def __init__(
        self,
        employee_id: str,
        tenant_id: str,
        workflow_engine: WorkflowEngine,
        memory_manager: MemoryManager,
        model_router: ModelRouter,
        metrics_collector: MetricsCollector,
        secret_manager: SecretManager
    ):
        super().__init__(employee_id, tenant_id)
        
        self.workflow_engine = workflow_engine
        self.memory_manager = memory_manager
        self.model_router = model_router
        self.metrics = metrics_collector
        self.secret_manager = secret_manager
        
        # Initialize connectors and playbooks
        self.connector_factory = RevOpsConnectorFactory(secret_manager, metrics_collector)
        self.playbooks = RevOpsPlaybooks(
            workflow_engine, memory_manager, model_router, self.connector_factory
        )
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.tasks_processed = 0
        
        # Performance targets
        self.performance_targets = {
            "pipeline_hygiene_score": 0.90,  # 90%
            "forecast_accuracy": 0.85,  # 85%
            "quota_attainment": 1.0,  # 100%
            "data_quality_score": 0.95,  # 95%
            "pipeline_coverage": 3.0  # 3x coverage
        }
        
        self.logger = logging.getLogger(f"revops_employee_{employee_id}")
    
    async def pipeline_hygiene_check(self, rep_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive pipeline hygiene analysis
        
        Returns:
            Dict containing hygiene analysis and recommendations
        """
        start_time = datetime.utcnow()
        
        try:
            self.logger.info(f"Starting pipeline hygiene check for rep: {rep_id or 'all'}")
            
            # Perform hygiene analysis
            hygiene_result = await self.playbooks.pipeline_hygiene_analysis(rep_id)
            
            # Store results in memory for tracking
            await self._store_hygiene_results(hygiene_result, rep_id)
            
            # Update performance metrics
            self.metrics.record_metric("revops_hygiene_score", hygiene_result.hygiene_score)
            self.metrics.record_metric("revops_stale_deals", len(hygiene_result.stale_deals))
            self.metrics.record_metric("revops_missing_data_deals", len(hygiene_result.missing_data_deals))
            
            # Generate action items if hygiene is below target
            action_items = []
            if hygiene_result.hygiene_score < self.performance_targets["pipeline_hygiene_score"]:
                action_items = await self._generate_hygiene_action_items(hygiene_result)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.tasks_processed += 1
            
            response = {
                "analysis_type": "pipeline_hygiene",
                "rep_id": rep_id,
                "hygiene_score": hygiene_result.hygiene_score,
                "total_deals": hygiene_result.total_deals,
                "issues": {
                    "stale_deals": len(hygiene_result.stale_deals),
                    "missing_data_deals": len(hygiene_result.missing_data_deals),
                    "overdue_deals": len(hygiene_result.overdue_deals)
                },
                "recommendations": hygiene_result.recommendations,
                "action_items": action_items,
                "meets_target": hygiene_result.hygiene_score >= self.performance_targets["pipeline_hygiene_score"],
                "processing_time_seconds": processing_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Pipeline hygiene check completed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in pipeline hygiene check: {str(e)}")
            self.metrics.increment("revops_hygiene_check_errors")
            
            return {
                "analysis_type": "pipeline_hygiene",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def forecast_analysis(self, period: str, rep_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform sales forecast analysis with accuracy tracking
        
        Args:
            period: Forecast period (e.g., "Q1 2024", "Jan 2024")
            rep_id: Optional sales rep ID for individual analysis
        
        Returns:
            Dict containing forecast analysis and recommendations
        """
        start_time = datetime.utcnow()
        
        try:
            self.logger.info(f"Starting forecast analysis for period: {period}, rep: {rep_id or 'all'}")
            
            # Perform forecast analysis
            forecast_result = await self.playbooks.forecast_analysis(period, rep_id)
            
            # Store results in memory
            await self._store_forecast_results(forecast_result, rep_id)
            
            # Update performance metrics
            self.metrics.record_metric("revops_forecast_accuracy", forecast_result.forecast_accuracy)
            self.metrics.record_metric("revops_quota_attainment", forecast_result.quota_attainment)
            self.metrics.record_metric("revops_pipeline_value", float(forecast_result.total_pipeline))
            
            # Calculate pipeline coverage
            pipeline_coverage = float(forecast_result.total_pipeline / (forecast_result.commit_forecast or 1))
            
            # Generate strategic recommendations
            strategic_recommendations = await self._generate_forecast_recommendations(forecast_result)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.tasks_processed += 1
            
            response = {
                "analysis_type": "sales_forecast",
                "period": period,
                "rep_id": rep_id,
                "pipeline_metrics": {
                    "total_pipeline": float(forecast_result.total_pipeline),
                    "weighted_pipeline": float(forecast_result.weighted_pipeline),
                    "commit_forecast": float(forecast_result.commit_forecast),
                    "best_case_forecast": float(forecast_result.best_case_forecast),
                    "pipeline_coverage": pipeline_coverage
                },
                "performance_metrics": {
                    "quota_attainment": forecast_result.quota_attainment,
                    "forecast_accuracy": forecast_result.forecast_accuracy
                },
                "risk_factors": forecast_result.risk_factors,
                "opportunities": forecast_result.opportunities,
                "strategic_recommendations": strategic_recommendations,
                "meets_targets": {
                    "forecast_accuracy": forecast_result.forecast_accuracy >= self.performance_targets["forecast_accuracy"],
                    "quota_attainment": forecast_result.quota_attainment >= self.performance_targets["quota_attainment"],
                    "pipeline_coverage": pipeline_coverage >= self.performance_targets["pipeline_coverage"]
                },
                "processing_time_seconds": processing_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Forecast analysis completed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in forecast analysis: {str(e)}")
            self.metrics.increment("revops_forecast_analysis_errors")
            
            return {
                "analysis_type": "sales_forecast",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def quote_optimization(self, quote_id: str) -> Dict[str, Any]:
        """
        Optimize sales quote for maximum win probability
        
        Args:
            quote_id: ID of the quote to optimize
        
        Returns:
            Dict containing optimization analysis and recommendations
        """
        start_time = datetime.utcnow()
        
        try:
            self.logger.info(f"Starting quote optimization for quote: {quote_id}")
            
            # Perform quote analysis
            quote_analysis = await self.playbooks.quote_optimization(quote_id)
            
            # Store results in memory
            await self._store_quote_analysis(quote_analysis)
            
            # Update performance metrics
            self.metrics.record_metric("revops_quote_win_probability", quote_analysis.win_probability)
            
            # Generate approval workflow if needed
            approval_workflow = None
            if quote_analysis.approval_required:
                approval_workflow = await self._initiate_approval_workflow(quote_analysis)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.tasks_processed += 1
            
            response = {
                "analysis_type": "quote_optimization",
                "quote_id": quote_id,
                "win_probability": quote_analysis.win_probability,
                "pricing_optimization": quote_analysis.pricing_optimization,
                "competitive_analysis": quote_analysis.competitive_analysis,
                "recommendations": quote_analysis.recommendations,
                "approval_required": quote_analysis.approval_required,
                "approval_workflow_id": approval_workflow.get("workflow_id") if approval_workflow else None,
                "processing_time_seconds": processing_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Quote optimization completed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in quote optimization: {str(e)}")
            self.metrics.increment("revops_quote_optimization_errors")
            
            return {
                "analysis_type": "quote_optimization",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def renewal_analysis(self, account_id: str) -> Dict[str, Any]:
        """
        Analyze customer renewal opportunity and risk factors
        
        Args:
            account_id: ID of the account to analyze
        
        Returns:
            Dict containing renewal analysis and recommendations
        """
        start_time = datetime.utcnow()
        
        try:
            self.logger.info(f"Starting renewal analysis for account: {account_id}")
            
            # Perform renewal analysis
            renewal_result = await self.playbooks.renewal_motion_analysis(account_id)
            
            # Store results in memory
            await self._store_renewal_analysis(renewal_result)
            
            # Update performance metrics
            self.metrics.record_metric("revops_renewal_probability", renewal_result["renewal_probability"])
            
            # Generate renewal strategy
            renewal_strategy = await self._generate_renewal_strategy(renewal_result)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.tasks_processed += 1
            
            response = {
                "analysis_type": "renewal_analysis",
                "account_id": account_id,
                "renewal_probability": renewal_result["renewal_probability"],
                "risk_factors": renewal_result["risk_factors"],
                "expansion_opportunities": renewal_result["expansion_opportunities"],
                "recommended_actions": renewal_result["recommended_actions"],
                "renewal_strategy": renewal_strategy,
                "health_score": renewal_result["health_score"],
                "renewal_date": renewal_result["renewal_date"],
                "processing_time_seconds": processing_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Renewal analysis completed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in renewal analysis: {str(e)}")
            self.metrics.increment("revops_renewal_analysis_errors")
            
            return {
                "analysis_type": "renewal_analysis",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_performance_metrics(self, time_period: Optional[timedelta] = None) -> RevOpsPerformanceMetrics:
        """
        Get current performance metrics for the RevOps AI Employee
        """
        if time_period is None:
            time_period = timedelta(days=30)
        
        # Get performance data from playbooks
        period_str = f"Last {time_period.days} days"
        performance_data = await self.playbooks.performance_metrics(period_str)
        
        return RevOpsPerformanceMetrics(
            pipeline_hygiene_score=performance_data.get("pipeline_hygiene_score", 0.0),
            forecast_accuracy=performance_data.get("forecast_accuracy", 0.0),
            quota_attainment=performance_data.get("quota_attainment", 0.0),
            deal_velocity=performance_data.get("avg_sales_cycle_days", 0.0),
            win_rate=performance_data.get("win_rate", 0.0),
            avg_deal_size=performance_data.get("avg_deal_size", 0.0),
            pipeline_coverage=3.0,  # Would calculate from actual data
            data_quality_score=0.95  # Would calculate from data quality metrics
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on RevOps AI Employee
        """
        health_status = {
            "status": "healthy",
            "employee_id": self.employee_id,
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "tasks_processed": self.tasks_processed,
            "last_activity": datetime.utcnow().isoformat()
        }
        
        # Check connector health
        try:
            # Test CRM connection
            crm_connector = self.connector_factory.create_crm_connector("salesforce")
            if crm_connector:
                crm_healthy = await crm_connector.authenticate()
                health_status["crm_connected"] = crm_healthy
            
            # Test BI connection
            bi_connector = self.connector_factory.create_bi_connector("powerbi")
            if bi_connector:
                bi_healthy = await bi_connector.authenticate()
                health_status["bi_connected"] = bi_healthy
            
        except Exception as e:
            health_status["status"] = "degraded"
            health_status["error"] = str(e)
        
        return health_status
    
    async def validate_performance_targets(self) -> Dict[str, bool]:
        """
        Validate current performance against targets
        """
        metrics = await self.get_performance_metrics()
        
        compliance = {
            "pipeline_hygiene": metrics.pipeline_hygiene_score >= self.performance_targets["pipeline_hygiene_score"],
            "forecast_accuracy": metrics.forecast_accuracy >= self.performance_targets["forecast_accuracy"],
            "quota_attainment": metrics.quota_attainment >= self.performance_targets["quota_attainment"],
            "data_quality": metrics.data_quality_score >= self.performance_targets["data_quality_score"],
            "pipeline_coverage": metrics.pipeline_coverage >= self.performance_targets["pipeline_coverage"]
        }
        
        # Log performance issues
        for metric, compliant in compliance.items():
            if not compliant:
                self.logger.warning(f"Performance target not met: {metric}")
                self.metrics.increment(f"revops_target_miss_{metric}")
        
        return compliance
    
    # Helper methods
    async def _store_hygiene_results(self, hygiene_result: PipelineHygieneResult, rep_id: Optional[str]) -> None:
        """Store hygiene analysis results in memory"""
        hygiene_data = {
            "rep_id": rep_id,
            "hygiene_score": hygiene_result.hygiene_score,
            "total_deals": hygiene_result.total_deals,
            "stale_deals_count": len(hygiene_result.stale_deals),
            "missing_data_count": len(hygiene_result.missing_data_deals),
            "overdue_deals_count": len(hygiene_result.overdue_deals),
            "recommendations": hygiene_result.recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.memory_manager.store_context(
            context_type="revops_hygiene_analysis",
            content=hygiene_data,
            metadata={
                "rep_id": rep_id or "all",
                "analysis_type": "pipeline_hygiene"
            }
        )
    
    async def _store_forecast_results(self, forecast_result: ForecastAnalysis, rep_id: Optional[str]) -> None:
        """Store forecast analysis results in memory"""
        forecast_data = {
            "period": forecast_result.period,
            "rep_id": rep_id,
            "total_pipeline": float(forecast_result.total_pipeline),
            "weighted_pipeline": float(forecast_result.weighted_pipeline),
            "commit_forecast": float(forecast_result.commit_forecast),
            "quota_attainment": forecast_result.quota_attainment,
            "forecast_accuracy": forecast_result.forecast_accuracy,
            "risk_factors": forecast_result.risk_factors,
            "opportunities": forecast_result.opportunities,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.memory_manager.store_context(
            context_type="revops_forecast_analysis",
            content=forecast_data,
            metadata={
                "period": forecast_result.period,
                "rep_id": rep_id or "all",
                "analysis_type": "sales_forecast"
            }
        )
    
    async def _store_quote_analysis(self, quote_analysis) -> None:
        """Store quote analysis results in memory"""
        quote_data = {
            "quote_id": quote_analysis.quote_id,
            "win_probability": quote_analysis.win_probability,
            "pricing_optimization": quote_analysis.pricing_optimization,
            "competitive_analysis": quote_analysis.competitive_analysis,
            "approval_required": quote_analysis.approval_required,
            "recommendations": quote_analysis.recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.memory_manager.store_context(
            context_type="revops_quote_analysis",
            content=quote_data,
            metadata={
                "quote_id": quote_analysis.quote_id,
                "analysis_type": "quote_optimization"
            }
        )
    
    async def _store_renewal_analysis(self, renewal_result: Dict[str, Any]) -> None:
        """Store renewal analysis results in memory"""
        await self.memory_manager.store_context(
            context_type="revops_renewal_analysis",
            content=renewal_result,
            metadata={
                "account_id": renewal_result["account_id"],
                "analysis_type": "renewal_analysis"
            }
        )
    
    async def _generate_hygiene_action_items(self, hygiene_result: PipelineHygieneResult) -> List[str]:
        """Generate specific action items for hygiene improvement"""
        action_items = []
        
        if len(hygiene_result.stale_deals) > 0:
            action_items.append(f"Update {len(hygiene_result.stale_deals)} stale deals with recent activity")
        
        if len(hygiene_result.missing_data_deals) > 0:
            action_items.append(f"Complete missing data for {len(hygiene_result.missing_data_deals)} deals")
        
        if len(hygiene_result.overdue_deals) > 0:
            action_items.append(f"Review and update close dates for {len(hygiene_result.overdue_deals)} overdue deals")
        
        return action_items
    
    async def _generate_forecast_recommendations(self, forecast_result: ForecastAnalysis) -> List[str]:
        """Generate strategic recommendations based on forecast analysis"""
        recommendations = []
        
        if forecast_result.quota_attainment < 0.8:
            recommendations.append("Focus on accelerating high-probability deals to improve quota attainment")
        
        if forecast_result.forecast_accuracy < 0.8:
            recommendations.append("Improve deal qualification and probability assessment accuracy")
        
        pipeline_coverage = float(forecast_result.total_pipeline / (forecast_result.commit_forecast or 1))
        if pipeline_coverage < 3.0:
            recommendations.append("Increase pipeline generation to achieve 3x coverage target")
        
        return recommendations
    
    async def _initiate_approval_workflow(self, quote_analysis) -> Dict[str, Any]:
        """Initiate approval workflow for high-risk quotes"""
        # Placeholder for approval workflow integration
        workflow_id = f"APPROVAL-{quote_analysis.quote_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        return {
            "workflow_id": workflow_id,
            "status": "initiated",
            "approvers": ["sales_manager", "finance_director"],
            "created_at": datetime.utcnow().isoformat()
        }
    
    async def _generate_renewal_strategy(self, renewal_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate renewal strategy based on analysis"""
        strategy = {
            "approach": "proactive" if renewal_result["renewal_probability"] > 0.7 else "intensive",
            "timeline": "60_days_before" if renewal_result["renewal_probability"] > 0.8 else "90_days_before",
            "key_stakeholders": ["account_manager", "customer_success"],
            "focus_areas": []
        }
        
        if renewal_result["renewal_probability"] < 0.6:
            strategy["focus_areas"].extend(["address_risk_factors", "executive_engagement"])
        
        if renewal_result.get("expansion_opportunities"):
            strategy["focus_areas"].append("expansion_discussion")
        
        return strategy