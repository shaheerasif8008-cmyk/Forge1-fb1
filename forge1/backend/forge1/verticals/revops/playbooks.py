"""
Revenue Operations (RevOps) Playbooks
Standard Operating Procedures for RevOps AI Employee
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, date, timedelta
from decimal import Decimal
import asyncio

from forge1.backend.forge1.core.orchestration import WorkflowEngine
from forge1.backend.forge1.core.memory import MemoryManager
from forge1.backend.forge1.core.models import ModelRouter
from forge1.backend.forge1.verticals.revops.connectors import (
    Deal, Account, Quote, ForecastData, RevOpsConnectorFactory
)


class DealStage(Enum):
    PROSPECTING = "prospecting"
    QUALIFICATION = "qualification"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"


class ForecastCategory(Enum):
    PIPELINE = "pipeline"
    BEST_CASE = "best_case"
    COMMIT = "commit"
    CLOSED = "closed"


@dataclass
class PipelineHygieneResult:
    """Result of pipeline hygiene analysis"""
    total_deals: int
    stale_deals: List[str]  # Deal IDs
    missing_data_deals: List[str]
    overdue_deals: List[str]
    hygiene_score: float  # 0.0 to 1.0
    recommendations: List[str]


@dataclass
class ForecastAnalysis:
    """Sales forecast analysis result"""
    period: str
    total_pipeline: Decimal
    weighted_pipeline: Decimal
    commit_forecast: Decimal
    best_case_forecast: Decimal
    quota_attainment: float
    forecast_accuracy: float
    risk_factors: List[str]
    opportunities: List[str]


@dataclass
class QuoteAnalysis:
    """Quote analysis and optimization result"""
    quote_id: str
    win_probability: float
    pricing_optimization: Dict[str, Any]
    competitive_analysis: Dict[str, Any]
    approval_required: bool
    recommendations: List[str]


class RevOpsPlaybooks:
    """Revenue Operations playbooks and workflows"""
    
    def __init__(
        self,
        workflow_engine: WorkflowEngine,
        memory_manager: MemoryManager,
        model_router: ModelRouter,
        connector_factory: RevOpsConnectorFactory
    ):
        self.workflow_engine = workflow_engine
        self.memory_manager = memory_manager
        self.model_router = model_router
        self.connector_factory = connector_factory
        
        # Performance targets
        self.hygiene_targets = {
            "stale_deal_threshold": 30,  # days without activity
            "missing_data_threshold": 0.05,  # 5% max missing data
            "overdue_threshold": 0.10,  # 10% max overdue deals
            "overall_hygiene_target": 0.90  # 90% hygiene score
        }
        
        self.forecast_targets = {
            "accuracy_threshold": 0.85,  # 85% forecast accuracy
            "quota_attainment_target": 1.0,  # 100% quota attainment
            "pipeline_coverage": 3.0  # 3x pipeline coverage
        }
    
    async def pipeline_hygiene_analysis(self, rep_id: Optional[str] = None) -> PipelineHygieneResult:
        """
        Analyze pipeline hygiene and data quality
        """
        # Get deals from CRM
        crm_connector = self.connector_factory.create_crm_connector("salesforce")
        if not crm_connector:
            raise Exception("CRM connector not available")
        
        filters = {"owner_id": rep_id} if rep_id else {}
        deals = await crm_connector.get_deals(filters)
        
        # Analyze hygiene issues
        stale_deals = []
        missing_data_deals = []
        overdue_deals = []
        
        current_date = datetime.now().date()
        
        for deal in deals:
            # Check for stale deals (no activity in 30+ days)
            if deal.last_activity:
                days_since_activity = (datetime.now() - deal.last_activity).days
                if days_since_activity > self.hygiene_targets["stale_deal_threshold"]:
                    stale_deals.append(deal.id)
            else:
                stale_deals.append(deal.id)  # No activity recorded
            
            # Check for missing critical data
            missing_fields = []
            if not deal.amount or deal.amount <= 0:
                missing_fields.append("amount")
            if not deal.close_date:
                missing_fields.append("close_date")
            if deal.probability < 0 or deal.probability > 1:
                missing_fields.append("probability")
            
            if missing_fields:
                missing_data_deals.append(deal.id)
            
            # Check for overdue deals
            if deal.close_date < current_date and deal.stage not in ["closed_won", "closed_lost"]:
                overdue_deals.append(deal.id)
        
        # Calculate hygiene score
        total_deals = len(deals)
        if total_deals == 0:
            hygiene_score = 1.0
        else:
            stale_penalty = len(stale_deals) / total_deals * 0.4
            missing_penalty = len(missing_data_deals) / total_deals * 0.3
            overdue_penalty = len(overdue_deals) / total_deals * 0.3
            hygiene_score = max(0.0, 1.0 - stale_penalty - missing_penalty - overdue_penalty)
        
        # Generate recommendations using AI
        recommendations = await self._generate_hygiene_recommendations(
            deals, stale_deals, missing_data_deals, overdue_deals
        )
        
        return PipelineHygieneResult(
            total_deals=total_deals,
            stale_deals=stale_deals,
            missing_data_deals=missing_data_deals,
            overdue_deals=overdue_deals,
            hygiene_score=hygiene_score,
            recommendations=recommendations
        )
    
    async def forecast_analysis(self, period: str, rep_id: Optional[str] = None) -> ForecastAnalysis:
        """
        Analyze sales forecast for accuracy and risk factors
        """
        # Get deals for the forecast period
        crm_connector = self.connector_factory.create_crm_connector("salesforce")
        if not crm_connector:
            raise Exception("CRM connector not available")
        
        # Parse period (e.g., "Q1 2024", "Jan 2024")
        start_date, end_date = self._parse_forecast_period(period)
        
        filters = {
            "close_date_after": start_date.isoformat(),
            "close_date_before": end_date.isoformat()
        }
        if rep_id:
            filters["owner_id"] = rep_id
        
        deals = await crm_connector.get_deals(filters)
        
        # Calculate forecast metrics
        total_pipeline = sum(deal.amount for deal in deals if deal.stage not in ["closed_won", "closed_lost"])
        weighted_pipeline = sum(deal.amount * deal.probability for deal in deals if deal.stage not in ["closed_won", "closed_lost"])
        
        # Categorize deals by forecast category
        commit_deals = [d for d in deals if d.forecast_category == "Commit"]
        best_case_deals = [d for d in deals if d.forecast_category in ["Commit", "Best Case"]]
        closed_won_deals = [d for d in deals if d.stage == "closed_won"]
        
        commit_forecast = sum(deal.amount for deal in commit_deals)
        best_case_forecast = sum(deal.amount for deal in best_case_deals)
        closed_won_amount = sum(deal.amount for deal in closed_won_deals)
        
        # Get quota information (would typically come from CRM or separate system)
        quota = await self._get_quota_for_period(period, rep_id)
        quota_attainment = float(closed_won_amount / quota) if quota > 0 else 0.0
        
        # Calculate forecast accuracy (comparing to historical data)
        forecast_accuracy = await self._calculate_forecast_accuracy(period, rep_id)
        
        # Analyze risk factors and opportunities using AI
        risk_factors, opportunities = await self._analyze_forecast_risks_opportunities(
            deals, period, quota_attainment
        )
        
        return ForecastAnalysis(
            period=period,
            total_pipeline=total_pipeline,
            weighted_pipeline=weighted_pipeline,
            commit_forecast=commit_forecast,
            best_case_forecast=best_case_forecast,
            quota_attainment=quota_attainment,
            forecast_accuracy=forecast_accuracy,
            risk_factors=risk_factors,
            opportunities=opportunities
        )
    
    async def quote_optimization(self, quote_id: str) -> QuoteAnalysis:
        """
        Analyze and optimize sales quotes for win probability
        """
        # Get quote data (would integrate with CPQ system)
        quote_data = await self._get_quote_data(quote_id)
        if not quote_data:
            raise Exception(f"Quote {quote_id} not found")
        
        # Get related deal and account information
        crm_connector = self.connector_factory.create_crm_connector("salesforce")
        deals = await crm_connector.get_deals({"id": quote_data["deal_id"]})
        accounts = await crm_connector.get_accounts({"id": quote_data["account_id"]})
        
        if not deals or not accounts:
            raise Exception("Related deal or account not found")
        
        deal = deals[0]
        account = accounts[0]
        
        # Analyze win probability using AI
        win_probability = await self._calculate_win_probability(quote_data, deal, account)
        
        # Pricing optimization analysis
        pricing_optimization = await self._analyze_pricing_optimization(quote_data, deal, account)
        
        # Competitive analysis
        competitive_analysis = await self._perform_competitive_analysis(quote_data, deal, account)
        
        # Determine if approval is required
        approval_required = self._requires_approval(quote_data, pricing_optimization)
        
        # Generate recommendations
        recommendations = await self._generate_quote_recommendations(
            quote_data, deal, account, win_probability, pricing_optimization
        )
        
        return QuoteAnalysis(
            quote_id=quote_id,
            win_probability=win_probability,
            pricing_optimization=pricing_optimization,
            competitive_analysis=competitive_analysis,
            approval_required=approval_required,
            recommendations=recommendations
        )
    
    async def renewal_motion_analysis(self, account_id: str) -> Dict[str, Any]:
        """
        Analyze customer renewal opportunities and risks
        """
        # Get account information
        crm_connector = self.connector_factory.create_crm_connector("salesforce")
        accounts = await crm_connector.get_accounts({"id": account_id})
        
        if not accounts:
            raise Exception(f"Account {account_id} not found")
        
        account = accounts[0]
        
        # Get renewal history and usage data
        renewal_history = await self._get_renewal_history(account_id)
        usage_data = await self._get_usage_data(account_id)
        support_tickets = await self._get_support_history(account_id)
        
        # Analyze renewal probability using AI
        renewal_analysis_prompt = f"""
        Analyze customer renewal probability and strategy:
        
        Account: {account.name}
        Tier: {account.size}
        Health Score: {account.health_score}
        Renewal Date: {account.renewal_date}
        
        Renewal History: {renewal_history}
        Usage Trends: {usage_data}
        Support Issues: {support_tickets}
        
        Provide:
        1. Renewal probability (0.0-1.0)
        2. Key risk factors
        3. Expansion opportunities
        4. Recommended actions
        5. Optimal timing for outreach
        """
        
        analysis = await self.model_router.generate_response(
            prompt=renewal_analysis_prompt,
            model_preference="reasoning",
            max_tokens=600
        )
        
        # Parse AI analysis
        renewal_probability = self._extract_renewal_probability(analysis)
        risk_factors = self._extract_risk_factors(analysis)
        expansion_opportunities = self._extract_expansion_opportunities(analysis)
        recommended_actions = self._extract_recommended_actions(analysis)
        
        return {
            "account_id": account_id,
            "renewal_probability": renewal_probability,
            "risk_factors": risk_factors,
            "expansion_opportunities": expansion_opportunities,
            "recommended_actions": recommended_actions,
            "health_score": account.health_score,
            "renewal_date": account.renewal_date,
            "analysis_date": datetime.utcnow().isoformat()
        }
    
    async def performance_metrics(self, period: str, rep_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate RevOps performance metrics
        """
        # Get deals and accounts data
        crm_connector = self.connector_factory.create_crm_connector("salesforce")
        
        start_date, end_date = self._parse_forecast_period(period)
        filters = {
            "close_date_after": start_date.isoformat(),
            "close_date_before": end_date.isoformat()
        }
        if rep_id:
            filters["owner_id"] = rep_id
        
        deals = await crm_connector.get_deals(filters)
        
        # Calculate key metrics
        total_deals = len(deals)
        closed_won_deals = [d for d in deals if d.stage == "closed_won"]
        closed_lost_deals = [d for d in deals if d.stage == "closed_lost"]
        
        win_rate = len(closed_won_deals) / (len(closed_won_deals) + len(closed_lost_deals)) if (len(closed_won_deals) + len(closed_lost_deals)) > 0 else 0
        
        total_revenue = sum(deal.amount for deal in closed_won_deals)
        avg_deal_size = total_revenue / len(closed_won_deals) if closed_won_deals else Decimal(0)
        
        # Calculate sales cycle length
        avg_sales_cycle = self._calculate_avg_sales_cycle(closed_won_deals)
        
        # Pipeline hygiene
        hygiene_result = await self.pipeline_hygiene_analysis(rep_id)
        
        # Forecast accuracy
        forecast_accuracy = await self._calculate_forecast_accuracy(period, rep_id)
        
        return {
            "period": period,
            "total_deals": total_deals,
            "win_rate": win_rate,
            "total_revenue": float(total_revenue),
            "avg_deal_size": float(avg_deal_size),
            "avg_sales_cycle_days": avg_sales_cycle,
            "pipeline_hygiene_score": hygiene_result.hygiene_score,
            "forecast_accuracy": forecast_accuracy,
            "quota_attainment": await self._get_quota_attainment(period, rep_id)
        }
    
    # Helper methods
    async def _generate_hygiene_recommendations(
        self, 
        deals: List[Deal], 
        stale_deals: List[str], 
        missing_data_deals: List[str], 
        overdue_deals: List[str]
    ) -> List[str]:
        """Generate AI-powered hygiene recommendations"""
        
        recommendations_prompt = f"""
        Generate pipeline hygiene recommendations based on analysis:
        
        Total Deals: {len(deals)}
        Stale Deals: {len(stale_deals)} ({len(stale_deals)/len(deals)*100:.1f}%)
        Missing Data: {len(missing_data_deals)} ({len(missing_data_deals)/len(deals)*100:.1f}%)
        Overdue Deals: {len(overdue_deals)} ({len(overdue_deals)/len(deals)*100:.1f}%)
        
        Provide 3-5 specific, actionable recommendations to improve pipeline hygiene.
        Focus on the most impactful issues first.
        """
        
        response = await self.model_router.generate_response(
            prompt=recommendations_prompt,
            model_preference="concise",
            max_tokens=300
        )
        
        # Parse recommendations (simplified)
        recommendations = [line.strip() for line in response.split('\n') if line.strip() and not line.startswith('#')]
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _parse_forecast_period(self, period: str) -> Tuple[date, date]:
        """Parse forecast period string into start and end dates"""
        # Simplified parsing - would handle various formats in production
        current_year = datetime.now().year
        
        if "Q1" in period:
            return date(current_year, 1, 1), date(current_year, 3, 31)
        elif "Q2" in period:
            return date(current_year, 4, 1), date(current_year, 6, 30)
        elif "Q3" in period:
            return date(current_year, 7, 1), date(current_year, 9, 30)
        elif "Q4" in period:
            return date(current_year, 10, 1), date(current_year, 12, 31)
        else:
            # Default to current quarter
            current_month = datetime.now().month
            if current_month <= 3:
                return date(current_year, 1, 1), date(current_year, 3, 31)
            elif current_month <= 6:
                return date(current_year, 4, 1), date(current_year, 6, 30)
            elif current_month <= 9:
                return date(current_year, 7, 1), date(current_year, 9, 30)
            else:
                return date(current_year, 10, 1), date(current_year, 12, 31)
    
    async def _get_quota_for_period(self, period: str, rep_id: Optional[str]) -> Decimal:
        """Get quota for the specified period and rep"""
        try:
            # Parse period (e.g., "2024-Q1", "2024-01")
            if "Q" in period:
                year, quarter = period.split("-Q")
                # Calculate quarter start/end dates
                quarter_start_month = (int(quarter) - 1) * 3 + 1
                start_date = f"{year}-{quarter_start_month:02d}-01"
                end_month = quarter_start_month + 2
                end_date = f"{year}-{end_month:02d}-31"
            else:
                # Monthly period
                start_date = f"{period}-01"
                # Calculate last day of month
                year, month = period.split("-")
                if month == "12":
                    end_date = f"{int(year)+1}-01-01"
                else:
                    end_date = f"{year}-{int(month)+1:02d}-01"
            
            # Query quota from database or CRM
            crm_connector = self.connector_factory.create_crm_connector("salesforce")
            if crm_connector:
                # Query quota records from Salesforce
                quota_query = f"""
                SELECT Quota__c, StartDate, EndDate 
                FROM QuotaAssignment__c 
                WHERE StartDate <= '{start_date}' 
                AND EndDate >= '{end_date}'
                """
                
                if rep_id:
                    quota_query += f" AND OwnerId = '{rep_id}'"
                
                # This would be a real SOQL query in production
                # For now, return calculated quota based on rep tier
                if rep_id:
                    # Get rep details to determine quota tier
                    rep_details = await self._get_rep_details(rep_id)
                    if rep_details and rep_details.get("tier") == "enterprise":
                        return Decimal("2000000")  # $2M for enterprise reps
                    elif rep_details and rep_details.get("tier") == "senior":
                        return Decimal("1500000")  # $1.5M for senior reps
                    else:
                        return Decimal("1000000")  # $1M for standard reps
                else:
                    # Team quota - sum of individual quotas
                    return Decimal("10000000")  # $10M team quota
            
            # Fallback to default quota
            return Decimal("1000000")
            
        except Exception as e:
            self.logger.error(f"Error getting quota for period {period}: {e}")
            return Decimal("1000000")  # Default fallback
    
    async def _calculate_forecast_accuracy(self, period: str, rep_id: Optional[str]) -> float:
        """Calculate historical forecast accuracy"""
        try:
            # Get historical forecast data from CRM
            crm_connector = self.connector_factory.create_crm_connector("salesforce")
            if not crm_connector:
                return 0.85  # Default accuracy
            
            # Query historical forecasts and actual results
            # This would typically look at the last 3-6 periods for accuracy calculation
            historical_periods = self._get_historical_periods(period, 6)
            
            total_forecast = Decimal('0')
            total_actual = Decimal('0')
            accuracy_scores = []
            
            for hist_period in historical_periods:
                # Get forecast data for the period
                forecast_data = await self._get_forecast_for_period(hist_period, rep_id)
                actual_data = await self._get_actual_results_for_period(hist_period, rep_id)
                
                if forecast_data and actual_data:
                    forecast_amount = forecast_data.get("amount", Decimal('0'))
                    actual_amount = actual_data.get("amount", Decimal('0'))
                    
                    if forecast_amount > 0:
                        # Calculate accuracy as 1 - (abs(forecast - actual) / forecast)
                        accuracy = 1 - abs(forecast_amount - actual_amount) / forecast_amount
                        accuracy_scores.append(max(0, accuracy))  # Don't go below 0
                        
                        total_forecast += forecast_amount
                        total_actual += actual_amount
            
            if accuracy_scores:
                # Return average accuracy across periods
                avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
                return float(min(1.0, max(0.0, avg_accuracy)))  # Clamp between 0 and 1
            
            # Fallback calculation based on total forecast vs actual
            if total_forecast > 0:
                overall_accuracy = 1 - abs(total_forecast - total_actual) / total_forecast
                return float(min(1.0, max(0.0, overall_accuracy)))
            
            return 0.85  # Default accuracy if no data
            
        except Exception as e:
            self.logger.error(f"Error calculating forecast accuracy for period {period}: {e}")
            return 0.85  # Default accuracy on error
    
    async def _get_rep_details(self, rep_id: str) -> Optional[Dict[str, Any]]:
        """Get sales rep details from CRM"""
        try:
            crm_connector = self.connector_factory.create_crm_connector("salesforce")
            if crm_connector:
                # Query user details from Salesforce
                user_query = f"""
                SELECT Id, Name, Title, UserRole.Name, Profile.Name, 
                       IsActive, EmployeeNumber, Department
                FROM User 
                WHERE Id = '{rep_id}' AND IsActive = true
                """
                
                # This would be a real SOQL query in production
                # For now, return mock data based on rep_id patterns
                if "senior" in rep_id.lower() or "sr" in rep_id.lower():
                    return {"tier": "senior", "name": "Senior Rep", "department": "Sales"}
                elif "enterprise" in rep_id.lower() or "ent" in rep_id.lower():
                    return {"tier": "enterprise", "name": "Enterprise Rep", "department": "Enterprise Sales"}
                else:
                    return {"tier": "standard", "name": "Sales Rep", "department": "Sales"}
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting rep details for {rep_id}: {e}")
            return None
    
    def _get_historical_periods(self, current_period: str, num_periods: int) -> List[str]:
        """Get list of historical periods for analysis"""
        try:
            periods = []
            
            if "Q" in current_period:
                # Quarterly periods
                year, quarter = current_period.split("-Q")
                year = int(year)
                quarter = int(quarter)
                
                for i in range(1, num_periods + 1):
                    hist_quarter = quarter - i
                    hist_year = year
                    
                    if hist_quarter <= 0:
                        hist_quarter += 4
                        hist_year -= 1
                    
                    periods.append(f"{hist_year}-Q{hist_quarter}")
            else:
                # Monthly periods
                year, month = current_period.split("-")
                year = int(year)
                month = int(month)
                
                for i in range(1, num_periods + 1):
                    hist_month = month - i
                    hist_year = year
                    
                    if hist_month <= 0:
                        hist_month += 12
                        hist_year -= 1
                    
                    periods.append(f"{hist_year}-{hist_month:02d}")
            
            return periods
            
        except Exception as e:
            self.logger.error(f"Error generating historical periods: {e}")
            return []
    
    async def _get_forecast_for_period(self, period: str, rep_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Get forecast data for a specific period"""
        try:
            # This would query forecast records from CRM
            # For now, return mock data
            base_amount = Decimal("800000") if rep_id else Decimal("8000000")
            
            # Add some variance based on period
            import hashlib
            period_hash = int(hashlib.md5(period.encode()).hexdigest()[:8], 16)
            variance = (period_hash % 20 - 10) / 100  # -10% to +10% variance
            
            return {
                "amount": base_amount * (1 + Decimal(str(variance))),
                "period": period,
                "rep_id": rep_id
            }
            
        except Exception as e:
            self.logger.error(f"Error getting forecast for period {period}: {e}")
            return None
    
    async def _get_actual_results_for_period(self, period: str, rep_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Get actual results for a specific period"""
        try:
            # This would query closed deals from CRM for the period
            crm_connector = self.connector_factory.create_crm_connector("salesforce")
            if crm_connector:
                # Parse period to date range
                start_date, end_date = self._parse_period_to_dates(period)
                
                # Query closed won deals in the period
                filters = {
                    "stage": ["Closed Won"],
                    "close_date_start": start_date,
                    "close_date_end": end_date
                }
                
                if rep_id:
                    filters["owner_id"] = rep_id
                
                deals = await crm_connector.get_deals(filters)
                total_amount = sum(deal.amount for deal in deals)
                
                return {
                    "amount": total_amount,
                    "period": period,
                    "rep_id": rep_id,
                    "deal_count": len(deals)
                }
            
            # Fallback to mock data
            forecast_data = await self._get_forecast_for_period(period, rep_id)
            if forecast_data:
                # Simulate actual results with some variance from forecast
                import random
                variance = random.uniform(0.8, 1.2)  # 80% to 120% of forecast
                return {
                    "amount": forecast_data["amount"] * Decimal(str(variance)),
                    "period": period,
                    "rep_id": rep_id
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting actual results for period {period}: {e}")
            return None
    
    def _parse_period_to_dates(self, period: str) -> Tuple[str, str]:
        """Parse period string to start and end dates"""
        try:
            if "Q" in period:
                year, quarter = period.split("-Q")
                quarter = int(quarter)
                
                # Calculate quarter dates
                start_month = (quarter - 1) * 3 + 1
                end_month = start_month + 2
                
                start_date = f"{year}-{start_month:02d}-01"
                
                # Calculate last day of quarter
                if end_month == 12:
                    end_date = f"{year}-12-31"
                else:
                    # Get last day of end_month
                    import calendar
                    last_day = calendar.monthrange(int(year), end_month)[1]
                    end_date = f"{year}-{end_month:02d}-{last_day}"
                
                return start_date, end_date
            else:
                # Monthly period
                year, month = period.split("-")
                start_date = f"{year}-{month}-01"
                
                # Calculate last day of month
                import calendar
                last_day = calendar.monthrange(int(year), int(month))[1]
                end_date = f"{year}-{month}-{last_day}"
                
                return start_date, end_date
                
        except Exception as e:
            self.logger.error(f"Error parsing period {period}: {e}")
            # Return current month as fallback
            from datetime import datetime
            now = datetime.utcnow()
            return f"{now.year}-{now.month:02d}-01", f"{now.year}-{now.month:02d}-28"
    
    async def _analyze_forecast_risks_opportunities(
        self, 
        deals: List[Deal], 
        period: str, 
        quota_attainment: float
    ) -> Tuple[List[str], List[str]]:
        """Analyze forecast risks and opportunities using AI"""
        
        analysis_prompt = f"""
        Analyze sales forecast for risks and opportunities:
        
        Period: {period}
        Total Deals: {len(deals)}
        Quota Attainment: {quota_attainment:.1%}
        
        Deal Breakdown:
        - High probability deals: {len([d for d in deals if d.probability > 0.7])}
        - Medium probability deals: {len([d for d in deals if 0.3 <= d.probability <= 0.7])}
        - Low probability deals: {len([d for d in deals if d.probability < 0.3])}
        
        Identify:
        1. Top 3 risk factors that could impact forecast
        2. Top 3 opportunities to exceed forecast
        
        Be specific and actionable.
        """
        
        analysis = await self.model_router.generate_response(
            prompt=analysis_prompt,
            model_preference="reasoning",
            max_tokens=400
        )
        
        # Parse risks and opportunities (simplified)
        lines = analysis.split('\n')
        risks = []
        opportunities = []
        
        current_section = None
        for line in lines:
            line = line.strip()
            if 'risk' in line.lower():
                current_section = 'risks'
            elif 'opportunit' in line.lower():
                current_section = 'opportunities'
            elif line and line[0].isdigit():
                if current_section == 'risks':
                    risks.append(line)
                elif current_section == 'opportunities':
                    opportunities.append(line)
        
        return risks[:3], opportunities[:3]
    
    async def _get_quote_data(self, quote_id: str) -> Optional[Dict[str, Any]]:
        """Get quote data from CPQ system"""
        # Placeholder - would integrate with CPQ system
        return {
            "id": quote_id,
            "deal_id": "DEAL-123",
            "account_id": "ACCOUNT-456",
            "total_amount": 50000,
            "discount_percent": 15,
            "line_items": []
        }
    
    async def _calculate_win_probability(self, quote_data: Dict, deal: Deal, account: Account) -> float:
        """Calculate win probability using AI analysis"""
        
        win_prob_prompt = f"""
        Calculate win probability for this sales quote:
        
        Quote Amount: ${quote_data['total_amount']:,}
        Discount: {quote_data['discount_percent']}%
        
        Deal Stage: {deal.stage}
        Deal Probability: {deal.probability:.0%}
        
        Account: {account.name}
        Account Size: {account.size}
        Industry: {account.industry}
        Health Score: {account.health_score}
        
        Based on these factors, what is the realistic win probability (0.0-1.0)?
        Consider pricing competitiveness, account fit, and deal momentum.
        """
        
        response = await self.model_router.generate_response(
            prompt=win_prob_prompt,
            model_preference="analytical",
            max_tokens=200
        )
        
        # Extract probability (simplified parsing)
        try:
            # Look for percentage or decimal in response
            import re
            prob_match = re.search(r'(\d+(?:\.\d+)?)', response)
            if prob_match:
                prob = float(prob_match.group(1))
                if prob > 1:
                    prob = prob / 100  # Convert percentage to decimal
                return min(max(prob, 0.0), 1.0)
        except:
            pass
        
        return deal.probability  # Fallback to deal probability
    
    async def _analyze_pricing_optimization(self, quote_data: Dict, deal: Deal, account: Account) -> Dict[str, Any]:
        """Analyze pricing optimization opportunities"""
        # Simplified pricing analysis
        current_discount = quote_data['discount_percent']
        
        optimization = {
            "current_discount": current_discount,
            "recommended_discount": max(0, current_discount - 5),  # Reduce discount by 5%
            "price_elasticity": 0.8,  # Placeholder
            "competitive_position": "neutral"
        }
        
        return optimization
    
    async def _perform_competitive_analysis(self, quote_data: Dict, deal: Deal, account: Account) -> Dict[str, Any]:
        """Perform competitive analysis"""
        # Placeholder competitive analysis
        return {
            "primary_competitor": "Competitor A",
            "competitive_threats": ["Lower pricing", "Existing relationship"],
            "competitive_advantages": ["Better features", "Superior support"],
            "win_strategy": "Emphasize ROI and long-term value"
        }
    
    def _requires_approval(self, quote_data: Dict, pricing_optimization: Dict) -> bool:
        """Determine if quote requires approval"""
        # Require approval for high discounts
        return quote_data['discount_percent'] > 20
    
    async def _generate_quote_recommendations(
        self, 
        quote_data: Dict, 
        deal: Deal, 
        account: Account, 
        win_probability: float, 
        pricing_optimization: Dict
    ) -> List[str]:
        """Generate quote optimization recommendations"""
        
        recommendations = []
        
        if quote_data['discount_percent'] > 15:
            recommendations.append("Consider reducing discount to improve margins")
        
        if win_probability < 0.5:
            recommendations.append("Focus on value proposition and ROI justification")
        
        if account.size == "Enterprise" and quote_data['total_amount'] < 100000:
            recommendations.append("Explore upselling opportunities for enterprise account")
        
        return recommendations
    
    async def _get_renewal_history(self, account_id: str) -> List[Dict]:
        """Get customer renewal history"""
        # Placeholder - would query renewal data
        return []
    
    async def _get_usage_data(self, account_id: str) -> Dict[str, Any]:
        """Get customer usage data"""
        # Placeholder - would query usage analytics
        return {"usage_trend": "increasing", "feature_adoption": 0.75}
    
    async def _get_support_history(self, account_id: str) -> List[Dict]:
        """Get customer support ticket history"""
        # Placeholder - would query support system
        return []
    
    def _extract_renewal_probability(self, analysis: str) -> float:
        """Extract renewal probability from AI analysis"""
        # Simplified extraction
        import re
        prob_match = re.search(r'(\d+(?:\.\d+)?)', analysis)
        if prob_match:
            prob = float(prob_match.group(1))
            if prob > 1:
                prob = prob / 100
            return min(max(prob, 0.0), 1.0)
        return 0.7  # Default
    
    def _extract_risk_factors(self, analysis: str) -> List[str]:
        """Extract risk factors from AI analysis"""
        # Simplified extraction
        return ["Usage decline", "Support issues", "Budget constraints"]
    
    def _extract_expansion_opportunities(self, analysis: str) -> List[str]:
        """Extract expansion opportunities from AI analysis"""
        # Simplified extraction
        return ["Additional licenses", "Premium features", "Professional services"]
    
    def _extract_recommended_actions(self, analysis: str) -> List[str]:
        """Extract recommended actions from AI analysis"""
        # Simplified extraction
        return ["Schedule QBR", "Address support issues", "Present expansion proposal"]
    
    def _calculate_avg_sales_cycle(self, closed_deals: List[Deal]) -> float:
        """Calculate average sales cycle length"""
        if not closed_deals:
            return 0.0
        
        total_days = 0
        for deal in closed_deals:
            cycle_days = (deal.close_date - deal.created_date.date()).days
            total_days += cycle_days
        
        return total_days / len(closed_deals)
    
    async def _get_quota_attainment(self, period: str, rep_id: Optional[str]) -> float:
        """Get quota attainment for period"""
        # Placeholder - would calculate from actual vs quota
        return 0.85  # 85% attainment