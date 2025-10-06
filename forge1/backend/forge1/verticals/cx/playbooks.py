"""
Customer Experience (CX) Playbooks
Standard Operating Procedures for CX AI Employee
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime, timedelta

from forge1.backend.forge1.core.orchestration import WorkflowEngine
from forge1.backend.forge1.core.memory import MemoryManager
from forge1.backend.forge1.core.models import ModelRouter
from forge1.backend.forge1.verticals.cx.connectors import (
    CXTicket, CustomerProfile, CXConnectorFactory, N8nConnector
)


class TicketPriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TicketCategory(Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    GENERAL = "general"
    COMPLAINT = "complaint"
    FEATURE_REQUEST = "feature_request"


@dataclass
class TriageResult:
    """Result of ticket triage process"""
    priority: TicketPriority
    category: TicketCategory
    sentiment: str  # positive, neutral, negative
    urgency_score: float  # 0.0 to 1.0
    escalation_required: bool
    suggested_response: str
    confidence: float


@dataclass
class ResolutionResult:
    """Result of ticket resolution attempt"""
    resolved: bool
    resolution_text: str
    follow_up_required: bool
    escalation_needed: bool
    customer_satisfaction_predicted: float


class CXPlaybooks:
    """Customer Experience playbooks and workflows"""
    
    def __init__(
        self,
        workflow_engine: WorkflowEngine,
        memory_manager: MemoryManager,
        model_router: ModelRouter,
        connector_factory: CXConnectorFactory
    ):
        self.workflow_engine = workflow_engine
        self.memory_manager = memory_manager
        self.model_router = model_router
        self.connector_factory = connector_factory
        
        # SLA targets
        self.first_response_sla = timedelta(seconds=5)  # < 5s first response
        self.resolution_sla = {
            TicketPriority.URGENT: timedelta(hours=1),
            TicketPriority.HIGH: timedelta(hours=4),
            TicketPriority.NORMAL: timedelta(hours=24),
            TicketPriority.LOW: timedelta(hours=72)
        }
        self.deflection_target = 0.95  # 95%+ deflection rate
    
    async def triage_ticket(self, ticket: CXTicket, customer: CustomerProfile) -> TriageResult:
        """
        Intelligent ticket triage with sentiment analysis and priority assignment
        """
        # Analyze ticket content for sentiment and urgency
        triage_prompt = f"""
        Analyze this customer support ticket for triage:
        
        Customer: {customer.name} (Tier: {customer.tier}, LTV: ${customer.lifetime_value:,.2f})
        Subject: {ticket.subject}
        Description: {ticket.description}
        
        Provide triage analysis including:
        1. Priority level (low/normal/high/urgent)
        2. Category (technical/billing/general/complaint/feature_request)
        3. Sentiment (positive/neutral/negative)
        4. Urgency score (0.0-1.0)
        5. Whether escalation to human is required
        6. Suggested initial response
        7. Confidence in analysis (0.0-1.0)
        
        Consider customer tier and history for priority weighting.
        """
        
        # Get AI analysis
        analysis = await self.model_router.generate_response(
            prompt=triage_prompt,
            model_preference="fast_accurate",
            max_tokens=500
        )
        
        # Parse analysis (simplified - would use structured output in production)
        priority = self._extract_priority(analysis, customer.tier)
        category = self._extract_category(analysis)
        sentiment = self._extract_sentiment(analysis)
        urgency_score = self._calculate_urgency_score(ticket, customer, sentiment)
        escalation_required = self._requires_escalation(urgency_score, customer.tier, category)
        
        # Generate suggested response
        response_prompt = f"""
        Generate a professional, empathetic first response for this ticket:
        
        Customer: {customer.name}
        Issue: {ticket.subject}
        Sentiment: {sentiment}
        Priority: {priority.value}
        
        Response should:
        - Acknowledge the issue
        - Show empathy if negative sentiment
        - Provide initial guidance if possible
        - Set expectations for resolution time
        - Be concise but helpful
        """
        
        suggested_response = await self.model_router.generate_response(
            prompt=response_prompt,
            model_preference="conversational",
            max_tokens=200
        )
        
        return TriageResult(
            priority=priority,
            category=category,
            sentiment=sentiment,
            urgency_score=urgency_score,
            escalation_required=escalation_required,
            suggested_response=suggested_response,
            confidence=0.85  # Would calculate based on model confidence
        )
    
    async def resolve_ticket(self, ticket: CXTicket, customer: CustomerProfile, triage: TriageResult) -> ResolutionResult:
        """
        Attempt to resolve customer ticket using knowledge base and AI reasoning
        """
        # Retrieve relevant knowledge from memory
        context = await self.memory_manager.retrieve_context(
            query=f"{ticket.subject} {ticket.description}",
            context_type="cx_knowledge",
            max_results=5
        )
        
        # Get customer history
        customer_history = await self.memory_manager.retrieve_context(
            query=f"customer:{customer.id}",
            context_type="customer_interactions",
            max_results=10
        )
        
        resolution_prompt = f"""
        Resolve this customer support ticket:
        
        Ticket: {ticket.subject}
        Description: {ticket.description}
        Customer: {customer.name} (Tier: {customer.tier})
        Category: {triage.category.value}
        Priority: {triage.priority.value}
        
        Knowledge Base Context:
        {self._format_context(context)}
        
        Customer History:
        {self._format_customer_history(customer_history)}
        
        Provide:
        1. Complete resolution if possible
        2. Step-by-step troubleshooting if technical
        3. Clear explanation of next steps
        4. Whether follow-up is needed
        5. Whether escalation to human is required
        
        Be thorough but concise. Prioritize customer satisfaction.
        """
        
        resolution = await self.model_router.generate_response(
            prompt=resolution_prompt,
            model_preference="reasoning",
            max_tokens=800
        )
        
        # Analyze resolution quality
        resolved = self._assess_resolution_completeness(resolution, triage.category)
        follow_up_required = self._requires_follow_up(resolution, triage.priority)
        escalation_needed = triage.escalation_required or not resolved
        
        # Predict customer satisfaction
        satisfaction_score = await self._predict_satisfaction(
            ticket, customer, triage, resolution
        )
        
        return ResolutionResult(
            resolved=resolved,
            resolution_text=resolution,
            follow_up_required=follow_up_required,
            escalation_needed=escalation_needed,
            customer_satisfaction_predicted=satisfaction_score
        )
    
    async def handle_escalation(self, ticket: CXTicket, customer: CustomerProfile, reason: str) -> bool:
        """
        Handle ticket escalation to human agents with proper context
        """
        # Prepare escalation context
        escalation_context = {
            "ticket_id": ticket.id,
            "customer_id": customer.id,
            "customer_tier": customer.tier,
            "escalation_reason": reason,
            "ai_analysis": await self._generate_escalation_summary(ticket, customer),
            "suggested_actions": await self._suggest_human_actions(ticket, customer),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Trigger n8n workflow for human notification
        n8n_connector = self.connector_factory.create_n8n_connector()
        workflow_triggered = await n8n_connector.trigger_workflow(
            workflow_id="cx_escalation",
            data=escalation_context
        )
        
        if workflow_triggered:
            # Update ticket status
            connector = self.connector_factory.create_connector("zendesk")  # or salesforce
            if connector:
                await connector.update_ticket(ticket.id, {
                    "status": "escalated",
                    "priority": "high",
                    "comment": f"Escalated to human agent: {reason}"
                })
        
        return workflow_triggered
    
    async def upsell_opportunity_detection(self, customer: CustomerProfile, interaction_history: List[str]) -> Optional[Dict[str, Any]]:
        """
        Detect upselling opportunities during customer interactions
        """
        upsell_prompt = f"""
        Analyze customer interaction for upselling opportunities:
        
        Customer: {customer.name}
        Tier: {customer.tier}
        Lifetime Value: ${customer.lifetime_value:,.2f}
        
        Recent Interactions:
        {chr(10).join(interaction_history[-5:])}
        
        Identify:
        1. Potential upsell products/services
        2. Customer pain points that upgrades could solve
        3. Timing appropriateness (0.0-1.0)
        4. Suggested approach
        5. Expected value of opportunity
        
        Only suggest if genuinely beneficial to customer.
        """
        
        analysis = await self.model_router.generate_response(
            prompt=upsell_prompt,
            model_preference="reasoning",
            max_tokens=400
        )
        
        # Parse and validate opportunity
        opportunity = self._parse_upsell_opportunity(analysis)
        
        if opportunity and opportunity["timing_score"] > 0.7:
            return opportunity
        
        return None
    
    async def measure_performance(self, time_period: timedelta) -> Dict[str, float]:
        """
        Measure CX performance against SLAs and targets
        """
        end_time = datetime.utcnow()
        start_time = end_time - time_period
        
        # Query performance metrics from memory/database
        metrics = await self._query_performance_metrics(start_time, end_time)
        
        return {
            "deflection_rate": metrics.get("deflection_rate", 0.0),
            "first_response_time_avg": metrics.get("first_response_time", 0.0),
            "resolution_time_avg": metrics.get("resolution_time", 0.0),
            "customer_satisfaction": metrics.get("csat_score", 0.0),
            "escalation_rate": metrics.get("escalation_rate", 0.0),
            "sla_compliance": metrics.get("sla_compliance", 0.0)
        }
    
    # Helper methods
    def _extract_priority(self, analysis: str, customer_tier: str) -> TicketPriority:
        """Extract priority from AI analysis"""
        analysis_lower = analysis.lower()
        
        # Boost priority for premium customers
        tier_boost = customer_tier.lower() in ["gold", "platinum"]
        
        if "urgent" in analysis_lower or (tier_boost and "high" in analysis_lower):
            return TicketPriority.URGENT
        elif "high" in analysis_lower:
            return TicketPriority.HIGH
        elif "low" in analysis_lower and not tier_boost:
            return TicketPriority.LOW
        else:
            return TicketPriority.NORMAL
    
    def _extract_category(self, analysis: str) -> TicketCategory:
        """Extract category from AI analysis"""
        analysis_lower = analysis.lower()
        
        if "technical" in analysis_lower or "bug" in analysis_lower:
            return TicketCategory.TECHNICAL
        elif "billing" in analysis_lower or "payment" in analysis_lower:
            return TicketCategory.BILLING
        elif "complaint" in analysis_lower or "angry" in analysis_lower:
            return TicketCategory.COMPLAINT
        elif "feature" in analysis_lower or "enhancement" in analysis_lower:
            return TicketCategory.FEATURE_REQUEST
        else:
            return TicketCategory.GENERAL
    
    def _extract_sentiment(self, analysis: str) -> str:
        """Extract sentiment from AI analysis"""
        analysis_lower = analysis.lower()
        
        if any(word in analysis_lower for word in ["angry", "frustrated", "disappointed", "negative"]):
            return "negative"
        elif any(word in analysis_lower for word in ["happy", "satisfied", "positive", "pleased"]):
            return "positive"
        else:
            return "neutral"
    
    def _calculate_urgency_score(self, ticket: CXTicket, customer: CustomerProfile, sentiment: str) -> float:
        """Calculate urgency score based on multiple factors"""
        base_score = 0.5
        
        # Customer tier impact
        tier_multipliers = {"bronze": 1.0, "silver": 1.2, "gold": 1.5, "platinum": 2.0}
        base_score *= tier_multipliers.get(customer.tier.lower(), 1.0)
        
        # Sentiment impact
        if sentiment == "negative":
            base_score += 0.3
        elif sentiment == "positive":
            base_score -= 0.1
        
        # Keywords in subject/description
        urgent_keywords = ["urgent", "critical", "down", "broken", "emergency", "asap"]
        text = f"{ticket.subject} {ticket.description}".lower()
        
        for keyword in urgent_keywords:
            if keyword in text:
                base_score += 0.2
                break
        
        return min(base_score, 1.0)
    
    def _requires_escalation(self, urgency_score: float, customer_tier: str, category: TicketCategory) -> bool:
        """Determine if ticket requires human escalation"""
        # Always escalate high urgency
        if urgency_score > 0.8:
            return True
        
        # Escalate premium customer complaints
        if customer_tier.lower() in ["gold", "platinum"] and category == TicketCategory.COMPLAINT:
            return True
        
        # Escalate complex technical issues
        if category == TicketCategory.TECHNICAL and urgency_score > 0.6:
            return True
        
        return False
    
    def _format_context(self, context: List[Any]) -> str:
        """Format knowledge base context for prompt"""
        if not context:
            return "No relevant knowledge found."
        
        formatted = []
        for item in context[:3]:  # Limit to top 3 results
            formatted.append(f"- {item.get('content', str(item))}")
        
        return "\n".join(formatted)
    
    def _format_customer_history(self, history: List[Any]) -> str:
        """Format customer history for prompt"""
        if not history:
            return "No previous interactions found."
        
        formatted = []
        for item in history[:5]:  # Limit to recent 5 interactions
            formatted.append(f"- {item.get('summary', str(item))}")
        
        return "\n".join(formatted)
    
    def _assess_resolution_completeness(self, resolution: str, category: TicketCategory) -> bool:
        """Assess if resolution appears complete"""
        resolution_lower = resolution.lower()
        
        # Check for resolution indicators
        resolution_indicators = [
            "resolved", "fixed", "completed", "solution", "answer",
            "here's how", "follow these steps", "this should resolve"
        ]
        
        return any(indicator in resolution_lower for indicator in resolution_indicators)
    
    def _requires_follow_up(self, resolution: str, priority: TicketPriority) -> bool:
        """Determine if follow-up is required"""
        resolution_lower = resolution.lower()
        
        # Always follow up on high priority
        if priority in [TicketPriority.HIGH, TicketPriority.URGENT]:
            return True
        
        # Follow up if resolution mentions it
        follow_up_indicators = ["follow up", "check back", "let us know", "contact us"]
        return any(indicator in resolution_lower for indicator in follow_up_indicators)
    
    async def _predict_satisfaction(self, ticket: CXTicket, customer: CustomerProfile, triage: TriageResult, resolution: str) -> float:
        """Predict customer satisfaction score"""
        # Simplified satisfaction prediction
        base_score = 0.7
        
        # Adjust based on resolution quality
        if len(resolution) > 200:  # Detailed response
            base_score += 0.1
        
        # Adjust based on sentiment
        if triage.sentiment == "negative":
            base_score -= 0.2
        elif triage.sentiment == "positive":
            base_score += 0.1
        
        # Adjust based on customer tier
        if customer.tier.lower() in ["gold", "platinum"]:
            base_score += 0.1
        
        return min(max(base_score, 0.0), 1.0)
    
    async def _generate_escalation_summary(self, ticket: CXTicket, customer: CustomerProfile) -> str:
        """Generate summary for human agent escalation"""
        summary_prompt = f"""
        Create a concise escalation summary for human agent:
        
        Customer: {customer.name} ({customer.tier} tier)
        Issue: {ticket.subject}
        Description: {ticket.description}
        
        Provide:
        - Key issue summary
        - Customer context
        - AI analysis results
        - Recommended next steps
        
        Keep under 150 words.
        """
        
        return await self.model_router.generate_response(
            prompt=summary_prompt,
            model_preference="concise",
            max_tokens=200
        )
    
    async def _suggest_human_actions(self, ticket: CXTicket, customer: CustomerProfile) -> List[str]:
        """Suggest actions for human agent"""
        # Simplified action suggestions
        actions = [
            f"Review customer tier ({customer.tier}) for appropriate handling",
            "Check customer history for similar issues",
            "Verify AI analysis accuracy"
        ]
        
        if customer.tier.lower() in ["gold", "platinum"]:
            actions.append("Consider proactive outreach to prevent churn")
        
        return actions
    
    def _parse_upsell_opportunity(self, analysis: str) -> Optional[Dict[str, Any]]:
        """Parse upsell opportunity from AI analysis"""
        # Simplified parsing - would use structured output in production
        if "no opportunity" in analysis.lower() or "not appropriate" in analysis.lower():
            return None
        
        return {
            "product": "Premium Support",  # Would extract from analysis
            "timing_score": 0.8,  # Would extract from analysis
            "expected_value": 1000,  # Would extract from analysis
            "approach": "Mention during resolution"  # Would extract from analysis
        }
    
    async def _query_performance_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, float]:
        """Query performance metrics from database"""
        # Placeholder - would query actual metrics from database
        return {
            "deflection_rate": 0.96,
            "first_response_time": 3.2,  # seconds
            "resolution_time": 1800,  # seconds
            "csat_score": 4.2,  # out of 5
            "escalation_rate": 0.04,
            "sla_compliance": 0.98
        }