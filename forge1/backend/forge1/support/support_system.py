"""
Forge 1 Support and Operations System
Tiered support, incident management, rollback procedures, and customer success
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
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


class SupportTier(Enum):
    """Support tier levels"""
    SELF_SERVICE = "self_service"
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class TicketPriority(Enum):
    """Support ticket priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class TicketStatus(Enum):
    """Support ticket status"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    WAITING_CUSTOMER = "waiting_customer"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class IncidentSeverity(Enum):
    """Incident severity levels"""
    SEV1 = "sev1"  # Critical - System down
    SEV2 = "sev2"  # High - Major functionality impacted
    SEV3 = "sev3"  # Medium - Minor functionality impacted
    SEV4 = "sev4"  # Low - Cosmetic or documentation issues


class RollbackType(Enum):
    """Types of rollback operations"""
    APPLICATION = "application"
    DATABASE = "database"
    CONFIGURATION = "configuration"
    INFRASTRUCTURE = "infrastructure"
    FULL_SYSTEM = "full_system"


@dataclass
class SupportTicket:
    """Customer support ticket"""
    ticket_id: str
    customer_id: str
    
    # Ticket details
    title: str
    description: str
    priority: TicketPriority
    status: TicketStatus = TicketStatus.OPEN
    
    # Assignment
    assigned_to: Optional[str] = None
    support_tier: SupportTier = SupportTier.BASIC
    
    # Contact information
    contact_email: str = ""
    contact_phone: Optional[str] = None
    
    # Categorization
    category: str = ""  # technical, billing, feature_request, etc.
    subcategory: str = ""
    product_area: str = ""
    
    # Resolution
    resolution: str = ""
    resolution_time_minutes: Optional[int] = None
    customer_satisfaction_score: Optional[int] = None
    
    # Timeline
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    
    # SLA tracking
    first_response_due: Optional[datetime] = None
    resolution_due: Optional[datetime] = None
    sla_breached: bool = False
    
    # Communication
    updates: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Incident:
    """System incident"""
    incident_id: str
    
    # Incident details
    title: str
    description: str
    severity: IncidentSeverity
    status: str = "investigating"  # investigating, identified, monitoring, resolved
    
    # Impact
    affected_services: List[str] = field(default_factory=list)
    affected_customers: int = 0
    business_impact: str = ""
    
    # Response team
    incident_commander: str = ""
    response_team: List[str] = field(default_factory=list)
    
    # Timeline
    detected_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Communication
    status_updates: List[Dict[str, Any]] = field(default_factory=list)
    customer_communication: List[Dict[str, Any]] = field(default_factory=list)
    
    # Post-incident
    root_cause: str = ""
    lessons_learned: str = ""
    action_items: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RollbackPlan:
    """System rollback plan"""
    plan_id: str
    name: str
    description: str
    rollback_type: RollbackType
    
    # Rollback details
    target_version: str
    current_version: str
    affected_components: List[str] = field(default_factory=list)
    
    # Execution steps
    pre_rollback_steps: List[str] = field(default_factory=list)
    rollback_steps: List[str] = field(default_factory=list)
    post_rollback_steps: List[str] = field(default_factory=list)
    
    # Validation
    validation_steps: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    
    # Risk assessment
    risk_level: str = "medium"  # low, medium, high
    estimated_downtime_minutes: int = 0
    rollback_window_start: Optional[datetime] = None
    rollback_window_end: Optional[datetime] = None
    
    # Approval
    requires_approval: bool = True
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    # Execution tracking
    executed: bool = False
    executed_at: Optional[datetime] = None
    executed_by: Optional[str] = None
    execution_log: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CustomerOnboarding:
    """Customer onboarding process"""
    onboarding_id: str
    customer_id: str
    
    # Onboarding details
    plan_type: str
    onboarding_tier: SupportTier
    
    # Progress tracking
    current_step: int = 1
    total_steps: int = 10
    completion_percentage: float = 0.0
    
    # Timeline
    started_at: datetime = field(default_factory=datetime.utcnow)
    target_completion_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Assigned resources
    customer_success_manager: Optional[str] = None
    technical_contact: Optional[str] = None
    
    # Milestones
    milestones_completed: List[str] = field(default_factory=list)
    milestones_pending: List[str] = field(default_factory=list)
    
    # Health score
    health_score: float = 100.0  # 0-100
    risk_factors: List[str] = field(default_factory=list)
    
    # Communication
    touchpoints: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class SupportSystem:
    """
    Comprehensive support and operations system
    
    Features:
    - Tiered customer support with SLA management
    - Incident response and management
    - Rollback procedures and automation
    - Customer onboarding and success tracking
    - Knowledge base and self-service
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
        self.logger = logging.getLogger("support_system")
        
        # Support data
        self.support_tickets: Dict[str, SupportTicket] = {}
        self.incidents: Dict[str, Incident] = {}
        self.rollback_plans: Dict[str, RollbackPlan] = {}
        self.customer_onboardings: Dict[str, CustomerOnboarding] = {}
        
        # Configuration
        self.sla_targets = {
            SupportTier.ENTERPRISE: {
                TicketPriority.EMERGENCY: {"first_response_minutes": 15, "resolution_hours": 4},
                TicketPriority.CRITICAL: {"first_response_minutes": 30, "resolution_hours": 8},
                TicketPriority.HIGH: {"first_response_minutes": 60, "resolution_hours": 24},
                TicketPriority.MEDIUM: {"first_response_minutes": 240, "resolution_hours": 72},
                TicketPriority.LOW: {"first_response_minutes": 480, "resolution_hours": 168}
            },
            SupportTier.PREMIUM: {
                TicketPriority.EMERGENCY: {"first_response_minutes": 30, "resolution_hours": 8},
                TicketPriority.CRITICAL: {"first_response_minutes": 60, "resolution_hours": 24},
                TicketPriority.HIGH: {"first_response_minutes": 120, "resolution_hours": 48},
                TicketPriority.MEDIUM: {"first_response_minutes": 480, "resolution_hours": 120},
                TicketPriority.LOW: {"first_response_minutes": 960, "resolution_hours": 240}
            },
            SupportTier.STANDARD: {
                TicketPriority.CRITICAL: {"first_response_minutes": 120, "resolution_hours": 48},
                TicketPriority.HIGH: {"first_response_minutes": 240, "resolution_hours": 72},
                TicketPriority.MEDIUM: {"first_response_minutes": 960, "resolution_hours": 168},
                TicketPriority.LOW: {"first_response_minutes": 1440, "resolution_hours": 336}
            },
            SupportTier.BASIC: {
                TicketPriority.HIGH: {"first_response_minutes": 480, "resolution_hours": 120},
                TicketPriority.MEDIUM: {"first_response_minutes": 1440, "resolution_hours": 240},
                TicketPriority.LOW: {"first_response_minutes": 2880, "resolution_hours": 480}
            }
        }
        
        # Initialize default rollback plans
        self._initialize_default_rollback_plans()
        
        self.logger.info("Initialized Support System")
    
    async def create_support_ticket(
        self,
        customer_id: str,
        title: str,
        description: str,
        priority: TicketPriority,
        contact_email: str,
        category: str = "technical",
        support_tier: SupportTier = SupportTier.STANDARD,
        contact_phone: Optional[str] = None
    ) -> SupportTicket:
        """Create a new support ticket"""
        
        ticket_id = f"ticket_{customer_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Calculate SLA targets
        sla_config = self.sla_targets.get(support_tier, {}).get(priority, {})
        
        first_response_due = None
        resolution_due = None
        
        if sla_config:
            first_response_due = datetime.utcnow() + timedelta(minutes=sla_config.get("first_response_minutes", 1440))
            resolution_due = datetime.utcnow() + timedelta(hours=sla_config.get("resolution_hours", 168))
        
        ticket = SupportTicket(
            ticket_id=ticket_id,
            customer_id=customer_id,
            title=title,
            description=description,
            priority=priority,
            contact_email=contact_email,
            contact_phone=contact_phone,
            category=category,
            support_tier=support_tier,
            first_response_due=first_response_due,
            resolution_due=resolution_due
        )
        
        # Store ticket
        self.support_tickets[ticket_id] = ticket
        
        # Auto-assign based on category and tier
        await self._auto_assign_ticket(ticket)
        
        # Store in memory
        await self._store_support_ticket(ticket)
        
        # Record metrics
        self.metrics.increment("support_ticket_created")
        self.metrics.increment(f"support_ticket_priority_{priority.value}")
        self.metrics.increment(f"support_ticket_tier_{support_tier.value}")
        
        self.logger.info(f"Created support ticket {ticket_id} for customer {customer_id}: {title}")
        
        return ticket
    
    async def update_ticket_status(
        self,
        ticket_id: str,
        new_status: TicketStatus,
        update_message: str,
        updated_by: str
    ) -> bool:
        """Update support ticket status"""
        
        if ticket_id not in self.support_tickets:
            return False
        
        ticket = self.support_tickets[ticket_id]
        old_status = ticket.status
        
        # Update status
        ticket.status = new_status
        ticket.updated_at = datetime.utcnow()
        
        # Add update to history
        update_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "updated_by": updated_by,
            "old_status": old_status.value,
            "new_status": new_status.value,
            "message": update_message
        }
        ticket.updates.append(update_entry)
        
        # Handle status-specific logic
        if new_status == TicketStatus.RESOLVED:
            ticket.resolved_at = datetime.utcnow()
            if ticket.created_at:
                ticket.resolution_time_minutes = int(
                    (ticket.resolved_at - ticket.created_at).total_seconds() / 60
                )
        elif new_status == TicketStatus.CLOSED:
            ticket.closed_at = datetime.utcnow()
        
        # Check SLA compliance
        await self._check_sla_compliance(ticket)
        
        # Store updated ticket
        await self._store_support_ticket(ticket)
        
        # Record metrics
        self.metrics.increment(f"support_ticket_status_{new_status.value}")
        
        self.logger.info(f"Updated ticket {ticket_id} status: {old_status.value} -> {new_status.value}")
        
        return True
    
    async def create_incident(
        self,
        title: str,
        description: str,
        severity: IncidentSeverity,
        affected_services: List[str],
        incident_commander: str,
        affected_customers: int = 0
    ) -> Incident:
        """Create a new incident"""
        
        incident_id = f"incident_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        
        incident = Incident(
            incident_id=incident_id,
            title=title,
            description=description,
            severity=severity,
            affected_services=affected_services,
            affected_customers=affected_customers,
            incident_commander=incident_commander
        )
        
        # Store incident
        self.incidents[incident_id] = incident
        
        # Auto-escalate based on severity
        await self._auto_escalate_incident(incident)
        
        # Store in memory
        await self._store_incident(incident)
        
        # Record metrics
        self.metrics.increment("incident_created")
        self.metrics.increment(f"incident_severity_{severity.value}")
        self.metrics.record_metric("incident_affected_customers", affected_customers)
        
        self.logger.warning(f"Created incident {incident_id}: {title} (Severity: {severity.value})")
        
        return incident
    
    async def execute_rollback(
        self,
        plan_id: str,
        executed_by: str,
        approval_override: bool = False
    ) -> Dict[str, Any]:
        """Execute a rollback plan"""
        
        if plan_id not in self.rollback_plans:
            raise ValueError(f"Rollback plan not found: {plan_id}")
        
        plan = self.rollback_plans[plan_id]
        
        # Check approval requirements
        if plan.requires_approval and not plan.approved_by and not approval_override:
            raise ValueError("Rollback plan requires approval before execution")
        
        # Check rollback window
        now = datetime.utcnow()
        if plan.rollback_window_start and plan.rollback_window_end:
            if not (plan.rollback_window_start <= now <= plan.rollback_window_end):
                raise ValueError("Current time is outside the approved rollback window")
        
        execution_start = datetime.utcnow()
        execution_log = []
        
        try:
            # Execute pre-rollback steps
            execution_log.append(f"Starting rollback execution at {execution_start.isoformat()}")
            
            for i, step in enumerate(plan.pre_rollback_steps, 1):
                execution_log.append(f"Pre-rollback step {i}: {step}")
                # In a real implementation, this would execute the actual step
                await asyncio.sleep(0.1)  # Simulate execution time
            
            # Execute rollback steps
            for i, step in enumerate(plan.rollback_steps, 1):
                execution_log.append(f"Rollback step {i}: {step}")
                # In a real implementation, this would execute the actual rollback
                await asyncio.sleep(0.1)  # Simulate execution time
            
            # Execute post-rollback steps
            for i, step in enumerate(plan.post_rollback_steps, 1):
                execution_log.append(f"Post-rollback step {i}: {step}")
                # In a real implementation, this would execute the actual step
                await asyncio.sleep(0.1)  # Simulate execution time
            
            # Execute validation steps
            validation_results = []
            for i, step in enumerate(plan.validation_steps, 1):
                execution_log.append(f"Validation step {i}: {step}")
                # In a real implementation, this would perform actual validation
                validation_results.append({"step": step, "result": "passed"})
            
            # Update plan status
            plan.executed = True
            plan.executed_at = datetime.utcnow()
            plan.executed_by = executed_by
            plan.execution_log = execution_log
            
            execution_duration = (datetime.utcnow() - execution_start).total_seconds()
            
            # Store updated plan
            await self._store_rollback_plan(plan)
            
            # Record metrics
            self.metrics.increment("rollback_executed")
            self.metrics.increment(f"rollback_type_{plan.rollback_type.value}")
            self.metrics.record_metric("rollback_duration_seconds", execution_duration)
            
            result = {
                "success": True,
                "plan_id": plan_id,
                "execution_duration_seconds": execution_duration,
                "steps_executed": len(plan.pre_rollback_steps) + len(plan.rollback_steps) + len(plan.post_rollback_steps),
                "validation_results": validation_results,
                "execution_log": execution_log
            }
            
            self.logger.info(f"Successfully executed rollback plan {plan_id} in {execution_duration:.2f}s")
            
            return result
            
        except Exception as e:
            execution_log.append(f"Rollback execution failed: {str(e)}")
            plan.execution_log = execution_log
            
            self.logger.error(f"Rollback execution failed for plan {plan_id}: {e}")
            
            return {
                "success": False,
                "plan_id": plan_id,
                "error": str(e),
                "execution_log": execution_log
            }
    
    async def start_customer_onboarding(
        self,
        customer_id: str,
        plan_type: str,
        support_tier: SupportTier,
        customer_success_manager: Optional[str] = None
    ) -> CustomerOnboarding:
        """Start customer onboarding process"""
        
        onboarding_id = f"onboarding_{customer_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Define onboarding milestones based on plan type
        milestones = self._get_onboarding_milestones(plan_type, support_tier)
        
        onboarding = CustomerOnboarding(
            onboarding_id=onboarding_id,
            customer_id=customer_id,
            plan_type=plan_type,
            onboarding_tier=support_tier,
            total_steps=len(milestones),
            customer_success_manager=customer_success_manager,
            milestones_pending=milestones,
            target_completion_date=datetime.utcnow() + timedelta(days=30)  # 30-day onboarding
        )
        
        # Store onboarding
        self.customer_onboardings[onboarding_id] = onboarding
        
        # Store in memory
        await self._store_customer_onboarding(onboarding)
        
        # Record metrics
        self.metrics.increment("customer_onboarding_started")
        self.metrics.increment(f"onboarding_plan_{plan_type}")
        
        self.logger.info(f"Started onboarding for customer {customer_id}: {plan_type}")
        
        return onboarding
    
    def get_support_dashboard(self) -> Dict[str, Any]:
        """Get support system dashboard data"""
        
        now = datetime.utcnow()
        
        # Ticket statistics
        open_tickets = [t for t in self.support_tickets.values() if t.status != TicketStatus.CLOSED]
        overdue_tickets = [
            t for t in open_tickets
            if t.resolution_due and t.resolution_due < now
        ]
        
        # SLA compliance
        resolved_tickets = [
            t for t in self.support_tickets.values()
            if t.status == TicketStatus.RESOLVED and t.resolved_at
        ]
        
        sla_compliant_tickets = [t for t in resolved_tickets if not t.sla_breached]
        sla_compliance_rate = (len(sla_compliant_tickets) / len(resolved_tickets) * 100) if resolved_tickets else 100
        
        # Average resolution time
        resolution_times = [t.resolution_time_minutes for t in resolved_tickets if t.resolution_time_minutes]
        avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
        
        # Incident statistics
        active_incidents = [i for i in self.incidents.values() if i.status != "resolved"]
        critical_incidents = [i for i in active_incidents if i.severity in [IncidentSeverity.SEV1, IncidentSeverity.SEV2]]
        
        # Onboarding statistics
        active_onboardings = [o for o in self.customer_onboardings.values() if not o.completed_at]
        at_risk_onboardings = [
            o for o in active_onboardings
            if o.health_score < 70 or (o.target_completion_date and o.target_completion_date < now)
        ]
        
        return {
            "tickets": {
                "total_open": len(open_tickets),
                "overdue": len(overdue_tickets),
                "sla_compliance_rate": sla_compliance_rate,
                "average_resolution_time_hours": avg_resolution_time / 60 if avg_resolution_time else 0,
                "by_priority": {
                    priority.value: len([t for t in open_tickets if t.priority == priority])
                    for priority in TicketPriority
                },
                "by_tier": {
                    tier.value: len([t for t in open_tickets if t.support_tier == tier])
                    for tier in SupportTier
                }
            },
            "incidents": {
                "total_active": len(active_incidents),
                "critical": len(critical_incidents),
                "by_severity": {
                    severity.value: len([i for i in active_incidents if i.severity == severity])
                    for severity in IncidentSeverity
                }
            },
            "onboarding": {
                "active_customers": len(active_onboardings),
                "at_risk": len(at_risk_onboardings),
                "average_health_score": sum(o.health_score for o in active_onboardings) / len(active_onboardings) if active_onboardings else 100
            },
            "rollbacks": {
                "plans_available": len(self.rollback_plans),
                "executed_today": len([
                    p for p in self.rollback_plans.values()
                    if p.executed_at and p.executed_at.date() == now.date()
                ])
            }
        }
    
    # Private helper methods
    def _initialize_default_rollback_plans(self) -> None:
        """Initialize default rollback plans"""
        
        default_plans = [
            RollbackPlan(
                plan_id="app_rollback_v1",
                name="Application Rollback to Previous Version",
                description="Rollback application deployment to the previous stable version",
                rollback_type=RollbackType.APPLICATION,
                target_version="v1.0.0",
                current_version="v1.1.0",
                affected_components=["web_app", "api_server", "worker_processes"],
                pre_rollback_steps=[
                    "Notify customers of maintenance window",
                    "Scale up backup instances",
                    "Create database backup"
                ],
                rollback_steps=[
                    "Stop current application instances",
                    "Deploy previous version containers",
                    "Update load balancer configuration",
                    "Start application instances with previous version"
                ],
                post_rollback_steps=[
                    "Verify application health checks",
                    "Run smoke tests",
                    "Monitor error rates and performance",
                    "Notify customers of completion"
                ],
                validation_steps=[
                    "Check application health endpoints",
                    "Verify database connectivity",
                    "Test critical user workflows",
                    "Confirm monitoring alerts are clear"
                ],
                success_criteria=[
                    "All health checks passing",
                    "Error rate < 0.1%",
                    "Response time < 200ms",
                    "No critical alerts"
                ],
                estimated_downtime_minutes=15
            ),
            RollbackPlan(
                plan_id="db_rollback_v1",
                name="Database Schema Rollback",
                description="Rollback database schema changes to previous version",
                rollback_type=RollbackType.DATABASE,
                target_version="schema_v1.0",
                current_version="schema_v1.1",
                affected_components=["primary_db", "read_replicas"],
                pre_rollback_steps=[
                    "Put application in maintenance mode",
                    "Create full database backup",
                    "Verify backup integrity"
                ],
                rollback_steps=[
                    "Stop application database connections",
                    "Execute rollback migration scripts",
                    "Update database version metadata",
                    "Restart read replicas"
                ],
                post_rollback_steps=[
                    "Verify schema integrity",
                    "Test database connectivity",
                    "Resume application connections",
                    "Exit maintenance mode"
                ],
                validation_steps=[
                    "Run database integrity checks",
                    "Verify all tables and indexes",
                    "Test application database queries",
                    "Check replication lag"
                ],
                success_criteria=[
                    "Schema matches target version",
                    "All integrity checks pass",
                    "Replication lag < 1 second",
                    "Application connects successfully"
                ],
                estimated_downtime_minutes=30,
                requires_approval=True
            )
        ]
        
        for plan in default_plans:
            self.rollback_plans[plan.plan_id] = plan
        
        self.logger.info(f"Initialized {len(default_plans)} default rollback plans")
    
    async def _auto_assign_ticket(self, ticket: SupportTicket) -> None:
        """Auto-assign ticket based on category and tier"""
        
        # Simple assignment logic - in production this would be more sophisticated
        assignment_rules = {
            "technical": "tech_support_team",
            "billing": "billing_team",
            "feature_request": "product_team",
            "security": "security_team"
        }
        
        assigned_to = assignment_rules.get(ticket.category, "general_support")
        
        # Escalate high priority tickets
        if ticket.priority in [TicketPriority.CRITICAL, TicketPriority.EMERGENCY]:
            assigned_to = "senior_support_team"
        
        ticket.assigned_to = assigned_to
        
        self.logger.debug(f"Auto-assigned ticket {ticket.ticket_id} to {assigned_to}")
    
    async def _auto_escalate_incident(self, incident: Incident) -> None:
        """Auto-escalate incident based on severity"""
        
        if incident.severity == IncidentSeverity.SEV1:
            # Page on-call engineer immediately
            incident.response_team.extend(["on_call_engineer", "engineering_manager", "cto"])
        elif incident.severity == IncidentSeverity.SEV2:
            # Notify engineering team
            incident.response_team.extend(["on_call_engineer", "engineering_manager"])
        
        self.logger.info(f"Auto-escalated incident {incident.incident_id} with {len(incident.response_team)} team members")
    
    async def _check_sla_compliance(self, ticket: SupportTicket) -> None:
        """Check and update SLA compliance for ticket"""
        
        now = datetime.utcnow()
        
        # Check first response SLA
        if ticket.first_response_due and now > ticket.first_response_due and not ticket.updates:
            ticket.sla_breached = True
        
        # Check resolution SLA
        if ticket.resolution_due and now > ticket.resolution_due and ticket.status != TicketStatus.RESOLVED:
            ticket.sla_breached = True
        
        if ticket.sla_breached:
            self.metrics.increment("support_sla_breach")
            self.logger.warning(f"SLA breached for ticket {ticket.ticket_id}")
    
    def _get_onboarding_milestones(self, plan_type: str, support_tier: SupportTier) -> List[str]:
        """Get onboarding milestones based on plan type and tier"""
        
        base_milestones = [
            "Account setup and verification",
            "Initial platform tour",
            "First AI employee creation",
            "Basic workflow configuration",
            "Integration setup",
            "Team member invitations",
            "Security configuration",
            "First successful automation",
            "Performance monitoring setup",
            "Onboarding completion review"
        ]
        
        if support_tier in [SupportTier.PREMIUM, SupportTier.ENTERPRISE]:
            base_milestones.extend([
                "Dedicated success manager introduction",
                "Custom workflow design session",
                "Advanced feature training",
                "Compliance setup (if applicable)",
                "Custom integration development"
            ])
        
        return base_milestones
    
    # Storage methods
    async def _store_support_ticket(self, ticket: SupportTicket) -> None:
        """Store support ticket in memory"""
        
        await self.memory_manager.store_context(
            context_type="support_ticket",
            content=ticket.__dict__,
            metadata={
                "ticket_id": ticket.ticket_id,
                "customer_id": ticket.customer_id,
                "priority": ticket.priority.value,
                "status": ticket.status.value,
                "support_tier": ticket.support_tier.value
            }
        )
    
    async def _store_incident(self, incident: Incident) -> None:
        """Store incident in memory"""
        
        await self.memory_manager.store_context(
            context_type="support_incident",
            content=incident.__dict__,
            metadata={
                "incident_id": incident.incident_id,
                "severity": incident.severity.value,
                "status": incident.status,
                "affected_customers": incident.affected_customers
            }
        )
    
    async def _store_rollback_plan(self, plan: RollbackPlan) -> None:
        """Store rollback plan in memory"""
        
        await self.memory_manager.store_context(
            context_type="support_rollback_plan",
            content=plan.__dict__,
            metadata={
                "plan_id": plan.plan_id,
                "rollback_type": plan.rollback_type.value,
                "executed": plan.executed,
                "risk_level": plan.risk_level
            }
        )
    
    async def _store_customer_onboarding(self, onboarding: CustomerOnboarding) -> None:
        """Store customer onboarding in memory"""
        
        await self.memory_manager.store_context(
            context_type="support_customer_onboarding",
            content=onboarding.__dict__,
            metadata={
                "onboarding_id": onboarding.onboarding_id,
                "customer_id": onboarding.customer_id,
                "completion_percentage": onboarding.completion_percentage,
                "health_score": onboarding.health_score
            }
        )