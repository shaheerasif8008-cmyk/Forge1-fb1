"""
Vertical AI Employee Registry
Central registry for all vertical AI employees with factory pattern
"""

from typing import Dict, Type, Optional, Any
from enum import Enum

from forge1.backend.forge1.core.agent_base import AgentBase
from forge1.backend.forge1.core.orchestration import WorkflowEngine
from forge1.backend.forge1.core.memory import MemoryManager
from forge1.backend.forge1.core.models import ModelRouter
from forge1.backend.forge1.core.monitoring import MetricsCollector
from forge1.backend.forge1.core.security import SecretManager

from forge1.backend.forge1.verticals.cx.employee import CXAIEmployee
from forge1.backend.forge1.verticals.revops.employee import RevOpsAIEmployee
from forge1.backend.forge1.verticals.finance.employee import FinanceAIEmployee
from forge1.backend.forge1.verticals.legal.employee import LegalAIEmployee
from forge1.backend.forge1.verticals.itops.employee import ITOpsAIEmployee
from forge1.backend.forge1.verticals.engineering.employee import EngineeringAIEmployee


class VerticalType(Enum):
    """Supported vertical AI employee types"""
    CUSTOMER_EXPERIENCE = "cx"
    REVENUE_OPERATIONS = "revops"
    FINANCE_FPA = "finance"
    LEGAL = "legal"
    IT_OPERATIONS = "itops"
    SOFTWARE_ENGINEERING = "engineering"


class VerticalEmployeeRegistry:
    """
    Registry for managing vertical AI employees
    
    Provides factory methods for creating and managing specialized AI employees
    across different business functions with consistent interfaces and monitoring.
    """
    
    def __init__(self):
        self._employee_classes: Dict[VerticalType, Type[AgentBase]] = {
            VerticalType.CUSTOMER_EXPERIENCE: CXAIEmployee,
            VerticalType.REVENUE_OPERATIONS: RevOpsAIEmployee,
            VerticalType.FINANCE_FPA: FinanceAIEmployee,
            VerticalType.LEGAL: LegalAIEmployee,
            VerticalType.IT_OPERATIONS: ITOpsAIEmployee,
            VerticalType.SOFTWARE_ENGINEERING: EngineeringAIEmployee
        }
        
        self._active_employees: Dict[str, AgentBase] = {}
    
    def create_employee(
        self,
        vertical_type: VerticalType,
        employee_id: str,
        tenant_id: str,
        workflow_engine: WorkflowEngine,
        memory_manager: MemoryManager,
        model_router: ModelRouter,
        metrics_collector: MetricsCollector,
        secret_manager: SecretManager
    ) -> AgentBase:
        """
        Create a new vertical AI employee instance
        
        Args:
            vertical_type: Type of vertical employee to create
            employee_id: Unique identifier for the employee
            tenant_id: Tenant identifier for multi-tenancy
            workflow_engine: Workflow orchestration engine
            memory_manager: Memory management system
            model_router: AI model routing system
            metrics_collector: Performance metrics collector
            secret_manager: Secrets management system
        
        Returns:
            Configured AI employee instance
        """
        if vertical_type not in self._employee_classes:
            raise ValueError(f"Unsupported vertical type: {vertical_type}")
        
        employee_class = self._employee_classes[vertical_type]
        
        employee = employee_class(
            employee_id=employee_id,
            tenant_id=tenant_id,
            workflow_engine=workflow_engine,
            memory_manager=memory_manager,
            model_router=model_router,
            metrics_collector=metrics_collector,
            secret_manager=secret_manager
        )
        
        # Register the employee
        self._active_employees[employee_id] = employee
        
        return employee
    
    def get_employee(self, employee_id: str) -> Optional[AgentBase]:
        """Get an active employee by ID"""
        return self._active_employees.get(employee_id)
    
    def list_employees(self, vertical_type: Optional[VerticalType] = None) -> Dict[str, AgentBase]:
        """List active employees, optionally filtered by vertical type"""
        if vertical_type is None:
            return self._active_employees.copy()
        
        filtered = {}
        target_class = self._employee_classes[vertical_type]
        
        for emp_id, employee in self._active_employees.items():
            if isinstance(employee, target_class):
                filtered[emp_id] = employee
        
        return filtered
    
    def remove_employee(self, employee_id: str) -> bool:
        """Remove an employee from the registry"""
        if employee_id in self._active_employees:
            del self._active_employees[employee_id]
            return True
        return False
    
    def get_supported_verticals(self) -> List[VerticalType]:
        """Get list of supported vertical types"""
        return list(self._employee_classes.keys())
    
    def get_vertical_capabilities(self, vertical_type: VerticalType) -> Dict[str, Any]:
        """Get capabilities description for a vertical type"""
        capabilities = {
            VerticalType.CUSTOMER_EXPERIENCE: {
                "name": "Customer Experience (CX)",
                "description": "Intelligent customer support with 95%+ deflection rate",
                "key_features": [
                    "Sub-5 second first response time",
                    "Intelligent ticket triage and prioritization",
                    "Automated resolution with escalation management",
                    "Upselling opportunity detection",
                    "Real-time performance monitoring"
                ],
                "integrations": ["Salesforce", "Zendesk", "ServiceNow", "n8n", "Zapier"],
                "sla_targets": {
                    "first_response_time": "< 5 seconds",
                    "deflection_rate": "> 95%",
                    "customer_satisfaction": "> 4.0/5.0",
                    "escalation_rate": "< 5%"
                }
            },
            VerticalType.REVENUE_OPERATIONS: {
                "name": "Revenue Operations (RevOps)",
                "description": "Pipeline hygiene, forecasting, and quote optimization",
                "key_features": [
                    "Pipeline hygiene monitoring and cleanup",
                    "Sales forecasting with accuracy tracking",
                    "Quote optimization and approval workflows",
                    "Renewal motion analysis",
                    "Performance analytics and reporting"
                ],
                "integrations": ["Salesforce", "HubSpot", "Power BI", "Looker"],
                "sla_targets": {
                    "pipeline_hygiene": "> 90%",
                    "forecast_accuracy": "> 85%",
                    "quote_turnaround": "< 24 hours"
                }
            },
            VerticalType.FINANCE_FPA: {
                "name": "Finance & Financial Planning & Analysis",
                "description": "Close assistance, variance analysis, and budget planning",
                "key_features": [
                    "Month-end close automation",
                    "Variance analysis with root cause identification",
                    "Budget planning and forecasting",
                    "Financial reporting and analytics",
                    "Compliance monitoring"
                ],
                "integrations": ["SAP", "Oracle", "NetSuite", "Excel", "Power BI"],
                "sla_targets": {
                    "close_cycle_time": "< 5 days",
                    "variance_accuracy": "> 95%",
                    "budget_accuracy": "> 90%"
                }
            },
            VerticalType.LEGAL: {
                "name": "Legal Operations",
                "description": "Contract management, risk analysis, and compliance",
                "key_features": [
                    "Contract lifecycle management",
                    "NDA/MSA/SOW templating",
                    "Clause extraction and risk scoring",
                    "Negotiation assistance with human oversight",
                    "Compliance monitoring"
                ],
                "integrations": ["Icertis", "Conga", "DocuSign", "Legal databases"],
                "sla_targets": {
                    "contract_cycle_time": "< 7 days",
                    "risk_scoring_precision": "> 90%",
                    "attorney_approval_rate": "> 95%"
                }
            },
            VerticalType.IT_OPERATIONS: {
                "name": "IT Operations & Security Operations",
                "description": "Incident management, provisioning, and security monitoring",
                "key_features": [
                    "Incident triage and automated resolution",
                    "User provisioning and access management",
                    "Vulnerability assessment and patch management",
                    "Security monitoring and threat detection",
                    "Emergency response and escalation"
                ],
                "integrations": ["ServiceNow", "Jira", "SIEM/SOAR", "Active Directory"],
                "sla_targets": {
                    "mttr": "< 30 minutes",
                    "false_positive_rate": "< 5%",
                    "automation_rate": "> 75%"
                }
            },
            VerticalType.SOFTWARE_ENGINEERING: {
                "name": "Software Engineering",
                "description": "Code generation, PR reviews, and testing automation",
                "key_features": [
                    "Specification to code generation",
                    "Pull request reviews and feedback",
                    "Automated test generation",
                    "Release notes and documentation",
                    "Code quality analysis"
                ],
                "integrations": ["GitHub", "GitLab", "CI/CD", "Issue trackers"],
                "sla_targets": {
                    "pr_throughput": "> 5 per day",
                    "defect_escape_rate": "< 2%",
                    "build_stability": "> 95%"
                }
            }
        }
        
        return capabilities.get(vertical_type, {})


# Global registry instance
vertical_registry = VerticalEmployeeRegistry()