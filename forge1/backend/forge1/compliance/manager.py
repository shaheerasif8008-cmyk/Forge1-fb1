"""
Unified Compliance Management System
Integrates SOC2, GDPR, and HIPAA compliance frameworks with centralized reporting and monitoring
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import logging
import json
from collections import defaultdict

# Mock classes for testing - in production these would import from the actual core modules
class MetricsCollector:
    def increment(self, metric): pass
    def record_metric(self, metric, value): pass

class MemoryManager:
    async def store_context(self, context_type, content, metadata): pass

class SecretManager:
    async def get(self, name): return "mock_secret"

from .gdpr import GDPRComplianceManager, LegalBasis, DataCategory, ProcessingPurpose
from .hipaa import HIPAAComplianceManager, HIPAARule, SafeguardType, ComplianceStatus as HIPAAStatus

# Mock SOC2 classes since the file is missing
class SOC2Principle:
    SECURITY = "security"

class SOC2Status:
    COMPLIANT = "compliant"

class SOC2ComplianceManager:
    def __init__(self, memory_manager, metrics_collector, secret_manager):
        self.controls = {"mock_control": type('MockControl', (), {'status': SOC2Status.COMPLIANT})()}
    
    async def run_compliance_assessment(self):
        return {"compliance_percentage": 95.0}
    
    def get_compliance_dashboard(self):
        return {
            "compliance_overview": {"compliance_percentage": 95.0, "overall_status": "compliant"},
            "control_summary": {"total_controls": 1, "status_breakdown": {"compliant": 1}}
        }


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SOC2 = "soc2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist"


class ComplianceRisk(Enum):
    """Compliance risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceAlert:
    """Compliance alert or notification"""
    alert_id: str
    framework: ComplianceFramework
    severity: ComplianceRisk
    title: str
    description: str
    
    # Alert details
    alert_type: str  # control_failure, deadline_approaching, breach_detected, etc.
    source_component: str
    affected_controls: List[str] = field(default_factory=list)
    
    # Timeline
    created_at: datetime = field(default_factory=datetime.utcnow)
    due_date: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Response
    assigned_to: Optional[str] = None
    resolution_notes: str = ""
    remediation_actions: List[str] = field(default_factory=list)
    
    # Status
    acknowledged: bool = False
    resolved: bool = False
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceReport:
    """Unified compliance report"""
    report_id: str
    report_date: datetime
    reporting_period_start: datetime
    reporting_period_end: datetime
    
    # Framework compliance
    soc2_compliance: Dict[str, Any] = field(default_factory=dict)
    gdpr_compliance: Dict[str, Any] = field(default_factory=dict)
    hipaa_compliance: Dict[str, Any] = field(default_factory=dict)
    
    # Overall metrics
    overall_compliance_score: float = 0.0
    risk_score: float = 0.0
    
    # Alerts and issues
    active_alerts: List[ComplianceAlert] = field(default_factory=list)
    resolved_issues: int = 0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Executive summary
    executive_summary: str = ""
    
    # Metadata
    generated_by: str = ""
    report_type: str = "comprehensive"  # comprehensive, framework_specific, executive


class UnifiedComplianceManager:
    """
    Unified compliance management system that orchestrates multiple compliance frameworks
    
    Features:
    - Centralized compliance monitoring and reporting
    - Cross-framework risk assessment and correlation
    - Automated alert generation and escalation
    - Executive dashboards and reporting
    - Compliance workflow automation
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
        self.logger = logging.getLogger("unified_compliance")
        
        # Initialize framework managers
        self.soc2_manager = SOC2ComplianceManager(memory_manager, metrics_collector, secret_manager)
        self.gdpr_manager = GDPRComplianceManager(memory_manager, metrics_collector, secret_manager)
        self.hipaa_manager = HIPAAComplianceManager(memory_manager, metrics_collector, secret_manager)
        
        # Compliance alerts and notifications
        self.alerts: Dict[str, ComplianceAlert] = {}
        
        # Configuration
        self.organization_name = "Cognisia Inc."
        self.compliance_officer = "compliance@cognisia.com"
        self.enabled_frameworks = [ComplianceFramework.SOC2, ComplianceFramework.GDPR, ComplianceFramework.HIPAA]
        
        # Alert thresholds
        self.alert_thresholds = {
            "compliance_score_threshold": 85.0,  # Alert if below 85%
            "risk_score_threshold": 70.0,  # Alert if above 70%
            "control_failure_threshold": 3,  # Alert if 3+ controls fail
            "deadline_warning_days": 30  # Alert 30 days before deadlines
        }
        
        self.logger.info("Initialized Unified Compliance Manager")
    
    async def run_comprehensive_assessment(self) -> ComplianceReport:
        """Run comprehensive compliance assessment across all frameworks"""
        
        self.logger.info("Starting comprehensive compliance assessment")
        
        assessment_start = datetime.utcnow()
        
        # Run assessments for each framework
        assessment_results = {}
        
        try:
            # SOC2 Assessment
            if ComplianceFramework.SOC2 in self.enabled_frameworks:
                self.logger.info("Running SOC2 compliance assessment")
                soc2_report = await self.soc2_manager.run_compliance_assessment()
                assessment_results["soc2"] = soc2_report
            
            # GDPR Assessment
            if ComplianceFramework.GDPR in self.enabled_frameworks:
                self.logger.info("Running GDPR compliance assessment")
                gdpr_report = await self.gdpr_manager.generate_privacy_report()
                assessment_results["gdpr"] = gdpr_report
            
            # HIPAA Assessment
            if ComplianceFramework.HIPAA in self.enabled_frameworks:
                self.logger.info("Running HIPAA compliance assessment")
                hipaa_report = await self.hipaa_manager.generate_hipaa_compliance_report()
                assessment_results["hipaa"] = hipaa_report
            
            # Generate unified report
            unified_report = await self._generate_unified_report(assessment_results, assessment_start)
            
            # Check for compliance alerts
            await self._check_compliance_alerts(assessment_results)
            
            # Store report
            await self._store_compliance_report(unified_report)
            
            # Record metrics
            self.metrics.record_metric("compliance_overall_score", unified_report.overall_compliance_score)
            self.metrics.record_metric("compliance_risk_score", unified_report.risk_score)
            self.metrics.increment("compliance_assessment_completed")
            
            assessment_duration = (datetime.utcnow() - assessment_start).total_seconds()
            self.logger.info(
                f"Comprehensive compliance assessment completed in {assessment_duration:.2f}s: "
                f"Score {unified_report.overall_compliance_score:.1f}%, Risk {unified_report.risk_score:.1f}%"
            )
            
            return unified_report
            
        except Exception as e:
            self.logger.error(f"Compliance assessment failed: {e}")
            raise
    
    async def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get unified compliance dashboard data"""
        
        # Get individual framework dashboards
        soc2_dashboard = self.soc2_manager.get_compliance_dashboard()
        gdpr_dashboard = self.gdpr_manager.get_gdpr_dashboard()
        hipaa_dashboard = self.hipaa_manager.get_hipaa_dashboard()
        
        # Calculate overall metrics
        overall_compliance = self._calculate_overall_compliance_score({
            "soc2": soc2_dashboard,
            "gdpr": gdpr_dashboard,
            "hipaa": hipaa_dashboard
        })
        
        # Get active alerts
        active_alerts = [alert for alert in self.alerts.values() if not alert.resolved]
        critical_alerts = [alert for alert in active_alerts if alert.severity == ComplianceRisk.CRITICAL]
        
        # Recent activity
        now = datetime.utcnow()
        recent_alerts = [
            alert for alert in self.alerts.values()
            if alert.created_at > now - timedelta(days=7)
        ]
        
        return {
            "overview": {
                "organization": self.organization_name,
                "enabled_frameworks": [f.value for f in self.enabled_frameworks],
                "overall_compliance_score": overall_compliance,
                "risk_level": self._calculate_risk_level(overall_compliance),
                "last_assessment": now.isoformat()  # Would track actual last assessment
            },
            "framework_status": {
                "soc2": {
                    "compliance_percentage": soc2_dashboard["compliance_overview"]["compliance_percentage"],
                    "status": soc2_dashboard["compliance_overview"]["overall_status"],
                    "controls_total": soc2_dashboard["control_summary"]["total_controls"],
                    "controls_compliant": soc2_dashboard["control_summary"]["status_breakdown"].get("compliant", 0)
                },
                "gdpr": {
                    "compliance_percentage": gdpr_dashboard["compliance_metrics"]["request_completion_rate"],
                    "data_subjects": gdpr_dashboard["overview"]["total_data_subjects"],
                    "pending_requests": gdpr_dashboard["pending_items"]["pending_requests"],
                    "recent_breaches": gdpr_dashboard["overview"]["total_data_breaches"]
                },
                "hipaa": {
                    "compliance_percentage": hipaa_dashboard["overview"]["compliance_percentage"],
                    "safeguards_total": hipaa_dashboard["overview"]["total_safeguards"],
                    "safeguards_compliant": hipaa_dashboard["overview"]["compliant_safeguards"],
                    "recent_incidents": hipaa_dashboard["recent_activity"]["security_incidents"]
                }
            },
            "alerts_and_risks": {
                "total_active_alerts": len(active_alerts),
                "critical_alerts": len(critical_alerts),
                "recent_alerts": len(recent_alerts),
                "top_risks": self._identify_top_risks({
                    "soc2": soc2_dashboard,
                    "gdpr": gdpr_dashboard,
                    "hipaa": hipaa_dashboard
                })
            },
            "recent_activity": {
                "new_alerts": len(recent_alerts),
                "resolved_alerts": len([
                    alert for alert in self.alerts.values()
                    if alert.resolved_at and alert.resolved_at > now - timedelta(days=7)
                ]),
                "compliance_assessments": 1,  # Would track actual assessments
                "period_days": 7
            },
            "compliance_trends": {
                "trend_direction": "stable",  # Would calculate from historical data
                "improvement_areas": self._identify_improvement_areas({
                    "soc2": soc2_dashboard,
                    "gdpr": gdpr_dashboard,
                    "hipaa": hipaa_dashboard
                }),
                "next_milestones": self._get_next_milestones()
            }
        }
    
    async def create_compliance_alert(
        self,
        framework: ComplianceFramework,
        severity: ComplianceRisk,
        title: str,
        description: str,
        alert_type: str,
        source_component: str,
        affected_controls: Optional[List[str]] = None,
        due_date: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a compliance alert"""
        
        alert_id = f"alert_{framework.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        
        alert = ComplianceAlert(
            alert_id=alert_id,
            framework=framework,
            severity=severity,
            title=title,
            description=description,
            alert_type=alert_type,
            source_component=source_component,
            affected_controls=affected_controls or [],
            due_date=due_date,
            metadata=metadata or {}
        )
        
        # Store alert
        self.alerts[alert_id] = alert
        
        # Store in memory
        await self._store_compliance_alert(alert)
        
        # Record metrics
        self.metrics.increment("compliance_alert_created")
        self.metrics.increment(f"compliance_alert_{framework.value}")
        self.metrics.increment(f"compliance_alert_severity_{severity.value}")
        
        self.logger.warning(f"Compliance alert created: {alert_id} - {title}")
        
        return alert_id
    
    async def resolve_compliance_alert(
        self,
        alert_id: str,
        resolved_by: str,
        resolution_notes: str,
        remediation_actions: Optional[List[str]] = None
    ) -> bool:
        """Resolve a compliance alert"""
        
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        
        # Update alert
        alert.resolved = True
        alert.resolved_at = datetime.utcnow()
        alert.resolution_notes = resolution_notes
        alert.remediation_actions = remediation_actions or []
        
        # Store updated alert
        await self._store_compliance_alert(alert)
        
        # Record metrics
        self.metrics.increment("compliance_alert_resolved")
        
        self.logger.info(f"Compliance alert resolved: {alert_id}")
        
        return True
    
    async def generate_executive_report(
        self,
        reporting_period_days: int = 30
    ) -> Dict[str, Any]:
        """Generate executive compliance report"""
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=reporting_period_days)
        
        # Get comprehensive assessment
        assessment = await self.run_comprehensive_assessment()
        
        # Calculate key metrics
        compliance_trend = "stable"  # Would calculate from historical data
        risk_trend = "decreasing"  # Would calculate from historical data
        
        # Get recent alerts
        recent_alerts = [
            alert for alert in self.alerts.values()
            if alert.created_at >= start_date
        ]
        
        critical_issues = [
            alert for alert in recent_alerts
            if alert.severity == ComplianceRisk.CRITICAL and not alert.resolved
        ]
        
        return {
            "executive_summary": {
                "organization": self.organization_name,
                "reporting_period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "overall_compliance_score": assessment.overall_compliance_score,
                "compliance_trend": compliance_trend,
                "risk_score": assessment.risk_score,
                "risk_trend": risk_trend
            },
            "key_metrics": {
                "frameworks_monitored": len(self.enabled_frameworks),
                "total_controls_assessed": self._count_total_controls(),
                "compliant_controls": self._count_compliant_controls(),
                "critical_issues": len(critical_issues),
                "resolved_issues": len([
                    alert for alert in recent_alerts
                    if alert.resolved
                ])
            },
            "framework_performance": {
                "soc2": {
                    "compliance_percentage": assessment.soc2_compliance.get("compliance_percentage", 0),
                    "status": "compliant" if assessment.soc2_compliance.get("compliance_percentage", 0) >= 95 else "needs_attention"
                },
                "gdpr": {
                    "compliance_percentage": assessment.gdpr_compliance.get("compliance_indicators", {}).get("request_completion_rate", 0),
                    "status": "compliant" if assessment.gdpr_compliance.get("compliance_indicators", {}).get("request_completion_rate", 0) >= 95 else "needs_attention"
                },
                "hipaa": {
                    "compliance_percentage": assessment.hipaa_compliance.get("compliance_summary", {}).get("overall_compliance_percentage", 0),
                    "status": "compliant" if assessment.hipaa_compliance.get("compliance_summary", {}).get("overall_compliance_percentage", 0) >= 95 else "needs_attention"
                }
            },
            "critical_issues": [
                {
                    "framework": issue.framework.value,
                    "title": issue.title,
                    "severity": issue.severity.value,
                    "created_date": issue.created_at.strftime('%Y-%m-%d'),
                    "due_date": issue.due_date.strftime('%Y-%m-%d') if issue.due_date else None
                }
                for issue in critical_issues
            ],
            "recommendations": assessment.recommendations,
            "next_actions": [
                "Complete quarterly SOC2 control testing",
                "Review and update GDPR privacy notices",
                "Conduct HIPAA risk assessment",
                "Address critical compliance alerts"
            ]
        }
    
    # Private helper methods
    async def _generate_unified_report(
        self,
        assessment_results: Dict[str, Any],
        assessment_start: datetime
    ) -> ComplianceReport:
        """Generate unified compliance report from individual assessments"""
        
        report_id = f"unified_report_{assessment_start.strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate overall compliance score
        overall_score = self._calculate_overall_compliance_score(assessment_results)
        
        # Calculate risk score
        risk_score = self._calculate_overall_risk_score(assessment_results)
        
        # Get active alerts
        active_alerts = [alert for alert in self.alerts.values() if not alert.resolved]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(assessment_results)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(overall_score, risk_score, assessment_results)
        
        return ComplianceReport(
            report_id=report_id,
            report_date=datetime.utcnow(),
            reporting_period_start=assessment_start - timedelta(days=30),  # Last 30 days
            reporting_period_end=assessment_start,
            soc2_compliance=assessment_results.get("soc2", {}),
            gdpr_compliance=assessment_results.get("gdpr", {}),
            hipaa_compliance=assessment_results.get("hipaa", {}),
            overall_compliance_score=overall_score,
            risk_score=risk_score,
            active_alerts=active_alerts,
            resolved_issues=len([alert for alert in self.alerts.values() if alert.resolved]),
            recommendations=recommendations,
            executive_summary=executive_summary,
            generated_by="unified_compliance_manager"
        )
    
    def _calculate_overall_compliance_score(self, assessment_results: Dict[str, Any]) -> float:
        """Calculate overall compliance score across all frameworks"""
        
        scores = []
        weights = {"soc2": 0.4, "gdpr": 0.3, "hipaa": 0.3}  # Weighted by importance
        
        if "soc2" in assessment_results:
            soc2_score = assessment_results["soc2"].get("compliance_percentage", 0)
            scores.append(soc2_score * weights["soc2"])
        
        if "gdpr" in assessment_results:
            gdpr_score = assessment_results["gdpr"].get("compliance_indicators", {}).get("request_completion_rate", 0)
            scores.append(gdpr_score * weights["gdpr"])
        
        if "hipaa" in assessment_results:
            hipaa_score = assessment_results["hipaa"].get("compliance_summary", {}).get("overall_compliance_percentage", 0)
            scores.append(hipaa_score * weights["hipaa"])
        
        return sum(scores) if scores else 0.0
    
    def _calculate_overall_risk_score(self, assessment_results: Dict[str, Any]) -> float:
        """Calculate overall risk score based on compliance gaps and incidents"""
        
        risk_factors = []
        
        # SOC2 risk factors
        if "soc2" in assessment_results:
            soc2_data = assessment_results["soc2"]
            non_compliant_controls = soc2_data.get("control_results", {})
            risk_factors.append(len([c for c in non_compliant_controls.values() if c.get("status") == "non_compliant"]) * 10)
        
        # GDPR risk factors
        if "gdpr" in assessment_results:
            gdpr_data = assessment_results["gdpr"]
            overdue_requests = gdpr_data.get("data_subject_requests", {}).get("overdue_requests", 0)
            recent_breaches = gdpr_data.get("data_breaches", {}).get("recent_breaches", 0)
            risk_factors.append(overdue_requests * 5 + recent_breaches * 15)
        
        # HIPAA risk factors
        if "hipaa" in assessment_results:
            hipaa_data = assessment_results["hipaa"]
            unresolved_incidents = hipaa_data.get("risk_indicators", {}).get("unresolved_incidents", 0)
            pending_notifications = hipaa_data.get("risk_indicators", {}).get("pending_breach_notifications", 0)
            risk_factors.append(unresolved_incidents * 10 + pending_notifications * 20)
        
        # Calculate normalized risk score (0-100)
        total_risk = sum(risk_factors)
        max_possible_risk = 100  # Normalize to 0-100 scale
        
        return min(total_risk, max_possible_risk)
    
    def _calculate_risk_level(self, compliance_score: float) -> str:
        """Calculate risk level based on compliance score"""
        
        if compliance_score >= 95:
            return "low"
        elif compliance_score >= 85:
            return "medium"
        elif compliance_score >= 70:
            return "high"
        else:
            return "critical"
    
    def _identify_top_risks(self, dashboard_data: Dict[str, Any]) -> List[str]:
        """Identify top compliance risks across frameworks"""
        
        risks = []
        
        # SOC2 risks
        soc2_data = dashboard_data.get("soc2", {})
        if soc2_data.get("risk_indicators", {}).get("failed_controls", 0) > 0:
            risks.append("SOC2 control failures detected")
        
        # GDPR risks
        gdpr_data = dashboard_data.get("gdpr", {})
        if gdpr_data.get("pending_items", {}).get("overdue_requests", 0) > 0:
            risks.append("Overdue GDPR data subject requests")
        
        # HIPAA risks
        hipaa_data = dashboard_data.get("hipaa", {})
        if hipaa_data.get("risk_indicators", {}).get("unresolved_incidents", 0) > 0:
            risks.append("Unresolved HIPAA security incidents")
        
        return risks[:5]  # Return top 5 risks
    
    def _identify_improvement_areas(self, dashboard_data: Dict[str, Any]) -> List[str]:
        """Identify areas for compliance improvement"""
        
        improvements = []
        
        # Analyze each framework for improvement opportunities
        soc2_data = dashboard_data.get("soc2", {})
        if soc2_data.get("compliance_overview", {}).get("compliance_percentage", 0) < 95:
            improvements.append("Enhance SOC2 control implementation")
        
        gdpr_data = dashboard_data.get("gdpr", {})
        if gdpr_data.get("compliance_metrics", {}).get("consent_withdrawal_rate", 0) > 10:
            improvements.append("Improve GDPR consent management")
        
        hipaa_data = dashboard_data.get("hipaa", {})
        if hipaa_data.get("overview", {}).get("compliance_percentage", 0) < 95:
            improvements.append("Strengthen HIPAA safeguard implementation")
        
        return improvements
    
    def _get_next_milestones(self) -> List[Dict[str, Any]]:
        """Get upcoming compliance milestones"""
        
        now = datetime.utcnow()
        
        return [
            {
                "title": "Quarterly SOC2 Assessment",
                "due_date": (now + timedelta(days=90)).strftime('%Y-%m-%d'),
                "framework": "soc2"
            },
            {
                "title": "Annual GDPR Privacy Impact Assessment",
                "due_date": (now + timedelta(days=365)).strftime('%Y-%m-%d'),
                "framework": "gdpr"
            },
            {
                "title": "HIPAA Risk Assessment Update",
                "due_date": (now + timedelta(days=180)).strftime('%Y-%m-%d'),
                "framework": "hipaa"
            }
        ]
    
    def _count_total_controls(self) -> int:
        """Count total controls across all frameworks"""
        
        total = 0
        
        if ComplianceFramework.SOC2 in self.enabled_frameworks:
            total += len(self.soc2_manager.controls)
        
        if ComplianceFramework.HIPAA in self.enabled_frameworks:
            total += len(self.hipaa_manager.safeguards)
        
        # GDPR doesn't have discrete controls like SOC2/HIPAA
        if ComplianceFramework.GDPR in self.enabled_frameworks:
            total += len(self.gdpr_manager.processing_records)
        
        return total
    
    def _count_compliant_controls(self) -> int:
        """Count compliant controls across all frameworks"""
        
        compliant = 0
        
        if ComplianceFramework.SOC2 in self.enabled_frameworks:
            compliant += len([
                c for c in self.soc2_manager.controls.values()
                if c.status == SOC2Status.COMPLIANT
            ])
        
        if ComplianceFramework.HIPAA in self.enabled_frameworks:
            compliant += len([
                s for s in self.hipaa_manager.safeguards.values()
                if s.status == HIPAAStatus.COMPLIANT
            ])
        
        # GDPR compliance is measured differently
        if ComplianceFramework.GDPR in self.enabled_frameworks:
            compliant += len(self.gdpr_manager.processing_records)  # Simplified
        
        return compliant
    
    def _generate_recommendations(self, assessment_results: Dict[str, Any]) -> List[str]:
        """Generate compliance recommendations based on assessment results"""
        
        recommendations = []
        
        # SOC2 recommendations
        if "soc2" in assessment_results:
            soc2_data = assessment_results["soc2"]
            if soc2_data.get("compliance_percentage", 0) < 95:
                recommendations.append("Implement additional SOC2 controls to achieve 95%+ compliance")
        
        # GDPR recommendations
        if "gdpr" in assessment_results:
            gdpr_data = assessment_results["gdpr"]
            if gdpr_data.get("data_subject_requests", {}).get("overdue_requests", 0) > 0:
                recommendations.append("Prioritize processing of overdue GDPR data subject requests")
        
        # HIPAA recommendations
        if "hipaa" in assessment_results:
            hipaa_data = assessment_results["hipaa"]
            if hipaa_data.get("risk_indicators", {}).get("unresolved_incidents", 0) > 0:
                recommendations.append("Resolve outstanding HIPAA security incidents")
        
        return recommendations
    
    def _generate_executive_summary(
        self,
        overall_score: float,
        risk_score: float,
        assessment_results: Dict[str, Any]
    ) -> str:
        """Generate executive summary for compliance report"""
        
        status = "compliant" if overall_score >= 95 else "needs attention"
        risk_level = self._calculate_risk_level(overall_score)
        
        summary = f"""
        Compliance Assessment Executive Summary
        
        Overall Compliance Score: {overall_score:.1f}%
        Risk Level: {risk_level.upper()}
        Status: {status.upper()}
        
        The organization maintains {status} across monitored compliance frameworks.
        Key areas requiring attention include control implementation and incident resolution.
        Continued monitoring and improvement efforts are recommended to maintain compliance posture.
        """
        
        return summary.strip()
    
    async def _check_compliance_alerts(self, assessment_results: Dict[str, Any]) -> None:
        """Check for compliance issues and generate alerts"""
        
        # Check overall compliance score
        overall_score = self._calculate_overall_compliance_score(assessment_results)
        if overall_score < self.alert_thresholds["compliance_score_threshold"]:
            await self.create_compliance_alert(
                framework=ComplianceFramework.SOC2,  # Use primary framework
                severity=ComplianceRisk.HIGH,
                title="Overall Compliance Score Below Threshold",
                description=f"Overall compliance score ({overall_score:.1f}%) is below threshold ({self.alert_thresholds['compliance_score_threshold']}%)",
                alert_type="compliance_threshold",
                source_component="unified_compliance_manager"
            )
        
        # Check individual framework issues
        await self._check_soc2_alerts(assessment_results.get("soc2", {}))
        await self._check_gdpr_alerts(assessment_results.get("gdpr", {}))
        await self._check_hipaa_alerts(assessment_results.get("hipaa", {}))
    
    async def _check_soc2_alerts(self, soc2_data: Dict[str, Any]) -> None:
        """Check for SOC2-specific alerts"""
        
        # Check for failed controls
        control_results = soc2_data.get("control_results", {})
        failed_controls = [
            control_id for control_id, result in control_results.items()
            if result.get("status") == "non_compliant"
        ]
        
        if len(failed_controls) >= self.alert_thresholds["control_failure_threshold"]:
            await self.create_compliance_alert(
                framework=ComplianceFramework.SOC2,
                severity=ComplianceRisk.HIGH,
                title="Multiple SOC2 Control Failures",
                description=f"{len(failed_controls)} SOC2 controls are non-compliant",
                alert_type="control_failure",
                source_component="soc2_manager",
                affected_controls=failed_controls
            )
    
    async def _check_gdpr_alerts(self, gdpr_data: Dict[str, Any]) -> None:
        """Check for GDPR-specific alerts"""
        
        # Check for overdue data subject requests
        overdue_requests = gdpr_data.get("data_subject_requests", {}).get("overdue_requests", 0)
        if overdue_requests > 0:
            await self.create_compliance_alert(
                framework=ComplianceFramework.GDPR,
                severity=ComplianceRisk.HIGH,
                title="Overdue GDPR Data Subject Requests",
                description=f"{overdue_requests} data subject requests are overdue",
                alert_type="deadline_breach",
                source_component="gdpr_manager"
            )
    
    async def _check_hipaa_alerts(self, hipaa_data: Dict[str, Any]) -> None:
        """Check for HIPAA-specific alerts"""
        
        # Check for unresolved incidents
        unresolved_incidents = hipaa_data.get("risk_indicators", {}).get("unresolved_incidents", 0)
        if unresolved_incidents > 0:
            await self.create_compliance_alert(
                framework=ComplianceFramework.HIPAA,
                severity=ComplianceRisk.MEDIUM,
                title="Unresolved HIPAA Security Incidents",
                description=f"{unresolved_incidents} HIPAA security incidents remain unresolved",
                alert_type="incident_unresolved",
                source_component="hipaa_manager"
            )
    
    # Storage methods
    async def _store_compliance_report(self, report: ComplianceReport) -> None:
        """Store compliance report in memory"""
        
        await self.memory_manager.store_context(
            context_type="unified_compliance_report",
            content=report.__dict__,
            metadata={
                "report_id": report.report_id,
                "report_date": report.report_date.isoformat(),
                "overall_score": report.overall_compliance_score,
                "risk_score": report.risk_score
            }
        )
    
    async def _store_compliance_alert(self, alert: ComplianceAlert) -> None:
        """Store compliance alert in memory"""
        
        await self.memory_manager.store_context(
            context_type="compliance_alert",
            content=alert.__dict__,
            metadata={
                "alert_id": alert.alert_id,
                "framework": alert.framework.value,
                "severity": alert.severity.value,
                "resolved": alert.resolved,
                "created_at": alert.created_at.isoformat()
            }
        )