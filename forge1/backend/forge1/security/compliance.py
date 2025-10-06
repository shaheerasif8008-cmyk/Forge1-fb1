"""
Compliance Programs Management
SOC 2 Type II / ISO 27001 scaffolding, evidence collection automation, and policy management
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import logging
import json
from pathlib import Path

from forge1.backend.forge1.core.monitoring import MetricsCollector
from forge1.backend.forge1.core.memory import MemoryManager
from forge1.backend.forge1.core.security import SecretManager


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SOC2_TYPE_II = "soc2_type_ii"
    ISO_27001 = "iso_27001"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    NIST_CSF = "nist_csf"


class ControlStatus(Enum):
    """Status of compliance controls"""
    NOT_IMPLEMENTED = "not_implemented"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    TESTED = "tested"
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"


class EvidenceType(Enum):
    """Types of compliance evidence"""
    POLICY_DOCUMENT = "policy_document"
    PROCEDURE_DOCUMENT = "procedure_document"
    SCREENSHOT = "screenshot"
    LOG_FILE = "log_file"
    CONFIGURATION_EXPORT = "configuration_export"
    AUDIT_REPORT = "audit_report"
    TRAINING_RECORD = "training_record"
    INCIDENT_REPORT = "incident_report"


@dataclass
class ComplianceControl:
    """Individual compliance control"""
    control_id: str
    framework: ComplianceFramework
    title: str
    description: str
    status: ControlStatus
    
    # Control details
    control_family: str
    control_type: str  # preventive, detective, corrective
    implementation_guidance: str
    testing_procedures: str
    
    # Ownership and responsibility
    owner: str
    responsible_party: str
    
    # Implementation tracking
    implementation_date: Optional[datetime] = None
    last_tested_date: Optional[datetime] = None
    next_test_date: Optional[datetime] = None
    
    # Evidence and documentation
    evidence_required: List[EvidenceType] = field(default_factory=list)
    evidence_collected: List[str] = field(default_factory=list)  # File paths
    
    # Risk and exceptions
    risk_rating: str = "medium"  # low, medium, high, critical
    exceptions: List[str] = field(default_factory=list)
    compensating_controls: List[str] = field(default_factory=list)
    
    # Metadata
    created_date: datetime = field(default_factory=datetime.utcnow)
    updated_date: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceEvidence:
    """Compliance evidence artifact"""
    evidence_id: str
    control_id: str
    evidence_type: EvidenceType
    title: str
    description: str
    
    # File information
    file_path: str
    file_size: int
    file_hash: str
    
    # Collection details
    collected_date: datetime
    collected_by: str
    collection_method: str  # automated, manual, system_generated
    
    # Validation
    validated: bool = False
    validated_by: Optional[str] = None
    validated_date: Optional[datetime] = None
    
    # Retention
    retention_period: timedelta = field(default_factory=lambda: timedelta(days=2555))  # 7 years
    expiration_date: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceAssessment:
    """Compliance assessment results"""
    assessment_id: str
    framework: ComplianceFramework
    assessment_type: str  # self_assessment, external_audit, certification
    
    # Timing
    start_date: datetime
    end_date: Optional[datetime] = None
    
    # Scope
    scope_description: str
    controls_assessed: List[str] = field(default_factory=list)
    
    # Results
    total_controls: int = 0
    compliant_controls: int = 0
    non_compliant_controls: int = 0
    not_applicable_controls: int = 0
    
    # Findings
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Assessor information
    assessor: str = ""
    assessor_organization: str = ""
    
    # Certification
    certification_status: str = "pending"  # pending, certified, not_certified
    certificate_number: Optional[str] = None
    certificate_expiry: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplianceManager:
    """
    Comprehensive compliance management system
    
    Manages:
    - Multiple compliance frameworks (SOC 2, ISO 27001, GDPR, etc.)
    - Control implementation and testing
    - Evidence collection and validation
    - Assessment and audit preparation
    - Continuous monitoring and reporting
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
        self.logger = logging.getLogger("compliance_manager")
        
        # Compliance data
        self.controls: Dict[str, ComplianceControl] = {}
        self.evidence: Dict[str, ComplianceEvidence] = {}
        self.assessments: Dict[str, ComplianceAssessment] = {}
        
        # Framework definitions
        self.framework_controls = self._initialize_framework_controls()
        
        # Evidence collection automation
        self.evidence_collectors = self._initialize_evidence_collectors()
    
    async def initialize_compliance_framework(
        self,
        framework: ComplianceFramework,
        organization_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Initialize compliance framework with standard controls
        
        Args:
            framework: Compliance framework to initialize
            organization_info: Organization-specific information
        
        Returns:
            Initialization results
        """
        
        self.logger.info(f"Initializing compliance framework: {framework.value}")
        
        # Get framework control definitions
        framework_controls = self.framework_controls.get(framework, [])
        
        initialized_controls = []
        
        for control_def in framework_controls:
            # Create control instance
            control = ComplianceControl(
                control_id=control_def["control_id"],
                framework=framework,
                title=control_def["title"],
                description=control_def["description"],
                status=ControlStatus.NOT_IMPLEMENTED,
                control_family=control_def["control_family"],
                control_type=control_def["control_type"],
                implementation_guidance=control_def["implementation_guidance"],
                testing_procedures=control_def["testing_procedures"],
                owner=organization_info.get("default_owner", "compliance_team"),
                responsible_party=organization_info.get("default_responsible_party", "security_team"),
                evidence_required=control_def["evidence_required"],
                risk_rating=control_def.get("risk_rating", "medium")
            )
            
            # Store control
            self.controls[control.control_id] = control
            await self._store_control(control)
            
            initialized_controls.append(control.control_id)
        
        # Record metrics
        self.metrics.increment(f"compliance_framework_initialized_{framework.value}")
        self.metrics.record_metric(f"compliance_controls_initialized_{framework.value}", len(initialized_controls))
        
        self.logger.info(f"Initialized {len(initialized_controls)} controls for {framework.value}")
        
        return {
            "framework": framework.value,
            "controls_initialized": len(initialized_controls),
            "control_ids": initialized_controls,
            "next_steps": self._generate_implementation_plan(framework)
        }
    
    async def update_control_status(
        self,
        control_id: str,
        status: ControlStatus,
        updated_by: str,
        notes: Optional[str] = None
    ) -> bool:
        """Update the status of a compliance control"""
        
        if control_id not in self.controls:
            raise ValueError(f"Control not found: {control_id}")
        
        control = self.controls[control_id]
        old_status = control.status
        
        # Update control
        control.status = status
        control.updated_date = datetime.utcnow()
        
        if status == ControlStatus.IMPLEMENTED:
            control.implementation_date = datetime.utcnow()
        elif status == ControlStatus.TESTED:
            control.last_tested_date = datetime.utcnow()
            # Set next test date (annual testing)
            control.next_test_date = datetime.utcnow() + timedelta(days=365)
        
        if notes:
            if "notes" not in control.metadata:
                control.metadata["notes"] = []
            control.metadata["notes"].append({
                "date": datetime.utcnow().isoformat(),
                "updated_by": updated_by,
                "note": notes
            })
        
        # Store updated control
        await self._store_control(control)
        
        # Record metrics
        self.metrics.increment(f"compliance_control_status_updated_{status.value}")
        
        self.logger.info(f"Updated control {control_id} status: {old_status.value} -> {status.value}")
        
        return True
    
    async def collect_evidence(
        self,
        control_id: str,
        evidence_type: EvidenceType,
        collection_method: str = "automated"
    ) -> Optional[ComplianceEvidence]:
        """
        Collect evidence for a compliance control
        
        Args:
            control_id: Control requiring evidence
            evidence_type: Type of evidence to collect
            collection_method: How evidence is collected
        
        Returns:
            Collected evidence or None if collection failed
        """
        
        if control_id not in self.controls:
            raise ValueError(f"Control not found: {control_id}")
        
        control = self.controls[control_id]
        
        self.logger.info(f"Collecting evidence for control {control_id}: {evidence_type.value}")
        
        try:
            # Use appropriate evidence collector
            collector = self.evidence_collectors.get(evidence_type)
            if not collector:
                raise ValueError(f"No collector available for evidence type: {evidence_type}")
            
            # Collect evidence
            evidence_data = await collector(control_id, control)
            
            if evidence_data:
                # Create evidence record
                evidence_id = f"{control_id}_{evidence_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                
                evidence = ComplianceEvidence(
                    evidence_id=evidence_id,
                    control_id=control_id,
                    evidence_type=evidence_type,
                    title=evidence_data["title"],
                    description=evidence_data["description"],
                    file_path=evidence_data["file_path"],
                    file_size=evidence_data["file_size"],
                    file_hash=evidence_data["file_hash"],
                    collected_date=datetime.utcnow(),
                    collected_by=evidence_data.get("collected_by", "system"),
                    collection_method=collection_method,
                    metadata=evidence_data.get("metadata", {})
                )
                
                # Store evidence
                self.evidence[evidence_id] = evidence
                await self._store_evidence(evidence)
                
                # Update control with evidence reference
                control.evidence_collected.append(evidence_id)
                await self._store_control(control)
                
                # Record metrics
                self.metrics.increment(f"compliance_evidence_collected_{evidence_type.value}")
                
                self.logger.info(f"Evidence collected successfully: {evidence_id}")
                
                return evidence
        
        except Exception as e:
            self.logger.error(f"Failed to collect evidence for {control_id}: {e}")
            self.metrics.increment("compliance_evidence_collection_failed")
        
        return None
    
    async def run_compliance_assessment(
        self,
        framework: ComplianceFramework,
        assessment_type: str = "self_assessment",
        assessor: str = "internal_team"
    ) -> ComplianceAssessment:
        """
        Run comprehensive compliance assessment
        
        Args:
            framework: Framework to assess
            assessment_type: Type of assessment
            assessor: Who is conducting the assessment
        
        Returns:
            Assessment results
        """
        
        assessment_id = f"{framework.value}_{assessment_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting compliance assessment: {assessment_id}")
        
        # Get framework controls
        framework_controls = [
            control for control in self.controls.values()
            if control.framework == framework
        ]
        
        assessment = ComplianceAssessment(
            assessment_id=assessment_id,
            framework=framework,
            assessment_type=assessment_type,
            start_date=datetime.utcnow(),
            scope_description=f"Full {framework.value} compliance assessment",
            controls_assessed=[c.control_id for c in framework_controls],
            assessor=assessor
        )
        
        # Assess each control
        for control in framework_controls:
            assessment.total_controls += 1
            
            # Evaluate control compliance
            compliance_result = await self._evaluate_control_compliance(control)
            
            if compliance_result["compliant"]:
                assessment.compliant_controls += 1
            else:
                assessment.non_compliant_controls += 1
                
                # Add finding
                finding = {
                    "control_id": control.control_id,
                    "title": control.title,
                    "finding": compliance_result["finding"],
                    "severity": compliance_result["severity"],
                    "recommendation": compliance_result["recommendation"]
                }
                assessment.findings.append(finding)
        
        # Generate recommendations
        assessment.recommendations = self._generate_assessment_recommendations(assessment)
        
        # Complete assessment
        assessment.end_date = datetime.utcnow()
        
        # Determine certification status
        compliance_rate = assessment.compliant_controls / assessment.total_controls
        if compliance_rate >= 0.95:
            assessment.certification_status = "certified"
        elif compliance_rate >= 0.80:
            assessment.certification_status = "conditional"
        else:
            assessment.certification_status = "not_certified"
        
        # Store assessment
        self.assessments[assessment_id] = assessment
        await self._store_assessment(assessment)
        
        # Record metrics
        self.metrics.increment(f"compliance_assessment_completed_{framework.value}")
        self.metrics.record_metric(f"compliance_rate_{framework.value}", compliance_rate)
        
        self.logger.info(
            f"Assessment completed: {assessment_id} - "
            f"{assessment.compliant_controls}/{assessment.total_controls} compliant "
            f"({compliance_rate:.1%})"
        )
        
        return assessment
    
    def get_compliance_dashboard(self, framework: Optional[ComplianceFramework] = None) -> Dict[str, Any]:
        """Get comprehensive compliance dashboard"""
        
        # Filter controls by framework if specified
        if framework:
            controls = [c for c in self.controls.values() if c.framework == framework]
        else:
            controls = list(self.controls.values())
        
        # Calculate status distribution
        status_counts = {}
        for status in ControlStatus:
            status_counts[status.value] = len([c for c in controls if c.status == status])
        
        # Calculate compliance rate
        compliant_controls = len([c for c in controls if c.status == ControlStatus.COMPLIANT])
        compliance_rate = compliant_controls / len(controls) if controls else 0
        
        # Evidence collection status
        total_evidence_required = sum(len(c.evidence_required) for c in controls)
        total_evidence_collected = sum(len(c.evidence_collected) for c in controls)
        evidence_collection_rate = total_evidence_collected / total_evidence_required if total_evidence_required else 0
        
        # Overdue controls
        overdue_controls = []
        for control in controls:
            if (control.next_test_date and 
                datetime.utcnow() > control.next_test_date and
                control.status != ControlStatus.TESTED):
                overdue_controls.append(control.control_id)
        
        # Recent assessments
        recent_assessments = [
            a for a in self.assessments.values()
            if a.start_date > datetime.utcnow() - timedelta(days=90)
        ]
        
        return {
            "framework": framework.value if framework else "all",
            "total_controls": len(controls),
            "compliance_rate": compliance_rate,
            "status_distribution": status_counts,
            "evidence_collection_rate": evidence_collection_rate,
            "overdue_controls": len(overdue_controls),
            "recent_assessments": len(recent_assessments),
            "certification_status": self._get_overall_certification_status(framework),
            "next_actions": self._get_next_actions(controls)
        }
    
    # Private methods
    def _initialize_framework_controls(self) -> Dict[ComplianceFramework, List[Dict[str, Any]]]:
        """Initialize standard controls for each framework"""
        
        return {
            ComplianceFramework.SOC2_TYPE_II: [
                {
                    "control_id": "CC1.1",
                    "title": "Control Environment - Integrity and Ethical Values",
                    "description": "The entity demonstrates a commitment to integrity and ethical values",
                    "control_family": "Control Environment",
                    "control_type": "preventive",
                    "implementation_guidance": "Establish code of conduct, ethics training, and disciplinary procedures",
                    "testing_procedures": "Review policies, interview personnel, test disciplinary actions",
                    "evidence_required": [EvidenceType.POLICY_DOCUMENT, EvidenceType.TRAINING_RECORD],
                    "risk_rating": "high"
                },
                {
                    "control_id": "CC2.1",
                    "title": "Communication and Information - Internal Communication",
                    "description": "The entity obtains or generates and uses relevant, quality information",
                    "control_family": "Communication and Information",
                    "control_type": "detective",
                    "implementation_guidance": "Implement information systems and communication processes",
                    "testing_procedures": "Review information systems, test communication effectiveness",
                    "evidence_required": [EvidenceType.PROCEDURE_DOCUMENT, EvidenceType.CONFIGURATION_EXPORT],
                    "risk_rating": "medium"
                }
                # Additional SOC 2 controls would be added here
            ],
            ComplianceFramework.ISO_27001: [
                {
                    "control_id": "A.5.1.1",
                    "title": "Information Security Policy",
                    "description": "An information security policy shall be defined, approved by management",
                    "control_family": "Information Security Policies",
                    "control_type": "preventive",
                    "implementation_guidance": "Develop comprehensive information security policy",
                    "testing_procedures": "Review policy document, verify management approval",
                    "evidence_required": [EvidenceType.POLICY_DOCUMENT],
                    "risk_rating": "high"
                }
                # Additional ISO 27001 controls would be added here
            ]
        }
    
    def _initialize_evidence_collectors(self) -> Dict[EvidenceType, Any]:
        """Initialize automated evidence collectors"""
        
        return {
            EvidenceType.POLICY_DOCUMENT: self._collect_policy_evidence,
            EvidenceType.CONFIGURATION_EXPORT: self._collect_configuration_evidence,
            EvidenceType.LOG_FILE: self._collect_log_evidence,
            EvidenceType.AUDIT_REPORT: self._collect_audit_evidence
        }
    
    async def _collect_policy_evidence(self, control_id: str, control: ComplianceControl) -> Dict[str, Any]:
        """Collect policy document evidence"""
        
        # Look for policy documents related to the control
        policy_path = f"policies/{control.control_family.lower().replace(' ', '_')}_policy.md"
        
        if Path(policy_path).exists():
            file_size = Path(policy_path).stat().st_size
            
            return {
                "title": f"Policy Document for {control.title}",
                "description": f"Policy document supporting control {control_id}",
                "file_path": policy_path,
                "file_size": file_size,
                "file_hash": "sha256:placeholder_hash",  # Would calculate actual hash
                "collected_by": "automated_collector"
            }
        
        return None
    
    async def _collect_configuration_evidence(self, control_id: str, control: ComplianceControl) -> Dict[str, Any]:
        """Collect system configuration evidence"""
        
        # Export relevant system configurations
        config_data = {
            "control_id": control_id,
            "timestamp": datetime.utcnow().isoformat(),
            "configurations": {
                "security_settings": "placeholder_config",
                "access_controls": "placeholder_config"
            }
        }
        
        config_path = f"evidence/configurations/{control_id}_config.json"
        
        # Would write actual configuration to file
        return {
            "title": f"Configuration Export for {control.title}",
            "description": f"System configuration supporting control {control_id}",
            "file_path": config_path,
            "file_size": len(json.dumps(config_data)),
            "file_hash": "sha256:placeholder_hash",
            "collected_by": "automated_collector"
        }
    
    async def _collect_log_evidence(self, control_id: str, control: ComplianceControl) -> Dict[str, Any]:
        """Collect log file evidence"""
        
        # Collect relevant log files
        log_path = f"evidence/logs/{control_id}_logs.txt"
        
        return {
            "title": f"Log Evidence for {control.title}",
            "description": f"System logs supporting control {control_id}",
            "file_path": log_path,
            "file_size": 1024,  # Placeholder
            "file_hash": "sha256:placeholder_hash",
            "collected_by": "automated_collector"
        }
    
    async def _collect_audit_evidence(self, control_id: str, control: ComplianceControl) -> Dict[str, Any]:
        """Collect audit report evidence"""
        
        # Generate audit report
        audit_path = f"evidence/audits/{control_id}_audit.pdf"
        
        return {
            "title": f"Audit Report for {control.title}",
            "description": f"Audit findings supporting control {control_id}",
            "file_path": audit_path,
            "file_size": 2048,  # Placeholder
            "file_hash": "sha256:placeholder_hash",
            "collected_by": "automated_collector"
        }
    
    async def _evaluate_control_compliance(self, control: ComplianceControl) -> Dict[str, Any]:
        """Evaluate compliance status of a control"""
        
        # Check if control is implemented and tested
        if control.status == ControlStatus.COMPLIANT:
            return {"compliant": True}
        
        # Check evidence collection
        evidence_complete = len(control.evidence_collected) >= len(control.evidence_required)
        
        # Check if testing is current
        testing_current = (
            control.last_tested_date and
            datetime.utcnow() - control.last_tested_date < timedelta(days=365)
        )
        
        if control.status == ControlStatus.IMPLEMENTED and evidence_complete and testing_current:
            return {"compliant": True}
        
        # Determine specific finding
        if control.status == ControlStatus.NOT_IMPLEMENTED:
            finding = "Control has not been implemented"
            severity = "high"
            recommendation = "Implement control according to implementation guidance"
        elif not evidence_complete:
            finding = "Insufficient evidence collected"
            severity = "medium"
            recommendation = "Collect all required evidence types"
        elif not testing_current:
            finding = "Control testing is overdue"
            severity = "medium"
            recommendation = "Perform annual control testing"
        else:
            finding = "Control implementation incomplete"
            severity = "medium"
            recommendation = "Complete control implementation and testing"
        
        return {
            "compliant": False,
            "finding": finding,
            "severity": severity,
            "recommendation": recommendation
        }
    
    def _generate_assessment_recommendations(self, assessment: ComplianceAssessment) -> List[str]:
        """Generate recommendations based on assessment findings"""
        
        recommendations = []
        
        # High-level recommendations
        if assessment.non_compliant_controls > assessment.compliant_controls:
            recommendations.append("Focus on implementing basic controls before pursuing certification")
        
        # Specific recommendations based on findings
        high_severity_findings = [f for f in assessment.findings if f.get("severity") == "high"]
        if high_severity_findings:
            recommendations.append(f"Address {len(high_severity_findings)} high-severity findings immediately")
        
        # Evidence collection recommendations
        controls_needing_evidence = [
            c for c in self.controls.values()
            if c.framework == assessment.framework and len(c.evidence_collected) < len(c.evidence_required)
        ]
        
        if controls_needing_evidence:
            recommendations.append(f"Complete evidence collection for {len(controls_needing_evidence)} controls")
        
        return recommendations
    
    def _get_overall_certification_status(self, framework: Optional[ComplianceFramework]) -> str:
        """Get overall certification status"""
        
        # Get recent assessment
        if framework:
            recent_assessments = [
                a for a in self.assessments.values()
                if a.framework == framework and a.start_date > datetime.utcnow() - timedelta(days=180)
            ]
        else:
            recent_assessments = [
                a for a in self.assessments.values()
                if a.start_date > datetime.utcnow() - timedelta(days=180)
            ]
        
        if recent_assessments:
            latest_assessment = max(recent_assessments, key=lambda x: x.start_date)
            return latest_assessment.certification_status
        
        return "not_assessed"
    
    def _get_next_actions(self, controls: List[ComplianceControl]) -> List[str]:
        """Get next recommended actions"""
        
        actions = []
        
        # Controls needing implementation
        not_implemented = [c for c in controls if c.status == ControlStatus.NOT_IMPLEMENTED]
        if not_implemented:
            actions.append(f"Implement {len(not_implemented)} pending controls")
        
        # Controls needing testing
        need_testing = [
            c for c in controls
            if c.status == ControlStatus.IMPLEMENTED and not c.last_tested_date
        ]
        if need_testing:
            actions.append(f"Test {len(need_testing)} implemented controls")
        
        # Overdue testing
        overdue_testing = [
            c for c in controls
            if (c.next_test_date and datetime.utcnow() > c.next_test_date)
        ]
        if overdue_testing:
            actions.append(f"Update testing for {len(overdue_testing)} overdue controls")
        
        return actions[:5]  # Limit to top 5 actions
    
    def _generate_implementation_plan(self, framework: ComplianceFramework) -> List[str]:
        """Generate implementation plan for framework"""
        
        return [
            "Review and customize control implementations for your organization",
            "Assign control owners and responsible parties",
            "Begin implementing high-risk controls first",
            "Set up automated evidence collection where possible",
            "Schedule regular control testing and reviews"
        ]
    
    async def _store_control(self, control: ComplianceControl) -> None:
        """Store compliance control in memory"""
        
        await self.memory_manager.store_context(
            context_type="compliance_control",
            content=control.__dict__,
            metadata={
                "control_id": control.control_id,
                "framework": control.framework.value,
                "status": control.status.value
            }
        )
    
    async def _store_evidence(self, evidence: ComplianceEvidence) -> None:
        """Store compliance evidence in memory"""
        
        await self.memory_manager.store_context(
            context_type="compliance_evidence",
            content=evidence.__dict__,
            metadata={
                "evidence_id": evidence.evidence_id,
                "control_id": evidence.control_id,
                "evidence_type": evidence.evidence_type.value
            }
        )
    
    async def _store_assessment(self, assessment: ComplianceAssessment) -> None:
        """Store compliance assessment in memory"""
        
        await self.memory_manager.store_context(
            context_type="compliance_assessment",
            content=assessment.__dict__,
            metadata={
                "assessment_id": assessment.assessment_id,
                "framework": assessment.framework.value,
                "assessment_type": assessment.assessment_type
            }
        )