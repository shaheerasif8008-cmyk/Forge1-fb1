"""
GDPR Compliance Framework
Comprehensive GDPR compliance with data subject rights, consent management, and privacy controls
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import logging
import json
import hashlib
from collections import defaultdict

# Mock classes for testing - in production these would import from the actual core modules
class MetricsCollector:
    def increment(self, metric): pass
    def record_metric(self, metric, value): pass

class MemoryManager:
    async def store_context(self, context_type, content, metadata): pass

class SecretManager:
    async def get(self, name): return "mock_secret"


class LegalBasis(Enum):
    """GDPR legal basis for processing"""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class DataCategory(Enum):
    """Categories of personal data"""
    BASIC_IDENTITY = "basic_identity"
    CONTACT_DETAILS = "contact_details"
    DEMOGRAPHIC = "demographic"
    FINANCIAL = "financial"
    PROFESSIONAL = "professional"
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    SPECIAL_CATEGORY = "special_category"  # Sensitive data


class ProcessingPurpose(Enum):
    """Purposes for data processing"""
    SERVICE_PROVISION = "service_provision"
    CUSTOMER_SUPPORT = "customer_support"
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    SECURITY = "security"
    LEGAL_COMPLIANCE = "legal_compliance"
    RESEARCH = "research"


class DataSubjectRightType(Enum):
    """GDPR data subject rights"""
    ACCESS = "access"  # Article 15
    RECTIFICATION = "rectification"  # Article 16
    ERASURE = "erasure"  # Article 17 (Right to be forgotten)
    RESTRICT_PROCESSING = "restrict_processing"  # Article 18
    DATA_PORTABILITY = "data_portability"  # Article 20
    OBJECT = "object"  # Article 21
    WITHDRAW_CONSENT = "withdraw_consent"  # Article 7(3)


class RequestStatus(Enum):
    """Status of data subject requests"""
    RECEIVED = "received"
    UNDER_REVIEW = "under_review"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    PARTIALLY_FULFILLED = "partially_fulfilled"


@dataclass
class DataProcessingRecord:
    """Record of Processing Activities (ROPA) entry"""
    record_id: str
    controller_name: str
    controller_contact: str
    
    # Processing details
    processing_purpose: ProcessingPurpose
    legal_basis: LegalBasis
    data_categories: List[DataCategory]
    data_subjects: List[str]  # Categories of data subjects
    
    # Recipients and transfers
    recipients: List[str] = field(default_factory=list)
    third_country_transfers: List[str] = field(default_factory=list)
    safeguards: List[str] = field(default_factory=list)
    
    # Retention and security
    retention_period: str = ""
    security_measures: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    
    # Additional information
    description: str = ""
    automated_decision_making: bool = False
    profiling: bool = False
    
    # Custom fields
    custom_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsentRecord:
    """Individual consent record"""
    consent_id: str
    data_subject_id: str
    
    # Consent details
    processing_purposes: List[ProcessingPurpose]
    data_categories: List[DataCategory]
    consent_given: bool
    consent_timestamp: datetime
    
    # Consent mechanism
    consent_method: str  # web_form, email, phone, etc.
    consent_text: str
    opt_in_explicit: bool = True
    
    # Withdrawal
    withdrawn: bool = False
    withdrawal_timestamp: Optional[datetime] = None
    withdrawal_method: Optional[str] = None
    
    # Validity
    expires_at: Optional[datetime] = None
    valid: bool = True
    
    # Audit trail
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DataSubjectRequest:
    """Data subject rights request"""
    request_id: str
    data_subject_id: str
    request_type: DataSubjectRightType
    
    # Request details
    description: str
    legal_basis: Optional[str] = None
    
    # Contact information
    contact_email: str = ""
    contact_phone: Optional[str] = None
    
    # Processing
    status: RequestStatus = RequestStatus.RECEIVED
    received_at: datetime = field(default_factory=datetime.utcnow)
    due_date: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=30))
    completed_at: Optional[datetime] = None
    
    # Response
    response_data: Optional[Dict[str, Any]] = None
    response_files: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    
    # Processing notes
    processing_notes: List[str] = field(default_factory=list)
    assigned_to: Optional[str] = None
    
    # Verification
    identity_verified: bool = False
    verification_method: Optional[str] = None
    verification_timestamp: Optional[datetime] = None
    
    # Metadata
    created_by: str = ""
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DataBreach:
    """Data breach incident record"""
    breach_id: str
    
    # Breach details
    description: str
    breach_type: str  # confidentiality, integrity, availability
    severity: str  # low, medium, high, critical
    
    # Timeline
    discovered_at: datetime
    occurred_at: Optional[datetime] = None
    contained_at: Optional[datetime] = None
    
    # Impact assessment
    data_subjects_affected: int = 0
    data_categories_affected: List[DataCategory] = field(default_factory=list)
    likely_consequences: str = ""
    
    # Response
    containment_measures: List[str] = field(default_factory=list)
    notification_required: bool = False
    authority_notified: bool = False
    authority_notification_date: Optional[datetime] = None
    data_subjects_notified: bool = False
    data_subject_notification_date: Optional[datetime] = None
    
    # Investigation
    root_cause: str = ""
    lessons_learned: str = ""
    preventive_measures: List[str] = field(default_factory=list)
    
    # Metadata
    reported_by: str = ""
    assigned_to: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class GDPRComplianceManager:
    """
    Comprehensive GDPR compliance management system
    
    Manages:
    - Data subject rights and request processing
    - Consent management and tracking
    - Records of Processing Activities (ROPA)
    - Data breach notification and management
    - Privacy impact assessments
    - Data protection by design and default
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
        self.logger = logging.getLogger("gdpr_compliance")
        
        # Data processing records
        self.processing_records: Dict[str, DataProcessingRecord] = {}
        
        # Consent management
        self.consent_records: Dict[str, List[ConsentRecord]] = defaultdict(list)  # data_subject_id -> consents
        
        # Data subject requests
        self.data_subject_requests: Dict[str, DataSubjectRequest] = {}
        
        # Data breaches
        self.data_breaches: Dict[str, DataBreach] = {}
        
        # Configuration
        self.dpo_contact = "dpo@cognisia.com"
        self.controller_name = "Cognisia Inc."
        self.controller_address = "123 AI Street, Tech City, TC 12345"
        
        # Initialize default processing records
        self._initialize_default_processing_records()
    
    def _initialize_default_processing_records(self) -> None:
        """Initialize default processing records for the platform"""
        
        default_records = [
            DataProcessingRecord(
                record_id="ROPA-001",
                controller_name=self.controller_name,
                controller_contact=self.dpo_contact,
                processing_purpose=ProcessingPurpose.SERVICE_PROVISION,
                legal_basis=LegalBasis.CONTRACT,
                data_categories=[DataCategory.BASIC_IDENTITY, DataCategory.CONTACT_DETAILS, DataCategory.PROFESSIONAL],
                data_subjects=["customers", "users"],
                recipients=["cloud_providers", "support_staff"],
                retention_period="Duration of contract + 7 years",
                security_measures=["encryption", "access_controls", "audit_logging"],
                description="Processing customer data for AI employee service provision"
            ),
            DataProcessingRecord(
                record_id="ROPA-002",
                controller_name=self.controller_name,
                controller_contact=self.dpo_contact,
                processing_purpose=ProcessingPurpose.CUSTOMER_SUPPORT,
                legal_basis=LegalBasis.LEGITIMATE_INTERESTS,
                data_categories=[DataCategory.CONTACT_DETAILS, DataCategory.TECHNICAL],
                data_subjects=["customers"],
                recipients=["support_staff"],
                retention_period="3 years after last contact",
                security_measures=["encryption", "access_controls"],
                description="Processing data for customer support and issue resolution"
            ),
            DataProcessingRecord(
                record_id="ROPA-003",
                controller_name=self.controller_name,
                controller_contact=self.dpo_contact,
                processing_purpose=ProcessingPurpose.SECURITY,
                legal_basis=LegalBasis.LEGITIMATE_INTERESTS,
                data_categories=[DataCategory.TECHNICAL, DataCategory.BEHAVIORAL],
                data_subjects=["users", "visitors"],
                recipients=["security_team"],
                retention_period="1 year",
                security_measures=["encryption", "pseudonymization", "access_controls"],
                description="Processing data for security monitoring and fraud prevention"
            )
        ]
        
        for record in default_records:
            self.processing_records[record.record_id] = record
        
        self.logger.info(f"Initialized {len(default_records)} default processing records")
    
    async def record_consent(
        self,
        data_subject_id: str,
        processing_purposes: List[ProcessingPurpose],
        data_categories: List[DataCategory],
        consent_method: str,
        consent_text: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> str:
        """Record consent from a data subject"""
        
        consent_id = f"consent_{data_subject_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        
        consent_record = ConsentRecord(
            consent_id=consent_id,
            data_subject_id=data_subject_id,
            processing_purposes=processing_purposes,
            data_categories=data_categories,
            consent_given=True,
            consent_timestamp=datetime.utcnow(),
            consent_method=consent_method,
            consent_text=consent_text,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at
        )
        
        # Add to consent records
        self.consent_records[data_subject_id].append(consent_record)
        
        # Store in memory
        await self._store_consent_record(consent_record)
        
        # Record metrics
        self.metrics.increment("gdpr_consent_recorded")
        self.metrics.increment(f"gdpr_consent_method_{consent_method}")
        
        self.logger.info(f"Recorded consent {consent_id} for data subject {data_subject_id}")
        
        return consent_id
    
    async def withdraw_consent(
        self,
        data_subject_id: str,
        consent_id: Optional[str] = None,
        withdrawal_method: str = "web_form"
    ) -> bool:
        """Withdraw consent for a data subject"""
        
        consents = self.consent_records.get(data_subject_id, [])
        
        if consent_id:
            # Withdraw specific consent
            for consent in consents:
                if consent.consent_id == consent_id and not consent.withdrawn:
                    consent.withdrawn = True
                    consent.withdrawal_timestamp = datetime.utcnow()
                    consent.withdrawal_method = withdrawal_method
                    consent.valid = False
                    
                    await self._store_consent_record(consent)
                    
                    self.logger.info(f"Withdrew consent {consent_id} for data subject {data_subject_id}")
                    return True
        else:
            # Withdraw all active consents
            withdrawn_count = 0
            for consent in consents:
                if not consent.withdrawn:
                    consent.withdrawn = True
                    consent.withdrawal_timestamp = datetime.utcnow()
                    consent.withdrawal_method = withdrawal_method
                    consent.valid = False
                    
                    await self._store_consent_record(consent)
                    withdrawn_count += 1
            
            if withdrawn_count > 0:
                self.logger.info(f"Withdrew {withdrawn_count} consents for data subject {data_subject_id}")
                return True
        
        # Record metrics
        self.metrics.increment("gdpr_consent_withdrawn")
        
        return False
    
    async def submit_data_subject_request(
        self,
        data_subject_id: str,
        request_type: DataSubjectRightType,
        description: str,
        contact_email: str,
        contact_phone: Optional[str] = None,
        created_by: str = "system"
    ) -> str:
        """Submit a data subject rights request"""
        
        request_id = f"dsr_{request_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        
        request = DataSubjectRequest(
            request_id=request_id,
            data_subject_id=data_subject_id,
            request_type=request_type,
            description=description,
            contact_email=contact_email,
            contact_phone=contact_phone,
            created_by=created_by
        )
        
        # Store request
        self.data_subject_requests[request_id] = request
        
        # Store in memory
        await self._store_data_subject_request(request)
        
        # Record metrics
        self.metrics.increment("gdpr_data_subject_request_submitted")
        self.metrics.increment(f"gdpr_request_type_{request_type.value}")
        
        self.logger.info(f"Submitted data subject request {request_id} of type {request_type.value}")
        
        return request_id
    
    async def process_data_subject_request(
        self,
        request_id: str,
        assigned_to: str,
        processing_notes: Optional[str] = None
    ) -> bool:
        """Process a data subject rights request"""
        
        if request_id not in self.data_subject_requests:
            return False
        
        request = self.data_subject_requests[request_id]
        
        # Update request status
        request.status = RequestStatus.IN_PROGRESS
        request.assigned_to = assigned_to
        request.updated_at = datetime.utcnow()
        
        if processing_notes:
            request.processing_notes.append(f"{datetime.utcnow().isoformat()}: {processing_notes}")
        
        # Process based on request type
        try:
            if request.request_type == DataSubjectRightType.ACCESS:
                response_data = await self._process_access_request(request.data_subject_id)
                request.response_data = response_data
            
            elif request.request_type == DataSubjectRightType.ERASURE:
                await self._process_erasure_request(request.data_subject_id)
                request.response_data = {"status": "data_erased", "timestamp": datetime.utcnow().isoformat()}
            
            elif request.request_type == DataSubjectRightType.RECTIFICATION:
                # Would implement data rectification logic
                request.response_data = {"status": "rectification_completed"}
            
            elif request.request_type == DataSubjectRightType.DATA_PORTABILITY:
                portable_data = await self._process_portability_request(request.data_subject_id)
                request.response_data = portable_data
            
            elif request.request_type == DataSubjectRightType.RESTRICT_PROCESSING:
                await self._restrict_processing(request.data_subject_id)
                request.response_data = {"status": "processing_restricted"}
            
            elif request.request_type == DataSubjectRightType.WITHDRAW_CONSENT:
                await self.withdraw_consent(request.data_subject_id)
                request.response_data = {"status": "consent_withdrawn"}
            
            # Mark as completed
            request.status = RequestStatus.COMPLETED
            request.completed_at = datetime.utcnow()
            
            # Store updated request
            await self._store_data_subject_request(request)
            
            # Record metrics
            self.metrics.increment("gdpr_data_subject_request_completed")
            
            self.logger.info(f"Completed data subject request {request_id}")
            
            return True
            
        except Exception as e:
            request.status = RequestStatus.REJECTED
            request.rejection_reason = str(e)
            request.updated_at = datetime.utcnow()
            
            self.logger.error(f"Failed to process data subject request {request_id}: {e}")
            
            return False
    
    async def report_data_breach(
        self,
        description: str,
        breach_type: str,
        severity: str,
        discovered_at: datetime,
        reported_by: str,
        data_subjects_affected: int = 0,
        data_categories_affected: Optional[List[DataCategory]] = None
    ) -> str:
        """Report a data breach incident"""
        
        breach_id = f"breach_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        
        breach = DataBreach(
            breach_id=breach_id,
            description=description,
            breach_type=breach_type,
            severity=severity,
            discovered_at=discovered_at,
            data_subjects_affected=data_subjects_affected,
            data_categories_affected=data_categories_affected or [],
            reported_by=reported_by
        )
        
        # Determine notification requirements
        if data_subjects_affected > 0 and severity in ["high", "critical"]:
            breach.notification_required = True
        
        # Store breach
        self.data_breaches[breach_id] = breach
        
        # Store in memory
        await self._store_data_breach(breach)
        
        # Record metrics
        self.metrics.increment("gdpr_data_breach_reported")
        self.metrics.increment(f"gdpr_breach_severity_{severity}")
        
        self.logger.warning(f"Data breach reported: {breach_id} - {description}")
        
        return breach_id
    
    async def get_consent_status(self, data_subject_id: str) -> Dict[str, Any]:
        """Get current consent status for a data subject"""
        
        consents = self.consent_records.get(data_subject_id, [])
        
        # Get active consents
        active_consents = [c for c in consents if c.valid and not c.withdrawn]
        
        # Check for expired consents
        now = datetime.utcnow()
        for consent in active_consents:
            if consent.expires_at and consent.expires_at < now:
                consent.valid = False
        
        # Refresh active consents
        active_consents = [c for c in consents if c.valid and not c.withdrawn]
        
        # Aggregate consent information
        consent_summary = {
            "data_subject_id": data_subject_id,
            "has_active_consent": len(active_consents) > 0,
            "total_consents": len(consents),
            "active_consents": len(active_consents),
            "withdrawn_consents": len([c for c in consents if c.withdrawn]),
            "expired_consents": len([c for c in consents if c.expires_at and c.expires_at < now]),
            "processing_purposes": list(set([
                purpose.value for consent in active_consents 
                for purpose in consent.processing_purposes
            ])),
            "data_categories": list(set([
                category.value for consent in active_consents 
                for category in consent.data_categories
            ])),
            "latest_consent_date": max([c.consent_timestamp for c in consents]).isoformat() if consents else None,
            "latest_withdrawal_date": max([
                c.withdrawal_timestamp for c in consents 
                if c.withdrawal_timestamp
            ]).isoformat() if any(c.withdrawal_timestamp for c in consents) else None
        }
        
        return consent_summary
    
    def get_gdpr_dashboard(self) -> Dict[str, Any]:
        """Get GDPR compliance dashboard data"""
        
        now = datetime.utcnow()
        
        # Recent activity (last 30 days)
        recent_consents = sum([
            len([c for c in consents if c.consent_timestamp > now - timedelta(days=30)])
            for consents in self.consent_records.values()
        ])
        
        recent_withdrawals = sum([
            len([c for c in consents if c.withdrawal_timestamp and c.withdrawal_timestamp > now - timedelta(days=30)])
            for consents in self.consent_records.values()
        ])
        
        recent_requests = len([
            r for r in self.data_subject_requests.values()
            if r.received_at > now - timedelta(days=30)
        ])
        
        # Pending items
        pending_requests = len([
            r for r in self.data_subject_requests.values()
            if r.status in [RequestStatus.RECEIVED, RequestStatus.UNDER_REVIEW, RequestStatus.IN_PROGRESS]
        ])
        
        overdue_requests = len([
            r for r in self.data_subject_requests.values()
            if r.due_date < now and r.status not in [RequestStatus.COMPLETED, RequestStatus.REJECTED]
        ])
        
        # Compliance metrics
        total_requests = len(self.data_subject_requests)
        completed_requests = len([
            r for r in self.data_subject_requests.values()
            if r.status == RequestStatus.COMPLETED
        ])
        
        completion_rate = (completed_requests / total_requests * 100) if total_requests > 0 else 100
        
        return {
            "overview": {
                "total_data_subjects": len(self.consent_records),
                "active_processing_records": len(self.processing_records),
                "total_data_subject_requests": total_requests,
                "total_data_breaches": len(self.data_breaches)
            },
            "recent_activity": {
                "new_consents": recent_consents,
                "consent_withdrawals": recent_withdrawals,
                "new_requests": recent_requests,
                "period_days": 30
            },
            "pending_items": {
                "pending_requests": pending_requests,
                "overdue_requests": overdue_requests,
                "breach_notifications_due": len([
                    b for b in self.data_breaches.values()
                    if b.notification_required and not b.authority_notified
                ])
            },
            "compliance_metrics": {
                "request_completion_rate": completion_rate,
                "average_response_time_days": self._calculate_average_response_time(),
                "consent_withdrawal_rate": self._calculate_consent_withdrawal_rate(),
                "breach_notification_compliance": self._calculate_breach_notification_compliance()
            },
            "risk_indicators": {
                "high_severity_breaches": len([
                    b for b in self.data_breaches.values()
                    if b.severity in ["high", "critical"]
                ]),
                "expired_consents": self._count_expired_consents(),
                "processing_without_consent": self._count_processing_without_consent()
            }
        }
    
    # Private helper methods
    async def _process_access_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Process data access request (Article 15)"""
        
        # Collect all data for the data subject
        personal_data = {
            "data_subject_id": data_subject_id,
            "consents": [
                {
                    "consent_id": c.consent_id,
                    "purposes": [p.value for p in c.processing_purposes],
                    "categories": [cat.value for cat in c.data_categories],
                    "given_at": c.consent_timestamp.isoformat(),
                    "withdrawn": c.withdrawn,
                    "withdrawal_date": c.withdrawal_timestamp.isoformat() if c.withdrawal_timestamp else None
                }
                for c in self.consent_records.get(data_subject_id, [])
            ],
            "processing_activities": [
                {
                    "record_id": r.record_id,
                    "purpose": r.processing_purpose.value,
                    "legal_basis": r.legal_basis.value,
                    "categories": [cat.value for cat in r.data_categories],
                    "retention_period": r.retention_period
                }
                for r in self.processing_records.values()
            ],
            "requests_history": [
                {
                    "request_id": r.request_id,
                    "type": r.request_type.value,
                    "date": r.received_at.isoformat(),
                    "status": r.status.value
                }
                for r in self.data_subject_requests.values()
                if r.data_subject_id == data_subject_id
            ]
        }
        
        return personal_data
    
    async def _process_erasure_request(self, data_subject_id: str) -> None:
        """Process right to be forgotten request (Article 17)"""
        
        # Mark all consents as withdrawn
        for consent in self.consent_records.get(data_subject_id, []):
            if not consent.withdrawn:
                consent.withdrawn = True
                consent.withdrawal_timestamp = datetime.utcnow()
                consent.withdrawal_method = "erasure_request"
                consent.valid = False
        
        # In a real implementation, this would:
        # 1. Delete/anonymize personal data from all systems
        # 2. Notify third parties to delete data
        # 3. Update backup systems
        # 4. Maintain audit trail of deletion
        
        self.logger.info(f"Processed erasure request for data subject {data_subject_id}")
    
    async def _process_portability_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Process data portability request (Article 20)"""
        
        # Export data in structured format
        portable_data = await self._process_access_request(data_subject_id)
        
        # Add export metadata
        portable_data["export_metadata"] = {
            "export_date": datetime.utcnow().isoformat(),
            "format": "JSON",
            "version": "1.0"
        }
        
        return portable_data
    
    async def _restrict_processing(self, data_subject_id: str) -> None:
        """Restrict processing for data subject (Article 18)"""
        
        # Mark processing as restricted
        # In a real implementation, this would update all systems
        # to restrict processing while maintaining storage
        
        self.logger.info(f"Restricted processing for data subject {data_subject_id}")
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time for completed requests"""
        
        completed_requests = [
            r for r in self.data_subject_requests.values()
            if r.status == RequestStatus.COMPLETED and r.completed_at
        ]
        
        if not completed_requests:
            return 0.0
        
        total_time = sum([
            (r.completed_at - r.received_at).total_seconds() / 86400
            for r in completed_requests
        ])
        
        return total_time / len(completed_requests)
    
    def _calculate_consent_withdrawal_rate(self) -> float:
        """Calculate consent withdrawal rate"""
        
        total_consents = sum(len(consents) for consents in self.consent_records.values())
        withdrawn_consents = sum([
            len([c for c in consents if c.withdrawn])
            for consents in self.consent_records.values()
        ])
        
        return (withdrawn_consents / total_consents * 100) if total_consents > 0 else 0.0
    
    def _calculate_breach_notification_compliance(self) -> float:
        """Calculate breach notification compliance rate"""
        
        if not self.data_breaches:
            return 100.0
        
        compliant_breaches = len([
            b for b in self.data_breaches.values()
            if not b.notification_required or b.authority_notified
        ])
        
        return (compliant_breaches / len(self.data_breaches)) * 100
    
    def _count_expired_consents(self) -> int:
        """Count expired consents"""
        
        now = datetime.utcnow()
        expired_count = 0
        
        for consents in self.consent_records.values():
            for consent in consents:
                if consent.expires_at and consent.expires_at < now and not consent.withdrawn:
                    expired_count += 1
        
        return expired_count
    
    def _count_processing_without_consent(self) -> int:
        """Count processing activities without valid consent"""
        
        # This would check for processing activities that require consent
        # but don't have valid consent records
        # Simplified implementation
        
        consent_required_records = [
            r for r in self.processing_records.values()
            if r.legal_basis == LegalBasis.CONSENT
        ]
        
        return len(consent_required_records)  # Simplified - would check actual consent status
    
    # Storage methods
    async def _store_consent_record(self, consent: ConsentRecord) -> None:
        """Store consent record in memory"""
        
        await self.memory_manager.store_context(
            context_type="gdpr_consent_record",
            content=consent.__dict__,
            metadata={
                "consent_id": consent.consent_id,
                "data_subject_id": consent.data_subject_id,
                "consent_given": consent.consent_given,
                "withdrawn": consent.withdrawn
            }
        )
    
    async def _store_data_subject_request(self, request: DataSubjectRequest) -> None:
        """Store data subject request in memory"""
        
        await self.memory_manager.store_context(
            context_type="gdpr_data_subject_request",
            content=request.__dict__,
            metadata={
                "request_id": request.request_id,
                "data_subject_id": request.data_subject_id,
                "request_type": request.request_type.value,
                "status": request.status.value
            }
        )
    
    async def _store_data_breach(self, breach: DataBreach) -> None:
        """Store data breach record in memory"""
        
        await self.memory_manager.store_context(
            context_type="gdpr_data_breach",
            content=breach.__dict__,
            metadata={
                "breach_id": breach.breach_id,
                "severity": breach.severity,
                "notification_required": breach.notification_required,
                "discovered_at": breach.discovered_at.isoformat()
            }
        )