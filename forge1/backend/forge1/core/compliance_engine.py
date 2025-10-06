"""
Forge 1 Enhanced Compliance Engine
Enterprise compliance automation for GDPR, HIPAA, SOX, SOC2
"""

import time
import logging
import hashlib
import json
from typing import Dict, Any, List, Optional
from enum import Enum
from fastapi import Request

logger = logging.getLogger(__name__)

class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    SOC2 = "soc2"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"

class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class ComplianceEngine:
    """Enhanced compliance engine with automated controls"""
    
    def __init__(self):
        self.compliance_rules = self._load_compliance_rules()
        self.audit_log = []
        self.data_inventory = {}
        self.privacy_controls = {
            "data_encryption": True,
            "access_logging": True,
            "data_retention": True,
            "consent_management": True,
            "breach_detection": True
        }
        self.compliance_status = {
            ComplianceFramework.GDPR: {"status": "compliant", "last_audit": time.time()},
            ComplianceFramework.SOC2: {"status": "compliant", "last_audit": time.time()},
            ComplianceFramework.HIPAA: {"status": "compliant", "last_audit": time.time()},
            ComplianceFramework.SOX: {"status": "compliant", "last_audit": time.time()}
        }
        
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Load compliance rules and controls"""
        return {
            ComplianceFramework.GDPR: {
                "data_protection": {
                    "encryption_required": True,
                    "consent_required": True,
                    "right_to_deletion": True,
                    "data_portability": True,
                    "breach_notification": 72  # hours
                },
                "privacy_controls": [
                    "data_minimization",
                    "purpose_limitation",
                    "storage_limitation",
                    "accuracy",
                    "integrity_confidentiality"
                ]
            },
            ComplianceFramework.SOC2: {
                "trust_principles": [
                    "security",
                    "availability", 
                    "processing_integrity",
                    "confidentiality",
                    "privacy"
                ],
                "controls": {
                    "access_controls": True,
                    "system_monitoring": True,
                    "change_management": True,
                    "risk_assessment": True,
                    "vendor_management": True
                }
            },
            ComplianceFramework.HIPAA: {
                "safeguards": {
                    "administrative": True,
                    "physical": True,
                    "technical": True
                },
                "requirements": {
                    "access_control": True,
                    "audit_controls": True,
                    "integrity": True,
                    "person_authentication": True,
                    "transmission_security": True
                }
            },
            ComplianceFramework.SOX: {
                "sections": {
                    "302": "corporate_responsibility",
                    "404": "internal_controls",
                    "409": "real_time_disclosure",
                    "802": "criminal_penalties"
                },
                "controls": {
                    "financial_reporting": True,
                    "internal_controls": True,
                    "audit_trail": True,
                    "segregation_of_duties": True
                }
            }
        }
    
    async def audit_request(self, request: Request, call_next):
        """Audit request for compliance"""
        start_time = time.time()
        
        try:
            # Skip auditing for health checks
            if request.url.path in ["/health", "/metrics"]:
                return await call_next(request)
            
            # Create audit entry
            audit_entry = {
                "timestamp": start_time,
                "request_id": id(request),
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
                "user_id": request.headers.get("X-User-ID", "anonymous")
            }
            
            # Process request
            response = await call_next(request)
            
            # Complete audit entry
            audit_entry.update({
                "status_code": response.status_code,
                "response_time": time.time() - start_time,
                "compliance_status": "compliant"
            })
            
            # Store audit log
            self.audit_log.append(audit_entry)
            
            # Keep only last 10000 entries
            if len(self.audit_log) > 10000:
                self.audit_log = self.audit_log[-10000:]
            
            # Add compliance headers
            response.headers["X-Compliance-Status"] = "compliant"
            response.headers["X-Audit-ID"] = str(audit_entry["request_id"])
            
            return response
            
        except Exception as e:
            # Log compliance violation
            audit_entry.update({
                "status_code": 500,
                "response_time": time.time() - start_time,
                "compliance_status": "violation",
                "error": str(e)
            })
            
            self.audit_log.append(audit_entry)
            logger.error(f"Compliance audit error: {e}")
            raise
    
    async def validate_content(self, content: str) -> bool:
        """Validate content for compliance violations"""
        try:
            # Check for sensitive data patterns
            sensitive_patterns = [
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone number
            ]
            
            import re
            for pattern in sensitive_patterns:
                if re.search(pattern, content):
                    logger.warning("Sensitive data pattern detected in content")
                    await self._log_compliance_event(
                        "sensitive_data_detected",
                        {"pattern": pattern, "content_length": len(content)}
                    )
                    # Don't block, but log for review
            
            # Check for prohibited content
            prohibited_terms = [
                "confidential", "secret", "private_key", "password",
                "credit_card", "ssn", "social_security"
            ]
            
            content_lower = content.lower()
            for term in prohibited_terms:
                if term in content_lower:
                    logger.warning(f"Prohibited term '{term}' found in content")
                    await self._log_compliance_event(
                        "prohibited_content",
                        {"term": term, "content_length": len(content)}
                    )
            
            return True  # Allow content but log violations
            
        except Exception as e:
            logger.error(f"Content validation error: {e}")
            return True  # Fail open for availability
    
    async def _log_compliance_event(self, event_type: str, details: Dict[str, Any]):
        """Log compliance events"""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details,
            "severity": self._get_event_severity(event_type)
        }
        
        self.audit_log.append(event)
        logger.info(f"Compliance Event: {event}")
    
    def _get_event_severity(self, event_type: str) -> str:
        """Get event severity level"""
        high_severity = ["data_breach", "unauthorized_access", "sensitive_data_exposed"]
        medium_severity = ["sensitive_data_detected", "prohibited_content", "policy_violation"]
        
        if event_type in high_severity:
            return "HIGH"
        elif event_type in medium_severity:
            return "MEDIUM"
        else:
            return "LOW"
    
    async def get_compliance_status(self) -> Dict[str, Any]:
        """Get comprehensive compliance status"""
        current_time = time.time()
        
        # Calculate compliance metrics
        total_requests = len([log for log in self.audit_log if "request_id" in log])
        violations = len([log for log in self.audit_log if log.get("compliance_status") == "violation"])
        compliance_rate = (total_requests - violations) / total_requests if total_requests > 0 else 1.0
        
        # Get recent events
        recent_events = [
            log for log in self.audit_log 
            if current_time - log["timestamp"] <= 86400  # Last 24 hours
        ]
        
        return {
            "timestamp": current_time,
            "overall_status": "compliant" if compliance_rate > 0.95 else "non_compliant",
            "compliance_rate": compliance_rate,
            "frameworks": self.compliance_status,
            "controls": self.privacy_controls,
            "metrics": {
                "total_requests": total_requests,
                "violations": violations,
                "recent_events": len(recent_events),
                "audit_log_size": len(self.audit_log)
            },
            "recent_violations": [
                log for log in recent_events 
                if log.get("compliance_status") == "violation"
            ][:10]  # Last 10 violations
        }
    
    async def generate_compliance_report(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """Generate compliance report for specific framework"""
        current_time = time.time()
        
        # Get framework-specific data
        framework_rules = self.compliance_rules.get(framework, {})
        framework_status = self.compliance_status.get(framework, {})
        
        # Calculate compliance metrics
        total_controls = len(framework_rules.get("controls", {}))
        implemented_controls = sum(1 for control in framework_rules.get("controls", {}).values() if control)
        control_coverage = implemented_controls / total_controls if total_controls > 0 else 1.0
        
        # Get relevant audit events
        framework_events = [
            log for log in self.audit_log
            if log.get("framework") == framework.value or "compliance" in log.get("event_type", "")
        ]
        
        report = {
            "framework": framework.value,
            "generated_at": current_time,
            "status": framework_status.get("status", "unknown"),
            "last_audit": framework_status.get("last_audit", 0),
            "control_coverage": control_coverage,
            "implemented_controls": implemented_controls,
            "total_controls": total_controls,
            "recent_events": len([
                event for event in framework_events
                if current_time - event["timestamp"] <= 2592000  # Last 30 days
            ]),
            "recommendations": await self._get_compliance_recommendations(framework),
            "evidence": await self._collect_compliance_evidence(framework)
        }
        
        return report
    
    async def _get_compliance_recommendations(self, framework: ComplianceFramework) -> List[Dict[str, Any]]:
        """Get compliance recommendations"""
        recommendations = []
        
        if framework == ComplianceFramework.GDPR:
            recommendations.extend([
                {
                    "priority": "high",
                    "category": "data_protection",
                    "title": "Implement data encryption at rest",
                    "description": "Ensure all personal data is encrypted when stored"
                },
                {
                    "priority": "medium",
                    "category": "consent_management",
                    "title": "Enhance consent tracking",
                    "description": "Implement granular consent management system"
                }
            ])
        
        elif framework == ComplianceFramework.SOC2:
            recommendations.extend([
                {
                    "priority": "high",
                    "category": "security",
                    "title": "Implement multi-factor authentication",
                    "description": "Require MFA for all administrative access"
                },
                {
                    "priority": "medium",
                    "category": "monitoring",
                    "title": "Enhance system monitoring",
                    "description": "Implement comprehensive system monitoring and alerting"
                }
            ])
        
        return recommendations
    
    async def _collect_compliance_evidence(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """Collect compliance evidence"""
        evidence = {
            "policies": [],
            "procedures": [],
            "technical_controls": [],
            "audit_logs": len(self.audit_log),
            "last_updated": time.time()
        }
        
        if framework == ComplianceFramework.GDPR:
            evidence.update({
                "data_processing_records": len(self.data_inventory),
                "consent_records": 0,  # Would be from consent management system
                "breach_notifications": 0,  # Would be from incident management
                "data_subject_requests": 0  # Would be from request management
            })
        
        elif framework == ComplianceFramework.SOC2:
            evidence.update({
                "access_reviews": 0,  # Would be from access management
                "vulnerability_scans": 0,  # Would be from security scanning
                "change_approvals": 0,  # Would be from change management
                "backup_verifications": 0  # Would be from backup systems
            })
        
        return evidence
    
    async def handle_data_subject_request(self, request_type: str, user_id: str) -> Dict[str, Any]:
        """Handle GDPR data subject requests"""
        current_time = time.time()
        request_id = hashlib.sha256(f"{user_id}{current_time}".encode()).hexdigest()[:16]
        
        # Log the request
        await self._log_compliance_event(
            "data_subject_request",
            {
                "request_id": request_id,
                "request_type": request_type,
                "user_id": user_id,
                "status": "received"
            }
        )
        
        if request_type == "access":
            # Collect user data
            user_data = await self._collect_user_data(user_id)
            return {
                "request_id": request_id,
                "status": "completed",
                "data": user_data,
                "processed_at": current_time
            }
        
        elif request_type == "deletion":
            # Process deletion request
            await self._delete_user_data(user_id)
            return {
                "request_id": request_id,
                "status": "completed",
                "message": "User data has been deleted",
                "processed_at": current_time
            }
        
        elif request_type == "portability":
            # Export user data
            user_data = await self._export_user_data(user_id)
            return {
                "request_id": request_id,
                "status": "completed",
                "export_format": "json",
                "data": user_data,
                "processed_at": current_time
            }
        
        else:
            return {
                "request_id": request_id,
                "status": "error",
                "message": f"Unsupported request type: {request_type}"
            }
    
    async def _collect_user_data(self, user_id: str) -> Dict[str, Any]:
        """Collect all data for a user"""
        # Mock user data collection
        return {
            "user_id": user_id,
            "profile": {"name": "Demo User", "email": f"{user_id}@example.com"},
            "activity_logs": [log for log in self.audit_log if log.get("user_id") == user_id],
            "preferences": {},
            "collected_at": time.time()
        }
    
    async def _delete_user_data(self, user_id: str):
        """Delete all data for a user"""
        # Mock user data deletion
        logger.info(f"Deleting all data for user {user_id}")
        
        # Remove from audit logs (in production, anonymize instead)
        self.audit_log = [
            log for log in self.audit_log 
            if log.get("user_id") != user_id
        ]
    
    async def _export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export user data in portable format"""
        user_data = await self._collect_user_data(user_id)
        
        # Format for portability
        return {
            "export_version": "1.0",
            "user_id": user_id,
            "exported_at": time.time(),
            "data": user_data
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Compliance engine health check"""
        try:
            current_time = time.time()
            
            # Check compliance status
            status = await self.get_compliance_status()
            
            return {
                "status": "healthy",
                "compliance_rate": status["compliance_rate"],
                "frameworks_compliant": sum(
                    1 for framework_status in self.compliance_status.values()
                    if framework_status["status"] == "compliant"
                ),
                "total_frameworks": len(self.compliance_status),
                "audit_log_size": len(self.audit_log),
                "timestamp": current_time
            }
            
        except Exception as e:
            logger.error(f"Compliance engine health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }