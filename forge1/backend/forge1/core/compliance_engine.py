# forge1/backend/forge1/core/compliance_engine.py
"""
Compliance Engine for Forge 1

Enterprise compliance with:
- GDPR, CCPA, HIPAA, SOX compliance
- Content validation and filtering
- Audit logging and reporting
- Real-time compliance monitoring
- Automated alert generation
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from fastapi import Request, Response
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ComplianceStatus(Enum):
    """Compliance status enumeration"""
    COMPLIANT = "compliant"
    WARNING = "warning"
    NON_COMPLIANT = "non_compliant"
    UNKNOWN = "unknown"

class AlertPriority(Enum):
    """Alert priority enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ComplianceAlert:
    """Compliance alert data structure"""
    id: str
    framework: str
    title: str
    description: str
    priority: AlertPriority
    created_at: datetime
    due_date: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    assigned_to: Optional[str] = None

@dataclass
class AuditEntry:
    """Audit trail entry data structure"""
    id: str
    timestamp: datetime
    framework: str
    event_type: str
    user_id: Optional[str]
    resource: str
    action: str
    result: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class ComplianceEngine:
    """Enterprise compliance engine with enhanced monitoring and alerting"""
    
    def __init__(self):
        self.audit_log: List[AuditEntry] = []
        self.active_alerts: List[ComplianceAlert] = []
        self.compliance_rules = self._load_compliance_rules()
        self.framework_scores = self._initialize_framework_scores()
        self.monitoring_enabled = True
        
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Load compliance rules with enhanced configuration"""
        return {
            "gdpr": {
                "data_retention_days": 365,
                "requires_consent": True,
                "right_to_deletion": True,
                "privacy_by_design": True,
                "data_portability": True,
                "breach_notification_hours": 72,
                "monitoring_rules": {
                    "data_access_logging": True,
                    "consent_tracking": True,
                    "deletion_requests": True
                }
            },
            "ccpa": {
                "data_retention_days": 365,
                "right_to_know": True,
                "right_to_delete": True,
                "right_to_opt_out": True,
                "non_discrimination": True,
                "monitoring_rules": {
                    "consumer_requests": True,
                    "data_sales_tracking": True,
                    "opt_out_compliance": True
                }
            },
            "hipaa": {
                "encryption_required": True,
                "audit_logging": True,
                "access_controls": True,
                "minimum_necessary": True,
                "business_associate_agreements": True,
                "breach_notification_days": 60,
                "monitoring_rules": {
                    "phi_access_logging": True,
                    "unauthorized_access_detection": True,
                    "encryption_compliance": True
                }
            },
            "sox": {
                "financial_controls": True,
                "audit_trail": True,
                "segregation_of_duties": True,
                "change_management": True,
                "documentation_requirements": True,
                "monitoring_rules": {
                    "financial_transaction_logging": True,
                    "control_testing": True,
                    "change_approval_tracking": True
                }
            },
            "pci_dss": {
                "network_security": True,
                "cardholder_data_protection": True,
                "vulnerability_management": True,
                "access_control_measures": True,
                "network_monitoring": True,
                "security_testing": True,
                "monitoring_rules": {
                    "payment_processing_logging": True,
                    "cardholder_data_access": True,
                    "security_incident_detection": True
                }
            }
        }
    
    def _initialize_framework_scores(self) -> Dict[str, float]:
        """Initialize framework compliance scores"""
        return {
            "gdpr": 98.5,
            "hipaa": 96.8,
            "sox": 97.2,
            "pci_dss": 92.1,
            "ccpa": 95.3
        }
    
    async def audit_request(self, request: Request, call_next) -> Response:
        """Audit request for compliance"""
        
        # Log request for audit trail
        audit_entry = {
            "timestamp": "now",
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host,
            "user_agent": request.headers.get("user-agent", "unknown")
        }
        
        self.audit_log.append(audit_entry)
        
        # Process request
        response = await call_next(request)
        
        # Log response
        audit_entry["status_code"] = response.status_code
        
        return response
    
    async def validate_content(self, content: str) -> bool:
        """Validate content for compliance"""
        
        # Basic content validation
        # In real implementation, this would check for:
        # - PII data
        # - Sensitive information
        # - Compliance violations
        
        prohibited_patterns = [
            "social security number",
            "ssn",
            "credit card",
            "password",
            "confidential"
        ]
        
        content_lower = content.lower()
        for pattern in prohibited_patterns:
            if pattern in content_lower:
                logger.warning(f"Content contains prohibited pattern: {pattern}")
                return False
        
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Compliance engine health check"""
        try:
            return {
                "status": "healthy",
                "compliance_rules_loaded": len(self.compliance_rules),
                "audit_log_entries": len(self.audit_log),
                "regulations_supported": ["GDPR", "CCPA", "HIPAA", "SOX"]
            }
        except Exception as e:
            logger.error(f"Compliance engine health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }    
    async def log_audit_event(self, 
                             framework: str,
                             event_type: str,
                             user_id: Optional[str],
                             resource: str,
                             action: str,
                             result: str,
                             request: Optional[Request] = None,
                             details: Optional[Dict[str, Any]] = None) -> str:
        """Log an audit event for compliance tracking"""
        
        audit_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        audit_entry = AuditEntry(
            id=audit_id,
            timestamp=datetime.now(),
            framework=framework,
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            ip_address=request.client.host if request else None,
            user_agent=request.headers.get("user-agent") if request else None,
            details=details or {}
        )
        
        self.audit_log.append(audit_entry)
        
        # Check for compliance violations
        await self._check_compliance_violations(audit_entry)
        
        logger.info(f"Audit event logged: {audit_id} - {framework} {event_type}")
        return audit_id
    
    async def _check_compliance_violations(self, audit_entry: AuditEntry):
        """Check audit entry for potential compliance violations"""
        
        framework_rules = self.compliance_rules.get(audit_entry.framework.lower())
        if not framework_rules:
            return
        
        # Check for suspicious patterns
        violations = []
        
        # Check for failed access attempts
        if audit_entry.result == "FAILED" and audit_entry.action in ["READ", "WRITE", "DELETE"]:
            violations.append({
                "type": "unauthorized_access_attempt",
                "severity": "high",
                "description": f"Failed {audit_entry.action} attempt on {audit_entry.resource}"
            })
        
        # Check for sensitive data access outside business hours
        if audit_entry.timestamp.hour < 6 or audit_entry.timestamp.hour > 22:
            if "sensitive" in audit_entry.resource.lower() or "patient" in audit_entry.resource.lower():
                violations.append({
                    "type": "after_hours_access",
                    "severity": "medium",
                    "description": f"Sensitive data access outside business hours: {audit_entry.resource}"
                })
        
        # Generate alerts for violations
        for violation in violations:
            await self._generate_compliance_alert(
                framework=audit_entry.framework,
                title=f"Compliance Violation: {violation['type']}",
                description=violation['description'],
                priority=AlertPriority.HIGH if violation['severity'] == "high" else AlertPriority.MEDIUM
            )
    
    async def _generate_compliance_alert(self,
                                       framework: str,
                                       title: str,
                                       description: str,
                                       priority: AlertPriority,
                                       due_date: Optional[datetime] = None,
                                       assigned_to: Optional[str] = None):
        """Generate a compliance alert"""
        
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        alert = ComplianceAlert(
            id=alert_id,
            framework=framework,
            title=title,
            description=description,
            priority=priority,
            created_at=datetime.now(),
            due_date=due_date,
            assigned_to=assigned_to
        )
        
        self.active_alerts.append(alert)
        
        logger.warning(f"Compliance alert generated: {alert_id} - {title}")
        
        # In production, this would trigger notifications (email, Slack, etc.)
        await self._send_alert_notification(alert)
    
    async def _send_alert_notification(self, alert: ComplianceAlert):
        """Send alert notification to relevant stakeholders"""
        
        # Mock notification - in production this would integrate with:
        # - Email systems
        # - Slack/Teams
        # - SMS alerts
        # - Dashboard notifications
        
        logger.info(f"Alert notification sent for {alert.id}: {alert.title}")
    
    async def get_compliance_score(self, framework: str) -> float:
        """Get current compliance score for a framework"""
        
        return self.framework_scores.get(framework.lower(), 0.0)
    
    async def update_compliance_score(self, framework: str, score: float):
        """Update compliance score for a framework"""
        
        old_score = self.framework_scores.get(framework.lower(), 0.0)
        self.framework_scores[framework.lower()] = score
        
        # Generate alert if score drops significantly
        if score < old_score - 5.0:
            await self._generate_compliance_alert(
                framework=framework,
                title=f"Compliance Score Drop: {framework}",
                description=f"Compliance score dropped from {old_score:.1f}% to {score:.1f}%",
                priority=AlertPriority.HIGH
            )
    
    async def get_audit_trail(self,
                            framework: Optional[str] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            limit: int = 100) -> List[AuditEntry]:
        """Get filtered audit trail entries"""
        
        filtered_entries = self.audit_log
        
        if framework:
            filtered_entries = [e for e in filtered_entries if e.framework.lower() == framework.lower()]
        
        if start_date:
            filtered_entries = [e for e in filtered_entries if e.timestamp >= start_date]
        
        if end_date:
            filtered_entries = [e for e in filtered_entries if e.timestamp <= end_date]
        
        # Sort by timestamp (newest first) and apply limit
        filtered_entries.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_entries[:limit]
    
    async def get_active_alerts(self,
                              framework: Optional[str] = None,
                              priority: Optional[AlertPriority] = None) -> List[ComplianceAlert]:
        """Get active compliance alerts with optional filtering"""
        
        filtered_alerts = [a for a in self.active_alerts if not a.resolved_at]
        
        if framework:
            filtered_alerts = [a for a in filtered_alerts if a.framework.lower() == framework.lower()]
        
        if priority:
            filtered_alerts = [a for a in filtered_alerts if a.priority == priority]
        
        # Sort by priority and creation time
        priority_order = {AlertPriority.CRITICAL: 0, AlertPriority.HIGH: 1, AlertPriority.MEDIUM: 2, AlertPriority.LOW: 3}
        filtered_alerts.sort(key=lambda x: (priority_order[x.priority], x.created_at), reverse=True)
        
        return filtered_alerts
    
    async def resolve_alert(self, alert_id: str, resolved_by: Optional[str] = None, resolution_note: Optional[str] = None) -> bool:
        """Resolve a compliance alert"""
        
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.resolved_at = datetime.now()
                logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
                return True
        
        return False
    
    async def generate_compliance_report(self,
                                       framework: str,
                                       start_date: datetime,
                                       end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        
        # Get audit entries for the period
        audit_entries = await self.get_audit_trail(framework, start_date, end_date)
        
        # Get alerts for the period
        period_alerts = [a for a in self.active_alerts 
                        if a.framework.lower() == framework.lower() 
                        and start_date <= a.created_at <= end_date]
        
        # Calculate metrics
        total_events = len(audit_entries)
        failed_events = len([e for e in audit_entries if e.result == "FAILED"])
        success_rate = ((total_events - failed_events) / total_events * 100) if total_events > 0 else 100
        
        # Generate findings and recommendations
        findings = []
        recommendations = []
        
        if failed_events > 0:
            findings.append({
                "type": "security_concern",
                "description": f"{failed_events} failed access attempts detected",
                "severity": "medium" if failed_events < 10 else "high"
            })
            recommendations.append({
                "priority": "high",
                "description": "Review and strengthen access controls"
            })
        
        if len(period_alerts) > 5:
            findings.append({
                "type": "alert_volume",
                "description": f"High alert volume: {len(period_alerts)} alerts generated",
                "severity": "medium"
            })
            recommendations.append({
                "priority": "medium",
                "description": "Review alert thresholds and compliance processes"
            })
        
        return {
            "framework": framework,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "metrics": {
                "total_events": total_events,
                "failed_events": failed_events,
                "success_rate": round(success_rate, 2),
                "alerts_generated": len(period_alerts),
                "compliance_score": await self.get_compliance_score(framework)
            },
            "findings": findings,
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat()
        }
    
    async def start_monitoring(self):
        """Start compliance monitoring background tasks"""
        
        if not self.monitoring_enabled:
            return
        
        # Start background monitoring tasks
        asyncio.create_task(self._periodic_compliance_check())
        asyncio.create_task(self._alert_escalation_monitor())
        
        logger.info("Compliance monitoring started")
    
    async def _periodic_compliance_check(self):
        """Periodic compliance check background task"""
        
        while self.monitoring_enabled:
            try:
                # Check for overdue alerts
                overdue_alerts = [a for a in self.active_alerts 
                                if a.due_date and a.due_date < datetime.now() and not a.resolved_at]
                
                for alert in overdue_alerts:
                    await self._generate_compliance_alert(
                        framework=alert.framework,
                        title=f"Overdue Alert: {alert.title}",
                        description=f"Alert {alert.id} is overdue and requires immediate attention",
                        priority=AlertPriority.CRITICAL
                    )
                
                # Check compliance scores
                for framework, score in self.framework_scores.items():
                    if score < 90.0:
                        await self._generate_compliance_alert(
                            framework=framework,
                            title=f"Low Compliance Score: {framework.upper()}",
                            description=f"Compliance score ({score:.1f}%) is below acceptable threshold",
                            priority=AlertPriority.HIGH
                        )
                
                # Wait 1 hour before next check
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in periodic compliance check: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _alert_escalation_monitor(self):
        """Monitor alerts for escalation"""
        
        while self.monitoring_enabled:
            try:
                # Check for critical alerts that need escalation
                critical_alerts = [a for a in self.active_alerts 
                                 if a.priority == AlertPriority.CRITICAL 
                                 and not a.resolved_at
                                 and (datetime.now() - a.created_at).total_seconds() > 3600]  # 1 hour
                
                for alert in critical_alerts:
                    # Escalate to management
                    logger.critical(f"ESCALATION: Critical alert {alert.id} unresolved for over 1 hour")
                    # In production, this would trigger executive notifications
                
                # Wait 30 minutes before next check
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"Error in alert escalation monitor: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def audit_request(self, request: Request, call_next) -> Response:
        """Enhanced audit request for compliance with automatic logging"""
        
        start_time = datetime.now()
        
        # Log request for audit trail
        audit_entry = {
            "timestamp": start_time.isoformat(),
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host,
            "user_agent": request.headers.get("user-agent", "unknown")
        }
        
        # Process request
        response = await call_next(request)
        
        # Log response
        audit_entry["status_code"] = response.status_code
        audit_entry["duration_ms"] = (datetime.now() - start_time).total_seconds() * 1000
        
        # Determine compliance framework based on request path
        framework = self._determine_framework_from_path(request.url.path)
        
        if framework:
            await self.log_audit_event(
                framework=framework,
                event_type="api_request",
                user_id=request.headers.get("x-user-id"),
                resource=request.url.path,
                action=request.method,
                result="SUCCESS" if response.status_code < 400 else "FAILED",
                request=request,
                details=audit_entry
            )
        
        return response
    
    def _determine_framework_from_path(self, path: str) -> Optional[str]:
        """Determine compliance framework based on request path"""
        
        path_lower = path.lower()
        
        if any(keyword in path_lower for keyword in ["patient", "medical", "health", "phi"]):
            return "HIPAA"
        elif any(keyword in path_lower for keyword in ["payment", "card", "transaction", "billing"]):
            return "PCI_DSS"
        elif any(keyword in path_lower for keyword in ["financial", "accounting", "audit", "sox"]):
            return "SOX"
        elif any(keyword in path_lower for keyword in ["personal", "privacy", "gdpr", "consent"]):
            return "GDPR"
        elif any(keyword in path_lower for keyword in ["consumer", "ccpa", "california"]):
            return "CCPA"
        
        return None
    
    async def analyze_content(self, content: str, framework: Optional[str] = None) -> Dict[str, Any]:
        """Analyze content for compliance with detailed results"""
        
        violations = []
        warnings = []
        
        # Basic content validation patterns
        prohibited_patterns = {
            "ssn": r"\b\d{3}-?\d{2}-?\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
        }
        
        content_lower = content.lower()
        
        # Check for prohibited patterns
        for pattern_name, pattern in prohibited_patterns.items():
            import re
            if re.search(pattern, content):
                violations.append({
                    "type": "pii_detected",
                    "pattern": pattern_name,
                    "description": f"Potential {pattern_name.replace('_', ' ')} detected in content"
                })
        
        # Check for sensitive keywords
        sensitive_keywords = [
            "confidential", "secret", "password", "private", "restricted",
            "internal only", "do not distribute", "proprietary"
        ]
        
        for keyword in sensitive_keywords:
            if keyword in content_lower:
                warnings.append({
                    "type": "sensitive_content",
                    "keyword": keyword,
                    "description": f"Sensitive keyword '{keyword}' found in content"
                })
        
        # Framework-specific validation
        if framework:
            framework_violations = await self._validate_framework_specific(content, framework)
            violations.extend(framework_violations)
        
        is_compliant = len(violations) == 0
        
        if not is_compliant:
            logger.warning(f"Content validation failed: {len(violations)} violations found")
        
        return {
            "compliant": is_compliant,
            "violations": violations,
            "warnings": warnings,
            "framework": framework,
            "validated_at": datetime.now().isoformat()
        }
    
    async def _validate_framework_specific(self, content: str, framework: str) -> List[Dict[str, Any]]:
        """Validate content against framework-specific rules"""
        
        violations = []
        content_lower = content.lower()
        
        if framework.upper() == "HIPAA":
            # HIPAA-specific validation
            phi_indicators = ["patient", "medical record", "diagnosis", "treatment", "prescription"]
            if any(indicator in content_lower for indicator in phi_indicators):
                violations.append({
                    "type": "phi_detected",
                    "framework": "HIPAA",
                    "description": "Potential Protected Health Information (PHI) detected"
                })
        
        elif framework.upper() == "PCI_DSS":
            # PCI DSS-specific validation
            if "cardholder" in content_lower or "pan" in content_lower:
                violations.append({
                    "type": "cardholder_data",
                    "framework": "PCI_DSS",
                    "description": "Potential cardholder data detected"
                })
        
        elif framework.upper() == "SOX":
            # SOX-specific validation
            financial_terms = ["revenue", "earnings", "financial statement", "audit", "internal control"]
            if any(term in content_lower for term in financial_terms):
                violations.append({
                    "type": "financial_data",
                    "framework": "SOX",
                    "description": "Potential financial data requiring SOX controls detected"
                })
        
        return violations
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced compliance engine health check"""
        try:
            return {
                "status": "healthy",
                "compliance_rules_loaded": len(self.compliance_rules),
                "audit_log_entries": len(self.audit_log),
                "active_alerts": len([a for a in self.active_alerts if not a.resolved_at]),
                "monitoring_enabled": self.monitoring_enabled,
                "frameworks_monitored": list(self.compliance_rules.keys()),
                "average_compliance_score": round(sum(self.framework_scores.values()) / len(self.framework_scores), 1),
                "last_check": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Compliance engine health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
