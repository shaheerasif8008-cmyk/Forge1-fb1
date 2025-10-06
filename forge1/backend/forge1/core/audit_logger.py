# forge1/backend/forge1/core/audit_logger.py
"""
Audit Logger for Employee Lifecycle System

Provides comprehensive audit logging for all employee interactions and management operations.
Ensures compliance with security and regulatory requirements.

Requirements: 8.3, 8.4
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from enum import Enum

from forge1.core.database_config import DatabaseManager, get_database_manager

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events"""
    # Authentication and Authorization
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    
    # Client Management
    CLIENT_CREATED = "client_created"
    CLIENT_UPDATED = "client_updated"
    CLIENT_DELETED = "client_deleted"
    CLIENT_ACCESSED = "client_accessed"
    
    # Employee Management
    EMPLOYEE_CREATED = "employee_created"
    EMPLOYEE_UPDATED = "employee_updated"
    EMPLOYEE_DELETED = "employee_deleted"
    EMPLOYEE_ACCESSED = "employee_accessed"
    EMPLOYEE_CONFIGURATION_CHANGED = "employee_configuration_changed"
    
    # Employee Interactions
    EMPLOYEE_INTERACTION = "employee_interaction"
    EMPLOYEE_MEMORY_ACCESSED = "employee_memory_accessed"
    EMPLOYEE_MEMORY_EXPORTED = "employee_memory_exported"
    EMPLOYEE_MEMORY_BACKUP = "employee_memory_backup"
    EMPLOYEE_MEMORY_RESTORED = "employee_memory_restored"
    
    # Security Events
    SECURITY_VIOLATION = "security_violation"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    
    # System Events
    SYSTEM_ERROR = "system_error"
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIGURATION_CHANGE = "configuration_change"
    
    # Request Tracking
    REQUEST_START = "request_start"
    REQUEST_SUCCESS = "request_success"
    REQUEST_FAILURE = "request_failure"


class AuditSeverity(Enum):
    """Severity levels for audit events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditLogger:
    """
    Comprehensive audit logger for the employee lifecycle system.
    
    Features:
    - Structured audit logging with consistent format
    - Multiple output destinations (database, files, external systems)
    - Real-time security event alerting
    - Compliance reporting capabilities
    - Tamper-evident logging with checksums
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager
        self._initialized = False
        self.log_to_file = os.getenv("AUDIT_LOG_TO_FILE", "true").lower() == "true"
        self.log_to_database = os.getenv("AUDIT_LOG_TO_DATABASE", "true").lower() == "true"
        self.log_file_path = os.getenv("AUDIT_LOG_FILE_PATH", "/var/log/forge1/audit.log")
        
        # Security event thresholds
        self.security_alert_threshold = {
            AuditEventType.UNAUTHORIZED_ACCESS: 5,  # 5 attempts in 10 minutes
            AuditEventType.RATE_LIMIT_EXCEEDED: 3,  # 3 rate limit violations in 10 minutes
            AuditEventType.LOGIN_FAILURE: 10,       # 10 failed logins in 10 minutes
        }
        
        # Recent events cache for pattern detection
        self.recent_events = {}
        self.alert_window_minutes = 10
    
    async def initialize(self):
        """Initialize the audit logger"""
        if self._initialized:
            return
        
        if not self.db_manager:
            self.db_manager = await get_database_manager()
        
        # Create audit log table if it doesn't exist
        if self.log_to_database:
            await self._create_audit_table()
        
        # Ensure log directory exists
        if self.log_to_file:
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        
        self._initialized = True
        logger.info("Audit logger initialized")
    
    async def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity = AuditSeverity.MEDIUM,
        context: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None
    ):
        """
        Log an audit event with full context and details.
        
        Args:
            event_type: Type of event being logged
            severity: Severity level of the event
            context: Security context or user context
            details: Additional event details
            resource_id: ID of the resource being accessed/modified
            resource_type: Type of resource (client, employee, etc.)
        """
        
        if not self._initialized:
            await self.initialize()
        
        # Create audit record
        audit_record = await self._create_audit_record(
            event_type, severity, context, details, resource_id, resource_type
        )
        
        # Log to different destinations
        await asyncio.gather(
            self._log_to_database_async(audit_record) if self.log_to_database else self._noop(),
            self._log_to_file_async(audit_record) if self.log_to_file else self._noop(),
            self._check_security_patterns(audit_record)
        )
    
    # Convenience methods for common audit events
    
    async def log_client_event(
        self,
        event_type: AuditEventType,
        client_id: str,
        context: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: AuditSeverity = AuditSeverity.MEDIUM
    ):
        """Log client-related audit event"""
        await self.log_event(
            event_type=event_type,
            severity=severity,
            context=context,
            details=details,
            resource_id=client_id,
            resource_type="client"
        )
    
    async def log_employee_event(
        self,
        event_type: AuditEventType,
        employee_id: str,
        client_id: str,
        context: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: AuditSeverity = AuditSeverity.MEDIUM
    ):
        """Log employee-related audit event"""
        event_details = details or {}
        event_details["client_id"] = client_id
        
        await self.log_event(
            event_type=event_type,
            severity=severity,
            context=context,
            details=event_details,
            resource_id=employee_id,
            resource_type="employee"
        )
    
    async def log_interaction_event(
        self,
        employee_id: str,
        client_id: str,
        interaction_id: str,
        context: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log employee interaction event"""
        event_details = details or {}
        event_details.update({
            "client_id": client_id,
            "interaction_id": interaction_id
        })
        
        await self.log_event(
            event_type=AuditEventType.EMPLOYEE_INTERACTION,
            severity=AuditSeverity.LOW,
            context=context,
            details=event_details,
            resource_id=employee_id,
            resource_type="employee"
        )
    
    async def log_security_violation(
        self,
        event_type: AuditEventType,
        context: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log security violation with high severity"""
        await self.log_event(
            event_type=event_type,
            severity=AuditSeverity.HIGH,
            context=context,
            details=details,
            resource_type="security"
        )
    
    async def log_system_event(
        self,
        event_type: AuditEventType,
        context: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: AuditSeverity = AuditSeverity.MEDIUM
    ):
        """Log system-level event"""
        await self.log_event(
            event_type=event_type,
            severity=severity,
            context=context,
            details=details,
            resource_type="system"
        )
    
    async def log_request(
        self,
        event_type: AuditEventType,
        context: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log HTTP request event"""
        await self.log_event(
            event_type=event_type,
            severity=AuditSeverity.LOW,
            context=context,
            details=details,
            resource_type="request"
        )
    
    # Query methods for audit data
    
    async def get_audit_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        severity: Optional[AuditSeverity] = None,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Query audit events with filters"""
        
        if not self.log_to_database:
            return []
        
        # Build query conditions
        conditions = []
        params = []
        param_count = 0
        
        if start_date:
            param_count += 1
            conditions.append(f"timestamp >= ${param_count}")
            params.append(start_date)
        
        if end_date:
            param_count += 1
            conditions.append(f"timestamp <= ${param_count}")
            params.append(end_date)
        
        if event_types:
            param_count += 1
            conditions.append(f"event_type = ANY(${param_count})")
            params.append([et.value for et in event_types])
        
        if severity:
            param_count += 1
            conditions.append(f"severity = ${param_count}")
            params.append(severity.value)
        
        if resource_id:
            param_count += 1
            conditions.append(f"resource_id = ${param_count}")
            params.append(resource_id)
        
        if resource_type:
            param_count += 1
            conditions.append(f"resource_type = ${param_count}")
            params.append(resource_type)
        
        if tenant_id:
            param_count += 1
            conditions.append(f"tenant_id = ${param_count}")
            params.append(tenant_id)
        
        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        
        query = f"""
            SELECT 
                id, event_type, severity, timestamp, tenant_id, user_id,
                resource_id, resource_type, details, ip_address, user_agent,
                request_id, checksum
            FROM audit_log 
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ${param_count + 1} OFFSET ${param_count + 2}
        """
        params.extend([limit, offset])
        
        try:
            async with self.db_manager.get_connection() as conn:
                rows = await conn.fetch(query, *params)
                
                return [
                    {
                        "id": str(row["id"]),
                        "event_type": row["event_type"],
                        "severity": row["severity"],
                        "timestamp": row["timestamp"].isoformat(),
                        "tenant_id": row["tenant_id"],
                        "user_id": row["user_id"],
                        "resource_id": row["resource_id"],
                        "resource_type": row["resource_type"],
                        "details": row["details"],
                        "ip_address": row["ip_address"],
                        "user_agent": row["user_agent"],
                        "request_id": row["request_id"],
                        "checksum": row["checksum"]
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to query audit events: {e}")
            return []
    
    async def get_security_summary(
        self,
        hours: int = 24,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get security event summary for the specified time period"""
        
        if not self.log_to_database:
            return {"error": "Database logging not enabled"}
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time.replace(hour=end_time.hour - hours)
        
        try:
            async with self.db_manager.get_connection() as conn:
                # Security event counts
                security_events = await conn.fetch("""
                    SELECT event_type, severity, COUNT(*) as count
                    FROM audit_log 
                    WHERE timestamp >= $1 AND timestamp <= $2
                    AND ($3::text IS NULL OR tenant_id = $3)
                    AND event_type IN ('security_violation', 'unauthorized_access', 'rate_limit_exceeded', 'login_failure')
                    GROUP BY event_type, severity
                    ORDER BY count DESC
                """, start_time, end_time, tenant_id)
                
                # Top IP addresses with security events
                top_ips = await conn.fetch("""
                    SELECT ip_address, COUNT(*) as count
                    FROM audit_log 
                    WHERE timestamp >= $1 AND timestamp <= $2
                    AND ($3::text IS NULL OR tenant_id = $3)
                    AND event_type IN ('security_violation', 'unauthorized_access', 'rate_limit_exceeded')
                    GROUP BY ip_address
                    ORDER BY count DESC
                    LIMIT 10
                """, start_time, end_time, tenant_id)
                
                # Failed login attempts
                failed_logins = await conn.fetchval("""
                    SELECT COUNT(*) FROM audit_log 
                    WHERE timestamp >= $1 AND timestamp <= $2
                    AND ($3::text IS NULL OR tenant_id = $3)
                    AND event_type = 'login_failure'
                """, start_time, end_time, tenant_id)
                
                return {
                    "period": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                        "hours": hours
                    },
                    "security_events": [
                        {
                            "event_type": row["event_type"],
                            "severity": row["severity"],
                            "count": row["count"]
                        }
                        for row in security_events
                    ],
                    "top_suspicious_ips": [
                        {
                            "ip_address": row["ip_address"],
                            "event_count": row["count"]
                        }
                        for row in top_ips
                    ],
                    "failed_login_attempts": failed_logins,
                    "tenant_id": tenant_id
                }
        except Exception as e:
            logger.error(f"Failed to get security summary: {e}")
            return {"error": str(e)}
    
    # Private methods
    
    async def _create_audit_record(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        context: Optional[Any],
        details: Optional[Dict[str, Any]],
        resource_id: Optional[str],
        resource_type: Optional[str]
    ) -> Dict[str, Any]:
        """Create a structured audit record"""
        
        timestamp = datetime.now(timezone.utc)
        
        # Extract context information
        tenant_id = None
        user_id = None
        ip_address = None
        user_agent = None
        request_id = None
        
        if context:
            if hasattr(context, 'tenant_id'):
                tenant_id = context.tenant_id
            if hasattr(context, 'user_id'):
                user_id = context.user_id
            if hasattr(context, 'ip_address'):
                ip_address = context.ip_address
            if hasattr(context, 'user_agent'):
                user_agent = context.user_agent
            if hasattr(context, 'request_id'):
                request_id = context.request_id
        
        # Create base record
        record = {
            "event_type": event_type.value,
            "severity": severity.value,
            "timestamp": timestamp,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "resource_id": resource_id,
            "resource_type": resource_type,
            "details": details or {},
            "ip_address": ip_address,
            "user_agent": user_agent,
            "request_id": request_id
        }
        
        # Add checksum for tamper detection
        record["checksum"] = self._calculate_checksum(record)
        
        return record
    
    def _calculate_checksum(self, record: Dict[str, Any]) -> str:
        """Calculate checksum for audit record integrity"""
        import hashlib
        
        # Create deterministic string representation
        checksum_data = {
            k: v for k, v in record.items() 
            if k != "checksum" and v is not None
        }
        
        # Convert to JSON string with sorted keys
        json_str = json.dumps(checksum_data, sort_keys=True, default=str)
        
        # Calculate SHA-256 hash
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    async def _create_audit_table(self):
        """Create audit log table if it doesn't exist"""
        try:
            async with self.db_manager.get_connection() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        event_type VARCHAR(100) NOT NULL,
                        severity VARCHAR(20) NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        tenant_id VARCHAR(255),
                        user_id VARCHAR(255),
                        resource_id VARCHAR(255),
                        resource_type VARCHAR(100),
                        details JSONB,
                        ip_address INET,
                        user_agent TEXT,
                        request_id VARCHAR(255),
                        checksum VARCHAR(64) NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # Create indexes for performance
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON audit_log(event_type);
                    CREATE INDEX IF NOT EXISTS idx_audit_log_tenant_id ON audit_log(tenant_id);
                    CREATE INDEX IF NOT EXISTS idx_audit_log_resource ON audit_log(resource_type, resource_id);
                    CREATE INDEX IF NOT EXISTS idx_audit_log_severity ON audit_log(severity);
                """)
                
        except Exception as e:
            logger.error(f"Failed to create audit table: {e}")
    
    async def _log_to_database_async(self, record: Dict[str, Any]):
        """Log audit record to database"""
        try:
            async with self.db_manager.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO audit_log (
                        event_type, severity, timestamp, tenant_id, user_id,
                        resource_id, resource_type, details, ip_address, user_agent,
                        request_id, checksum
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, 
                    record["event_type"], record["severity"], record["timestamp"],
                    record["tenant_id"], record["user_id"], record["resource_id"],
                    record["resource_type"], json.dumps(record["details"]),
                    record["ip_address"], record["user_agent"], record["request_id"],
                    record["checksum"]
                )
        except Exception as e:
            logger.error(f"Failed to log to database: {e}")
    
    async def _log_to_file_async(self, record: Dict[str, Any]):
        """Log audit record to file"""
        try:
            log_line = json.dumps(record, default=str) + "\n"
            
            # In production, use proper async file I/O
            with open(self.log_file_path, "a") as f:
                f.write(log_line)
        except Exception as e:
            logger.error(f"Failed to log to file: {e}")
    
    async def _check_security_patterns(self, record: Dict[str, Any]):
        """Check for security patterns and trigger alerts"""
        event_type = AuditEventType(record["event_type"])
        
        if event_type in self.security_alert_threshold:
            # Track recent events for pattern detection
            key = f"{event_type.value}:{record.get('ip_address', 'unknown')}"
            current_time = datetime.now(timezone.utc)
            
            if key not in self.recent_events:
                self.recent_events[key] = []
            
            # Add current event
            self.recent_events[key].append(current_time)
            
            # Remove old events outside the window
            window_start = current_time.replace(
                minute=current_time.minute - self.alert_window_minutes
            )
            self.recent_events[key] = [
                event_time for event_time in self.recent_events[key]
                if event_time >= window_start
            ]
            
            # Check if threshold exceeded
            if len(self.recent_events[key]) >= self.security_alert_threshold[event_type]:
                await self._trigger_security_alert(event_type, record)
    
    async def _trigger_security_alert(self, event_type: AuditEventType, record: Dict[str, Any]):
        """Trigger security alert for suspicious patterns"""
        logger.warning(f"Security alert triggered: {event_type.value} threshold exceeded")
        
        # In production, this would:
        # - Send alerts to security team
        # - Trigger automated responses (IP blocking, etc.)
        # - Update security monitoring dashboards
        # - Create incident tickets
        
        alert_details = {
            "alert_type": "security_pattern_detected",
            "event_type": event_type.value,
            "threshold_exceeded": self.security_alert_threshold[event_type],
            "window_minutes": self.alert_window_minutes,
            "triggering_record": record
        }
        
        await self.log_event(
            event_type=AuditEventType.SUSPICIOUS_ACTIVITY,
            severity=AuditSeverity.CRITICAL,
            details=alert_details,
            resource_type="security_alert"
        )
    
    async def _noop(self):
        """No-op async function for conditional execution"""
        pass