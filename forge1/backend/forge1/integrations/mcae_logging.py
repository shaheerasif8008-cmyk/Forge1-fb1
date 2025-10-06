"""
MCAE Integration Logging

Enhanced logging and audit trail system for MCAE integration operations
with compliance, security, and performance monitoring.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from enum import Enum
from contextvars import ContextVar

from forge1.core.tenancy import get_current_tenant
from forge1.integrations.workflow_context_injector import get_context

# Context variable for correlation IDs
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class LogLevel(str, Enum):
    """Enhanced log levels for MCAE operations"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"
    AUDIT = "AUDIT"
    PERFORMANCE = "PERFORMANCE"


class LogCategory(str, Enum):
    """Log categories for filtering and analysis"""
    WORKFLOW = "workflow"
    AGENT = "agent"
    TOOL = "tool"
    MEMORY = "memory"
    MODEL = "model"
    SECURITY = "security"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    ERROR = "error"


class MCAELogger:
    """
    Enhanced logger for MCAE integration with structured logging,
    audit trails, and compliance features.
    """
    
    def __init__(self, name: str = "mcae_integration"):
        self.logger = logging.getLogger(name)
        self.audit_entries = []
        self.performance_metrics = []
        self.security_events = []
        
        # Configure structured logging
        self._setup_structured_logging()
        
        # Statistics
        self.stats = {
            "total_logs": 0,
            "logs_by_level": {},
            "logs_by_category": {},
            "audit_entries": 0,
            "security_events": 0,
            "performance_entries": 0
        }
    
    def _setup_structured_logging(self):
        """Set up structured logging format"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Ensure handler exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log(
        self,
        level: LogLevel,
        category: LogCategory,
        message: str,
        **kwargs
    ) -> None:
        """
        Enhanced logging with structured data and context.
        
        Args:
            level: Log level
            category: Log category
            message: Log message
            **kwargs: Additional structured data
        """
        # Get current context
        context = get_context()
        tenant_id = get_current_tenant()
        correlation = correlation_id.get()
        
        # Build structured log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.value,
            "category": category.value,
            "message": message,
            "tenant_id": tenant_id,
            "correlation_id": correlation,
            **kwargs
        }
        
        # Add workflow context if available
        if context:
            log_entry.update({
                "workflow_context": {
                    "session_id": context.get("session_id"),
                    "employee_id": context.get("employee_id"),
                    "user_id": context.get("user_id")
                }
            })
        
        # Log the structured entry
        log_message = f"[{category.value.upper()}] {message} | {json.dumps(kwargs)}"
        
        if level == LogLevel.DEBUG:
            self.logger.debug(log_message)
        elif level == LogLevel.INFO:
            self.logger.info(log_message)
        elif level == LogLevel.WARNING:
            self.logger.warning(log_message)
        elif level == LogLevel.ERROR:
            self.logger.error(log_message)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(log_message)
        elif level == LogLevel.SECURITY:
            self.logger.critical(f"SECURITY: {log_message}")
            self._log_security_event(log_entry)
        elif level == LogLevel.AUDIT:
            self.logger.info(f"AUDIT: {log_message}")
            self._log_audit_entry(log_entry)
        elif level == LogLevel.PERFORMANCE:
            self.logger.info(f"PERF: {log_message}")
            self._log_performance_metric(log_entry)
        
        # Update statistics
        self._update_stats(level, category)
    
    def audit(self, operation: str, **kwargs) -> None:
        """Log audit entry for compliance"""
        self.log(
            LogLevel.AUDIT,
            LogCategory.INTEGRATION,
            f"Audit: {operation}",
            operation=operation,
            **kwargs
        )
    
    def security(self, event: str, **kwargs) -> None:
        """Log security event"""
        self.log(
            LogLevel.SECURITY,
            LogCategory.SECURITY,
            f"Security: {event}",
            security_event=event,
            **kwargs
        )
    
    def performance(self, metric: str, value: float, **kwargs) -> None:
        """Log performance metric"""
        self.log(
            LogLevel.PERFORMANCE,
            LogCategory.PERFORMANCE,
            f"Performance: {metric} = {value}",
            metric=metric,
            value=value,
            **kwargs
        )
    
    def workflow_start(self, workflow_id: str, **kwargs) -> None:
        """Log workflow start"""
        self.log(
            LogLevel.INFO,
            LogCategory.WORKFLOW,
            f"Workflow started: {workflow_id}",
            workflow_id=workflow_id,
            event="workflow_start",
            **kwargs
        )
    
    def workflow_complete(self, workflow_id: str, duration: float, **kwargs) -> None:
        """Log workflow completion"""
        self.log(
            LogLevel.INFO,
            LogCategory.WORKFLOW,
            f"Workflow completed: {workflow_id} in {duration:.2f}s",
            workflow_id=workflow_id,
            event="workflow_complete",
            duration=duration,
            **kwargs
        )
    
    def workflow_error(self, workflow_id: str, error: str, **kwargs) -> None:
        """Log workflow error"""
        self.log(
            LogLevel.ERROR,
            LogCategory.WORKFLOW,
            f"Workflow failed: {workflow_id} - {error}",
            workflow_id=workflow_id,
            event="workflow_error",
            error=error,
            **kwargs
        )
    
    def agent_action(self, agent_name: str, action: str, **kwargs) -> None:
        """Log agent action"""
        self.log(
            LogLevel.INFO,
            LogCategory.AGENT,
            f"Agent {agent_name} performed {action}",
            agent_name=agent_name,
            action=action,
            **kwargs
        )
    
    def tool_access(self, tool_name: str, granted: bool, **kwargs) -> None:
        """Log tool access attempt"""
        level = LogLevel.INFO if granted else LogLevel.WARNING
        self.log(
            level,
            LogCategory.TOOL,
            f"Tool access {'granted' if granted else 'denied'}: {tool_name}",
            tool_name=tool_name,
            access_granted=granted,
            **kwargs
        )
    
    def memory_operation(self, operation: str, **kwargs) -> None:
        """Log memory operation"""
        self.log(
            LogLevel.DEBUG,
            LogCategory.MEMORY,
            f"Memory operation: {operation}",
            operation=operation,
            **kwargs
        )
    
    def model_request(self, model: str, tokens: int, **kwargs) -> None:
        """Log model request"""
        self.log(
            LogLevel.DEBUG,
            LogCategory.MODEL,
            f"Model request: {model} ({tokens} tokens)",
            model=model,
            tokens=tokens,
            **kwargs
        )
    
    def tenant_violation(self, attempted_tenant: str, current_tenant: str, **kwargs) -> None:
        """Log tenant isolation violation"""
        self.security(
            "tenant_isolation_violation",
            attempted_tenant=attempted_tenant,
            current_tenant=current_tenant,
            severity="critical",
            **kwargs
        )
    
    def _log_audit_entry(self, log_entry: Dict[str, Any]) -> None:
        """Store audit entry for compliance"""
        self.audit_entries.append(log_entry)
        self.stats["audit_entries"] += 1
        
        # Limit audit log size
        if len(self.audit_entries) > 10000:
            self.audit_entries = self.audit_entries[-5000:]
    
    def _log_security_event(self, log_entry: Dict[str, Any]) -> None:
        """Store security event"""
        self.security_events.append(log_entry)
        self.stats["security_events"] += 1
        
        # Limit security log size
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-500:]
    
    def _log_performance_metric(self, log_entry: Dict[str, Any]) -> None:
        """Store performance metric"""
        self.performance_metrics.append(log_entry)
        self.stats["performance_entries"] += 1
        
        # Limit performance log size
        if len(self.performance_metrics) > 5000:
            self.performance_metrics = self.performance_metrics[-2500:]
    
    def _update_stats(self, level: LogLevel, category: LogCategory) -> None:
        """Update logging statistics"""
        self.stats["total_logs"] += 1
        
        # Update level stats
        if level.value not in self.stats["logs_by_level"]:
            self.stats["logs_by_level"][level.value] = 0
        self.stats["logs_by_level"][level.value] += 1
        
        # Update category stats
        if category.value not in self.stats["logs_by_category"]:
            self.stats["logs_by_category"][category.value] = 0
        self.stats["logs_by_category"][category.value] += 1
    
    def get_audit_trail(self, tenant_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit trail entries"""
        entries = self.audit_entries
        
        if tenant_id:
            entries = [e for e in entries if e.get("tenant_id") == tenant_id]
        
        return entries[-limit:]
    
    def get_security_events(self, tenant_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get security events"""
        events = self.security_events
        
        if tenant_id:
            events = [e for e in events if e.get("tenant_id") == tenant_id]
        
        return events[-limit:]
    
    def get_performance_metrics(self, metric_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get performance metrics"""
        metrics = self.performance_metrics
        
        if metric_name:
            metrics = [m for m in metrics if m.get("metric") == metric_name]
        
        return metrics[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            **self.stats,
            "audit_trail_size": len(self.audit_entries),
            "security_events_size": len(self.security_events),
            "performance_metrics_size": len(self.performance_metrics)
        }
    
    def export_logs(self, format: str = "json", tenant_id: Optional[str] = None) -> str:
        """Export logs for compliance or analysis"""
        export_data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "tenant_id": tenant_id,
            "audit_entries": self.get_audit_trail(tenant_id),
            "security_events": self.get_security_events(tenant_id),
            "performance_metrics": self.get_performance_metrics(),
            "statistics": self.get_stats()
        }
        
        if format == "json":
            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global logger instance
mcae_logger = MCAELogger()


# Convenience functions
def log_workflow_start(workflow_id: str, **kwargs):
    """Log workflow start"""
    mcae_logger.workflow_start(workflow_id, **kwargs)


def log_workflow_complete(workflow_id: str, duration: float, **kwargs):
    """Log workflow completion"""
    mcae_logger.workflow_complete(workflow_id, duration, **kwargs)


def log_workflow_error(workflow_id: str, error: str, **kwargs):
    """Log workflow error"""
    mcae_logger.workflow_error(workflow_id, error, **kwargs)


def log_agent_action(agent_name: str, action: str, **kwargs):
    """Log agent action"""
    mcae_logger.agent_action(agent_name, action, **kwargs)


def log_tool_access(tool_name: str, granted: bool, **kwargs):
    """Log tool access"""
    mcae_logger.tool_access(tool_name, granted, **kwargs)


def log_security_event(event: str, **kwargs):
    """Log security event"""
    mcae_logger.security(event, **kwargs)


def log_audit(operation: str, **kwargs):
    """Log audit entry"""
    mcae_logger.audit(operation, **kwargs)


def log_performance(metric: str, value: float, **kwargs):
    """Log performance metric"""
    mcae_logger.performance(metric, value, **kwargs)


def log_tenant_violation(attempted_tenant: str, current_tenant: str, **kwargs):
    """Log tenant violation"""
    mcae_logger.tenant_violation(attempted_tenant, current_tenant, **kwargs)


# Decorators for automatic logging
def log_workflow_execution(func):
    """Decorator to automatically log workflow execution"""
    async def wrapper(*args, **kwargs):
        workflow_id = kwargs.get('workflow_id', 'unknown')
        start_time = time.time()
        
        log_workflow_start(workflow_id, function=func.__name__)
        
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            log_workflow_complete(workflow_id, duration, function=func.__name__)
            return result
        except Exception as e:
            duration = time.time() - start_time
            log_workflow_error(workflow_id, str(e), function=func.__name__, duration=duration)
            raise
    
    return wrapper


def log_agent_operations(func):
    """Decorator to automatically log agent operations"""
    async def wrapper(*args, **kwargs):
        agent_name = kwargs.get('agent_name', args[0].__class__.__name__ if args else 'unknown')
        operation = func.__name__
        
        log_agent_action(agent_name, operation, function=func.__name__)
        
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            mcae_logger.log(
                LogLevel.ERROR,
                LogCategory.AGENT,
                f"Agent operation failed: {agent_name}.{operation}",
                agent_name=agent_name,
                operation=operation,
                error=str(e)
            )
            raise
    
    return wrapper


def log_performance_metrics(metric_name: str):
    """Decorator to automatically log performance metrics"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                log_performance(metric_name, duration, function=func.__name__)
                return result
            except Exception as e:
                duration = time.time() - start_time
                log_performance(f"{metric_name}_error", duration, function=func.__name__, error=str(e))
                raise
        
        return wrapper
    return decorator


# Context managers for correlation tracking
class CorrelationContext:
    """Context manager for correlation ID tracking"""
    
    def __init__(self, correlation_id_value: str):
        self.correlation_id_value = correlation_id_value
        self.token = None
    
    def __enter__(self):
        self.token = correlation_id.set(self.correlation_id_value)
        return self.correlation_id_value
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        correlation_id.reset(self.token)


def with_correlation_id(correlation_id_value: str):
    """Context manager for correlation ID"""
    return CorrelationContext(correlation_id_value)


# Health check function
async def health_check() -> Dict[str, Any]:
    """Health check for MCAE logging system"""
    try:
        stats = mcae_logger.get_stats()
        
        return {
            "status": "healthy",
            "logging_stats": stats,
            "recent_audit_entries": len(mcae_logger.get_audit_trail(limit=10)),
            "recent_security_events": len(mcae_logger.get_security_events(limit=10))
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }