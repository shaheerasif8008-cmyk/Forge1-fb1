"""
Usage Tracking and Metrics Collection System

Comprehensive system for tracking LLM usage, tool execution metrics,
and workflow performance with CSV export capabilities for billing.
"""

import asyncio
import csv
import io
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json

from forge1.core.memory_manager import MemoryManager
from forge1.core.audit_logger import AuditLogger

logger = logging.getLogger(__name__)

@dataclass
class UsageEvent:
    """Individual usage event for tracking"""
    timestamp: str
    tenant_id: str
    employee_id: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_estimate: float
    tool_calls: str  # JSON string of tool calls
    request_id: Optional[str] = None
    workflow_id: Optional[str] = None
    step_name: Optional[str] = None

class UsageTracker:
    """Tracks usage metrics for billing and monitoring"""
    
    def __init__(self, memory_manager: MemoryManager, audit_logger: AuditLogger):
        self.memory_manager = memory_manager
        self.audit_logger = audit_logger
        self.usage_events: List[UsageEvent] = []
        self.batch_size = 100
        self.flush_interval = 300  # 5 minutes
        
        # Start background flush task
        asyncio.create_task(self._periodic_flush())
    
    async def track_llm_usage(
        self,
        tenant_id: str,
        employee_id: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        cost_estimate: float = 0.0,
        request_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        step_name: Optional[str] = None
    ) -> None:
        """Track LLM usage event"""
        
        event = UsageEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            tenant_id=tenant_id,
            employee_id=employee_id,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_estimate=cost_estimate,
            tool_calls="[]",  # No tool calls for direct LLM usage
            request_id=request_id,
            workflow_id=workflow_id,
            step_name=step_name
        )
        
        await self._add_usage_event(event)
    
    async def track_tool_usage(
        self,
        tenant_id: str,
        employee_id: str,
        tool_name: str,
        execution_time_ms: float,
        success: bool,
        tool_calls: List[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        step_name: Optional[str] = None
    ) -> None:
        """Track tool usage event"""
        
        event = UsageEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            tenant_id=tenant_id,
            employee_id=employee_id,
            model=f"tool:{tool_name}",
            tokens_in=0,  # Tools don't use tokens directly
            tokens_out=0,
            latency_ms=execution_time_ms,
            cost_estimate=0.0,  # Tool cost would be calculated separately
            tool_calls=json.dumps(tool_calls or []),
            request_id=request_id,
            workflow_id=workflow_id,
            step_name=step_name
        )
        
        await self._add_usage_event(event)
    
    async def _add_usage_event(self, event: UsageEvent) -> None:
        """Add usage event to tracking"""
        self.usage_events.append(event)
        
        # Flush if batch size reached
        if len(self.usage_events) >= self.batch_size:
            await self._flush_events()
    
    async def _flush_events(self) -> None:
        """Flush usage events to storage"""
        if not self.usage_events:
            return
        
        try:
            # Store events in memory system
            for event in self.usage_events:
                await self.memory_manager.store_memory(
                    content=asdict(event),
                    memory_type="usage_event",
                    metadata={
                        "tenant_id": event.tenant_id,
                        "employee_id": event.employee_id,
                        "model": event.model,
                        "workflow_id": event.workflow_id,
                        "timestamp": event.timestamp
                    }
                )
            
            logger.info(f"Flushed {len(self.usage_events)} usage events")
            self.usage_events.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush usage events: {e}")
    
    async def _periodic_flush(self) -> None:
        """Periodically flush events"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_events()
            except Exception as e:
                logger.error(f"Periodic flush failed: {e}")
    
    async def export_usage_csv(
        self,
        tenant_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> str:
        """Export usage data as CSV string"""
        
        try:
            # Query usage events from memory system
            # This would use the memory manager's search functionality
            events = await self._query_usage_events(tenant_id, start_date, end_date)
            
            # Create CSV
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                'timestamp', 'tenant_id', 'employee_id', 'model',
                'tokens_in', 'tokens_out', 'latency_ms', 'cost_estimate',
                'tool_calls', 'request_id', 'workflow_id', 'step_name'
            ])
            
            # Write data
            for event in events:
                writer.writerow([
                    event.timestamp, event.tenant_id, event.employee_id, event.model,
                    event.tokens_in, event.tokens_out, event.latency_ms, event.cost_estimate,
                    event.tool_calls, event.request_id, event.workflow_id, event.step_name
                ])
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to export usage CSV: {e}")
            return ""
    
    async def _query_usage_events(
        self,
        tenant_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[UsageEvent]:
        """Query usage events from storage"""
        
        # This would implement actual querying from memory system
        # For now, return current in-memory events
        filtered_events = []
        
        for event in self.usage_events:
            # Apply filters
            if tenant_id and event.tenant_id != tenant_id:
                continue
            
            event_time = datetime.fromisoformat(event.timestamp.replace('Z', '+00:00'))
            if start_date and event_time < start_date:
                continue
            if end_date and event_time > end_date:
                continue
            
            filtered_events.append(event)
        
        return filtered_events
    
    async def get_usage_summary(
        self,
        tenant_id: Optional[str] = None,
        time_period_hours: int = 24
    ) -> Dict[str, Any]:
        """Get usage summary for a time period"""
        
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=time_period_hours)
            
            events = await self._query_usage_events(tenant_id, start_time, end_time)
            
            # Calculate summary statistics
            total_tokens_in = sum(event.tokens_in for event in events)
            total_tokens_out = sum(event.tokens_out for event in events)
            total_cost = sum(event.cost_estimate for event in events)
            total_requests = len(events)
            
            # Model breakdown
            model_usage = {}
            for event in events:
                if event.model not in model_usage:
                    model_usage[event.model] = {
                        "requests": 0,
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "cost": 0.0,
                        "avg_latency": 0.0
                    }
                
                model_usage[event.model]["requests"] += 1
                model_usage[event.model]["tokens_in"] += event.tokens_in
                model_usage[event.model]["tokens_out"] += event.tokens_out
                model_usage[event.model]["cost"] += event.cost_estimate
            
            # Calculate average latencies
            for model, stats in model_usage.items():
                model_events = [e for e in events if e.model == model]
                if model_events:
                    stats["avg_latency"] = sum(e.latency_ms for e in model_events) / len(model_events)
            
            return {
                "time_period_hours": time_period_hours,
                "total_requests": total_requests,
                "total_tokens_in": total_tokens_in,
                "total_tokens_out": total_tokens_out,
                "total_cost": total_cost,
                "model_breakdown": model_usage,
                "tenant_id": tenant_id,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get usage summary: {e}")
            return {}

# Global usage tracker instance
usage_tracker = None

def get_usage_tracker() -> Optional[UsageTracker]:
    """Get global usage tracker instance"""
    return usage_tracker

def initialize_usage_tracker(memory_manager: MemoryManager, audit_logger: AuditLogger) -> UsageTracker:
    """Initialize global usage tracker"""
    global usage_tracker
    usage_tracker = UsageTracker(memory_manager, audit_logger)
    return usage_tracker