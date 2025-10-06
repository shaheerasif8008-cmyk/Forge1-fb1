"""
OpenMeter Integration Adapter

Provides usage metering and billing integration with OpenMeter
for comprehensive resource tracking and cost management.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum

import httpx
from kafka import KafkaProducer

from forge1.integrations.base_adapter import BaseAdapter, HealthCheckResult, AdapterStatus, ExecutionContext, TenantContext
from forge1.config.integration_settings import IntegrationType, settings_manager
from forge1.core.tenancy import get_current_tenant
from forge1.core.dlp import redact_payload

logger = logging.getLogger(__name__)

class UsageEventType(Enum):
    """Types of usage events"""
    MODEL_REQUEST = "model_request"
    TOOL_EXECUTION = "tool_execution"
    STORAGE_OPERATION = "storage_operation"
    COMPUTE_TIME = "compute_time"
    API_REQUEST = "api_request"
    MEMORY_OPERATION = "memory_operation"
    VECTOR_OPERATION = "vector_operation"

@dataclass
class UsageEvent:
    """Usage event for metering"""
    event_type: str
    tenant_id: str
    employee_id: str
    resource_type: str
    quantity: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any]
    cost_estimate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_type": self.event_type,
            "tenant_id": self.tenant_id,
            "employee_id": self.employee_id,
            "resource_type": self.resource_type,
            "quantity": self.quantity,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "cost_estimate": self.cost_estimate
        }

@dataclass
class UsageSummary:
    """Usage summary for a period"""
    tenant_id: str
    employee_id: Optional[str]
    period_start: datetime
    period_end: datetime
    total_events: int
    total_cost: float
    resource_breakdown: Dict[str, Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tenant_id": self.tenant_id,
            "employee_id": self.employee_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_events": self.total_events,
            "total_cost": self.total_cost,
            "resource_breakdown": self.resource_breakdown
        }

@dataclass
class TimePeriod:
    """Time period for usage queries"""
    start: datetime
    end: datetime
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary"""
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat()
        }

class OpenMeterAdapter(BaseAdapter):
    """OpenMeter integration adapter for usage metering"""
    
    def __init__(self):
        config = settings_manager.get_config(IntegrationType.METERING)
        super().__init__("openmeter", config)
        
        self.openmeter_config = config
        self.http_client: Optional[httpx.AsyncClient] = None
        self.kafka_producer: Optional[KafkaProducer] = None
        
        # Event batching
        self._event_batch: List[UsageEvent] = []
        self._batch_lock = asyncio.Lock()
        self._last_flush = time.time()
        
        # Statistics
        self._events_sent = 0
        self._events_failed = 0
        self._batch_flushes = 0
    
    async def initialize(self) -> bool:
        """Initialize OpenMeter client connections"""
        try:
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(
                base_url=self.openmeter_config.api_endpoint,
                headers={
                    "Authorization": f"Bearer {self.openmeter_config.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=httpx.Timeout(self.openmeter_config.timeout_seconds)
            )
            
            # Initialize Kafka producer if configured
            if self.openmeter_config.kafka_brokers:
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=self.openmeter_config.kafka_brokers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None,
                    retries=self.openmeter_config.retry_attempts,
                    batch_size=self.openmeter_config.batch_size * 1024,  # Convert to bytes
                    linger_ms=self.openmeter_config.flush_interval_seconds * 1000
                )
            
            # Test connection
            await self._test_connection()
            
            # Start background flush task
            asyncio.create_task(self._background_flush_task())
            
            logger.info("OpenMeter adapter initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenMeter adapter: {e}")
            return False
    
    async def health_check(self) -> HealthCheckResult:
        """Perform health check of OpenMeter connection"""
        start_time = time.time()
        
        try:
            if not self.http_client:
                return HealthCheckResult(
                    status=AdapterStatus.UNHEALTHY,
                    message="OpenMeter client not initialized",
                    details={},
                    timestamp=time.time(),
                    response_time_ms=0
                )
            
            # Test HTTP connection
            http_healthy = False
            try:
                response = await self.http_client.get("/health")
                http_healthy = response.status_code == 200
            except Exception as e:
                logger.warning(f"OpenMeter HTTP health check failed: {e}")
            
            # Test Kafka connection if configured
            kafka_healthy = True
            if self.kafka_producer:
                try:
                    # Simple test - check if producer is not closed
                    kafka_healthy = not self.kafka_producer._closed
                except Exception as e:
                    logger.warning(f"OpenMeter Kafka health check failed: {e}")
                    kafka_healthy = False
            
            # Determine overall status
            if http_healthy and kafka_healthy:
                status = AdapterStatus.HEALTHY
                message = "OpenMeter adapter healthy"
            elif http_healthy or kafka_healthy:
                status = AdapterStatus.DEGRADED
                message = "OpenMeter adapter partially healthy"
            else:
                status = AdapterStatus.UNHEALTHY
                message = "OpenMeter adapter unhealthy"
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                status=status,
                message=message,
                details={
                    "http_healthy": http_healthy,
                    "kafka_healthy": kafka_healthy,
                    "kafka_enabled": self.kafka_producer is not None,
                    "events_sent": self._events_sent,
                    "events_failed": self._events_failed,
                    "batch_flushes": self._batch_flushes,
                    "pending_events": len(self._event_batch)
                },
                timestamp=time.time(),
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=AdapterStatus.UNHEALTHY,
                message=f"OpenMeter health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
                response_time_ms=response_time
            )
    
    async def cleanup(self) -> bool:
        """Clean up OpenMeter resources"""
        try:
            # Flush remaining events
            await self._flush_events()
            
            # Close HTTP client
            if self.http_client:
                await self.http_client.aclose()
            
            # Close Kafka producer
            if self.kafka_producer:
                self.kafka_producer.close()
            
            logger.info("OpenMeter adapter cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup OpenMeter adapter: {e}")
            return False
    
    async def emit_usage_event(self, event: UsageEvent) -> bool:
        """Emit a usage event"""
        
        if not await self.ensure_initialized():
            raise RuntimeError("OpenMeter adapter not initialized")
        
        try:
            # Apply DLP redaction to event metadata
            safe_metadata, violations = redact_payload(event.metadata)
            event.metadata = safe_metadata
            
            if violations:
                event.metadata["dlp_violations"] = len(violations)
            
            # Add to batch
            async with self._batch_lock:
                self._event_batch.append(event)
                
                # Check if batch is full
                if len(self._event_batch) >= self.openmeter_config.batch_size:
                    await self._flush_events()
            
            return True
            
        except Exception as e:
            self._events_failed += 1
            logger.error(f"Failed to emit usage event: {e}")
            return False
    
    async def get_usage_summary(self, tenant_id: str, period: TimePeriod, 
                              employee_id: Optional[str] = None) -> Optional[UsageSummary]:
        """Get usage summary for a tenant and period"""
        
        if not await self.ensure_initialized():
            raise RuntimeError("OpenMeter adapter not initialized")
        
        try:
            # Build query parameters
            params = {
                "tenant_id": tenant_id,
                "start": period.start.isoformat(),
                "end": period.end.isoformat()
            }
            
            if employee_id:
                params["employee_id"] = employee_id
            
            # Make API request
            response = await self.http_client.get("/usage/summary", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert response to UsageSummary
            summary = UsageSummary(
                tenant_id=data["tenant_id"],
                employee_id=data.get("employee_id"),
                period_start=datetime.fromisoformat(data["period_start"]),
                period_end=datetime.fromisoformat(data["period_end"]),
                total_events=data["total_events"],
                total_cost=data["total_cost"],
                resource_breakdown=data["resource_breakdown"]
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get usage summary for tenant {tenant_id}: {e}")
            return None
    
    async def export_usage_csv(self, tenant_id: str, month: str, 
                             employee_id: Optional[str] = None) -> Optional[str]:
        """Export usage data as CSV for a specific month"""
        
        if not await self.ensure_initialized():
            raise RuntimeError("OpenMeter adapter not initialized")
        
        try:
            # Parse month (format: YYYY-MM)
            year, month_num = month.split("-")
            start_date = datetime(int(year), int(month_num), 1, tzinfo=timezone.utc)
            
            # Calculate end date (first day of next month)
            if int(month_num) == 12:
                end_date = datetime(int(year) + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end_date = datetime(int(year), int(month_num) + 1, 1, tzinfo=timezone.utc)
            
            # Build query parameters
            params = {
                "tenant_id": tenant_id,
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "format": "csv"
            }
            
            if employee_id:
                params["employee_id"] = employee_id
            
            # Make API request
            response = await self.http_client.get("/usage/export", params=params)
            response.raise_for_status()
            
            # Return CSV content
            return response.text
            
        except Exception as e:
            logger.error(f"Failed to export usage CSV for tenant {tenant_id}, month {month}: {e}")
            return None
    
    async def _test_connection(self):
        """Test connection to OpenMeter"""
        if self.http_client:
            try:
                response = await self.http_client.get("/health")
                if response.status_code != 200:
                    raise ConnectionError(f"OpenMeter health check failed: {response.status_code}")
            except Exception as e:
                logger.warning(f"OpenMeter connection test failed: {e}")
    
    async def _flush_events(self):
        """Flush batched events to OpenMeter"""
        
        async with self._batch_lock:
            if not self._event_batch:
                return
            
            events_to_send = self._event_batch.copy()
            self._event_batch.clear()
        
        try:
            # Send via Kafka if available, otherwise HTTP
            if self.kafka_producer:
                await self._send_events_kafka(events_to_send)
            else:
                await self._send_events_http(events_to_send)
            
            self._events_sent += len(events_to_send)
            self._batch_flushes += 1
            self._last_flush = time.time()
            
            logger.debug(f"Flushed {len(events_to_send)} usage events")
            
        except Exception as e:
            self._events_failed += len(events_to_send)
            logger.error(f"Failed to flush usage events: {e}")
            
            # Re-add events to batch for retry (with limit to prevent memory issues)
            async with self._batch_lock:
                if len(self._event_batch) < self.openmeter_config.batch_size * 2:
                    self._event_batch.extend(events_to_send)
    
    async def _send_events_kafka(self, events: List[UsageEvent]):
        """Send events via Kafka"""
        
        for event in events:
            try:
                # Use tenant_id as partition key for better distribution
                self.kafka_producer.send(
                    topic=self.openmeter_config.kafka_topic,
                    key=event.tenant_id,
                    value=event.to_dict()
                )
            except Exception as e:
                logger.error(f"Failed to send event via Kafka: {e}")
                raise
        
        # Ensure all messages are sent
        self.kafka_producer.flush()
    
    async def _send_events_http(self, events: List[UsageEvent]):
        """Send events via HTTP API"""
        
        # Convert events to API format
        event_data = [event.to_dict() for event in events]
        
        # Send batch request
        response = await self.http_client.post("/events/batch", json={"events": event_data})
        response.raise_for_status()
    
    async def _background_flush_task(self):
        """Background task to flush events periodically"""
        
        while True:
            try:
                await asyncio.sleep(self.openmeter_config.flush_interval_seconds)
                
                # Check if flush is needed
                current_time = time.time()
                if (current_time - self._last_flush) >= self.openmeter_config.flush_interval_seconds:
                    await self._flush_events()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background flush task error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get adapter statistics"""
        return {
            "events_sent": self._events_sent,
            "events_failed": self._events_failed,
            "batch_flushes": self._batch_flushes,
            "pending_events": len(self._event_batch),
            "last_flush": self._last_flush
        }

# Global OpenMeter adapter instance
openmeter_adapter = OpenMeterAdapter()