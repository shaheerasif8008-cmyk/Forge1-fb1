"""
Usage Event Service

Standardized service for generating and emitting usage events across
all Forge1 operations with cost estimation and tenant-aware tracking.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from forge1.integrations.metering.openmeter_client import OpenMeterAdapter, UsageEvent, UsageEventType
from forge1.integrations.base_adapter import ExecutionContext, TenantContext
from forge1.core.tenancy import get_current_tenant

logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Types of resources for usage tracking"""
    MODEL_TOKENS = "model_tokens"
    TOOL_EXECUTION = "tool_execution"
    STORAGE_BYTES = "storage_bytes"
    COMPUTE_SECONDS = "compute_seconds"
    API_REQUESTS = "api_requests"
    MEMORY_OPERATIONS = "memory_operations"
    VECTOR_OPERATIONS = "vector_operations"
    POLICY_EVALUATIONS = "policy_evaluations"

@dataclass
class ModelUsage:
    """Model usage information"""
    model_name: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    provider: str
    cost_per_token: float = 0.0

@dataclass
class ToolUsage:
    """Tool execution usage information"""
    tool_name: str
    execution_time_ms: float
    parameters_size_bytes: int
    result_size_bytes: int
    success: bool

@dataclass
class StorageUsage:
    """Storage usage information"""
    operation_type: str  # read, write, delete
    bytes_processed: int
    duration_ms: float
    storage_type: str  # redis, weaviate, postgres

class UsageEventService:
    """Service for standardized usage event generation and emission"""
    
    def __init__(self, openmeter_adapter: Optional[OpenMeterAdapter] = None):
        self.openmeter = openmeter_adapter or OpenMeterAdapter()
        
        # Cost estimation tables (would be loaded from configuration)
        self.cost_tables = {
            "gpt-4o": {"input": 0.00001, "output": 0.00003},
            "gpt-4-turbo": {"input": 0.00001, "output": 0.00003},
            "claude-3-opus": {"input": 0.000015, "output": 0.000075},
            "claude-3-sonnet": {"input": 0.000003, "output": 0.000015},
            "gemini-pro": {"input": 0.000001, "output": 0.000002},
            "text-embedding-ada-002": {"input": 0.0000001, "output": 0.0}
        }
        
        # Tool execution costs (per second)
        self.tool_costs = {
            "document_search": 0.001,
            "text_generation": 0.002,
            "data_analysis": 0.003,
            "default": 0.001
        }
        
        # Storage costs (per GB per month, prorated)
        self.storage_costs = {
            "redis": 0.10,
            "weaviate": 0.15,
            "postgres": 0.08
        }
        
        # Statistics
        self._events_generated = 0
        self._events_emitted = 0
        self._total_cost_estimated = 0.0
    
    async def record_model_usage(self, usage: ModelUsage, context: Optional[ExecutionContext] = None) -> bool:
        """Record model usage event with cost estimation"""
        
        try:
            if not context:
                context = self._create_default_context()
            
            # Calculate cost
            cost_estimate = self._estimate_model_cost(usage)
            
            # Create usage event
            event = UsageEvent(
                event_type=UsageEventType.MODEL_REQUEST.value,
                tenant_id=context.tenant_context.tenant_id,
                employee_id=context.tenant_context.employee_id or "unknown",
                resource_type=ResourceType.MODEL_TOKENS.value,
                quantity=usage.tokens_input + usage.tokens_output,
                unit="tokens",
                timestamp=datetime.now(timezone.utc),
                cost_estimate=cost_estimate,
                metadata={
                    "model_name": usage.model_name,
                    "tokens_input": usage.tokens_input,
                    "tokens_output": usage.tokens_output,
                    "latency_ms": usage.latency_ms,
                    "provider": usage.provider,
                    "request_id": context.request_id,
                    "session_id": context.session_id
                }
            )
            
            # Emit event
            success = await self.openmeter.emit_usage_event(event)
            
            if success:
                self._events_generated += 1
                self._events_emitted += 1
                self._total_cost_estimated += cost_estimate
                
                logger.info(f"Model usage recorded: {usage.model_name}, tokens: {usage.tokens_input + usage.tokens_output}, cost: ${cost_estimate:.6f}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to record model usage: {e}")
            return False
    
    async def record_tool_execution(self, usage: ToolUsage, context: Optional[ExecutionContext] = None) -> bool:
        """Record tool execution event with cost estimation"""
        
        try:
            if not context:
                context = self._create_default_context()
            
            # Calculate cost
            cost_estimate = self._estimate_tool_cost(usage)
            
            # Create usage event
            event = UsageEvent(
                event_type=UsageEventType.TOOL_EXECUTION.value,
                tenant_id=context.tenant_context.tenant_id,
                employee_id=context.tenant_context.employee_id or "unknown",
                resource_type=ResourceType.TOOL_EXECUTION.value,
                quantity=usage.execution_time_ms / 1000.0,  # Convert to seconds
                unit="seconds",
                timestamp=datetime.now(timezone.utc),
                cost_estimate=cost_estimate,
                metadata={
                    "tool_name": usage.tool_name,
                    "execution_time_ms": usage.execution_time_ms,
                    "parameters_size_bytes": usage.parameters_size_bytes,
                    "result_size_bytes": usage.result_size_bytes,
                    "success": usage.success,
                    "request_id": context.request_id,
                    "session_id": context.session_id
                }
            )
            
            # Emit event
            success = await self.openmeter.emit_usage_event(event)
            
            if success:
                self._events_generated += 1
                self._events_emitted += 1
                self._total_cost_estimated += cost_estimate
                
                logger.info(f"Tool usage recorded: {usage.tool_name}, duration: {usage.execution_time_ms}ms, cost: ${cost_estimate:.6f}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to record tool usage: {e}")
            return False
    
    async def record_storage_usage(self, usage: StorageUsage, context: Optional[ExecutionContext] = None) -> bool:
        """Record storage operation event with cost estimation"""
        
        try:
            if not context:
                context = self._create_default_context()
            
            # Calculate cost
            cost_estimate = self._estimate_storage_cost(usage)
            
            # Create usage event
            event = UsageEvent(
                event_type=UsageEventType.STORAGE_OPERATION.value,
                tenant_id=context.tenant_context.tenant_id,
                employee_id=context.tenant_context.employee_id or "unknown",
                resource_type=ResourceType.STORAGE_BYTES.value,
                quantity=usage.bytes_processed,
                unit="bytes",
                timestamp=datetime.now(timezone.utc),
                cost_estimate=cost_estimate,
                metadata={
                    "operation_type": usage.operation_type,
                    "bytes_processed": usage.bytes_processed,
                    "duration_ms": usage.duration_ms,
                    "storage_type": usage.storage_type,
                    "request_id": context.request_id,
                    "session_id": context.session_id
                }
            )
            
            # Emit event
            success = await self.openmeter.emit_usage_event(event)
            
            if success:
                self._events_generated += 1
                self._events_emitted += 1
                self._total_cost_estimated += cost_estimate
                
                logger.debug(f"Storage usage recorded: {usage.operation_type} {usage.bytes_processed} bytes, cost: ${cost_estimate:.6f}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to record storage usage: {e}")
            return False
    
    async def record_api_request(self, endpoint: str, method: str, duration_ms: float, 
                                status_code: int, context: Optional[ExecutionContext] = None) -> bool:
        """Record API request usage"""
        
        try:
            if not context:
                context = self._create_default_context()
            
            # Simple cost estimation for API requests
            cost_estimate = 0.0001  # $0.0001 per API request
            
            # Create usage event
            event = UsageEvent(
                event_type=UsageEventType.API_REQUEST.value,
                tenant_id=context.tenant_context.tenant_id,
                employee_id=context.tenant_context.employee_id or "unknown",
                resource_type=ResourceType.API_REQUESTS.value,
                quantity=1,
                unit="requests",
                timestamp=datetime.now(timezone.utc),
                cost_estimate=cost_estimate,
                metadata={
                    "endpoint": endpoint,
                    "method": method,
                    "duration_ms": duration_ms,
                    "status_code": status_code,
                    "request_id": context.request_id,
                    "session_id": context.session_id
                }
            )
            
            # Emit event
            success = await self.openmeter.emit_usage_event(event)
            
            if success:
                self._events_generated += 1
                self._events_emitted += 1
                self._total_cost_estimated += cost_estimate
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to record API request usage: {e}")
            return False
    
    async def record_memory_operation(self, operation: str, memory_size_bytes: int, 
                                    duration_ms: float, context: Optional[ExecutionContext] = None) -> bool:
        """Record memory operation usage"""
        
        try:
            if not context:
                context = self._create_default_context()
            
            # Cost estimation based on memory size and operation complexity
            base_cost = 0.00001  # Base cost per operation
            size_cost = (memory_size_bytes / 1024 / 1024) * 0.000001  # Cost per MB
            cost_estimate = base_cost + size_cost
            
            # Create usage event
            event = UsageEvent(
                event_type=UsageEventType.MEMORY_OPERATION.value,
                tenant_id=context.tenant_context.tenant_id,
                employee_id=context.tenant_context.employee_id or "unknown",
                resource_type=ResourceType.MEMORY_OPERATIONS.value,
                quantity=1,
                unit="operations",
                timestamp=datetime.now(timezone.utc),
                cost_estimate=cost_estimate,
                metadata={
                    "operation": operation,
                    "memory_size_bytes": memory_size_bytes,
                    "duration_ms": duration_ms,
                    "request_id": context.request_id,
                    "session_id": context.session_id
                }
            )
            
            # Emit event
            success = await self.openmeter.emit_usage_event(event)
            
            if success:
                self._events_generated += 1
                self._events_emitted += 1
                self._total_cost_estimated += cost_estimate
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to record memory operation usage: {e}")
            return False
    
    async def record_vector_operation(self, operation: str, vector_count: int, 
                                    duration_ms: float, context: Optional[ExecutionContext] = None) -> bool:
        """Record vector database operation usage"""
        
        try:
            if not context:
                context = self._create_default_context()
            
            # Cost estimation based on vector operations
            cost_per_vector = 0.000001  # $0.000001 per vector operation
            cost_estimate = vector_count * cost_per_vector
            
            # Create usage event
            event = UsageEvent(
                event_type=UsageEventType.VECTOR_OPERATION.value,
                tenant_id=context.tenant_context.tenant_id,
                employee_id=context.tenant_context.employee_id or "unknown",
                resource_type=ResourceType.VECTOR_OPERATIONS.value,
                quantity=vector_count,
                unit="vectors",
                timestamp=datetime.now(timezone.utc),
                cost_estimate=cost_estimate,
                metadata={
                    "operation": operation,
                    "vector_count": vector_count,
                    "duration_ms": duration_ms,
                    "request_id": context.request_id,
                    "session_id": context.session_id
                }
            )
            
            # Emit event
            success = await self.openmeter.emit_usage_event(event)
            
            if success:
                self._events_generated += 1
                self._events_emitted += 1
                self._total_cost_estimated += cost_estimate
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to record vector operation usage: {e}")
            return False
    
    def _estimate_model_cost(self, usage: ModelUsage) -> float:
        """Estimate cost for model usage"""
        
        model_costs = self.cost_tables.get(usage.model_name.lower(), {
            "input": 0.00001,  # Default cost
            "output": 0.00003
        })
        
        input_cost = usage.tokens_input * model_costs.get("input", 0.0)
        output_cost = usage.tokens_output * model_costs.get("output", 0.0)
        
        return input_cost + output_cost
    
    def _estimate_tool_cost(self, usage: ToolUsage) -> float:
        """Estimate cost for tool execution"""
        
        cost_per_second = self.tool_costs.get(usage.tool_name, self.tool_costs["default"])
        execution_seconds = usage.execution_time_ms / 1000.0
        
        return execution_seconds * cost_per_second
    
    def _estimate_storage_cost(self, usage: StorageUsage) -> float:
        """Estimate cost for storage operations"""
        
        # Convert bytes to GB
        gb_processed = usage.bytes_processed / (1024 ** 3)
        
        # Get cost per GB per month for storage type
        cost_per_gb_month = self.storage_costs.get(usage.storage_type, 0.10)
        
        # Prorate cost based on operation (very simplified)
        if usage.operation_type == "write":
            # Assume data is stored for average of 1 month
            return gb_processed * cost_per_gb_month
        elif usage.operation_type == "read":
            # Small cost for read operations
            return gb_processed * cost_per_gb_month * 0.01
        else:
            # Delete operations have minimal cost
            return gb_processed * cost_per_gb_month * 0.001
    
    def _create_default_context(self) -> ExecutionContext:
        """Create default execution context when none provided"""
        
        tenant_id = get_current_tenant() or "unknown"
        
        tenant_context = TenantContext(
            tenant_id=tenant_id,
            user_id="system",
            employee_id="system"
        )
        
        return ExecutionContext(
            tenant_context=tenant_context,
            request_id=f"usage_event_{int(time.time())}"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get usage event service statistics"""
        
        return {
            "events_generated": self._events_generated,
            "events_emitted": self._events_emitted,
            "total_cost_estimated": self._total_cost_estimated,
            "emission_success_rate": self._events_emitted / max(1, self._events_generated),
            "supported_resource_types": [rt.value for rt in ResourceType],
            "supported_event_types": [et.value for et in UsageEventType]
        }
    
    def reset_statistics(self):
        """Reset usage statistics"""
        
        self._events_generated = 0
        self._events_emitted = 0
        self._total_cost_estimated = 0.0

# Global usage event service
usage_event_service = UsageEventService()