"""
LlamaIndex Integration Adapter for Forge1

Provides LlamaIndex integration as a tool provider while maintaining Forge1's
architectural boundaries. All LLM calls are routed through Forge1's Model Router,
memory operations use Forge1's Memory Manager, and RBAC is enforced for all operations.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Type
from enum import Enum
from dataclasses import dataclass
import json

# LlamaIndex imports
from llama_index.core.tools import BaseTool, ToolMetadata
from llama_index.core.llms.llm import LLM
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document, TextNode
from llama_index.core.vector_stores import VectorStore

# Forge1 imports
from forge1.core.tenancy import get_current_tenant, set_current_tenant
from forge1.core.model_router import ModelRouter
from forge1.core.memory_manager import MemoryManager
from forge1.core.security_manager import SecurityManager
from forge1.core.secret_manager import SecretManager
from forge1.core.audit_logger import AuditLogger
from forge1.core.dlp import redact_payload

logger = logging.getLogger(__name__)

class ToolType(Enum):
    """Available LlamaIndex tool types"""
    DOCUMENT_PARSER = "document_parser"
    KB_SEARCH = "kb_search"
    DRIVE_FETCH = "drive_fetch"
    SLACK_POST = "slack_post"

@dataclass
class ExecutionContext:
    """Context passed to all LlamaIndex operations"""
    tenant_id: str
    employee_id: str
    role: str
    request_id: str
    case_id: str
    session_id: Optional[str] = None
    trace_id: Optional[str] = None

@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    data: Any
    error: Optional[str] = None
    usage_metrics: Optional[Dict[str, Any]] = None
    execution_time_ms: float = 0.0

class LlamaIndexIntegrationError(Exception):
    """Base exception for LlamaIndex integration errors"""
    pass

class RBACViolationError(LlamaIndexIntegrationError):
    """RBAC permission violation"""
    pass

class TenantIsolationError(LlamaIndexIntegrationError):
    """Tenant isolation violation"""
    pass

class LlamaIndexTool(BaseTool):
    """Base class for Forge1-aware LlamaIndex tools"""
    
    def __init__(
        self,
        tool_type: ToolType,
        security_manager: SecurityManager,
        secret_manager: SecretManager,
        audit_logger: AuditLogger,
        **kwargs
    ):
        self.tool_type = tool_type
        self.security_manager = security_manager
        self.secret_manager = secret_manager
        self.audit_logger = audit_logger
        self.tool_id = f"llamaindex_{tool_type.value}_{uuid.uuid4().hex[:8]}"
        
        # Initialize with tool metadata
        metadata = ToolMetadata(
            name=tool_type.value,
            description=f"Forge1-integrated {tool_type.value} tool",
        )
        super().__init__(metadata=metadata, **kwargs)
    
    async def _enforce_rbac(self, context: ExecutionContext, permission: str) -> None:
        """Enforce RBAC permissions for tool access"""
        try:
            # Set tenant context for security check
            set_current_tenant(context.tenant_id)
            
            # Check permission
            has_permission = await self.security_manager.check_permission(
                user_id=context.employee_id,
                permission=permission,
                resource_context={
                    "tenant_id": context.tenant_id,
                    "tool_type": self.tool_type.value,
                    "case_id": context.case_id
                }
            )
            
            if not has_permission:
                # Audit the violation
                await self.audit_logger.log_security_event(
                    event_type="rbac_violation",
                    user_id=context.employee_id,
                    tenant_id=context.tenant_id,
                    details={
                        "permission": permission,
                        "tool_type": self.tool_type.value,
                        "request_id": context.request_id,
                        "case_id": context.case_id
                    }
                )
                
                raise RBACViolationError(
                    f"Permission denied: {permission} for user {context.employee_id} "
                    f"in tenant {context.tenant_id}"
                )
            
            # Log successful access
            await self.audit_logger.log_access_event(
                event_type="tool_access",
                user_id=context.employee_id,
                tenant_id=context.tenant_id,
                details={
                    "permission": permission,
                    "tool_type": self.tool_type.value,
                    "tool_id": self.tool_id,
                    "request_id": context.request_id
                }
            )
            
        except RBACViolationError:
            raise
        except Exception as e:
            logger.error(f"RBAC enforcement failed: {e}")
            raise RBACViolationError(f"RBAC check failed: {e}")
    
    async def _get_tenant_secret(self, secret_name: str, context: ExecutionContext) -> str:
        """Get tenant-scoped secret from Key/Secret broker"""
        try:
            secret_key = f"tenant:{context.tenant_id}:{secret_name}"
            secret_value = await self.secret_manager.get_secret(secret_key)
            
            if not secret_value:
                raise LlamaIndexIntegrationError(
                    f"Secret not found: {secret_name} for tenant {context.tenant_id}"
                )
            
            return secret_value
            
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            raise LlamaIndexIntegrationError(f"Secret retrieval failed: {e}")
    
    async def _redact_output(self, output: Any) -> Any:
        """Apply PII/DLP redaction to tool output"""
        try:
            if isinstance(output, str):
                redacted_output, violations = redact_payload(output)
                return redacted_output
            elif isinstance(output, dict):
                redacted_dict = {}
                for key, value in output.items():
                    if isinstance(value, str):
                        redacted_value, _ = redact_payload(value)
                        redacted_dict[key] = redacted_value
                    else:
                        redacted_dict[key] = value
                return redacted_dict
            else:
                return output
        except Exception as e:
            logger.warning(f"DLP redaction failed: {e}")
            return output
    
    def call(self, *args, **kwargs) -> Any:
        """Synchronous call wrapper - not recommended for async operations"""
        return asyncio.run(self.acall(*args, **kwargs))
    
    async def acall(self, *args, **kwargs) -> Any:
        """Async call method - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement acall method")

class LlamaIndexAdapter:
    """Main adapter for LlamaIndex integration with Forge1"""
    
    def __init__(
        self,
        model_router: ModelRouter,
        memory_manager: MemoryManager,
        security_manager: SecurityManager,
        secret_manager: SecretManager,
        audit_logger: AuditLogger
    ):
        self.model_router = model_router
        self.memory_manager = memory_manager
        self.security_manager = security_manager
        self.secret_manager = secret_manager
        self.audit_logger = audit_logger
        
        # Tool registry
        self.tools: Dict[str, LlamaIndexTool] = {}
        self.tool_types: Dict[ToolType, Type[LlamaIndexTool]] = {}
        
        # Performance metrics
        self.metrics = {
            "tools_created": 0,
            "tools_executed": 0,
            "tools_failed": 0,
            "average_execution_time": 0.0,
            "rbac_violations": 0
        }
        
        self._initialized = False
        logger.info("LlamaIndex adapter initialized")
    
    async def initialize(self) -> None:
        """Initialize the adapter and register default tools"""
        if self._initialized:
            return
        
        try:
            # Register default tool types (implementations will be added in subsequent tasks)
            from forge1.integrations.llamaindex_tools import (
                DocumentParserTool,
                KBSearchTool,
                DriveFetchTool,
                SlackPostTool
            )
            
            self.tool_types = {
                ToolType.DOCUMENT_PARSER: DocumentParserTool,
                ToolType.KB_SEARCH: KBSearchTool,
                ToolType.DRIVE_FETCH: DriveFetchTool,
                ToolType.SLACK_POST: SlackPostTool
            }
            
            self._initialized = True
            logger.info("LlamaIndex adapter initialized with tool types")
            
        except ImportError:
            # Tools not yet implemented - will be added in later tasks
            logger.warning("Tool implementations not yet available")
            self._initialized = True
    
    async def create_tool(
        self,
        tool_type: ToolType,
        config: Dict[str, Any],
        context: ExecutionContext
    ) -> str:
        """Create a new LlamaIndex tool instance"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Validate tenant context
            if context.tenant_id != get_current_tenant():
                raise TenantIsolationError(
                    f"Tenant context mismatch: {context.tenant_id} != {get_current_tenant()}"
                )
            
            # Get tool class
            tool_class = self.tool_types.get(tool_type)
            if not tool_class:
                raise LlamaIndexIntegrationError(f"Tool type not supported: {tool_type}")
            
            # Create tool instance
            tool = tool_class(
                tool_type=tool_type,
                security_manager=self.security_manager,
                secret_manager=self.secret_manager,
                audit_logger=self.audit_logger,
                model_router=self.model_router,
                memory_manager=self.memory_manager,
                **config
            )
            
            # Register tool
            self.tools[tool.tool_id] = tool
            self.metrics["tools_created"] += 1
            
            logger.info(f"Created {tool_type.value} tool with ID {tool.tool_id}")
            return tool.tool_id
            
        except Exception as e:
            logger.error(f"Failed to create tool {tool_type.value}: {e}")
            raise LlamaIndexIntegrationError(f"Tool creation failed: {e}")
    
    async def execute_tool(
        self,
        tool_id: str,
        context: ExecutionContext,
        params: Dict[str, Any]
    ) -> ToolResult:
        """Execute a tool with given parameters"""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Validate tool exists
            if tool_id not in self.tools:
                return ToolResult(
                    success=False,
                    error=f"Tool not found: {tool_id}",
                    execution_time_ms=0.0
                )
            
            tool = self.tools[tool_id]
            
            # Set tenant context
            set_current_tenant(context.tenant_id)
            
            # Execute tool
            result = await tool.acall(context=context, **params)
            
            # Apply DLP redaction
            redacted_result = await tool._redact_output(result)
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Update metrics
            self.metrics["tools_executed"] += 1
            current_avg = self.metrics["average_execution_time"]
            executed_count = self.metrics["tools_executed"]
            self.metrics["average_execution_time"] = (
                (current_avg * (executed_count - 1) + execution_time) / executed_count
            )
            
            return ToolResult(
                success=True,
                data=redacted_result,
                execution_time_ms=execution_time
            )
            
        except RBACViolationError as e:
            self.metrics["rbac_violations"] += 1
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return ToolResult(
                success=False,
                error=f"Permission denied: {e}",
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            self.metrics["tools_failed"] += 1
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.error(f"Tool execution failed for {tool_id}: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )
    
    async def get_available_tools(self, context: ExecutionContext) -> List[Dict[str, Any]]:
        """Get list of available tools for the current context"""
        try:
            set_current_tenant(context.tenant_id)
            
            available_tools = []
            for tool_id, tool in self.tools.items():
                # Check if user has access to this tool type
                try:
                    permission = f"{tool.tool_type.value}:read"
                    has_permission = await self.security_manager.check_permission(
                        user_id=context.employee_id,
                        permission=permission,
                        resource_context={"tenant_id": context.tenant_id}
                    )
                    
                    if has_permission:
                        available_tools.append({
                            "tool_id": tool_id,
                            "tool_type": tool.tool_type.value,
                            "name": tool.metadata.name,
                            "description": tool.metadata.description
                        })
                        
                except Exception as e:
                    logger.warning(f"Permission check failed for tool {tool_id}: {e}")
                    continue
            
            return available_tools
            
        except Exception as e:
            logger.error(f"Failed to get available tools: {e}")
            return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter performance metrics"""
        return {
            **self.metrics,
            "active_tools": len(self.tools),
            "registered_tool_types": len(self.tool_types)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on LlamaIndex integration"""
        try:
            health_status = {
                "status": "healthy" if self._initialized else "unhealthy",
                "initialized": self._initialized,
                "active_tools": len(self.tools),
                "metrics": self.get_metrics()
            }
            
            # Check core dependencies
            try:
                from llama_index.core import __version__ as llamaindex_version
                health_status["llamaindex_version"] = llamaindex_version
            except ImportError:
                health_status["status"] = "unhealthy"
                health_status["error"] = "LlamaIndex not available"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self._initialized
            }