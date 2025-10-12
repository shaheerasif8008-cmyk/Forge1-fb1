"""
LlamaIndex Tools Implementation for Forge1

Production-ready tools that integrate LlamaIndex capabilities with Forge1's
security, tenancy, and observability systems. Each tool enforces RBAC,
applies DLP redaction, and tracks usage metrics.
"""

import base64
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from forge1.integrations.automation_connectors import (
    DocumentParserAdapter,
    DriveConnector,
    KBSearchAdapter,
    KBSearchResult,
    SlackConnector,
)

# Forge1 imports
from forge1.integrations.llamaindex_adapter import (
    LlamaIndexTool, ToolType, ExecutionContext
)
from forge1.core.memory_manager import MemoryManager, MemoryType
from forge1.core.model_router import ModelRouter
from forge1.integrations.llamaindex_model_shim import ModelShimFactory

logger = logging.getLogger(__name__)

class DocumentParserTool(LlamaIndexTool):
    """Document parser tool with OCR fallback capability"""

    def __init__(
        self,
        model_router: ModelRouter,
        memory_manager: MemoryManager,
        parser_adapter: Optional[DocumentParserAdapter] = None,
        **kwargs,
    ):
        super().__init__(tool_type=ToolType.DOCUMENT_PARSER, **kwargs)
        self.model_router = model_router
        self.memory_manager = memory_manager
        self.parser_adapter = parser_adapter or DocumentParserAdapter()

        # Initialize model shim factory
        self.model_shim_factory = ModelShimFactory(
            model_router=model_router,
            audit_logger=self.audit_logger
        )
    
    async def acall(self, context: ExecutionContext, **kwargs) -> Dict[str, Any]:
        """Parse document with OCR fallback"""
        try:
            # Enforce RBAC
            await self._enforce_rbac(context, "document:read")
            
            # Extract parameters
            document_path = kwargs.get("document_path")
            document_content = kwargs.get("document_content")  # Base64 encoded
            document_format = kwargs.get("document_format", "auto")
            use_ocr = kwargs.get("use_ocr", True)
            
            if not document_path and not document_content:
                raise ValueError("Either document_path or document_content must be provided")
            
            # Decode inline content when provided
            document_bytes = base64.b64decode(document_content) if document_content else None

            # Parse document via adapter (handles temp files internally)
            parsed_content = await self._parse_document(
                document_path=document_path,
                document_format=document_format,
                document_bytes=document_bytes,
                use_ocr=use_ocr,
            )

            # Normalise reported format for downstream consumers
            document_format = parsed_content.get("document_format", document_format)
            
            # Store parsed content in memory for future reference
            await self._store_parsed_content(parsed_content, context)
            
            return {
                "success": True,
                "parsed_content": parsed_content,
                "document_format": document_format,
                "node_count": len(parsed_content.get("nodes", [])),
                "text_length": len(parsed_content.get("text", ""))
            }
            
        except Exception as e:
            logger.error(f"Document parsing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _parse_document(
        self,
        document_path: Optional[str],
        document_format: str,
        document_bytes: Optional[bytes],
        use_ocr: bool,
    ) -> Dict[str, Any]:
        return await self.parser_adapter.parse(
            document_path=document_path,
            document_bytes=document_bytes,
            document_format=document_format,
            use_ocr=use_ocr,
        )
    
    async def _store_parsed_content(self, parsed_content: Dict[str, Any], context: ExecutionContext) -> None:
        """Store parsed content in Forge1 memory system"""
        try:
            # Store the parsed document content
            await self.memory_manager.store_memory(
                content=parsed_content,
                memory_type=MemoryType.DOCUMENT,
                metadata={
                    "tool_type": "document_parser",
                    "tenant_id": context.tenant_id,
                    "employee_id": context.employee_id,
                    "case_id": context.case_id,
                    "extraction_method": parsed_content.get("extraction_method"),
                    "node_count": len(parsed_content.get("nodes", []))
                }
            )
            
        except Exception as e:
            logger.warning(f"Failed to store parsed content in memory: {e}")

class KBSearchTool(LlamaIndexTool):
    """Knowledge base search tool with vector/hybrid retrieval"""

    def __init__(
        self,
        model_router: ModelRouter,
        memory_manager: MemoryManager,
        kb_adapter: Optional[KBSearchAdapter] = None,
        **kwargs,
    ):
        super().__init__(tool_type=ToolType.KB_SEARCH, **kwargs)
        self.model_router = model_router
        self.memory_manager = memory_manager
        self.kb_adapter = kb_adapter or KBSearchAdapter(memory_manager=memory_manager)

        # Initialize model shim factory
        self.model_shim_factory = ModelShimFactory(
            model_router=model_router,
            audit_logger=self.audit_logger
        )
    
    async def acall(self, context: ExecutionContext, **kwargs) -> Dict[str, Any]:
        """Search knowledge base using vector/hybrid retrieval"""
        try:
            # Enforce RBAC
            await self._enforce_rbac(context, "knowledge_base:search")
            
            # Extract parameters
            query = kwargs.get("query", "")
            max_results = kwargs.get("max_results", 10)
            similarity_threshold = kwargs.get("similarity_threshold", 0.7)
            search_types = kwargs.get("search_types", [MemoryType.DOCUMENT, MemoryType.CONVERSATION])
            
            if not query:
                raise ValueError("Query parameter is required")
            
            # Perform search using the adapter (Weaviate or memory manager fallback)
            search_result: KBSearchResult = await self.kb_adapter.search(
                tenant_id=context.tenant_id,
                employee_id=context.employee_id,
                query=query,
                limit=max_results,
                similarity_threshold=similarity_threshold,
                search_types=search_types,
            )

            return {
                "success": True,
                "query": query,
                "results": search_result.results,
                "total_count": search_result.total_count,
                "query_time_ms": search_result.query_time_ms,
            }
            
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": kwargs.get("query", "")
            }

class DriveFetchTool(LlamaIndexTool):
    """Google Drive fetch tool with read-only access"""

    def __init__(
        self,
        model_router: ModelRouter,
        memory_manager: MemoryManager,
        drive_connector: Optional[DriveConnector] = None,
        **kwargs,
    ):
        super().__init__(tool_type=ToolType.DRIVE_FETCH, **kwargs)
        self.model_router = model_router
        self.memory_manager = memory_manager
        self.drive_connector = drive_connector or DriveConnector(auth_manager=self.security_manager)

    async def acall(self, context: ExecutionContext, **kwargs) -> Dict[str, Any]:
        """Fetch document from Google Drive"""
        try:
            # Enforce RBAC
            await self._enforce_rbac(context, "drive:read")

            # Extract parameters
            file_id = kwargs.get("file_id")
            file_path = kwargs.get("file_path")  # Alternative: path-based lookup

            if not file_id and not file_path:
                raise ValueError("Either file_id or file_path must be provided")

            # Get tenant-scoped Google Drive credentials when available
            credentials_json: Optional[str] = None
            try:
                credentials_json = await self._get_tenant_secret("google_drive_credentials", context)
            except Exception:
                credentials_json = None

            # Fetch file via connector (local-first with optional Google Drive)
            file_data = await self.drive_connector.fetch_file(
                file_id=file_id,
                file_path=file_path,
                credentials_json=credentials_json,
            )

            # Store access log
            await self._log_drive_access(context, file_data)

            return {
                "success": True,
                "file_data": file_data,
                "access_time": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Google Drive fetch failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _log_drive_access(self, context: ExecutionContext, file_data: Dict[str, Any]) -> None:
        """Log Google Drive access for audit"""
        try:
            await self.audit_logger.log_access_event(
                event_type="drive_access",
                user_id=context.employee_id,
                tenant_id=context.tenant_id,
                details={
                    "file_id": file_data.get("file_id"),
                    "file_name": file_data.get("name"),
                    "file_size": file_data.get("size"),
                    "case_id": context.case_id,
                    "request_id": context.request_id
                }
            )
        except Exception as e:
            logger.warning(f"Failed to log Drive access: {e}")

class SlackPostTool(LlamaIndexTool):
    """Slack post tool for tenant-scoped channels"""

    def __init__(
        self,
        model_router: ModelRouter,
        memory_manager: MemoryManager,
        slack_connector: Optional[SlackConnector] = None,
        **kwargs,
    ):
        super().__init__(tool_type=ToolType.SLACK_POST, **kwargs)
        self.model_router = model_router
        self.memory_manager = memory_manager
        self.slack_connector = slack_connector or SlackConnector(auth_manager=self.secret_manager)
    
    async def acall(self, context: ExecutionContext, **kwargs) -> Dict[str, Any]:
        """Post message to Slack channel"""
        try:
            # Enforce RBAC
            await self._enforce_rbac(context, "slack:post")

            # Extract parameters
            channel = kwargs.get("channel")
            message = kwargs.get("message", "")
            attachments = kwargs.get("attachments", [])
            thread_ts = kwargs.get("thread_ts")  # For threaded replies

            if not channel or not message:
                raise ValueError("Both channel and message parameters are required")

            # Get tenant-scoped Slack token when available
            slack_token: Optional[str] = None
            try:
                slack_token = await self._get_tenant_secret("slack_bot_token", context)
            except Exception:
                slack_token = None

            # Post message via connector
            message_ts = await self.slack_connector.send_message(
                channel=channel,
                text=message,
                attachments=attachments,
                thread_ts=thread_ts,
                token=slack_token,
            )

            # Log the post
            await self._log_slack_post(context, channel, message, message_ts)

            return {
                "success": True,
                "message_ts": message_ts,
                "channel": channel,
                "message": message,
                "post_time": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Slack post failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _log_slack_post(
        self,
        context: ExecutionContext,
        channel: str,
        message: str,
        message_ts: str,
    ) -> None:
        """Log Slack post for audit"""
        try:
            await self.audit_logger.log_access_event(
                event_type="slack_post",
                user_id=context.employee_id,
                tenant_id=context.tenant_id,
                details={
                    "channel": channel,
                    "message_length": len(message),
                    "message_ts": message_ts,
                    "case_id": context.case_id,
                    "request_id": context.request_id
                }
            )
        except Exception as e:
            logger.warning(f"Failed to log Slack post: {e}")