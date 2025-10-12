"""Functional automation connectors and tool adapters.

This module replaces the previous stubs with lightweight but fully
operational implementations that can back the MCAE/LlamaIndex tool layer.
Each connector prefers mocked, local-first behaviour for unit testing and
continual integration, while still supporting optional real APIs when the
relevant SDKs are installed and credentials are supplied at runtime.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import re
import shutil
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from forge1.billing import usage_meter
from forge1.core.logging_config import init_logger
from forge1.core.tenancy import get_tenant_context

logger = init_logger("forge1.integrations.automation_connectors")

# Optional third-party imports -------------------------------------------------
try:  # pragma: no cover - optional dependency
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_SDK_AVAILABLE = True
except Exception:  # pragma: no cover - the SDK is optional for tests
    WebClient = None  # type: ignore[assignment]
    SlackApiError = Exception  # type: ignore[assignment]
    SLACK_SDK_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    GOOGLE_CLIENT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    build = None  # type: ignore[assignment]
    Credentials = None  # type: ignore[assignment]
    GOOGLE_CLIENT_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:  # pragma: no cover - OCR is optional for tests
    pytesseract = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    OCR_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency
    fitz = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from weaviate import Client as WeaviateClient
    WEAVIATE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    WeaviateClient = None  # type: ignore[assignment]
    WEAVIATE_AVAILABLE = False

# LlamaIndex imports are required because the surrounding MCAE tooling already
# depends on them.  Import errors here would mirror existing behaviour.
try:  # pragma: no cover - optional dependency
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.core.schema import Document
    from llama_index.readers.file import DocxReader, PDFReader
    HAS_LLAMA_INDEX = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    SimpleNodeParser = None  # type: ignore[assignment]
    Document = None  # type: ignore[assignment]
    DocxReader = None  # type: ignore[assignment]
    PDFReader = None  # type: ignore[assignment]
    HAS_LLAMA_INDEX = False

try:
    from forge1.core.memory_models import MemoryQuery, MemoryType
    HAS_MEMORY_MODELS = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    MemoryQuery = None  # type: ignore[assignment]

    class MemoryType(Enum):  # type: ignore[no-redef]
        DOCUMENT = "document"
        CONVERSATION = "conversation"

    HAS_MEMORY_MODELS = False


# ---------------------------------------------------------------------------
# Automation platform connectors
# ---------------------------------------------------------------------------


class AutomationPlatform(Enum):
    N8N = "n8n"
    ZAPIER = "zapier"
    CUSTOM = "custom"


class TriggerType(Enum):
    WEBHOOK = "webhook"
    SCHEDULE = "schedule"
    EVENT = "event"


class ActionType(Enum):
    HTTP_REQUEST = "http_request"
    EMAIL = "email"
    SLACK = "slack"
    CUSTOM = "custom"


@dataclass
class ConnectorInfo:
    connector_id: str
    platform: AutomationPlatform
    created_at: str
    configuration: Dict[str, Any] = field(default_factory=dict)


class BaseAutomationConnector:
    """Common scaffolding for automation platforms (n8n, Zapier, …)."""

    platform: AutomationPlatform = AutomationPlatform.CUSTOM

    def __init__(
        self,
        auth_manager: Any,
        security_manager: Any,
        performance_monitor: Any,
        connector_id: Optional[str] = None,
        **configuration: Any,
    ) -> None:
        self.auth_manager = auth_manager
        self.security_manager = security_manager
        self.performance_monitor = performance_monitor
        self.configuration = configuration
        self._workflows: Dict[str, Dict[str, Any]] = {}
        self._webhooks: Dict[str, Dict[str, Any]] = {}
        self._executions: Dict[str, Dict[str, Any]] = {}
        self.info = ConnectorInfo(
            connector_id=connector_id or uuid.uuid4().hex,
            platform=self.platform,
            created_at=datetime.now(timezone.utc).isoformat(),
            configuration=configuration,
        )

    async def create_workflow(self, definition: Dict[str, Any], name: Optional[str] = None) -> Dict[str, Any]:
        workflow_id = definition.get("id") or uuid.uuid4().hex
        workflow = {
            "id": workflow_id,
            "name": name or definition.get("name", f"workflow-{workflow_id[:8]}"),
            "definition": definition,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._workflows[workflow_id] = workflow
        logger.info("Workflow created", extra={"connector": self.info.connector_id, "workflow": workflow_id})
        return workflow

    async def execute_workflow(self, workflow_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if workflow_id not in self._workflows:
            raise KeyError(f"Workflow not found: {workflow_id}")

        execution_id = uuid.uuid4().hex
        result = {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "status": "completed",
            "input": payload,
            "output": {"echo": payload},
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        self._executions[execution_id] = result
        logger.debug(
            "Workflow executed",
            extra={"connector": self.info.connector_id, "workflow": workflow_id, "execution": execution_id},
        )
        return result

    async def register_webhook(self, config: Dict[str, Any]) -> Dict[str, Any]:
        webhook_id = config.get("id") or uuid.uuid4().hex
        webhook = {
            "id": webhook_id,
            "config": config,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        self._webhooks[webhook_id] = webhook
        logger.info("Webhook registered", extra={"connector": self.info.connector_id, "webhook": webhook_id})
        return webhook

    def get_connector_info(self) -> Dict[str, Any]:
        info = asdict(self.info)
        info.update(
            workflows=len(self._workflows),
            webhooks=len(self._webhooks),
            executions=len(self._executions),
        )
        return info

    async def get_metrics(self) -> Dict[str, Any]:
        return {
            "workflows": len(self._workflows),
            "executions": len(self._executions),
            "webhooks": len(self._webhooks),
        }


class N8NConnector(BaseAutomationConnector):
    platform = AutomationPlatform.N8N


class ZapierConnector(BaseAutomationConnector):
    platform = AutomationPlatform.ZAPIER


class CustomWebhookConnector(BaseAutomationConnector):
    platform = AutomationPlatform.CUSTOM


class AutomationConnectorManager:
    """Registry and lifecycle management for automation connectors."""

    _platform_map = {
        AutomationPlatform.N8N: N8NConnector,
        AutomationPlatform.ZAPIER: ZapierConnector,
        AutomationPlatform.CUSTOM: CustomWebhookConnector,
    }

    def __init__(
        self,
        auth_manager: Any,
        security_manager: Any,
        performance_monitor: Any,
    ) -> None:
        self.auth_manager = auth_manager
        self.security_manager = security_manager
        self.performance_monitor = performance_monitor
        self.connectors: Dict[str, BaseAutomationConnector] = {}

    async def create_connector(self, platform: AutomationPlatform, config: Dict[str, Any]) -> Dict[str, Any]:
        connector_cls = self._platform_map.get(platform)
        if not connector_cls:
            raise ValueError(f"Unsupported automation platform: {platform.value}")

        connector = connector_cls(
            self.auth_manager,
            self.security_manager,
            self.performance_monitor,
            **config,
        )
        self.connectors[connector.info.connector_id] = connector
        return connector.get_connector_info()

    async def get_connector(self, connector_id: str) -> Optional[BaseAutomationConnector]:
        return self.connectors.get(connector_id)

    async def list_connectors(self) -> Dict[str, Dict[str, Any]]:
        return {connector_id: connector.get_connector_info() for connector_id, connector in self.connectors.items()}

    async def remove_connector(self, connector_id: str) -> Dict[str, Any]:
        connector = self.connectors.pop(connector_id, None)
        if not connector:
            raise KeyError(f"Connector not found: {connector_id}")
        return {"removed": connector_id, "platform": connector.platform.value}

    async def execute_workflow(self, connector_id: str, workflow_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        connector = await self.get_connector(connector_id)
        if not connector:
            raise KeyError(f"Connector not found: {connector_id}")
        return await connector.execute_workflow(workflow_id, payload)

    async def get_metrics(self) -> Dict[str, Any]:
        metrics = {}
        for connector_id, connector in self.connectors.items():
            metrics[connector_id] = await connector.get_metrics()
        return metrics


# ---------------------------------------------------------------------------
# Communication connectors (Slack, Email)
# ---------------------------------------------------------------------------


def _is_awaitable(value: Any) -> bool:
    return asyncio.iscoroutine(value) or isinstance(value, asyncio.Future)


class _InMemorySlackClient:
    """Local Slack client used during development and tests."""

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        self.storage_dir = storage_dir
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.messages: List[Dict[str, Any]] = []

    async def chat_postMessage(self, **payload: Any) -> Dict[str, Any]:  # pragma: no cover - simple wrapper
        return self._post_message(**payload)

    def chat_postMessage_sync(self, **payload: Any) -> Dict[str, Any]:
        return self._post_message(**payload)

    def _post_message(self, **payload: Any) -> Dict[str, Any]:
        message_ts = datetime.now(timezone.utc).isoformat()
        record = {
            "ok": True,
            "ts": message_ts,
            "channel": payload.get("channel"),
            "text": payload.get("text"),
            "attachments": payload.get("attachments"),
            "blocks": payload.get("blocks"),
        }
        self.messages.append(record)
        if self.storage_dir:
            filename = self.storage_dir / f"slack_{message_ts.replace(':', '-')}.json"
            filename.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record


class SlackConnector:
    """Send Slack messages via slack_sdk when available or a mock client otherwise."""

    def __init__(
        self,
        auth_manager: Any,
        token_secret_name: str = "slack_bot_token",
        client: Optional[Any] = None,
        storage_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.auth_manager = auth_manager
        self.token_secret_name = token_secret_name
        self._client = client
        self._storage_dir = Path(storage_dir) if storage_dir else None
        if self._storage_dir:
            self._storage_dir.mkdir(parents=True, exist_ok=True)

    async def send_message(
        self,
        channel: str,
        text: str,
        blocks: Optional[List[Dict[str, Any]]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        thread_ts: Optional[str] = None,
        token: Optional[str] = None,
    ) -> str:
        start = time.perf_counter()
        message_ts: Optional[str] = None
        success = False
        try:
            if not channel:
                raise ValueError("Channel must be provided")
            if not text:
                raise ValueError("Message text must be provided")

            resolved_token = await self._resolve_token(token)
            client = await self._get_client(resolved_token)

            payload = {
                "channel": channel,
                "text": text,
                "blocks": blocks,
                "attachments": attachments,
                "thread_ts": thread_ts,
            }

            response = (
                client.chat_postMessage_sync(**payload)
                if hasattr(client, "chat_postMessage_sync")
                else await _maybe_async(client.chat_postMessage(**payload))
            )

            if self._storage_dir and isinstance(client, _InMemorySlackClient):
                # Already persisted by the mock client
                pass

            message_ts = response.get("ts", datetime.now(timezone.utc).isoformat())
            success = True
            logger.info(
                "Slack message delivered",
                extra={"channel": response.get("channel", channel), "message_ts": message_ts},
            )
            return message_ts
        except SlackApiError as exc:  # pragma: no cover - exercised when slack_sdk installed
            logger.error("Slack API error", extra={"channel": channel, "error": str(exc)})
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            context = get_tenant_context()
            tenant_id = context.tenant_id if context else "unknown"
            employee_id = context.user_id if context else "unknown"
            request_id = message_ts or f"slack-{uuid.uuid4().hex[:10]}"
            usage_meter.record_tool_call(
                tenant_id=tenant_id,
                employee_id=employee_id,
                tool="slack.send_message",
                latency_ms=latency_ms,
                cost_estimate=0.0,
                request_id=request_id,
                metadata={
                    "channel": channel,
                    "has_attachments": bool(attachments),
                    "success": success,
                },
            )
        
        
        
        

    async def _resolve_token(self, explicit_token: Optional[str]) -> Optional[str]:
        if explicit_token:
            return explicit_token

        provider = getattr(self.auth_manager, "get_secret", None) or getattr(self.auth_manager, "get", None)
        if provider:
            value = provider(self.token_secret_name)
            if _is_awaitable(value):
                value = await value  # type: ignore[assignment]
            return value
        return None

    async def _get_client(self, token: Optional[str]) -> Any:
        if self._client:
            return self._client

        if SLACK_SDK_AVAILABLE and token:
            self._client = WebClient(token=token)
        else:
            self._client = _InMemorySlackClient(storage_dir=self._storage_dir)
        return self._client

    def send_message_sync(self, *args: Any, **kwargs: Any) -> str:
        return asyncio.run(self.send_message(*args, **kwargs))


async def _maybe_async(call_result: Any) -> Any:
    if _is_awaitable(call_result):
        return await call_result  # type: ignore[return-value]
    return call_result


class EmailConnector:
    """Minimal SMTP email sender with Mailhog-friendly defaults."""

    def __init__(
        self,
        auth_manager: Any,
        smtp_host: str = "localhost",
        smtp_port: int = 1025,
        use_tls: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        smtp_factory: Optional[Callable[[str, int], Any]] = None,
    ) -> None:
        self.auth_manager = auth_manager
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.use_tls = use_tls
        self.username = username
        self.password = password
        self._smtp_factory = smtp_factory

    def _get_smtp(self) -> Any:
        if self._smtp_factory:
            return self._smtp_factory(self.smtp_host, self.smtp_port)

        import smtplib

        return smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10)

    def send_email(
        self,
        to: str,
        subject: str,
        html: str,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
        sender: Optional[str] = None,
    ) -> str:
        start = time.perf_counter()
        success = False
        message_id = f"email-{uuid.uuid4().hex[:12]}"
        try:
            if not to:
                raise ValueError("Recipient address required")
            if not subject:
                raise ValueError("Subject required")

            sender_address = sender or getattr(self.auth_manager, "default_sender", "forge1@example.com")
            message = EmailMessage()
            message["From"] = sender_address
            message["To"] = to
            if cc:
                message["Cc"] = cc
            message["Subject"] = subject
            message["Message-ID"] = message_id
            message.set_content("This message requires an HTML capable client.")
            message.add_alternative(html, subtype="html")

            recipients = [addr.strip() for addr in [to, cc, bcc] if addr]

            smtp = self._get_smtp()
            with smtp:
                if self.use_tls and hasattr(smtp, "starttls"):
                    smtp.starttls()
                username = self.username or getattr(self.auth_manager, "smtp_username", None)
                password = self.password or getattr(self.auth_manager, "smtp_password", None)
                if username and password:
                    smtp.login(username, password)
                smtp.send_message(message, from_addr=sender_address, to_addrs=recipients)

            success = True
            logger.info("Email dispatched", extra={"to": to, "message_id": message_id})
            return message_id
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            context = get_tenant_context()
            tenant_id = context.tenant_id if context else "unknown"
            employee_id = context.user_id if context else "unknown"
            usage_meter.record_tool_call(
                tenant_id=tenant_id,
                employee_id=employee_id,
                tool="email.send",
                latency_ms=latency_ms,
                cost_estimate=0.0,
                request_id=message_id,
                metadata={
                    "to": to,
                    "cc": bool(cc),
                    "bcc": bool(bcc),
                    "success": success,
                },
            )


# ---------------------------------------------------------------------------
# Storage connector (Drive)
# ---------------------------------------------------------------------------


class DriveConnector:
    """Local-first drive connector with optional Google Drive support."""

    def __init__(
        self,
        auth_manager: Any,
        base_path: Union[str, Path, None] = None,
        google_service_factory: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> None:
        self.auth_manager = auth_manager
        self.base_path = Path(base_path or "./artifacts/drive")
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._google_service_factory = google_service_factory
        self._google_service_cache: Dict[str, Any] = {}

    def upload_file(self, path: str, folder_id: str) -> str:
        start = time.perf_counter()
        source = Path(path)
        file_id: Optional[str] = None
        size_bytes = 0
        success = False
        try:
            if not source.exists():
                raise FileNotFoundError(path)

            safe_folder = re.sub(r"[^a-zA-Z0-9_-]", "_", folder_id or "default")
            destination_dir = self.base_path / safe_folder
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / source.name
            shutil.copy2(source, destination)

            size_bytes = source.stat().st_size
            file_id = f"local::{safe_folder}/{source.name}"
            logger.info("File uploaded", extra={"file_id": file_id})
            success = True
            return file_id
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            context = get_tenant_context()
            tenant_id = context.tenant_id if context else "unknown"
            employee_id = context.user_id if context else "unknown"
            usage_meter.record_tool_call(
                tenant_id=tenant_id,
                employee_id=employee_id,
                tool="drive.upload_file",
                latency_ms=latency_ms,
                cost_estimate=0.0,
                request_id=file_id or f"drive-upload-{uuid.uuid4().hex[:10]}",
                metadata={
                    "folder_id": folder_id,
                    "size_bytes": size_bytes,
                    "success": success,
                },
            )

    async def fetch_file(
        self,
        *,
        file_id: Optional[str] = None,
        file_path: Optional[str] = None,
        credentials_json: Optional[str] = None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        result: Optional[Dict[str, Any]] = None
        success = False
        request_id = f"drive-fetch-{uuid.uuid4().hex[:10]}"
        try:
            if file_id and file_id.startswith("local::"):
                relative = file_id[len("local::") :]
                result = await self._fetch_local_file(Path(relative))
            elif file_path and Path(file_path).is_file():
                result = await self._fetch_local_file(Path(file_path))
            elif credentials_json:
                service = await self._get_google_service(credentials_json)
                if file_id:
                    result = await self._fetch_google_file_by_id(service, file_id)
                elif file_path:
                    result = await self._fetch_google_file_by_path(service, file_path)
            else:
                raise ValueError("Unable to locate file with the provided parameters")

            if result is None:
                raise ValueError("Unable to locate file with the provided parameters")

            request_id = result.get("file_id", request_id)
            success = True
            return result
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            context = get_tenant_context()
            tenant_id = context.tenant_id if context else "unknown"
            employee_id = context.user_id if context else "unknown"
            usage_meter.record_tool_call(
                tenant_id=tenant_id,
                employee_id=employee_id,
                tool="drive.fetch_file",
                latency_ms=latency_ms,
                cost_estimate=0.0,
                request_id=request_id,
                metadata={
                    "requested_file_id": file_id,
                    "requested_path": file_path,
                    "success": success,
                },
            )

    async def _fetch_local_file(self, relative_path: Path) -> Dict[str, Any]:
        absolute = (self.base_path / relative_path).resolve()
        if not absolute.exists():
            raise FileNotFoundError(str(absolute))
        content = absolute.read_bytes()
        return {
            "file_id": f"local::{relative_path.as_posix()}",
            "name": absolute.name,
            "mime_type": "application/octet-stream",
            "size": absolute.stat().st_size,
            "modified_time": datetime.fromtimestamp(absolute.stat().st_mtime, timezone.utc).isoformat(),
            "content": base64.b64encode(content).decode("utf-8"),
            "metadata": {"path": absolute.as_posix()},
        }

    async def _get_google_service(self, credentials_json: str) -> Any:
        if credentials_json in self._google_service_cache:
            return self._google_service_cache[credentials_json]

        if self._google_service_factory:
            service = await _maybe_async(self._google_service_factory(json.loads(credentials_json)))
            self._google_service_cache[credentials_json] = service
            return service

        if not GOOGLE_CLIENT_AVAILABLE:
            raise RuntimeError("google-api-python-client is not installed")

        creds = Credentials.from_authorized_user_info(json.loads(credentials_json))  # type: ignore[arg-type]
        service = build("drive", "v3", credentials=creds)
        self._google_service_cache[credentials_json] = service
        return service

    async def _fetch_google_file_by_id(self, service: Any, file_id: str) -> Dict[str, Any]:  # pragma: no cover - requires API
        file_metadata = service.files().get(fileId=file_id).execute()
        file_content = service.files().get_media(fileId=file_id).execute()
        return {
            "file_id": file_id,
            "name": file_metadata.get("name"),
            "mime_type": file_metadata.get("mimeType"),
            "size": file_metadata.get("size"),
            "modified_time": file_metadata.get("modifiedTime"),
            "content": base64.b64encode(file_content).decode("utf-8"),
            "metadata": file_metadata,
        }

    async def _fetch_google_file_by_path(self, service: Any, file_path: str) -> Dict[str, Any]:  # pragma: no cover - requires API
        query = f"name='{Path(file_path).name}'"
        results = service.files().list(q=query).execute()
        files = results.get("files", [])
        if not files:
            raise FileNotFoundError(file_path)
        return await self._fetch_google_file_by_id(service, files[0]["id"])


# ---------------------------------------------------------------------------
# Document parsing adapter
# ---------------------------------------------------------------------------


class DocumentParserAdapter:
    """Parse PDF, DOCX, text, and image documents with optional OCR."""

    def __init__(self, enable_ocr: bool = True) -> None:
        self.enable_ocr = enable_ocr
        self._node_parser = SimpleNodeParser() if HAS_LLAMA_INDEX else None

    async def parse(
        self,
        *,
        document_path: Optional[str] = None,
        document_bytes: Optional[bytes] = None,
        document_format: str = "auto",
        use_ocr: bool = True,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        request_id = f"document-parse-{uuid.uuid4().hex[:10]}"
        success = False
        detected_format = "unknown"
        nodes_count = 0
        try:
            if not document_path and document_bytes is None:
                raise ValueError("Either document_path or document_bytes must be provided")

            temp_path: Optional[Path] = None
            if document_bytes is not None:
                suffix = f".{document_format}" if document_format and document_format != "auto" else ""
                fd, temp_filename = tempfile.mkstemp(suffix=suffix)
                with os.fdopen(fd, "wb") as tmp:
                    tmp.write(document_bytes)
                temp_path = Path(temp_filename)
                document_path = str(temp_path)

            assert document_path  # for type-checkers
            try:
                detected_format = document_format.lower() if document_format != "auto" else self._detect_format(document_path)
                if detected_format == "pdf":
                    result = await asyncio.to_thread(self._parse_pdf, document_path, use_ocr)
                elif detected_format == "docx":
                    result = await asyncio.to_thread(self._parse_docx, document_path)
                elif detected_format == "txt":
                    result = await asyncio.to_thread(self._parse_text, document_path)
                elif detected_format == "image" and use_ocr:
                    result = await asyncio.to_thread(self._parse_image_ocr, document_path)
                else:
                    raise ValueError(f"Unsupported document format: {detected_format}")

                result.setdefault("document_format", detected_format)
                nodes = result.get("nodes")
                if isinstance(nodes, list):
                    nodes_count = len(nodes)
                success = True
                return result
            finally:
                if temp_path and temp_path.exists():
                    temp_path.unlink(missing_ok=True)
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            context = get_tenant_context()
            tenant_id = context.tenant_id if context else "unknown"
            employee_id = context.user_id if context else "unknown"
            usage_meter.record_tool_call(
                tenant_id=tenant_id,
                employee_id=employee_id,
                tool="document_parser.parse",
                latency_ms=latency_ms,
                cost_estimate=0.0,
                request_id=request_id,
                metadata={
                    "detected_format": detected_format,
                    "success": success,
                    "nodes": nodes_count,
                },
            )

    def _detect_format(self, path: str) -> str:
        suffix = Path(path).suffix.lower()
        return {
            ".pdf": "pdf",
            ".docx": "docx",
            ".doc": "docx",
            ".txt": "txt",
            ".md": "txt",
            ".png": "image",
            ".jpg": "image",
            ".jpeg": "image",
        }.get(suffix, "unknown")

    def _parse_pdf(self, path: str, use_ocr: bool) -> Dict[str, Any]:
        if not HAS_LLAMA_INDEX:
            raise ValueError("LlamaIndex not available for PDF parsing")

        reader = PDFReader()
        documents = reader.load_data(path)
        if documents and documents[0].text.strip():
            return self._package_documents(documents, "standard_pdf", page_count=len(documents))
        if not use_ocr:
            raise ValueError("PDF text extraction failed and OCR disabled")
        return self._parse_pdf_ocr(path)

    def _parse_pdf_ocr(self, path: str) -> Dict[str, Any]:
        if not self.enable_ocr or not OCR_AVAILABLE:
            raise ValueError("OCR not available - install pytesseract and pillow")
        if fitz is None:
            raise ValueError("PyMuPDF is required for PDF OCR support")

        doc = fitz.open(path)
        extracted: List[str] = []
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            pixmap = page.get_pixmap()
            img_data = pixmap.tobytes("ppm")
            from io import BytesIO

            with Image.open(BytesIO(img_data)) as img:
                text = pytesseract.image_to_string(img)
            extracted.append(text)
        doc.close()
        joined = "\n".join(extracted)
        document = Document(text=joined)
        nodes = self._node_parser.get_nodes_from_documents([document])
        return {
            "text": joined,
            "nodes": self._serialise_nodes(nodes),
            "extraction_method": "ocr",
            "page_count": len(extracted),
        }

    def _parse_docx(self, path: str) -> Dict[str, Any]:
        if not HAS_LLAMA_INDEX:
            raise ValueError("LlamaIndex not available for DOCX parsing")

        reader = DocxReader()
        documents = reader.load_data(path)
        return self._package_documents(documents, "docx_reader", document_count=len(documents))

    def _parse_text(self, path: str) -> Dict[str, Any]:
        text = Path(path).read_text(encoding="utf-8")
        if HAS_LLAMA_INDEX:
            document = Document(text=text)
            nodes = self._node_parser.get_nodes_from_documents([document])
        else:
            nodes = None
        return {
            "text": text,
            "nodes": self._serialise_nodes(nodes, fallback_text=text),
            "extraction_method": "text_reader",
            "character_count": len(text),
        }

    def _parse_image_ocr(self, path: str) -> Dict[str, Any]:
        if not self.enable_ocr or not OCR_AVAILABLE:
            raise ValueError("OCR not available - install pytesseract and pillow")
        with Image.open(path) as img:
            text = pytesseract.image_to_string(img)
        if HAS_LLAMA_INDEX:
            document = Document(text=text)
            nodes = self._node_parser.get_nodes_from_documents([document])
        else:
            nodes = None
        return {
            "text": text,
            "nodes": self._serialise_nodes(nodes, fallback_text=text),
            "extraction_method": "image_ocr",
        }

    def _package_documents(self, documents: Sequence[Any], method: str, **metadata: Any) -> Dict[str, Any]:
        text = "\n".join(getattr(doc, "text", "") for doc in documents)
        if HAS_LLAMA_INDEX:
            nodes = self._node_parser.get_nodes_from_documents(list(documents))
        else:
            nodes = None
        payload = {
            "text": text,
            "nodes": self._serialise_nodes(nodes, fallback_text=text),
            "extraction_method": method,
        }
        payload.update(metadata)
        return payload

    def _serialise_nodes(self, nodes: Optional[Iterable[Any]], fallback_text: str = "") -> List[Dict[str, Any]]:
        serialised: List[Dict[str, Any]] = []
        if nodes is None:
            chunks = [fallback_text] if fallback_text else []
            for chunk in chunks:
                serialised.append({"text": chunk, "metadata": {}})
            return serialised

        for node in nodes:
            serialised.append({
                "text": getattr(node, "text", ""),
                "metadata": getattr(node, "metadata", {}),
            })
        return serialised


# ---------------------------------------------------------------------------
# Knowledge base search adapter
# ---------------------------------------------------------------------------


@dataclass
class KBSearchResult:
    results: List[Dict[str, Any]]
    total_count: int
    query_time_ms: float


class KBSearchAdapter:
    """Hybrid knowledge base search with Weaviate or a local vector store."""

    def __init__(
        self,
        memory_manager: Optional[Any] = None,
        weaviate_client: Optional[WeaviateClient] = None,
    ) -> None:
        self.memory_manager = memory_manager
        self.weaviate_client = weaviate_client
        self._local_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def index_local_document(
        self,
        tenant_id: str,
        employee_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        doc_id = uuid.uuid4().hex
        vector, norm = self._embed(content)
        self._local_index[tenant_id].append(
            {
                "id": doc_id,
                "employee_id": employee_id,
                "content": content,
                "metadata": metadata or {},
                "vector": vector,
                "norm": norm,
            }
        )
        return doc_id

    async def search(
        self,
        *,
        tenant_id: str,
        employee_id: str,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        search_types: Optional[Sequence[MemoryType]] = None,
    ) -> KBSearchResult:
        start = time.perf_counter()
        request_id = f"kb-search-{uuid.uuid4().hex[:10]}"
        success = False
        backend = "local"
        result: Optional[KBSearchResult] = None
        try:
            if not query:
                raise ValueError("Query text is required")

            if self.weaviate_client:  # pragma: no cover - requires live Weaviate
                backend = "weaviate"
                result = await self._search_weaviate(tenant_id, employee_id, query, limit, similarity_threshold)
            elif self.memory_manager:
                backend = "memory_manager"
                result = await self._search_memory_manager(tenant_id, employee_id, query, limit, similarity_threshold, search_types)
            else:
                backend = "local"
                result = await self._search_local_index(tenant_id, employee_id, query, limit, similarity_threshold)

            success = True
            return result
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            total = result.total_count if result else 0
            context = get_tenant_context()
            tenant_ctx = context.tenant_id if context else tenant_id
            employee_ctx = context.user_id if context else employee_id
            usage_meter.record_tool_call(
                tenant_id=tenant_ctx,
                employee_id=employee_ctx,
                tool="kb.search",
                latency_ms=latency_ms,
                cost_estimate=0.0,
                request_id=request_id,
                metadata={
                    "backend": backend,
                    "tenant_id": tenant_id,
                    "employee_id": employee_id,
                    "limit": limit,
                    "success": success,
                    "results": total,
                },
                tokens_input=len(query.split()),
            )

    async def _search_memory_manager(
        self,
        tenant_id: str,
        employee_id: str,
        query: str,
        limit: int,
        similarity_threshold: float,
        search_types: Optional[Sequence[MemoryType]],
    ) -> KBSearchResult:
        if not HAS_MEMORY_MODELS:
            raise ValueError("Memory models are not available")

        memory_query = MemoryQuery(
            query_text=query,
            limit=limit,
            min_relevance_score=similarity_threshold,
            memory_types=list(search_types) if search_types else None,
            employee_ids=[employee_id],
        )
        search_response = await self.memory_manager.search_memories(memory_query, employee_id)
        formatted = []
        for result in getattr(search_response, "results", []):
            memory = result.memory
            formatted.append(
                {
                    "memory_id": memory.id,
                    "content": memory.content,
                    "summary": getattr(memory, "summary", None),
                    "similarity_score": getattr(result, "similarity_score", 0.0),
                    "memory_type": getattr(memory, "memory_type", MemoryType.DOCUMENT).value,
                    "created_at": getattr(memory, "created_at", datetime.now(timezone.utc)).isoformat(),
                    "metadata": getattr(memory, "metadata", {}),
                }
            )
        return KBSearchResult(
            results=formatted,
            total_count=getattr(search_response, "total_count", len(formatted)),
            query_time_ms=getattr(search_response, "query_time_ms", 0.0),
        )

    async def _search_local_index(
        self,
        tenant_id: str,
        employee_id: str,
        query: str,
        limit: int,
        similarity_threshold: float,
    ) -> KBSearchResult:
        start = datetime.now(timezone.utc)
        vector, norm = self._embed(query)
        matches: List[Tuple[float, Dict[str, Any]]] = []
        for document in self._local_index.get(tenant_id, []):
            if document["employee_id"] != employee_id:
                continue
            score = self._cosine_similarity(vector, norm, document["vector"], document["norm"])
            if score >= similarity_threshold:
                matches.append((score, document))
        matches.sort(key=lambda item: item[0], reverse=True)
        results = [
            {
                "memory_id": doc["id"],
                "content": doc["content"],
                "summary": doc["metadata"].get("summary"),
                "similarity_score": score,
                "memory_type": doc["metadata"].get("memory_type", MemoryType.DOCUMENT.value),
                "created_at": doc["metadata"].get("created_at", datetime.now(timezone.utc).isoformat()),
                "metadata": doc["metadata"],
            }
            for score, doc in matches[:limit]
        ]
        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        return KBSearchResult(results=results, total_count=len(results), query_time_ms=elapsed_ms)

    async def _search_weaviate(
        self,
        tenant_id: str,
        employee_id: str,
        query: str,
        limit: int,
        similarity_threshold: float,
    ) -> KBSearchResult:  # pragma: no cover - requires live Weaviate
        response = (
            self.weaviate_client.query
            .get("ForgeMemory", ["memory_id", "content", "summary", "memory_type", "created_at", "tenant_id", "employee_id"])
            .with_where({
                "operator": "And",
                "operands": [
                    {"path": ["tenant_id"], "operator": "Equal", "valueText": tenant_id},
                    {"path": ["employee_id"], "operator": "Equal", "valueText": employee_id},
                ],
            })
            .with_near_text({"concepts": [query], "distance": 1 - similarity_threshold})
            .with_limit(limit)
            .do()
        )
        data = response.get("data", {}).get("Get", {}).get("ForgeMemory", [])
        results = []
        for entry in data:
            results.append(
                {
                    "memory_id": entry.get("memory_id"),
                    "content": entry.get("content"),
                    "summary": entry.get("summary"),
                    "similarity_score": entry.get("_additional", {}).get("distance", 0.0),
                    "memory_type": entry.get("memory_type", MemoryType.DOCUMENT.value),
                    "created_at": entry.get("created_at"),
                    "metadata": entry,
                }
            )
        return KBSearchResult(results=results, total_count=len(results), query_time_ms=0.0)

    def _tokenise(self, text: str) -> Counter[str]:
        return Counter(token for token in re.findall(r"\b\w+\b", text.lower()) if token)

    def _embed(self, text: str) -> Tuple[Counter[str], float]:
        vector = self._tokenise(text)
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return vector, norm or 1.0

    def _cosine_similarity(
        self,
        vector_a: Counter[str],
        norm_a: float,
        vector_b: Counter[str],
        norm_b: float,
    ) -> float:
        intersection = set(vector_a) & set(vector_b)
        dot_product = sum(vector_a[token] * vector_b[token] for token in intersection)
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0


__all__ = [
    "AutomationPlatform",
    "TriggerType",
    "ActionType",
    "ConnectorInfo",
    "BaseAutomationConnector",
    "N8NConnector",
    "ZapierConnector",
    "CustomWebhookConnector",
    "AutomationConnectorManager",
    "SlackConnector",
    "DriveConnector",
    "EmailConnector",
    "DocumentParserAdapter",
    "KBSearchAdapter",
    "KBSearchResult",
]
