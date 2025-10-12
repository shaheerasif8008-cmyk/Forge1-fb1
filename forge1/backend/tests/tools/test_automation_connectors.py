import asyncio
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from forge1.integrations.automation_connectors import (
    DocumentParserAdapter,
    DriveConnector,
    EmailConnector,
    KBSearchAdapter,
    SlackConnector,
)
try:
    from forge1.integrations.llamaindex_tools import (
        DocumentParserTool,
        DriveFetchTool,
        KBSearchTool,
        SlackPostTool,
    )
    HAS_LLAMA_TOOLS = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    DocumentParserTool = DriveFetchTool = KBSearchTool = SlackPostTool = None  # type: ignore[assignment]
    HAS_LLAMA_TOOLS = False
try:
    from forge1.integrations.llamaindex_adapter import ExecutionContext
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    from dataclasses import dataclass

    @dataclass
    class ExecutionContext:  # type: ignore[no-redef]
        tenant_id: str
        employee_id: str
        role: str
        request_id: str
        case_id: str
try:
    from forge1.core.memory_manager import MemoryType
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    from enum import Enum

    class MemoryType(Enum):  # type: ignore[no-redef]
        DOCUMENT = "document"
        CONVERSATION = "conversation"


class _StubSecurityManager:
    async def check_permission(self, **_: Any) -> bool:
        return True

    async def _log_security_event(self, **_: Any) -> None:  # pragma: no cover - compatibility hook
        return None


class _StubSecretManager:
    async def get_secret(self, name: str) -> Optional[str]:
        return None


class _StubAuditLogger:
    def __init__(self) -> None:
        self.access_events: List[Dict[str, Any]] = []
        self.security_events: List[Dict[str, Any]] = []

    async def log_access_event(self, **event: Any) -> None:
        self.access_events.append(event)

    async def log_security_event(self, **event: Any) -> None:
        self.security_events.append(event)


class _StubMemoryManager:
    def __init__(self) -> None:
        self.stored: List[Dict[str, Any]] = []

    async def store_memory(self, **payload: Any) -> str:
        self.stored.append(payload)
        return "stub-id"


class _StubModelRouter:
    pass


def test_document_parser_adapter_handles_text(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello forge1", encoding="utf-8")

    adapter = DocumentParserAdapter()
    result = asyncio.run(adapter.parse(document_path=str(file_path), document_format="auto", use_ocr=False))

    assert result["text"].strip() == "hello forge1"
    assert result["document_format"] == "txt"
    assert result["extraction_method"] == "text_reader"
    assert isinstance(result.get("nodes"), list)


def test_document_parser_adapter_bytes_roundtrip() -> None:
    adapter = DocumentParserAdapter()
    payload = b"plain text payload"
    result = asyncio.run(adapter.parse(document_bytes=payload, document_format="txt", use_ocr=False))
    assert "plain text" in result["text"]
    assert result["document_format"] == "txt"


def test_kb_search_adapter_local_index_filters_by_tenant() -> None:
    adapter = KBSearchAdapter(memory_manager=None)
    doc_a = adapter.index_local_document("tenant-a", "employee-1", "The quick brown fox", {"summary": "fox"})
    adapter.index_local_document("tenant-a", "employee-2", "Another tenant A document")
    adapter.index_local_document("tenant-b", "employee-1", "Completely different")

    result = asyncio.run(
        adapter.search(
            tenant_id="tenant-a",
            employee_id="employee-1",
            query="quick fox",
            limit=5,
            similarity_threshold=0.1,
        )
    )

    assert result.total_count == 1
    assert result.results[0]["memory_id"] == doc_a


def test_slack_connector_in_memory(tmp_path: Path) -> None:
    connector = SlackConnector(auth_manager=_StubSecretManager(), storage_dir=tmp_path)
    message_ts = asyncio.run(connector.send_message(channel="#alerts", text="System ready"))

    assert isinstance(message_ts, str)
    stored_files = list(tmp_path.glob("*.json"))
    assert stored_files, "slack connector should persist messages when storage_dir is provided"


class _SMTPRecorder:
    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []

    def __call__(self, host: str, port: int) -> "_SMTPRecorder":  # pragma: no cover - invoked by connector
        self.host = host
        self.port = port
        return self

    def __enter__(self) -> "_SMTPRecorder":  # pragma: no cover - context manager protocol
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - nothing to clean up
        return None

    def starttls(self) -> None:
        self.tls = True

    def login(self, username: str, password: str) -> None:
        self.credentials = (username, password)

    def send_message(self, message, from_addr: str, to_addrs: List[str]) -> None:
        self.messages.append({
            "from": from_addr,
            "to": to_addrs,
            "subject": message["Subject"],
        })


def test_email_connector_uses_custom_factory() -> None:
    recorder = _SMTPRecorder()
    connector = EmailConnector(
        auth_manager=type("Auth", (), {"default_sender": "forge1@example.com", "smtp_username": "user", "smtp_password": "pass"})(),
        smtp_host="localhost",
        smtp_port=1025,
        use_tls=True,
        smtp_factory=recorder,
    )

    message_id = connector.send_email(to="user@example.com", subject="Greetings", html="<p>Hello</p>")

    assert isinstance(message_id, str)
    assert recorder.messages[0]["subject"] == "Greetings"


def test_drive_connector_local_roundtrip(tmp_path: Path) -> None:
    base_dir = tmp_path / "drive"
    connector = DriveConnector(auth_manager=None, base_path=base_dir)

    source_file = tmp_path / "input.txt"
    source_file.write_text("payload", encoding="utf-8")

    file_id = connector.upload_file(str(source_file), folder_id="tenant-a")
    fetched = asyncio.run(connector.fetch_file(file_id=file_id))

    assert fetched["name"] == "input.txt"
    assert base64.b64decode(fetched["content"]).decode("utf-8") == "payload"


def test_document_parser_tool_delegates_to_adapter(tmp_path: Path) -> None:
    class StubAdapter(DocumentParserAdapter):
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        async def parse(self, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
            self.calls.append(kwargs)
            return {
                "text": "adapter output",
                "nodes": [],
                "extraction_method": "text_reader",
                "document_format": "txt",
            }

    adapter = StubAdapter()
    if not HAS_LLAMA_TOOLS:
        pytest.skip("LlamaIndex tools not available")
    tool = DocumentParserTool(
        model_router=_StubModelRouter(),
        memory_manager=_StubMemoryManager(),
        parser_adapter=adapter,
        security_manager=_StubSecurityManager(),
        secret_manager=_StubSecretManager(),
        audit_logger=_StubAuditLogger(),
    )

    file_path = tmp_path / "doc.txt"
    file_path.write_text("adapter", encoding="utf-8")

    context = ExecutionContext(
        tenant_id="tenant-a",
        employee_id="emp-1",
        role="user",
        request_id="req-1",
        case_id="case-1",
    )

    result = asyncio.run(tool.acall(context, document_path=str(file_path)))

    assert result["success"] is True
    assert adapter.calls, "adapter should receive parse invocation"


def test_slack_post_tool_uses_connector() -> None:
    class StubSlackConnector:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        async def send_message(self, **kwargs: Any) -> str:
            self.calls.append(kwargs)
            return "123.456"

    connector = StubSlackConnector()
    if not HAS_LLAMA_TOOLS:
        pytest.skip("LlamaIndex tools not available")
    tool = SlackPostTool(
        model_router=_StubModelRouter(),
        memory_manager=_StubMemoryManager(),
        slack_connector=connector,
        security_manager=_StubSecurityManager(),
        secret_manager=_StubSecretManager(),
        audit_logger=_StubAuditLogger(),
    )

    context = ExecutionContext(
        tenant_id="tenant-a",
        employee_id="emp-1",
        role="user",
        request_id="req-1",
        case_id="case-1",
    )

    result = asyncio.run(tool.acall(context, channel="#alerts", message="hi"))

    assert result["success"] is True
    assert connector.calls[0]["channel"] == "#alerts"


def test_drive_fetch_tool_uses_connector() -> None:
    class StubDriveConnector:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        async def fetch_file(self, **kwargs: Any) -> Dict[str, Any]:
            self.calls.append(kwargs)
            return {"file_id": "local::foo", "name": "foo", "size": 1, "content": base64.b64encode(b"x").decode("utf-8")}

    connector = StubDriveConnector()
    if not HAS_LLAMA_TOOLS:
        pytest.skip("LlamaIndex tools not available")
    tool = DriveFetchTool(
        model_router=_StubModelRouter(),
        memory_manager=_StubMemoryManager(),
        drive_connector=connector,
        security_manager=_StubSecurityManager(),
        secret_manager=_StubSecretManager(),
        audit_logger=_StubAuditLogger(),
    )

    context = ExecutionContext(
        tenant_id="tenant-a",
        employee_id="emp-1",
        role="user",
        request_id="req-1",
        case_id="case-1",
    )

    result = asyncio.run(tool.acall(context, file_id="local::foo"))

    assert result["success"] is True
    assert connector.calls[0]["file_id"] == "local::foo"


def test_kb_search_tool_uses_adapter() -> None:
    class StubKBAdapter(KBSearchAdapter):
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        async def search(self, **kwargs: Any) -> Any:  # type: ignore[override]
            self.calls.append(kwargs)
            return type("Result", (), {"results": [{"memory_id": "1"}], "total_count": 1, "query_time_ms": 0.5})()

    adapter = StubKBAdapter()
    if not HAS_LLAMA_TOOLS:
        pytest.skip("LlamaIndex tools not available")
    tool = KBSearchTool(
        model_router=_StubModelRouter(),
        memory_manager=_StubMemoryManager(),
        kb_adapter=adapter,
        security_manager=_StubSecurityManager(),
        secret_manager=_StubSecretManager(),
        audit_logger=_StubAuditLogger(),
    )

    context = ExecutionContext(
        tenant_id="tenant-a",
        employee_id="emp-1",
        role="user",
        request_id="req-1",
        case_id="case-1",
    )

    result = asyncio.run(
        tool.acall(
            context,
            query="hello",
            max_results=5,
            similarity_threshold=0.1,
            search_types=[MemoryType.DOCUMENT],
        )
    )

    assert result["success"] is True
    assert adapter.calls, "adapter should be invoked"
    assert result["results"][0]["memory_id"] == "1"
