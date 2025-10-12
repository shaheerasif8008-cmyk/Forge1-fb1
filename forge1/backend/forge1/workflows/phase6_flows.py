"""Phase 6 end-to-end flow executors.

This module provides lightweight, fully-executable implementations of the
Phase 6 readiness flows described in the engineering program:

1. Legal NDA review: drive_fetch → document_parser(OCR) → kb_search →
   compose → slack_post → email_send.
2. Finance P&L review: ERP ingest (mock) → analysis → report_builder →
   email_send.

The flows lean on the tool adapters that were implemented in earlier phases and
persist structured artifacts describing each run. They intentionally avoid
external network dependencies so they can execute inside unit tests and CI
while still exercising the connectors, usage metering, tenancy propagation, and
basic observability hooks.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import textwrap
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from forge1.billing import usage_meter
from forge1.core.tenancy import TenantContext, set_tenant_context, clear_tenant_context
from forge1.integrations.automation_connectors import (
    DocumentParserAdapter,
    DriveConnector,
    EmailConnector,
    KBSearchAdapter,
    KBSearchResult,
    SlackConnector,
)

JsonType = Union[str, int, float, bool, None, Dict[str, "JsonType"], List["JsonType"]]


@dataclass
class FlowStepRecord:
    """Chronological record for a single step in a Phase 6 flow."""

    name: str
    status: str
    started_at: datetime
    finished_at: datetime
    duration_ms: float
    output: JsonType

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["started_at"] = self.started_at.isoformat()
        payload["finished_at"] = self.finished_at.isoformat()
        return payload


@dataclass
class FlowExecutionResult:
    """Structured summary of an end-to-end flow execution."""

    flow_name: str
    tenant_id: str
    employee_id: str
    case_id: str
    request_id: str
    steps: List[FlowStepRecord]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flow_name": self.flow_name,
            "tenant_id": self.tenant_id,
            "employee_id": self.employee_id,
            "case_id": self.case_id,
            "request_id": self.request_id,
            "summary": self.summary,
            "steps": [step.to_dict() for step in self.steps],
        }


class _Phase6AuthManager:
    """Minimal auth/secret provider used by the flows."""

    def __init__(self, secrets: Optional[Mapping[str, str]] = None) -> None:
        self._secrets = dict(secrets or {})
        self.default_sender = "no-reply@forge1.local"

    def get_secret(self, name: str) -> Optional[str]:
        return self._secrets.get(name)


class _SMTPRecorder:
    """Context manager that records outbound messages instead of sending them."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.messages: List[Dict[str, Any]] = []
        self._counter = 0

    def __call__(self, host: str, port: int) -> "_SMTPRecorder":  # pragma: no cover - exercised indirectly
        self.host = host
        self.port = port
        return self

    def __enter__(self) -> "_SMTPRecorder":  # pragma: no cover - context manager protocol
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - context manager protocol
        return None

    # smtplib.SMTP compatible hooks -------------------------------------------------
    def starttls(self) -> None:  # pragma: no cover - compatibility shim
        return None

    def login(self, username: str, password: str) -> None:  # pragma: no cover - compatibility shim
        self.credentials = {"username": username, "password": password}

    def send_message(self, message, from_addr: str, to_addrs: Iterable[str]) -> None:
        self._counter += 1
        payload = {
            "from": from_addr,
            "to": list(to_addrs),
            "subject": message["Subject"],
            "body_preview": str(message.get_body(preferencelist=("html", "plain"))).strip()[:400],
        }
        self.messages.append(payload)
        output_file = self.base_dir / f"email_{self._counter:02d}.json"
        output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ensure_path(value: Union[str, Path]) -> Path:
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _serialise(value: Any) -> JsonType:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > 400:
            return value[:200] + "…" + value[-50:]
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): _serialise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialise(item) for item in value]
    return str(value)


def _run_step(
    name: str,
    func: Callable[[], Any],
    *,
    summariser: Optional[Callable[[Any], JsonType]] = None,
) -> Tuple[FlowStepRecord, Any]:
    """Execute a flow step while capturing timing metadata."""

    started_at = datetime.now(timezone.utc)
    started_perf = perf_counter()
    result = func()
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    finished_at = datetime.now(timezone.utc)
    duration_ms = (perf_counter() - started_perf) * 1000

    summary = summariser(result) if summariser else _serialise(result)
    record = FlowStepRecord(
        name=name,
        status="success",
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        output=_serialise(summary),
    )
    return record, result


def _summarise_parser_output(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = payload.get("text", "")
    return {
        "document_format": payload.get("document_format"),
        "text_preview": textwrap.shorten(str(text), width=200, placeholder="…"),
        "node_count": payload.get("node_count"),
    }


def _summarise_kb_result(result: KBSearchResult) -> Dict[str, Any]:
    top = result.results[:3]
    return {
        "total_count": result.total_count,
        "top_results": [
            {
                "memory_id": item.get("memory_id"),
                "summary": textwrap.shorten(item.get("summary") or item.get("content", ""), width=120, placeholder="…"),
                "similarity_score": round(float(item.get("similarity_score", 0.0)), 3),
            }
            for item in top
        ],
    }


def _compose_nda_summary(parsed: Dict[str, Any], kb_summary: Dict[str, Any]) -> str:
    preview = parsed.get("text", "")
    preview_snippet = textwrap.shorten(str(preview), width=240, placeholder="…")
    references = ", ".join(entry.get("memory_id", "") for entry in kb_summary.get("top_results", [])) or "n/a"
    return (
        "NDA review completed with OCR-enabled parsing, "
        f"capturing {parsed.get('node_count')} nodes. Key findings reference entries: {references}. "
        f"Preview: {preview_snippet}"
    )


def run_law_nda_flow(artifact_dir: Union[str, Path]) -> FlowExecutionResult:
    """Execute the legal NDA workflow end-to-end and persist artifacts."""

    artifact_path = _ensure_path(artifact_dir)
    inputs_dir = _ensure_path(artifact_path / "inputs")
    slack_dir = _ensure_path(artifact_path / "slack")
    email_dir = _ensure_path(artifact_path / "emails")

    tenant_context = TenantContext(tenant_id="tenant_legal", user_id="attorney_jane", role="legal_review")
    set_tenant_context(tenant_context)

    try:
        usage_meter.reset()
        auth_manager = _Phase6AuthManager()
        smtp_recorder = _SMTPRecorder(email_dir)
        slack_connector = SlackConnector(auth_manager=auth_manager, storage_dir=slack_dir)
        email_connector = EmailConnector(auth_manager=auth_manager, smtp_factory=smtp_recorder)
        drive_connector = DriveConnector(auth_manager=None, base_path=artifact_path / "drive")
        parser_adapter = DocumentParserAdapter()
        kb_adapter = KBSearchAdapter()

        kb_adapter.index_local_document(
            tenant_id=tenant_context.tenant_id,
            employee_id=tenant_context.user_id,
            content="Non-disclosure agreements require mutual confidentiality obligations and clear definitions of protected information.",
            metadata={"summary": "Confidentiality obligations"},
        )
        kb_adapter.index_local_document(
            tenant_id=tenant_context.tenant_id,
            employee_id=tenant_context.user_id,
            content="Standard NDAs often include term limits, governing law, and exceptions for information already known to the recipient.",
            metadata={"summary": "Standard clauses"},
        )

        nda_text = textwrap.dedent(
            """
            This Non-Disclosure Agreement (NDA) is made between Contoso Legal and Tailwind Ventures.
            The recipient agrees to keep all proprietary information confidential for a period of three years.
            Exceptions apply to information previously known or obtained independently.
            """
        ).strip()
        nda_path = inputs_dir / "nda_sample.txt"
        nda_path.write_text(nda_text, encoding="utf-8")

        steps: List[FlowStepRecord] = []

        upload_record, file_id = _run_step(
            "drive_upload",
            lambda: drive_connector.upload_file(str(nda_path), folder_id=tenant_context.tenant_id),
            summariser=lambda output: {"file_id": output},
        )
        steps.append(upload_record)

        fetch_record, drive_payload = _run_step(
            "drive_fetch",
            lambda: drive_connector.fetch_file(file_id=file_id),
            summariser=lambda payload: {
                "file_id": payload.get("file_id"),
                "name": payload.get("name"),
                "size": payload.get("size"),
            },
        )
        steps.append(fetch_record)

        document_bytes = base64.b64decode(drive_payload["content"].encode("utf-8"))
        parse_record, parsed_content = _run_step(
            "document_parser",
            lambda: parser_adapter.parse(document_bytes=document_bytes, document_format="txt", use_ocr=True),
            summariser=_summarise_parser_output,
        )
        steps.append(parse_record)

        kb_record, kb_result = _run_step(
            "kb_search",
            lambda: kb_adapter.search(
                tenant_id=tenant_context.tenant_id,
                employee_id=tenant_context.user_id,
                query="confidentiality obligations",
                limit=3,
                similarity_threshold=0.2,
            ),
            summariser=_summarise_kb_result,
        )
        steps.append(kb_record)

        nda_summary = _compose_nda_summary(parsed_content, kb_record.output)
        usage_meter.record_model_call(
            tenant_id=tenant_context.tenant_id,
            employee_id=tenant_context.user_id,
            model="mock-llm-legal",
            tokens_input=len(nda_text.split()),
            tokens_output=len(nda_summary.split()),
            latency_ms=42.0,
            cost_estimate=0.0001,
            request_id="nda-compose",
            metadata={"case_id": "nda-case-001"},
        )

        compose_record = FlowStepRecord(
            name="compose_draft",
            status="success",
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            duration_ms=1.0,
            output={"draft_preview": textwrap.shorten(nda_summary, width=200, placeholder="…")},
        )
        steps.append(compose_record)

        slack_record, slack_ts = _run_step(
            "slack_post",
            lambda: slack_connector.send_message(
                channel="#legal-reviews",
                text="NDA review ready",
                attachments=[{"text": nda_summary[:200]}],
            ),
            summariser=lambda ts: {"message_ts": ts},
        )
        steps.append(slack_record)

        email_record, message_id = _run_step(
            "email_send",
            lambda: email_connector.send_email(
                to="legal-team@example.com",
                subject="NDA Review Packet",
                html=f"<p>{nda_summary}</p>",
            ),
            summariser=lambda msg: {"message_id": msg, "recorded_messages": len(smtp_recorder.messages)},
        )
        steps.append(email_record)

        request_id = slack_ts or message_id
        summary = (
            "Legal NDA workflow executed successfully with Slack notification and email delivery. "
            f"Recorded {len(usage_meter.snapshot())} usage events."
        )

        flow_result = FlowExecutionResult(
            flow_name="law_nda",
            tenant_id=tenant_context.tenant_id,
            employee_id=tenant_context.user_id,
            case_id="nda-case-001",
            request_id=request_id,
            steps=steps,
            summary=summary,
        )

        (artifact_path / "flow.json").write_text(json.dumps(flow_result.to_dict(), indent=2), encoding="utf-8")
        (artifact_path / "summary.txt").write_text(nda_summary, encoding="utf-8")
        (artifact_path / "usage_events.json").write_text(
            json.dumps([event.to_dict() for event in usage_meter.snapshot()], indent=2),
            encoding="utf-8",
        )
        return flow_result
    finally:
        clear_tenant_context()


def run_finance_pnl_flow(artifact_dir: Union[str, Path]) -> FlowExecutionResult:
    """Execute the finance P&L workflow with deterministic sample data."""

    artifact_path = _ensure_path(artifact_dir)
    email_dir = _ensure_path(artifact_path / "emails")

    tenant_context = TenantContext(tenant_id="tenant_finance", user_id="analyst_lee", role="finance_analyst")
    set_tenant_context(tenant_context)

    try:
        auth_manager = _Phase6AuthManager()
        smtp_recorder = _SMTPRecorder(email_dir)
        email_connector = EmailConnector(auth_manager=auth_manager, smtp_factory=smtp_recorder)

        steps: List[FlowStepRecord] = []

        erp_payload = {
            "month": "2025-01",
            "revenue": [
                {"department": "Enterprise", "amount": 180000.0},
                {"department": "SMB", "amount": 75000.0},
            ],
            "expenses": [
                {"category": "Payroll", "amount": 90000.0},
                {"category": "Infrastructure", "amount": 30000.0},
                {"category": "Sales", "amount": 15000.0},
            ],
        }

        ingest_record, _ = _run_step(
            "erp_ingest",
            lambda: erp_payload,
            summariser=lambda payload: {
                "month": payload["month"],
                "revenue_entries": len(payload["revenue"]),
                "expense_entries": len(payload["expenses"]),
            },
        )
        steps.append(ingest_record)

        def _analyse() -> Dict[str, Any]:
            total_revenue = sum(item["amount"] for item in erp_payload["revenue"])
            total_expenses = sum(item["amount"] for item in erp_payload["expenses"])
            net_profit = total_revenue - total_expenses
            margin = net_profit / total_revenue if total_revenue else 0.0
            usage_meter.record_tool_call(
                tenant_id=tenant_context.tenant_id,
                employee_id=tenant_context.user_id,
                tool="finance.analysis",
                latency_ms=18.5,
                cost_estimate=0.0,
                request_id="pnl-analysis",
                metadata={"net_profit": net_profit, "margin": margin},
            )
            return {
                "total_revenue": total_revenue,
                "total_expenses": total_expenses,
                "net_profit": net_profit,
                "profit_margin": margin,
            }

        analysis_record, analysis = _run_step(
            "analysis",
            _analyse,
            summariser=lambda payload: {
                "total_revenue": payload["total_revenue"],
                "total_expenses": payload["total_expenses"],
                "net_profit": payload["net_profit"],
                "profit_margin": round(payload["profit_margin"], 4),
            },
        )
        steps.append(analysis_record)

        def _build_report() -> str:
            usage_meter.record_model_call(
                tenant_id=tenant_context.tenant_id,
                employee_id=tenant_context.user_id,
                model="mock-llm-finance",
                tokens_input=120,
                tokens_output=200,
                latency_ms=35.0,
                cost_estimate=0.0002,
                request_id="pnl-report",
            )
            return textwrap.dedent(
                f"""
                <h1>Monthly P&amp;L Summary - {erp_payload['month']}</h1>
                <p>Total Revenue: ${analysis['total_revenue']:,.2f}</p>
                <p>Total Expenses: ${analysis['total_expenses']:,.2f}</p>
                <p>Net Profit: ${analysis['net_profit']:,.2f}</p>
                <p>Profit Margin: {analysis['profit_margin']*100:.2f}%</p>
                """
            ).strip()

        report_record, report_html = _run_step(
            "report_builder",
            _build_report,
            summariser=lambda html: {"preview": textwrap.shorten(html, width=200, placeholder="…")},
        )
        steps.append(report_record)

        email_record, message_id = _run_step(
            "email_send",
            lambda: email_connector.send_email(
                to="finance-exec@example.com",
                subject="Monthly P&L Summary",
                html=report_html,
            ),
            summariser=lambda msg: {"message_id": msg, "recorded_messages": len(smtp_recorder.messages)},
        )
        steps.append(email_record)

        summary = (
            "Finance P&L workflow ingested ERP data, produced an analysis with "
            f"net profit ${analysis['net_profit']:,.2f}, and distributed the report via email."
        )

        flow_result = FlowExecutionResult(
            flow_name="finance_pnl",
            tenant_id=tenant_context.tenant_id,
            employee_id=tenant_context.user_id,
            case_id="pnl-case-2025-01",
            request_id=message_id,
            steps=steps,
            summary=summary,
        )

        (artifact_path / "flow.json").write_text(json.dumps(flow_result.to_dict(), indent=2), encoding="utf-8")
        (artifact_path / "report.html").write_text(report_html, encoding="utf-8")
        (artifact_path / "usage_events.json").write_text(
            json.dumps([event.to_dict() for event in usage_meter.snapshot()], indent=2),
            encoding="utf-8",
        )
        return flow_result
    finally:
        clear_tenant_context()


def execute_phase6_flows(artifact_dir: Union[str, Path]) -> Dict[str, FlowExecutionResult]:
    """Run both Phase 6 flows and persist shared artifacts."""

    artifact_path = _ensure_path(artifact_dir)
    usage_meter.reset()

    nda_result = run_law_nda_flow(artifact_path / "law_nda")
    finance_result = run_finance_pnl_flow(artifact_path / "finance_pnl")

    usage_events = [event.to_dict() for event in usage_meter.snapshot()]
    (artifact_path / "usage_events.json").write_text(json.dumps(usage_events, indent=2), encoding="utf-8")

    summary_lines = [
        "# Phase 6 Flow Execution",
        "",
        f"- Legal NDA flow: {nda_result.summary}",
        f"- Finance P&L flow: {finance_result.summary}",
        f"- Usage events captured: {len(usage_events)}",
    ]
    (artifact_path / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    return {"law_nda": nda_result, "finance_pnl": finance_result}


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    artifacts_root = Path(os.getenv("PHASE6_ARTIFACT_DIR", "artifacts/phase6"))
    results = execute_phase6_flows(artifacts_root)
    print(json.dumps({name: result.to_dict() for name, result in results.items()}, indent=2))
