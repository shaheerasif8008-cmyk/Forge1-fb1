"""Tenant-aware usage metering utilities for billing reconciliation.

The goal for Phase 5 is to capture lightweight usage events for every
model and tool invocation, aggregate them per tenant/employee, and expose
summaries that can back CSV/JSON exports.  The implementation favours
in-memory storage so it remains usable in unit tests and local
bootstraps, while keeping the surface minimal and well typed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple


EventType = Literal["model", "tool"]


@dataclass(slots=True)
class UsageEvent:
    """A single model or tool invocation captured for billing purposes."""

    timestamp: datetime
    tenant_id: str
    employee_id: str
    event_type: EventType
    model: Optional[str] = None
    tool: Optional[str] = None
    tokens_input: int = 0
    tokens_output: int = 0
    latency_ms: float = 0.0
    cost_estimate: float = 0.0
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


class UsageMeter:
    """In-memory usage event collector with aggregation helpers."""

    def __init__(self) -> None:
        self._events: List[UsageEvent] = []
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------
    def record_model_call(
        self,
        *,
        tenant_id: str,
        employee_id: str,
        model: str,
        tokens_input: int,
        tokens_output: int,
        latency_ms: float,
        cost_estimate: float,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> UsageEvent:
        event = UsageEvent(
            timestamp=self._normalise_timestamp(timestamp),
            tenant_id=tenant_id,
            employee_id=employee_id,
            event_type="model",
            model=model,
            tool=None,
            tokens_input=max(0, int(tokens_input)),
            tokens_output=max(0, int(tokens_output)),
            latency_ms=max(0.0, float(latency_ms)),
            cost_estimate=float(cost_estimate),
            request_id=request_id,
            metadata=dict(metadata or {}),
        )
        self._store(event)
        return event

    def record_tool_call(
        self,
        *,
        tenant_id: str,
        employee_id: str,
        tool: str,
        latency_ms: float,
        cost_estimate: float,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        tokens_input: int = 0,
        tokens_output: int = 0,
    ) -> UsageEvent:
        event = UsageEvent(
            timestamp=self._normalise_timestamp(timestamp),
            tenant_id=tenant_id,
            employee_id=employee_id,
            event_type="tool",
            tool=tool,
            model=None,
            tokens_input=max(0, int(tokens_input)),
            tokens_output=max(0, int(tokens_output)),
            latency_ms=max(0.0, float(latency_ms)),
            cost_estimate=float(cost_estimate),
            request_id=request_id,
            metadata=dict(metadata or {}),
        )
        self._store(event)
        return event

    # ------------------------------------------------------------------
    # Public querying helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            self._events.clear()

    def snapshot(self) -> List[UsageEvent]:
        with self._lock:
            return list(self._events)

    def events_for_month(self, month: str) -> List[UsageEvent]:
        start, end = self._month_bounds(month)
        with self._lock:
            return [
                event
                for event in self._events
                if start <= event.timestamp < end
            ]

    def month_summary(self, month: str) -> Dict[str, Any]:
        events = self.events_for_month(month)
        tenants = self._group_by(events, key=lambda e: e.tenant_id)
        tenant_summaries: List[Dict[str, Any]] = []

        for tenant_id, tenant_events in tenants:
            employee_summaries: List[Dict[str, Any]] = []
            for employee_id, employee_events in self._group_by(
                tenant_events, key=lambda e: e.employee_id
            ):
                employee_summaries.append(
                    self._summarise_events(
                        employee_events,
                        extra={"employee_id": employee_id},
                    )
                )

            tenant_payload = self._summarise_events(
                tenant_events,
                extra={
                    "tenant_id": tenant_id,
                    "employees": employee_summaries,
                },
            )
            tenant_summaries.append(tenant_payload)

        summary_totals = self._summarise_events(events, extra={"tenants": len(tenant_summaries)})
        return {
            "month": month,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary_totals,
            "tenants": tenant_summaries,
        }

    def export_month_csv(self, month: str) -> str:
        events = sorted(self.events_for_month(month), key=lambda item: item.timestamp)
        header = (
            "timestamp,tenant_id,employee_id,event_type,model,tool,tokens_input," \
            "tokens_output,latency_ms,cost_estimate,request_id"
        )
        rows = [header]
        for event in events:
            rows.append(
                ",".join(
                    [
                        event.timestamp.isoformat(),
                        event.tenant_id,
                        event.employee_id,
                        event.event_type,
                        event.model or "",
                        event.tool or "",
                        str(event.tokens_input),
                        str(event.tokens_output),
                        f"{event.latency_ms:.2f}",
                        f"{event.cost_estimate:.6f}",
                        event.request_id or "",
                    ]
                )
            )
        return "\n".join(rows)

    def reconcile_month(self, month: str) -> Dict[str, Any]:
        events = self.events_for_month(month)
        summary = self._summarise_events(events)
        aggregated_count = summary.get("total_events", 0)
        actual_count = len(events)
        variance = abs(actual_count - aggregated_count)
        variance_pct = (variance / actual_count * 100.0) if actual_count else 0.0
        return {
            "month": month,
            "event_count": actual_count,
            "aggregated_count": aggregated_count,
            "variance": variance,
            "variance_pct": round(variance_pct, 4),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _store(self, event: UsageEvent) -> None:
        with self._lock:
            self._events.append(event)

    def _normalise_timestamp(self, timestamp: Optional[datetime]) -> datetime:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)

    def _month_bounds(self, month: str) -> Tuple[datetime, datetime]:
        try:
            start = datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Month must be in YYYY-MM format: {month}") from exc
        year = start.year
        next_month = 1 if start.month == 12 else start.month + 1
        next_year = year + 1 if start.month == 12 else year
        end = start.replace(year=next_year, month=next_month)
        return start, end

    def _group_by(
        self,
        events: Iterable[UsageEvent],
        *,
        key,
    ) -> List[Tuple[str, List[UsageEvent]]]:
        buckets: Dict[str, List[UsageEvent]] = {}
        for event in events:
            buckets.setdefault(key(event), []).append(event)
        grouped = []
        for bucket_key in sorted(buckets):
            grouped.append((bucket_key, sorted(buckets[bucket_key], key=lambda item: item.timestamp)))
        return grouped

    def _summarise_events(
        self,
        events: Iterable[UsageEvent],
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        events_list = list(events)
        summary = {
            "total_events": len(events_list),
            "model_calls": sum(1 for event in events_list if event.event_type == "model"),
            "tool_calls": sum(1 for event in events_list if event.event_type == "tool"),
            "tokens_input": sum(event.tokens_input for event in events_list),
            "tokens_output": sum(event.tokens_output for event in events_list),
            "total_latency_ms": round(sum(event.latency_ms for event in events_list), 4),
            "total_cost_estimate": round(sum(event.cost_estimate for event in events_list), 6),
        }
        if extra:
            summary.update(extra)
        return summary


usage_meter = UsageMeter()

__all__ = ["UsageEvent", "UsageMeter", "usage_meter"]
