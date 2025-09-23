"""
Vertical Playbook Engine (Phase C.1–C.6)

Loads vertical playbooks and executes them through the framework compatibility
layer (mocked here if unavailable). Supports dry-run, KPI binding, and HITL gates.
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from forge1.core.tenancy import get_current_tenant
from forge1.verticals.kpis import default_kpis

try:
    from forge1.integrations.framework_adapter import (
        FrameworkCompatibilityLayer,
        FrameworkType,
        UnifiedTaskType,
    )
    HAS_FRAMEWORK = True
except Exception:
    HAS_FRAMEWORK = False


@dataclass
class PlaybookStep:
    name: str
    task_type: str
    config: Dict[str, Any]
    requires_approval: bool = False


@dataclass
class Playbook:
    vertical: str
    name: str
    description: str
    steps: List[PlaybookStep]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vertical": self.vertical,
            "name": self.name,
            "description": self.description,
            "steps": [
                {
                    "name": s.name,
                    "task_type": s.task_type,
                    "config": s.config,
                    "requires_approval": s.requires_approval,
                }
                for s in self.steps
            ],
        }


def load_default_playbooks() -> Dict[str, List[Playbook]]:
    def mk(name: str, vertical: str, steps: List[PlaybookStep]) -> Playbook:
        return Playbook(vertical, name, f"Default {vertical.upper()} playbook: {name}", steps)

    return {
        "cx": [
            mk("triage_and_resolve", "cx", [
                PlaybookStep("triage", "agent_execution", {"task": "Classify and route inbound inquiry"}),
                PlaybookStep("resolve", "agent_execution", {"task": "Propose resolution and draft reply"}),
                PlaybookStep("escalate_if_needed", "workflow_orchestration", {"policy": "escalation_rules"}, True),
            ])
        ],
        "revops": [
            mk("forecast_and_cleanup", "revops", [
                PlaybookStep("pipeline_hygiene", "agent_execution", {"task": "Detect and fix pipeline hygiene issues"}),
                PlaybookStep("forecast", "agent_execution", {"task": "Generate forecast and annotate risks"}),
            ])
        ],
        "finance": [
            mk("close_support", "finance", [
                PlaybookStep("variance_analysis", "agent_execution", {"task": "Perform variance analysis"}),
                PlaybookStep("journal_review", "conversation_management", {"topic": "Postings requiring approvals"}, True),
            ])
        ],
        "legal": [
            mk("contract_assist", "legal", [
                PlaybookStep("clause_extraction", "document_processing", {"mode": "clauses"}),
                PlaybookStep("risk_scoring", "agent_execution", {"task": "Score risk and propose redlines"}, True),
            ])
        ],
        "itops": [
            mk("incident_triage", "itops", [
                PlaybookStep("triage", "agent_execution", {"task": "Assess incident and propose remediation"}),
                PlaybookStep("remediate", "workflow_orchestration", {"workflow": "patch_or_roll"}, True),
            ])
        ],
        "software": [
            mk("spec_to_pr", "software", [
                PlaybookStep("generate_spec", "agent_execution", {"task": "Draft design spec from requirements"}),
                PlaybookStep("code_draft", "agent_execution", {"task": "Generate code draft and tests"}),
                PlaybookStep("review", "conversation_management", {"topic": "PR review"}, True),
            ])
        ],
    }


class PlaybookEngine:
    def __init__(self, framework: Optional[FrameworkCompatibilityLayer] = None):
        self.framework = framework
        self.playbooks = load_default_playbooks()

    def list_verticals(self) -> List[str]:
        return list(self.playbooks.keys())

    def list_playbooks(self, vertical: str) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.playbooks.get(vertical, [])]

    def get_kpis(self, vertical: str) -> Dict[str, Any]:
        return default_kpis(vertical).to_dict()

    async def execute(self, vertical: str, name: str, context: Dict[str, Any], dry_run: bool = True) -> Dict[str, Any]:
        tenant = get_current_tenant()
        pbooks = self.playbooks.get(vertical, [])
        play = next((p for p in pbooks if p.name == name), None)
        if not play:
            return {"error": "playbook_not_found"}

        results: List[Dict[str, Any]] = []
        approvals_required = []
        for step in play.steps:
            if step.requires_approval:
                approvals_required.append(step.name)
            if dry_run or not HAS_FRAMEWORK or self.framework is None:
                results.append({"step": step.name, "status": "simulated"})
                continue
            # Map task types
            task_type = {
                "agent_execution": UnifiedTaskType.AGENT_EXECUTION,
                "workflow_orchestration": UnifiedTaskType.WORKFLOW_ORCHESTRATION,
                "conversation_management": UnifiedTaskType.CONVERSATION_MANAGEMENT,
                "document_processing": UnifiedTaskType.DOCUMENT_PROCESSING,
            }.get(step.task_type)
            if not task_type:
                results.append({"step": step.name, "status": "skipped", "reason": "unknown_task_type"})
                continue
            r = await self.framework.execute_unified_task(task_type, step.config)
            results.append({"step": step.name, "status": r.get("status", "completed"), "result": r})

        return {
            "vertical": vertical,
            "playbook": name,
            "tenant": tenant,
            "dry_run": dry_run,
            "approvals_required": approvals_required,
            "steps": results,
            "kpis": self.get_kpis(vertical),
        }

