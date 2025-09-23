# forge1/backend/forge1/orchestration/temporal_client.py
"""
Temporal adapter (skeleton).

Provides a thin wrapper API we can switch to when Temporal is deployed.
If the SDK isn't installed or env isn't set, methods no-op or raise.
"""

import os
from typing import Any, Dict, Optional


try:
    # from temporalio.client import Client
    HAS_TEMPORAL = False  # set True when SDK is added to deps
except Exception:
    HAS_TEMPORAL = False


class TemporalOrchestrator:
    def __init__(self, address: Optional[str] = None, namespace: Optional[str] = None):
        self.enabled = os.getenv("TEMPORAL_ENABLED", "false").lower() == "true"
        self.address = address or os.getenv("TEMPORAL_ADDRESS", "temporal-frontend:7233")
        self.namespace = namespace or os.getenv("TEMPORAL_NAMESPACE", "default")

    async def start(self):
        if not (self.enabled and HAS_TEMPORAL):
            return None
        # client = await Client.connect(self.address, namespace=self.namespace)
        # self.client = client
        return None

    async def submit_workflow(self, name: str, args: Dict[str, Any]) -> str:
        if not (self.enabled and HAS_TEMPORAL):
            raise RuntimeError("Temporal not enabled or SDK not installed")
        # handle = await self.client.start_workflow(name, **args)
        # return handle.id
        return ""

    async def signal_approval(self, workflow_id: str, decision: str, note: str = "") -> None:
        if not (self.enabled and HAS_TEMPORAL):
            raise RuntimeError("Temporal not enabled or SDK not installed")
        # await self.client.signal_workflow(workflow_id, "approval_decision", {"decision": decision, "note": note})
        return None


__all__ = ["TemporalOrchestrator", "HAS_TEMPORAL"]

