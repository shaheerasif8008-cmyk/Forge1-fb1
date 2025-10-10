"""Stub agent base definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from forge1.core.logging_config import init_logger

logger = init_logger("forge1.core.agent_base")


@dataclass
class AgentContext:
    session_id: str
    metadata: Dict[str, Any]


class BaseAgent:
    def __init__(self, context: AgentContext) -> None:
        self.context = context

    async def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("stub")


__all__ = ["AgentContext", "BaseAgent"]
