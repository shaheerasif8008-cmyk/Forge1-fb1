"""Stub security helpers for Forge1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from forge1.core.logging_config import init_logger

logger = init_logger("forge1.core.security")


@dataclass
class SecretManager:
    def get_secret(self, name: str) -> str:
        raise NotImplementedError("stub")

    def store_secret(self, name: str, value: str) -> None:
        raise NotImplementedError("stub")


__all__ = ["SecretManager"]
