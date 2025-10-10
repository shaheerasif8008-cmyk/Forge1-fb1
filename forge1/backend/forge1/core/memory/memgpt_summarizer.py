"""Minimal MemGPT summarizer shim used by the memory manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from forge1.core.logging_config import init_logger


logger = init_logger("forge1.core.memory.memgpt")


@dataclass(frozen=True)
class CompressionStrategy:
    name: str = "truncate"
    target_tokens: int = 2048


@dataclass(frozen=True)
class TTLPolicy:
    max_age_days: int = 30
    compression_threshold_tokens: int = 8000
    archive_after_days: int = 90
    delete_after_days: int = 365


@dataclass(frozen=True)
class CompressionResult:
    original_tokens: int
    compressed_tokens: int
    summary: str


class MemGPTSummarizer:
    """Placeholder summarizer until full MemGPT integration is available."""

    def __init__(
        self,
        model_router: Optional[object] = None,
        model_alias: str = "gpt-4o-mini",
        max_length: int = 2048,
    ):
        self.model_alias = model_alias
        self.max_length = max_length
        self.model_router = model_router

    def summarize(self, text: str, max_length: Optional[int] = None) -> str:
        """Return a truncated summary while logging fallback usage."""

        limit = max_length or self.max_length
        if len(text) <= limit:
            return text

        logger.debug(
            "MemGPTSummarizer fallback truncating text",
            extra={"model_alias": self.model_alias, "limit": limit, "original_len": len(text)},
        )
        return text[:limit] + " …"


__all__ = [
    "MemGPTSummarizer",
    "CompressionStrategy",
    "TTLPolicy",
    "CompressionResult",
]
