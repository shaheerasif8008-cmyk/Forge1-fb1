"""Centralized logging configuration for Forge1 services."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from forge1.core.dlp import redact_text


class _DLPScrubbingFilter(logging.Filter):
    """Ensure log messages never leak raw PII/PHI patterns."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - exercised in tests
        if isinstance(record.msg, str):
            scrubbed, _ = redact_text(record.getMessage())
            # When ``record.getMessage`` is called it formats args, so we replace both msg and args
            record.msg = scrubbed
            record.args = ()
        return True


def init_logger(name: str = "forge1", level: int = logging.INFO) -> logging.Logger:
    """Initialize and return a configured logger instance.

    Ensures handlers are registered only once so repeated calls are safe. The logger
    writes structured output to stdout to align with container logging best practices.
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        handler.addFilter(_DLPScrubbingFilter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False

    return logger


__all__ = ["init_logger"]
