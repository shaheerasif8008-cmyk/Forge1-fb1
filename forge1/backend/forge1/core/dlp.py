# forge1/backend/forge1/core/dlp.py
"""
Lightweight DLP/Classification/Redaction (Phase B.1)

Detects common PII/PHI patterns and redacts them before storage/transit.
Intended as a pluggable interface that can be replaced with a managed DLP.
"""

import re
from typing import Any, Dict, List, Tuple, Union


PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
}


def _redact_text(text: str) -> Tuple[str, List[Dict[str, str]]]:
    violations: List[Dict[str, str]] = []
    redacted = text
    for name, pattern in PII_PATTERNS.items():
        def repl(m):
            violations.append({"type": name, "match": m.group(0)})
            return f"[[REDACTED_{name.upper()}]]"
        redacted = pattern.sub(repl, redacted)
    return redacted, violations


def redact_payload(payload: Any) -> Tuple[Any, List[Dict[str, str]]]:
    """Recursively redact strings in dict/list/scalar payloads."""
    violations: List[Dict[str, str]] = []
    if isinstance(payload, str):
        r, v = _redact_text(payload)
        return r, v
    if isinstance(payload, list):
        result = []
        for item in payload:
            r, v = redact_payload(item)
            result.append(r)
            violations.extend(v)
        return result, violations
    if isinstance(payload, dict):
        result = {}
        for k, v in payload.items():
            r, rv = redact_payload(v)
            result[k] = r
            violations.extend(rv)
        return result, violations
    # Non-string scalar
    return payload, violations


def redact_text(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """Public helper that applies the regex based redaction to plain text."""

    return _redact_text(text)


def sanitize_vector_metadata(metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Redact metadata before it is persisted to the vector store."""

    redacted, violations = redact_payload(metadata)
    sanitized = dict(redacted)
    sanitized["redaction_count"] = len(violations)
    return sanitized, violations


def redact_for_export(payload: Any) -> Tuple[Any, List[Dict[str, str]]]:
    """Redact payloads that will be exported outside the service boundary."""

    return redact_payload(payload)


__all__ = [
    "redact_payload",
    "redact_text",
    "sanitize_vector_metadata",
    "redact_for_export",
]

