# forge1/backend/forge1/dlp/presidio_adapter.py
"""
Presidio DLP adapter

Uses Microsoft Presidio Analyzer/Anonymizer to detect and redact PII/PHI.
Falls back gracefully if Presidio isn't installed.
"""

from typing import Any, Dict, List, Tuple

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    HAS_PRESIDIO = True
except Exception:
    AnalyzerEngine = None
    AnonymizerEngine = None
    HAS_PRESIDIO = False


def _redact_text_presidio(text: str, language: str = "en") -> Tuple[str, List[Dict[str, str]]]:
    if not HAS_PRESIDIO:
        return text, []
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    results = analyzer.analyze(text=text, entities=[], language=language)
    if not results:
        return text, []
    res = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        anonymizers_config={},
    )
    violations: List[Dict[str, str]] = []
    for r in results:
        violations.append({"type": r.entity_type.lower(), "match": text[r.start : r.end]})
    return res.text, violations


def redact_payload_presidio(payload: Any, language: str = "en") -> Tuple[Any, List[Dict[str, str]]]:
    """Recursively apply Presidio redaction to payload (dict/list/str)."""
    violations: List[Dict[str, str]] = []
    if isinstance(payload, str):
        red, v = _redact_text_presidio(payload, language)
        return red, v
    if isinstance(payload, list):
        out = []
        for item in payload:
            r, v = redact_payload_presidio(item, language)
            out.append(r)
            violations.extend(v)
        return out, violations
    if isinstance(payload, dict):
        out = {}
        for k, v in payload.items():
            r, rv = redact_payload_presidio(v, language)
            out[k] = r
            violations.extend(rv)
        return out, violations
    return payload, violations


__all__ = ["HAS_PRESIDIO", "redact_payload_presidio"]

