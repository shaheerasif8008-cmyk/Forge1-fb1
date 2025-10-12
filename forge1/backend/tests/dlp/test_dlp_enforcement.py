from __future__ import annotations

import logging

from forge1.core.dlp import redact_for_export, sanitize_vector_metadata
from forge1.core.logging_config import init_logger


def test_vector_metadata_redacted() -> None:
    metadata = {
        "employee_id": "emp-123",
        "owner_id": "owner@example.com",
        "security_level": "high",
    }

    sanitized, violations = sanitize_vector_metadata(metadata)

    assert sanitized["owner_id"] == "[[REDACTED_EMAIL]]"
    assert sanitized["redaction_count"] == len(violations) == 1


def test_logging_filter_scrubs_sensitive_text(capsys) -> None:
    logger = init_logger("forge1.test.dlp")
    logger.info("User SSN 123-45-6789 contacted support")

    captured = capsys.readouterr().out
    assert "123-45-6789" not in captured
    assert "[[REDACTED_SSN]]" in captured


def test_export_redaction() -> None:
    payload = {"notes": "Reach out to patient at 555-123-4567"}
    redacted, violations = redact_for_export(payload)

    assert redacted["notes"] == "Reach out to patient at [[REDACTED_PHONE]]"
    assert len(violations) == 1
