import importlib

import pytest


@pytest.mark.importorskip("forge1")
def test_import_key_modules():
    modules = [
        "forge1.main",
        "forge1.auth.jwt",
        "forge1.core.security",
        "forge1.core.tenancy",
        "forge1.core.database_config",
        "forge1.middleware.tenant_context",
    ]

    failures = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - failure reporting only
            failures.append((name, repr(exc)))

    assert not failures, f"Import failures detected: {failures}"
