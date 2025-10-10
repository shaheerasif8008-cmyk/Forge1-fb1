"""Forge1 monorepo namespace package."""

from __future__ import annotations

from pathlib import Path

__all__: list[str] = []

_current_dir = Path(__file__).resolve().parent
backend_path = _current_dir / "backend"
if backend_path.exists():
    __path__.append(str(backend_path))  # type: ignore[name-defined]
    nested_pkg = backend_path / "forge1"
    if nested_pkg.exists():
        __path__.append(str(nested_pkg))  # type: ignore[name-defined]
