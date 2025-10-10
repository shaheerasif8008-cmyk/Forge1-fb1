"""Forge1 backend package marker."""

from __future__ import annotations

from pathlib import Path

__all__: list[str] = []

_current_dir = Path(__file__).resolve().parent
forge1_pkg = _current_dir / "forge1"
if forge1_pkg.exists():
    __path__.append(str(forge1_pkg))  # type: ignore[name-defined]
