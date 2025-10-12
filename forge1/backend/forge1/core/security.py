"""Security helpers for Forge1 core services.

This module previously shipped as a stub and therefore every caller that
attempted to fetch configuration secrets failed at runtime.  Phase 1 replaces
the stub with a lightweight, real implementation that supports the two
providers we rely on in development: process environment variables and mounted
secret files (``*_FILE`` conventions or an explicit secrets directory).

The :class:`SecretManager` intentionally avoids persisting or mutating secrets –
it only provides read access.  Writing secrets is intentionally unsupported in
order to keep the surface area minimal and to ensure we never log or echo
sensitive values during tests or health checks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, MutableMapping, Optional

from forge1.core.logging_config import init_logger

logger = init_logger("forge1.core.security")


class SecretNotFoundError(KeyError):
    """Raised when a requested secret is not available from any provider."""


@dataclass
class SecretManager:
    """Fetch secrets from the environment and optional secret files.

    Parameters
    ----------
    env
        Mapping used to resolve in-memory secrets. Defaults to ``os.environ``.
    secrets_dir
        Optional directory that contains files with secret values.  When
        provided, a secret named ``database_password`` can be stored in a file
        named ``database_password`` within that directory.

    The manager never logs raw secret material.  All log statements emit
    metadata only, such as which provider was used, to aid debugging without
    risking disclosure.
    """

    env: Mapping[str, str] | MutableMapping[str, str] = field(default_factory=lambda: os.environ)
    secrets_dir: Optional[Path] = None

    def __post_init__(self) -> None:  # pragma: no cover - simple path handling
        if isinstance(self.secrets_dir, str):
            self.secrets_dir = Path(self.secrets_dir)

    def get_secret(self, name: str) -> Optional[str]:
        """Return the secret value for ``name`` if available.

        Lookup order:

        1. ``name`` (or its upper-case variant) in the provided environment.
        2. ``{name}_FILE`` pointing at an arbitrary file on disk.
        3. ``secrets_dir / name`` when ``secrets_dir`` is configured.

        No log entry contains the resolved value – only the provider name.
        """

        if not name:
            raise ValueError("Secret name must be provided")

        candidates = {name, name.upper()}
        for candidate in candidates:
            if candidate in self.env:
                logger.debug("Secret resolved from environment", extra={"name": candidate})
                return self.env[candidate]

        file_env_key = f"{name}_FILE".upper()
        file_path = self.env.get(file_env_key)
        if file_path:
            path = Path(file_path)
            try:
                value = path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                logger.warning(
                    "Failed to read secret file", extra={"source": "env_file", "name": name, "error": str(exc)}
                )
            else:
                logger.debug("Secret resolved from explicit file", extra={"name": name})
                return value

        if self.secrets_dir:
            path = Path(self.secrets_dir) / name
            try:
                if path.exists():
                    value = path.read_text(encoding="utf-8").strip()
                    logger.debug("Secret resolved from secrets directory", extra={"name": name})
                    return value
            except OSError as exc:
                logger.warning(
                    "Failed to read secret from secrets directory",
                    extra={"name": name, "directory": str(self.secrets_dir), "error": str(exc)},
                )

        logger.debug("Secret not found", extra={"name": name})
        return None

    # ``store_secret`` intentionally omitted for now – avoiding writes keeps the
    # implementation minimal and prevents accidental disclosure.  Callers that
    # previously relied on the stub raising ``NotImplementedError`` still
    # receive a clear signal if they expect mutation support.


__all__ = ["SecretManager", "SecretNotFoundError"]
