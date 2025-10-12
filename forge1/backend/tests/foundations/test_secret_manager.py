from pathlib import Path

from forge1.core.security import SecretManager


def test_env_secret_lookup(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "abc123")
    manager = SecretManager()
    assert manager.get_secret("API_TOKEN") == "abc123"


def test_file_secret_lookup(tmp_path: Path):
    secret_file = tmp_path / "service_key"
    secret_file.write_text("value-from-file\n", encoding="utf-8")
    manager = SecretManager(env={}, secrets_dir=tmp_path)
    assert manager.get_secret("service_key") == "value-from-file"


def test_missing_secret_returns_none():
    manager = SecretManager(env={})
    assert manager.get_secret("does_not_exist") is None
