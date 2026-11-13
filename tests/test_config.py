"""Configuration contract (task 1.1b)."""

from __future__ import annotations

from pathlib import Path

import pytest

from finint.config import Settings, load_yaml


def test_defaults_resolve_paths_under_data_dir(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    assert settings.duckdb_path == tmp_path / "finint.duckdb"
    assert settings.audit_db_path == tmp_path / "audit.sqlite"
    assert settings.duckdb_path != settings.audit_db_path


def test_secrets_are_masked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "super-secret-value")
    settings = Settings()
    assert settings.google_api_key is not None
    assert "super-secret-value" not in repr(settings)
    assert settings.google_api_key.get_secret_value() == "super-secret-value"


def test_gemini_api_key_alias_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "from-gemini-alias")
    assert Settings().google_api_key is not None


def test_model_role_can_be_overridden_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FININT_REVIEWER__PROVIDER", "ollama")
    monkeypatch.setenv("FININT_REVIEWER__MODEL", "qwen3:8b")
    settings = Settings()
    assert settings.reviewer.provider == "ollama"
    assert settings.reviewer.model == "qwen3:8b"


def test_langfuse_disabled_without_both_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert Settings().langfuse_enabled is False
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert Settings().langfuse_enabled is True


def test_load_yaml_reads_mapping(tmp_path: Path) -> None:
    (tmp_path / "thresholds.yaml").write_text("capacity_per_day: 200\n", encoding="utf-8")
    assert load_yaml("thresholds", config_dir=tmp_path) == {"capacity_per_day": 200}


def test_load_yaml_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml("nope", config_dir=tmp_path)


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml("bad", config_dir=tmp_path)
