"""Single source of configuration (steering/tech.md: no scattered ``os.getenv``).

Two kinds of configuration:

* **Settings**: paths, secrets and runtime switches from environment variables
  (prefix ``FININT_``) or ``.env``. Provider API keys keep their standard names
  (``GOOGLE_API_KEY``, ``TAVILY_API_KEY``, ``LANGFUSE_*``), so the SDKs and our code
  agree on them.
* **Domain config**: thresholds, capacity, FX rates, etc., versioned as YAML in
  ``config/`` and loaded with :func:`load_yaml`. These are reviewed like code.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ModelRole(BaseModel):
    """LLM for one agent role (R3.13), with provider ids as ``init_chat_model`` expects."""

    provider: str = "google_genai"
    model: str = "gemini-3.5-flash-lite"
    temperature: float = 0.0


class Settings(BaseSettings):
    """Runtime settings. Secrets are ``SecretStr`` so they never print by accident."""

    model_config = SettingsConfigDict(
        env_prefix="FININT_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    config_dir: Path = PROJECT_ROOT / "config"
    reports_dir: Path = PROJECT_ROOT / "reports"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"

    # Reproducibility and time
    seed: int = 42
    as_of_date: date = date(2026, 11, 13)

    # LLM per agent role
    investigator: ModelRole = ModelRole()
    reviewer: ModelRole = ModelRole(model="gemini-3.5-flash")
    media: ModelRole = ModelRole()

    # Provider secrets (standard names, no prefix)
    google_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY")
    )
    tavily_api_key: SecretStr | None = Field(default=None, validation_alias="TAVILY_API_KEY")
    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: SecretStr | None = Field(
        default=None, validation_alias="LANGFUSE_SECRET_KEY"
    )
    langfuse_host: str | None = Field(default=None, validation_alias="LANGFUSE_HOST")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def duckdb_path(self) -> Path:
        return self.data_dir / "finint.duckdb"

    @property
    def audit_db_path(self) -> Path:
        """Audit trail lives apart from analytical data (ADR-01.1)."""
        return self.data_dir / "audit.sqlite"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (cached). Tests build ``Settings(...)`` directly instead."""
    return Settings()


def load_yaml(name: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Load ``config/<name>.yaml`` as a dict. A missing file is an error, not an empty config."""
    directory = config_dir or get_settings().config_dir
    path = directory / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level")
    return loaded
