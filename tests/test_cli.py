"""CLI smoke tests (task 1.1)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from finint import __version__
from finint.cli import app
from finint.config import get_settings

runner = CliRunner()


def test_version_prints_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_info_never_prints_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "do-not-leak-me")
    get_settings.cache_clear()
    try:
        result = runner.invoke(app, ["info"])
    finally:
        get_settings.cache_clear()
    assert result.exit_code == 0
    assert "do-not-leak-me" not in result.stdout
    assert "google_api_key" in result.stdout


def test_data_build_exits_2_with_download_hint_when_files_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FININT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        result = runner.invoke(app, ["data", "build"])
    finally:
        get_settings.cache_clear()
    assert result.exit_code == 2
    assert "finint data download" in result.output
