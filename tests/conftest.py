"""Shared test fixtures.

Tests must never see the developer's real ``.env`` or provider keys: a test could
otherwise call a paid LLM API. The session fixture runs before every module-scoped
fixture, so even fixtures that build ``Settings`` early are isolated.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

_ENV_KEYS = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "TAVILY_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
)


@pytest.fixture(scope="session", autouse=True)
def no_real_credentials(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    saved = {k: os.environ.pop(k) for k in _ENV_KEYS if k in os.environ}
    cwd = Path.cwd()
    os.chdir(tmp_path_factory.mktemp("cwd"))  # .env is resolved relative to the cwd
    try:
        yield
    finally:
        os.chdir(cwd)
        os.environ.update(saved)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Per test: own cwd and no provider keys (tests may set them via monkeypatch)."""
    monkeypatch.chdir(tmp_path)
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
