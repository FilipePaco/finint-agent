"""Model versioning (R2.8): ``{name}-{git_sha}-{yyyymmdd}``. No ML imports on purpose."""

from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime


def git_sha() -> str:
    git = shutil.which("git")
    if git is None:
        return "nogit"
    result = subprocess.run(  # noqa: S603 - fixed arguments, resolved git binary
        [git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "nogit"


def model_version(name: str, when: datetime | None = None) -> str:
    stamp = (when or datetime.now(UTC)).strftime("%Y%m%d")
    return f"{name}-{git_sha()}-{stamp}"
