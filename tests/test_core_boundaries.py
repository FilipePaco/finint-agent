"""Constitution X: compliance-critical code in finint.core stays framework-free."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

CORE = Path(__file__).parents[1] / "src" / "finint" / "core"
FORBIDDEN = (
    "langchain",
    "langgraph",
    "langfuse",
    "google",
    "finint.data",
    "finint.ml",
    "finint.agents",
    "finint.tools",
    "finint.app",
)


@pytest.mark.parametrize("module", sorted(CORE.glob("*.py")), ids=lambda p: p.name)
def test_core_module_imports_no_framework(module: Path) -> None:
    tree = ast.parse(module.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    bad = sorted(m for m in imported if m.startswith(FORBIDDEN))
    assert not bad, f"{module.name} imports {bad}"
