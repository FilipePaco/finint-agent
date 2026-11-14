"""Agent Skills loader (task 3.10 · R3.2): progressive disclosure of typology playbooks.

The investigator's prompt lists only each skill's name and description; the body is loaded
when the agent decides it is relevant, and each load is audited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str


class UnknownSkillError(KeyError):
    pass


def _parse(path: Path) -> Skill:
    match = _FRONTMATTER.match(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"{path}: missing YAML frontmatter")
    meta = yaml.safe_load(match.group(1))
    if meta.get("name") != path.parent.name:
        raise ValueError(f"{path}: frontmatter name must equal the folder name")
    return Skill(meta["name"], " ".join(str(meta["description"]).split()), match.group(2).strip())


class SkillLibrary:
    def __init__(self, root: Path) -> None:
        self._skills = {s.name: s for s in (_parse(p) for p in sorted(root.glob("*/SKILL.md")))}

    def index(self) -> str:
        """The only skill text that goes into the system prompt."""
        return "\n".join(f"- {s.name}: {s.description}" for s in self._skills.values())

    def names(self) -> list[str]:
        return list(self._skills)

    def load(self, name: str) -> Skill:
        # Lookup by known name only: a name is never used to build a file path.
        if name not in self._skills:
            raise UnknownSkillError(f"unknown skill {name!r}; available: {', '.join(self._skills)}")
        return self._skills[name]
