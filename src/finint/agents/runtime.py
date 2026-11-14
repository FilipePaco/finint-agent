"""Build the production runtime from settings (models, prompts, tools, audit, providers)."""

from __future__ import annotations

from datetime import date
from typing import Any

from finint.agents.graph import Runtime
from finint.agents.llm import load_prompt, make_model
from finint.agents.observability import run_config
from finint.config import Settings, load_yaml
from finint.core.audit import AuditLog
from finint.tools.media import FixtureProvider, MediaProvider, TavilyProvider
from finint.tools.skills import SkillLibrary

ROLES = ("investigator", "reviewer", "media")


def build_runtime(
    settings: Settings, *, live_media: bool = False, tool_budget: int = 20
) -> Runtime:
    specs = {role: getattr(settings, role) for role in ROLES}
    provider: MediaProvider
    if live_media and settings.tavily_api_key is not None:
        provider = TavilyProvider(settings.tavily_api_key.get_secret_value(), date.today())
    else:
        provider = FixtureProvider(
            settings.fixtures_dir / "media" / "documents.yaml", settings.as_of_date
        )

    def config(run_id: str, alert_id: str) -> dict[str, Any]:
        return run_config(settings, run_id, alert_id)

    return Runtime(
        settings=settings,
        audit=AuditLog(settings.audit_db_path),
        classification=load_yaml("data_classification", settings.config_dir)["tables"],
        skills=SkillLibrary(settings.config_dir.parent / "skills"),
        screening_config=load_yaml("screening", settings.config_dir),
        media_provider=provider,
        models={role: make_model(spec, settings) for role, spec in specs.items()},
        model_ids={role: spec.model for role, spec in specs.items()},
        prompts={role: load_prompt(role) for role in ROLES},
        tool_budget=tool_budget,
        config_factory=config,
    )
