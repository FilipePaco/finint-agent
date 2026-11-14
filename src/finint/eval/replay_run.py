"""Record or replay the CI evaluation on the synthetic world (task 4.4 · R4.4).

finint eval record   # live Gemini, writes eval/cassettes/synthetic.json (needs a key)
pytest tests/test_replay_eval.py   # replays it in CI, no key, no cost
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from finint.agents.graph import CHECKPOINT_TYPES, Runtime, build_graph, resume, start
from finint.agents.llm import load_prompt
from finint.config import Settings, load_yaml
from finint.core.audit import AuditLog
from finint.core.models import AnalystDecision
from finint.eval.replay import Cassette, RecordReplayChatModel
from finint.tools.media import FixtureProvider
from finint.tools.skills import SkillLibrary

ROOT = Path(__file__).resolve().parents[3]
CASSETTE = ROOT / "eval" / "cassettes" / "synthetic.json"
ROLES = ("investigator", "reviewer", "media")


def replay_runtime(settings: Settings, cassette: Cassette, live: dict[str, Any] | None) -> Runtime:
    models = {
        role: RecordReplayChatModel(role=role, cassette=cassette, inner=(live or {}).get(role))
        for role in ROLES
    }
    return Runtime(
        settings=settings,
        audit=AuditLog(settings.audit_db_path),
        classification=load_yaml("data_classification", ROOT / "config")["tables"],
        skills=SkillLibrary(ROOT / "skills"),
        screening_config=load_yaml("screening", ROOT / "config"),
        media_provider=FixtureProvider(
            ROOT / "data" / "fixtures" / "media" / "documents.yaml", settings.as_of_date
        ),
        models=models,  # type: ignore[arg-type]
        model_ids={
            "investigator": "gemini-3.5-flash-lite",
            "reviewer": "gemini-3.5-flash",
            "media": "gemini-3.5-flash-lite",
        },
        prompts={role: load_prompt(role) for role in ROLES},
    )


def run_alerts(rt: Runtime, alerts: list[str]) -> list[dict[str, Any]]:
    out = []
    for alert_id in alerts:
        serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
        graph = build_graph(replace(rt), InMemorySaver(serde=serde))
        run_id, payload = start(graph, alert_id, run_id=f"replay{alert_id[-4:]:0>28}")
        disposition = (payload.get("policy") or {}).get("disposition") or "NEEDS_MORE_INFO"
        resume(
            graph,
            run_id,
            AnalystDecision(
                action="APPROVE", analyst_id="replay-analyst", final_disposition=disposition
            ),
        )
        out.append({"alert_id": alert_id, "run_id": run_id, "payload": payload})
    return out
