"""Typology gate: code-flagged candidates must be decided before a case is accepted.

Task 3.23 · R3.15 · ADR-03.11. Driven through ``run_investigator`` with a scripted model and a
pre-filled evidence store, so no database or LLM is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from fake_llm import ScriptedChatModel, ai, call
from finint.agents.investigator import TYPOLOGY_GATE_PREFIX, InvestigatorDeps, run_investigator
from finint.core.audit import AuditLog
from finint.tools.investigation import ToolContext
from finint.tools.skills import SkillLibrary

ACCOUNT = "12:A00000008"


def _deps(tmp_path: Path, gate: bool = True) -> InvestigatorDeps:
    ctx = ToolContext(tmp_path / "unused.duckdb", {})
    ctx.store.add("get_alert", {"alert_id": "AL-1"}, {"account_key": ACCOUNT})
    ctx.store.add(
        "detect_patterns",
        {"account_key": ACCOUNT, "start": "2022-09-01", "end": "2022-09-11"},
        {
            "typology_candidates": [
                {
                    "typology": "scatter-gather",
                    "indicator": "multi.scatter_gather_intermediary.parallel_intermediaries",
                    "observed": 6,
                    "rule": ">= 5",
                }
            ]
        },
    )
    return InvestigatorDeps(
        ctx=ctx,
        skills=SkillLibrary(None),
        audit=AuditLog(tmp_path / "audit.sqlite"),
        run_id="run-1",
        alert_id="AL-1",
        screen=lambda _k: "unused",  # type: ignore[arg-type, return-value]
        media_check=lambda _k: "unused",
        on_llm_call=lambda *_a: None,
        alert_account=ACCOUNT,
        typology_gate=gate,
    )


def _case(hypotheses: list[dict[str, Any]]) -> Any:
    case = {
        "summary": "Two-stage flow through parallel intermediaries.",
        "hypotheses": hypotheses,
        "disposition": "ESCALATE",
        "confidence": "MEDIUM",
    }
    return ai(call("submit_case", case, f"submit-{len(hypotheses)}"))


def _rejections(deps: InvestigatorDeps) -> list[Any]:
    return [
        e
        for e in deps.audit.events(run_id="run-1")
        if e.event_type == "TOOL_CALL" and e.payload.get("rejected") == "typology gate"
    ]


def test_undecided_candidate_is_refused_once_then_decided(tmp_path: Path) -> None:
    """R3.15: a case silent on a flagged candidate is sent back, and the refusal is audited."""
    deps = _deps(tmp_path)
    decided = {"typology": "scatter-gather", "status": "CONFIRMED", "evidence_ids": ["ev-002"]}
    model = ScriptedChatModel(
        script=[_case([{"typology": "fan-out", "status": "CONFIRMED"}]), _case([decided])]
    )
    result = run_investigator(model, [HumanMessage("go")], deps)
    assert result.draft is not None
    assert [h.typology for h in result.draft.hypotheses] == ["scatter-gather"]
    refusals = [
        m
        for m in result.messages
        if isinstance(m, ToolMessage) and "typology candidates" in str(m.content)
    ]
    assert len(refusals) == 1
    assert "scatter-gather (ev-002)" in str(refusals[0].content)
    assert _rejections(deps)[0].payload["undecided"] == ["scatter-gather"]


def test_gate_refuses_at_most_once(tmp_path: Path) -> None:
    """R3.15: bounded cost: a second undecided submission is accepted, never looped."""
    deps = _deps(tmp_path)
    inconclusive = {"typology": "scatter-gather", "status": "INCONCLUSIVE"}
    model = ScriptedChatModel(script=[_case([inconclusive]), _case([inconclusive])])
    result = run_investigator(model, [HumanMessage("go")], deps)
    assert result.draft is not None
    assert len(_rejections(deps)) == 1
    assert str(result.messages[-1].content) == "case received"
    assert any(str(m.content).startswith(TYPOLOGY_GATE_PREFIX) for m in result.messages)


def test_gate_off_for_the_baseline(tmp_path: Path) -> None:
    """R3.15: with the switch off (v1 baseline) the case is accepted as submitted."""
    deps = _deps(tmp_path, gate=False)
    model = ScriptedChatModel(script=[_case([])])
    result = run_investigator(model, [HumanMessage("go")], deps)
    assert result.draft is not None
    assert not _rejections(deps)
