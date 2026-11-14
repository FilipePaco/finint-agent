"""The investigation graph end to end with scripted models (tasks 3.12-3.17 · R3.1-R3.14).

Covers the agent rows of test-strategy.md: grounding repair and blocking, off-allowlist
tools, injection reaching the policy guard, LLM outage, budget exhaustion, no final state
without the analyst, resume after a restart, and verbatim media quotes.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pytest
from langchain_core.messages import BaseMessage, ToolMessage

from fake_llm import ScriptedChatModel, ai, call
from finint.agents.graph import Runtime, build_graph, resume, sqlite_checkpointer, start
from finint.agents.llm import load_prompt
from finint.agents.media_agent import run_media_agent
from finint.config import Settings, load_yaml
from finint.core.audit import AuditLog
from finint.core.models import AnalystDecision
from finint.data.build import build
from finint.data.synthetic import write_dataset
from finint.tools.media import FixtureProvider
from finint.tools.skills import SkillLibrary

ROOT = Path(__file__).parents[1]
MEDIA_DOCS = ROOT / "data" / "fixtures" / "media" / "documents.yaml"


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("agents")
    settings = Settings(data_dir=root / "data", reports_dir=root / "reports")
    write_dataset(settings.raw_dir)
    report = build(settings, offline=True, log=lambda _m: None)
    inj_txn = report.injected_prompts["INJ-01"]
    with duckdb.connect(str(settings.duckdb_path)) as con:
        account, day = con.execute(
            "SELECT from_key, ts::DATE FROM transactions WHERE txn_id = ?", [inj_txn]
        ).fetchone()
        con.execute(
            "CREATE TABLE alerts AS SELECT 'AL-T-0001' AS alert_id, ? AS account_key, "
            "CAST(? AS DATE) AS alert_date, 'test' AS split, 0.9 AS score, 0.5 AS score_raw, "
            "0.45 AS threshold, 'm1-test' AS model_version, 1 AS priority_rank, "
            "CAST(? AS BIGINT) AS top_txn_id, CAST(? AS BIGINT[]) AS contributing_txn_ids, "
            "5 AS n_txn, '{}' AS features_json, ? AS shap_top5_json",
            [
                account,
                day,
                inj_txn,
                [inj_txn],
                json.dumps([{"feature": "out_sum_1h", "value": 1.0, "contribution": 0.4}]),
            ],
        )
    return {
        "settings": settings,
        "account": account,
        "day": day.isoformat(),
        "start": (day - timedelta(days=7)).isoformat(),
    }


def _runtime(
    world: dict[str, Any],
    tmp: Path,
    investigator: list[Any],
    verdicts: list[Any] | None = None,
    budget: int = 20,
) -> Runtime:
    settings = world["settings"]
    models = {
        "investigator": ScriptedChatModel(script=investigator, bound_tool_names=[]),
        "reviewer": ScriptedChatModel(script=[], verdicts=verdicts or [], bound_tool_names=[]),
        "media": ScriptedChatModel(script=[], bound_tool_names=[]),
    }
    return Runtime(
        settings=settings,
        audit=AuditLog(tmp / "audit.sqlite"),
        classification=load_yaml("data_classification", ROOT / "config")["tables"],
        skills=SkillLibrary(ROOT / "skills"),
        screening_config=load_yaml("screening", ROOT / "config"),
        media_provider=FixtureProvider(MEDIA_DOCS, date(2026, 11, 13)),
        models=models,
        model_ids={
            "investigator": "gemini-3.5-flash-lite",
            "reviewer": "gemini-3.5-flash",
            "media": "gemini-3.5-flash-lite",
        },
        prompts={r: load_prompt(r) for r in ("investigator", "reviewer", "media")},
        tool_budget=budget,
    )


def _evidence(messages: list[BaseMessage], tool_prefix: str) -> tuple[str, dict[str, Any]]:
    """Find the latest tool result whose JSON starts with a key from that tool's output."""
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and (
            m := re.match(r"\[(ev-\d+)\] (\{.*)", str(message.content), re.S)
        ):
            data = json.loads(m.group(2))
            if tool_prefix in data:
                return m.group(1), data
    raise AssertionError(f"no evidence with {tool_prefix!r}")


def _gather(world: dict[str, Any]) -> Any:
    a = {"account_key": world["account"], "start": world["start"], "end": world["day"]}
    return ai(
        call("detect_patterns", a, "c1"),
        call("get_counterparties", a, "c2"),
        call("trace_flows", a, "c3"),
        call("load_skill", {"name": "fan-out"}, "c4"),
        call("screen_sanctions", {"account_key": world["account"]}, "c5"),
    )


def _submit(amount: str | None = None, disposition: str = "ESCALATE") -> Any:
    def step(messages: list[BaseMessage]) -> Any:
        ev_id, data = _evidence(messages, "fan")
        total = amount or data["fan"]["total_out_usd"]
        case = {
            "summary": "Fan-out of funds to several new counterparties shortly after inflows.",
            "hypotheses": [{"typology": "FAN-OUT", "status": "CONFIRMED", "evidence_ids": [ev_id]}],
            "findings": [
                {
                    "claim": f"Outflows totalled {total} USD in the window.",
                    "typology": "FAN-OUT",
                    "evidence_ids": [ev_id],
                    "cited_values": [{"kind": "amount", "value": str(total)}],
                }
            ],
            "disposition": disposition,
            "confidence": "MEDIUM",
        }
        return ai(call("submit_case", case, f"submit-{len(messages)}"))

    return step


def _decision(
    final: str = "ESCALATE", action: str = "APPROVE", reason: str = ""
) -> AnalystDecision:
    return AnalystDecision(
        action=action, analyst_id="analyst-1", reason=reason, final_disposition=final
    )


def test_happy_path_is_grounded_reviewed_and_audited(world: dict[str, Any], tmp_path: Path) -> None:
    rt = _runtime(world, tmp_path, [_gather(world), _submit()])
    graph = build_graph(rt, sqlite_checkpointer(tmp_path / "ckpt.sqlite"))
    run_id, payload = start(graph, "AL-T-0001")
    assert payload["status"] == "GROUNDED"
    assert payload["grounding"]["passed"] is True
    assert payload["case"]["disposition"] == "ESCALATE"
    events = {e.event_type for e in rt.audit.events(run_id=run_id)}
    assert "RUN_COMPLETED" not in events
    assert "ANALYST_DECISION" not in events
    resume(graph, run_id, _decision())
    events_after = [e.event_type for e in rt.audit.events(run_id=run_id)]
    for required in (
        "RUN_STARTED",
        "TOOL_CALL",
        "SKILL_LOADED",
        "LLM_CALL",
        "GROUNDING_RESULT",
        "REVIEW_VERDICT",
        "CASE_DRAFTED",
        "ANALYST_DECISION",
        "RUN_COMPLETED",
    ):
        assert required in events_after, required
    assert rt.audit.verify() == len(rt.audit.events())
    completed = rt.audit.events(run_id=run_id)[-1].payload
    assert completed["final_disposition"] == "ESCALATE"
    assert completed["input_tokens"] > 0
    assert completed["cost_usd"] is not None


def test_injection_memo_forces_review_of_a_closure(world: dict[str, Any], tmp_path: Path) -> None:
    rt = _runtime(world, tmp_path, [_gather(world), _submit(disposition="CLOSE_FALSE_POSITIVE")])
    _, payload = start(build_graph(rt, None), "AL-T-0001")
    assert payload["policy"]["disposition"] == "NEEDS_MORE_INFO"
    assert "INJECTION_ATTEMPT" in payload["policy"]["flags"]
    assert payload["policy"]["overrides"][0]["rule_id"] == "POL-04"


def test_invented_amount_is_repaired_after_feedback(world: dict[str, Any], tmp_path: Path) -> None:
    rt = _runtime(world, tmp_path, [_gather(world), _submit(amount="123456.78"), _submit()])
    run_id, payload = start(build_graph(rt, None), "AL-T-0001")
    assert payload["status"] == "GROUNDED"
    results = [
        e.payload["passed"]
        for e in rt.audit.events(run_id=run_id)
        if e.event_type == "GROUNDING_RESULT"
    ]
    assert results == [False, True]


def test_ungrounded_after_retries_is_blocked(world: dict[str, Any], tmp_path: Path) -> None:
    bad = _submit(amount="999999.99")
    rt = _runtime(world, tmp_path, [_gather(world), bad, bad, bad])
    _, payload = start(build_graph(rt, None), "AL-T-0001")
    assert payload["status"] == "BLOCKED_UNGROUNDED"
    assert payload["policy"] is None  # never shown as a recommendation


def test_off_allowlist_tool_is_rejected_and_audited(world: dict[str, Any], tmp_path: Path) -> None:
    sneaky = ai(call("close_alert", {"alert_id": "AL-T-0001"}, "x1"))
    rt = _runtime(world, tmp_path, [sneaky, _gather(world), _submit()])
    run_id, _ = start(build_graph(rt, None), "AL-T-0001")
    rejected = [
        e
        for e in rt.audit.events(run_id=run_id)
        if e.event_type == "TOOL_CALL" and e.payload.get("rejected")
    ]
    assert rejected
    assert rejected[0].payload["tool"] == "close_alert"


def test_llm_unavailable_degrades_to_needs_more_info(world: dict[str, Any], tmp_path: Path) -> None:
    rt = _runtime(world, tmp_path, [ConnectionError("provider down")])
    run_id, payload = start(build_graph(rt, None), "AL-T-0001")
    assert payload["status"] == "FALLBACK"
    assert payload["case"]["disposition"] == "NEEDS_MORE_INFO"
    errors = [
        e.payload["error"] for e in rt.audit.events(run_id=run_id) if e.event_type == "LLM_CALL"
    ]
    assert errors
    assert "provider down" in errors[0]


def test_budget_exhaustion_offers_only_submit(world: dict[str, Any], tmp_path: Path) -> None:
    rt = _runtime(world, tmp_path, [_gather(world), _submit()], budget=2)
    _, payload = start(build_graph(rt, None), "AL-T-0001")
    assert rt.models["investigator"].bound_tool_names[-1] == ["submit_case"]  # type: ignore[attr-defined]
    assert payload["case"] is not None


def test_reviewer_revise_loops_back_once(world: dict[str, Any], tmp_path: Path) -> None:
    revise = {"verdict": "REVISE", "objections": ["check counterparties"], "evidence_requests": []}
    rt = _runtime(world, tmp_path, [_gather(world), _submit(), _submit()], verdicts=[revise])
    run_id, payload = start(build_graph(rt, None), "AL-T-0001")
    verdicts = [
        e.payload["verdict"]
        for e in rt.audit.events(run_id=run_id)
        if e.event_type == "REVIEW_VERDICT"
    ]
    assert verdicts == ["REVISE", "ACCEPT"]
    assert payload["review"]["verdict"] == "ACCEPT"


def test_paused_case_survives_restart(world: dict[str, Any], tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.sqlite"
    rt = _runtime(world, tmp_path, [_gather(world), _submit()])
    run_id, _ = start(build_graph(rt, sqlite_checkpointer(ckpt)), "AL-T-0001")
    fresh = build_graph(_runtime(world, tmp_path, []), sqlite_checkpointer(ckpt))
    result = resume(
        fresh, run_id, _decision(final="NEEDS_MORE_INFO", action="EDIT", reason="need KYC")
    )
    assert result["decision"].final_disposition == "NEEDS_MORE_INFO"
    decided = [e for e in rt.audit.events(run_id=run_id) if e.event_type == "ANALYST_DECISION"]
    assert decided[0].payload["changed"] is True


def test_media_agent_keeps_only_verbatim_quotes() -> None:
    url = "https://news.example/ledger/2026/09/velmora-shell-network"
    model = ScriptedChatModel(
        bound_tool_names=[],
        script=[
            ai(call("search_media", {"query": "Velmora Freight Ltd"}, "s1")),
            ai(call("fetch_document", {"url": url}, "f1")),
            ai(
                call(
                    "submit_findings",
                    {
                        "findings": [
                            {
                                "url": url,
                                "quote": "Former employees allege that invoices were issued",
                                "category": "ALLEGATION",
                                "same_entity": True,
                                "reason": "same company",
                            },
                            {
                                "url": url,
                                "quote": "Velmora was convicted of fraud by a London court",
                                "category": "CONVICTION",
                                "same_entity": True,
                                "reason": "invented",
                            },
                        ]
                    },
                    "sub",
                )
            ),
        ],
    )
    check = run_media_agent(
        model,
        FixtureProvider(MEDIA_DOCS, date(2026, 11, 13)),
        {"name": "Velmora Freight Ltd", "country": "UK", "kind": "company"},
        load_prompt("media").text,
        on_llm_call=lambda *_a: None,
    )
    assert [f["category"] for f in check.verified] == ["ALLEGATION"]
    assert check.dropped[0]["drop_reason"] == "quote not found verbatim in document"
    assert check.queries == ["Velmora Freight Ltd"]


def test_coverage_gate_refuses_a_shallow_case(world: dict[str, Any], tmp_path: Path) -> None:
    shallow = ai(
        call(
            "detect_patterns",
            {"account_key": world["account"], "start": world["start"], "end": world["day"]},
            "s1",
        )
    )
    rt = _runtime(world, tmp_path, [shallow, _submit(), _gather(world), _submit()])
    run_id, payload = start(build_graph(rt, None), "AL-T-0001")
    refused = [
        e
        for e in rt.audit.events(run_id=run_id)
        if e.event_type == "TOOL_CALL" and e.payload.get("rejected") == "coverage gate"
    ]
    assert refused
    assert set(refused[0].payload["missing"]) == {
        "get_counterparties",
        "trace_flows",
        "screen_sanctions",
    }
    assert payload["status"] == "GROUNDED"


def test_gate_can_be_disabled_for_the_baseline(world: dict[str, Any], tmp_path: Path) -> None:
    from dataclasses import replace

    shallow = ai(
        call(
            "detect_patterns",
            {"account_key": world["account"], "start": world["start"], "end": world["day"]},
            "s1",
        )
    )
    rt = replace(_runtime(world, tmp_path, [shallow, _submit()]), required_steps=())
    _, payload = start(build_graph(rt, None), "AL-T-0001")
    assert payload["status"] == "GROUNDED"
