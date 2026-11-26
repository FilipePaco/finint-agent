"""The investigation graph (task 3.17 · R3.1-R3.8, R3.12 · ADR-03.1, ADR-03.2).

    intake → screen_counterparties → investigate ⇄ grounding → review ⇄ investigate → policy
           → analyst → finalize

* **screen_counterparties**: code. Screens the alert's counterparties against OFAC (ADR-03.9).
* **investigate**: LLM agent (tools, skills, media sub-agent).
* **grounding**: code. Up to 2 repair rounds, then ``BLOCKED_UNGROUNDED`` (no recommendation).
* **review**: LLM agent. Up to 2 revise rounds, then remaining objections go to the analyst.
* **policy**: code. Raises scrutiny only; overrides are recorded.
* **analyst**: ``interrupt``. Nothing is final without an ``AnalystDecision``.

State is checkpointed (SQLite), so a case waiting for the analyst survives a restart. The
audit trail is the system of record; totals in ``RUN_COMPLETED`` are summed from it.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, TypedDict
from uuid import uuid4

import duckdb
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from finint.agents.investigator import (
    InvestigatorDeps,
    evidence_digest,
    fallback_case,
    run_investigator,
)
from finint.agents.llm import Prompt, record_llm_call
from finint.agents.media_agent import run_media_agent
from finint.agents.reviewer import ReviewVerdict, review_case
from finint.config import Settings
from finint.core.audit import AuditLog, EventType, NewEvent
from finint.core.evidence import EvidenceStore, agent_visible
from finint.core.grounding import GroundingReport, validate_grounding
from finint.core.models import AnalystDecision, CaseFile, Evidence
from finint.core.policy import PolicyInput, PolicyResult, apply_policy
from finint.tools.investigation import ToolContext, get_alert
from finint.tools.media import MediaProvider, SearchNotPermittedError, check_permitted
from finint.tools.sanctions import PERSON_TYPES, screen_counterparties, screen_party
from finint.tools.skills import SkillLibrary

MAX_GROUNDING_ATTEMPTS = 3  # first draft + 2 repairs
MAX_REVIEW_ROUNDS = 2

Status = Literal[
    "INVESTIGATING", "GROUNDED", "BLOCKED_UNGROUNDED", "FALLBACK", "AWAITING_ANALYST", "CLOSED"
]


class InvestigationState(TypedDict, total=False):
    run_id: str
    alert_id: str
    started_at: float
    evidence: dict[str, Evidence]
    messages: list[BaseMessage]
    draft: CaseFile | None
    grounding: GroundingReport | None
    grounding_attempts: int
    review: ReviewVerdict | None
    review_rounds: int
    policy: PolicyResult | None
    status: Status
    tool_calls_used: int
    skills_loaded: list[str]
    fallback_reason: str | None
    gathering_seconds: float
    decision: AnalystDecision | None


@dataclass
class Runtime:
    settings: Settings
    audit: AuditLog
    classification: dict[str, dict[str, str]]
    skills: SkillLibrary
    screening_config: dict[str, Any]
    media_provider: MediaProvider
    models: dict[str, BaseChatModel]  # investigator, reviewer, media
    model_ids: dict[str, str]
    prompts: dict[str, Prompt]
    tool_budget: int = 20
    config_factory: Callable[[str, str], dict[str, Any]] | None = None
    use_reviewer: bool = True  # ablation switch (R4.3)
    # Harness v2 coverage gate: tools that must run on the alerted account before submitting.
    required_steps: tuple[str, ...] = (
        "detect_patterns",
        "get_counterparties",
        "trace_flows",
        "screen_sanctions",
    )
    memo_overrides: dict[int, str] | None = None  # evaluation-only counterfactual (R4.2)
    # Harness screening of the alert's counterparties, in code (R3.4, ADR-03.9).
    screen_counterparties: bool = True

    def ctx(self, evidence: dict[str, Evidence] | None = None) -> ToolContext:
        return ToolContext(
            self.settings.duckdb_path,
            self.classification,
            EvidenceStore.from_dict(evidence or {}),
            memo_overrides=self.memo_overrides or {},
        )

    def llm_config(self, run_id: str, alert_id: str) -> Any:
        return self.config_factory(run_id, alert_id) if self.config_factory else None

    def recorder(self, role: Any, run_id: str, alert_id: str) -> Callable[..., None]:
        def record(response: AIMessage | None, started: float, error: str | None) -> None:
            record_llm_call(
                self.audit,
                None,
                run_id=run_id,
                alert_id=alert_id,
                role=role,
                model=self.model_ids[role],
                prompt=self.prompts[role],
                response=response,
                started=started,
                error=error,
            )

        return record

    def event(
        self, state: InvestigationState, event_type: EventType, actor: str, payload: dict[str, Any]
    ) -> None:
        self.audit.append(
            NewEvent(event_type, actor, payload, run_id=state["run_id"], alert_id=state["alert_id"])
        )


def build_graph(rt: Runtime, checkpointer: BaseCheckpointSaver[Any] | None) -> Any:
    graph: StateGraph[InvestigationState] = StateGraph(InvestigationState)

    def intake(state: InvestigationState) -> dict[str, Any]:
        ctx = rt.ctx()
        alert = get_alert(ctx, state["alert_id"])
        rt.event(
            state,
            "RUN_STARTED",
            "system:graph",
            {
                "models": rt.model_ids,
                "prompt_versions": {k: p.version for k, p in rt.prompts.items()},
                "tool_budget": rt.tool_budget,
                "alert_evidence_id": alert.evidence_id,
                "injection_hits": len(alert.injection_hits),
            },
        )
        system = rt.prompts["investigator"].text.replace("{skills_index}", rt.skills.index())
        brief = (
            f"Alert {state['alert_id']} on account {alert.data['account_key']} "
            f"(alert date {alert.data['alert_date']}).\n"
            f"[{alert.evidence_id}] {alert.summary}\n\nInvestigate it."
        )
        return {
            "evidence": ctx.store.as_dict(),
            "messages": [SystemMessage(system), HumanMessage(brief)],
            "started_at": time.time(),
            "status": "INVESTIGATING",
            "tool_calls_used": 0,
            "grounding_attempts": 0,
            "review_rounds": 0,
            "skills_loaded": [],
        }

    def screen_counterparties_node(state: InvestigationState) -> dict[str, Any]:
        """Harness step, no LLM (R3.4, ADR-03.9): screen the alert's counterparties in code.

        Results go to the evidence store under ``cp-`` ids, each screening is audited, and the
        policy guard reads their statuses. They are not shown to the agents.
        """
        if not rt.screen_counterparties:
            return {}
        ctx = rt.ctx(state["evidence"])
        alert = state["evidence"]["ev-001"].data
        subject = alert["account_key"]
        alert_date = date.fromisoformat(str(alert["alert_date"])[:10])
        for screen in screen_counterparties(ctx, subject, alert_date, rt.screening_config):
            payload: dict[str, Any] = {
                "tool": "screen_sanctions",
                "args": {"account_key": screen.selected.account_key},
                "selected_by": screen.selected.reasons,
                "counterparty_of": subject,
            }
            if screen.evidence is None:
                payload["error"] = screen.error
            else:
                payload |= {
                    "evidence_id": screen.evidence.evidence_id,
                    "status": screen.evidence.data["status"],
                }
            rt.event(state, "TOOL_CALL", "system:counterparty-screening", payload)
        return {"evidence": ctx.store.as_dict()}

    def investigate(state: InvestigationState) -> dict[str, Any]:
        ctx = rt.ctx(state["evidence"])
        run_id, alert_id = state["run_id"], state["alert_id"]
        deps = InvestigatorDeps(
            ctx=ctx,
            skills=rt.skills,
            audit=rt.audit,
            run_id=run_id,
            alert_id=alert_id,
            screen=lambda key: screen_party(ctx, key, rt.screening_config),
            media_check=lambda key: _media_check(rt, ctx, state, key),
            on_llm_call=rt.recorder("investigator", run_id, alert_id),
            config=rt.llm_config(run_id, alert_id),
            tool_budget=rt.tool_budget,
            alert_account=state["evidence"]["ev-001"].data.get("account_key"),
            required_steps=rt.required_steps,
        )
        result = run_investigator(
            rt.models["investigator"],
            list(state["messages"]),
            deps,
            state.get("tool_calls_used", 0),
        )
        update: dict[str, Any] = {
            "messages": result.messages,
            "evidence": ctx.store.as_dict(),
            "tool_calls_used": result.tool_calls_used,
            "skills_loaded": [*state.get("skills_loaded", []), *result.skills_loaded],
            "gathering_seconds": time.time() - state["started_at"],
        }
        if result.draft is None:
            reason = result.fallback_reason or "no case"
            update |= {
                "draft": fallback_case(alert_id, list(ctx.store), reason),
                "status": "FALLBACK",
                "fallback_reason": reason,
            }
        else:
            update |= {"draft": result.draft}
        return update

    def grounding(state: InvestigationState) -> dict[str, Any]:
        draft = state["draft"]
        assert draft is not None  # noqa: S101 - investigate always sets a draft
        # Only what the agent saw can ground its case (harness `cp-` items were never shown).
        report = validate_grounding(draft, agent_visible(state["evidence"]))
        attempts = state.get("grounding_attempts", 0) + 1
        rt.event(
            state,
            "GROUNDING_RESULT",
            "system:grounding",
            {
                "passed": report.passed,
                "attempt": attempts,
                "n_values_checked": report.n_values_checked,
                "violations": [v.model_dump() for v in report.violations],
            },
        )
        update: dict[str, Any] = {"grounding": report, "grounding_attempts": attempts}
        if report.passed:
            return update | {
                "status": "GROUNDED" if state.get("status") != "FALLBACK" else "FALLBACK"
            }
        if attempts < MAX_GROUNDING_ATTEMPTS:
            lines = "\n".join(
                f"- {v.location}: {v.kind} {v.value!r} ({v.reason})" for v in report.violations
            )
            feedback = HumanMessage(
                "The grounding check rejected your case file. Fix these values (use only "
                f"values from tool results, or fetch what you need) and submit again:\n{lines}"
            )
            return update | {"messages": [*state["messages"], feedback]}
        return update | {"status": "BLOCKED_UNGROUNDED"}

    def route_grounding(state: InvestigationState) -> str:
        if state["status"] == "BLOCKED_UNGROUNDED":
            return "analyst"
        if not state["grounding"] or not state["grounding"].passed:
            return "investigate"
        if state["status"] == "FALLBACK" or not rt.use_reviewer:
            return "policy"
        return "review"

    def review(state: InvestigationState) -> dict[str, Any]:
        draft = state["draft"]
        assert draft is not None  # noqa: S101
        verdict = review_case(
            rt.models["reviewer"],
            rt.prompts["reviewer"].text,
            draft,
            evidence_digest(list(agent_visible(state["evidence"]).values())),
            on_llm_call=rt.recorder("reviewer", state["run_id"], state["alert_id"]),
            config=rt.llm_config(state["run_id"], state["alert_id"]),
        )
        rounds = state.get("review_rounds", 0) + 1
        rt.event(
            state, "REVIEW_VERDICT", "agent:reviewer", {"round": rounds, **verdict.model_dump()}
        )
        update: dict[str, Any] = {"review": verdict, "review_rounds": rounds}
        if verdict.verdict == "REVISE" and rounds < MAX_REVIEW_ROUNDS:
            feedback = HumanMessage(
                "A senior reviewer asked for revisions. Address them (fetch evidence if needed) "
                "and call submit_case again.\nObjections:\n"
                + "\n".join(f"- {o}" for o in verdict.objections)
                + (
                    "\nEvidence requests:\n"
                    + "\n".join(f"- {r}" for r in verdict.evidence_requests)
                    if verdict.evidence_requests
                    else ""
                )
            )
            update |= {"messages": [*state["messages"], feedback], "grounding_attempts": 0}
        return update

    def route_review(state: InvestigationState) -> str:
        review_ = state["review"]
        revise = review_ is not None and review_.verdict == "REVISE"
        return "investigate" if revise and state["review_rounds"] < MAX_REVIEW_ROUNDS else "policy"

    def policy(state: InvestigationState) -> dict[str, Any]:
        draft = state["draft"]
        assert draft is not None  # noqa: S101
        evidence = list(state["evidence"].values())
        # Every screening counts, the harness's counterparty screens included (constitution III).
        statuses = [e.data["status"] for e in evidence if e.tool == "screen_sanctions"]
        categories = [
            f["category"]
            for e in evidence
            if e.tool == "request_media_check"
            for f in e.data.get("findings", [])
            if f.get("same_entity")
        ]
        result = apply_policy(
            PolicyInput(
                disposition=draft.disposition,
                sanctions_statuses=statuses,
                n_evidence=len(agent_visible(state["evidence"])),  # what the agent gathered
                injection_attempts=sum(len(e.injection_hits) for e in evidence),
                adverse_media_categories=categories,
            )
        )
        for override in result.overrides:
            rt.event(state, "POLICY_OVERRIDE", "system:policy", override.model_dump())
        rt.event(
            state,
            "CASE_DRAFTED",
            "system:graph",
            {
                "status": state["status"],
                "recommended_disposition": draft.disposition,
                "policy_disposition": result.disposition,
                "flags": result.flags,
                "case": draft.model_dump(mode="json"),
                "gathering_seconds": round(state.get("gathering_seconds", 0.0), 2),
            },
        )
        return {"policy": result}

    def analyst(state: InvestigationState) -> dict[str, Any]:
        payload = {
            "run_id": state["run_id"],
            "alert_id": state["alert_id"],
            "status": state["status"],
            "case": _dump(state.get("draft")),
            "policy": _dump(state.get("policy")),
            "review": _dump(state.get("review")),
            "grounding": _dump(state.get("grounding")),
        }
        decision = AnalystDecision.model_validate(interrupt(payload))
        policy_result = state.get("policy")
        recommended = policy_result.disposition if policy_result is not None else None
        rt.event(
            state,
            "ANALYST_DECISION",
            f"analyst:{decision.analyst_id}",
            {
                **decision.model_dump(),
                "recommended_disposition": recommended,
                "changed": recommended != decision.final_disposition
                or decision.action != "APPROVE",
            },
        )
        return {"decision": decision, "status": "CLOSED"}

    def finalize(state: InvestigationState) -> dict[str, Any]:
        calls = [e for e in rt.audit.events(run_id=state["run_id"]) if e.event_type == "LLM_CALL"]
        costs: list[float | None] = [e.payload.get("cost_usd") for e in calls]
        known = [c for c in costs if c is not None]
        decision = state.get("decision")
        rt.event(
            state,
            "RUN_COMPLETED",
            "system:graph",
            {
                "llm_calls": len(calls),
                "input_tokens": sum(int(e.payload.get("input_tokens", 0)) for e in calls),
                "output_tokens": sum(int(e.payload.get("output_tokens", 0)) for e in calls),
                "cost_usd": round(sum(known), 6) if len(known) == len(costs) else None,
                "tool_calls": state.get("tool_calls_used", 0),
                "evidence_items": len(state["evidence"]),
                "gathering_seconds": round(state.get("gathering_seconds", 0.0), 2),
                "final_disposition": decision.final_disposition if decision else None,
            },
        )
        return {}

    graph.add_node("intake", intake)
    graph.add_node("screen_counterparties", screen_counterparties_node)
    graph.add_node("investigate", investigate)
    graph.add_node("grounding", grounding)
    graph.add_node("review", review)
    graph.add_node("policy", policy)
    graph.add_node("analyst", analyst)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "screen_counterparties")
    graph.add_edge("screen_counterparties", "investigate")
    graph.add_edge("investigate", "grounding")
    graph.add_conditional_edges(
        "grounding", route_grounding, ["investigate", "review", "policy", "analyst"]
    )
    graph.add_conditional_edges("review", route_review, ["investigate", "policy"])
    graph.add_edge("policy", "analyst")
    graph.add_edge("analyst", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def _media_check(
    rt: Runtime, ctx: ToolContext, state: InvestigationState, account_key: str
) -> Evidence | str:
    with duckdb.connect(str(rt.settings.duckdb_path), read_only=True) as con:
        row = con.execute(
            "SELECT e.entity_id, e.synthetic_name, e.country, e.entity_type, e.is_injected "
            "FROM accounts a JOIN entities e USING (entity_id) WHERE a.account_key = ?",
            [account_key],
        ).fetchone()
    if row is None:
        return f"ERROR: unknown account {account_key}"
    entity_id, name, country, entity_type, is_injected = row
    try:
        check_permitted(bool(is_injected), entity_id)
    except SearchNotPermittedError:
        return (
            "NOT_PERMITTED: adverse-media search runs only for designated test entities; "
            "no media check was performed for this owner (not evidence either way)."
        )
    kind = "person" if entity_type in PERSON_TYPES else "company"
    check = run_media_agent(
        rt.models["media"],
        rt.media_provider,
        {"name": name, "country": country, "kind": kind},
        rt.prompts["media"].text,
        on_llm_call=rt.recorder("media", state["run_id"], state["alert_id"]),
        config=rt.llm_config(state["run_id"], state["alert_id"]),
    )
    for dropped in check.dropped:
        rt.event(state, "MEDIA_CLAIM_DROPPED", "system:media-verifier", dropped)
    data = {
        "account_key": account_key,
        "entity_id": entity_id,
        "queries": check.queries,
        "documents_read": list(check.fetched),
        "findings": check.verified,
        "n_findings_dropped": len(check.dropped),
        "provider": rt.media_provider.name,
        "submitted": check.submitted,
    }
    return ctx.store.add(
        "request_media_check",
        {"account_key": account_key},
        data,
        untrusted=True,
        injection_hits=check.injection_hits,
    )


def _dump(model: Any) -> dict[str, Any] | None:
    return model.model_dump(mode="json") if model is not None else None


# Only these project types may be rebuilt from a checkpoint (explicit allowlist instead of
# deserializing any importable class).
CHECKPOINT_TYPES = [
    ("finint.core.models", "Evidence"),
    ("finint.core.models", "EvidenceValues"),
    ("finint.core.models", "CaseFile"),
    ("finint.core.models", "Finding"),
    ("finint.core.models", "CitedValue"),
    ("finint.core.models", "Hypothesis"),
    ("finint.core.models", "AnalystDecision"),
    ("finint.core.grounding", "GroundingReport"),
    ("finint.core.grounding", "Violation"),
    ("finint.core.policy", "PolicyResult"),
    ("finint.core.policy", "Override"),
    ("finint.agents.reviewer", "ReviewVerdict"),
]


def sqlite_checkpointer(path: Path) -> SqliteSaver:
    path.parent.mkdir(parents=True, exist_ok=True)
    serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False), serde=serde)


def new_run_id() -> str:
    """32 hex chars: also valid as a Langfuse/OpenTelemetry trace id."""
    return uuid4().hex


def start(graph: Any, alert_id: str, run_id: str | None = None) -> tuple[str, dict[str, Any]]:
    """Run until the analyst interrupt. Returns ``(run_id, payload for the analyst)``."""
    run_id = run_id or new_run_id()
    config = {"configurable": {"thread_id": run_id}}
    out = graph.invoke({"run_id": run_id, "alert_id": alert_id}, config=config)
    interrupts = out.get("__interrupt__") or []
    return run_id, (interrupts[0].value if interrupts else {})


def resume(graph: Any, run_id: str, decision: AnalystDecision) -> dict[str, Any]:
    config = {"configurable": {"thread_id": run_id}}
    result: dict[str, Any] = graph.invoke(Command(resume=decision.model_dump()), config=config)
    return result
