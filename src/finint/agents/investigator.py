"""Investigator agent (task 3.15 · R3.1, R3.2 · ADR-03.1).

The model decides which tools to call, in which order, and when to stop; the code decides
what is allowed. Every tool call is executed by this module from an allowlist (an unknown
tool name is refused and audited), bounded by a tool-call budget, and recorded in the
audit trail. If the budget runs out the agent must submit; if the LLM is unavailable the
run degrades to a deterministic evidence pack with ``NEEDS_MORE_INFO``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, ValidationError

from finint.agents.media_agent import tool_spec
from finint.core.audit import AuditLog, NewEvent
from finint.core.models import CaseFile, Evidence
from finint.tools import investigation as inv
from finint.tools.skills import SkillLibrary, UnknownSkillError

DEFAULT_TOOL_BUDGET = 20
MAX_SUBMIT_ATTEMPTS = 3


class AccountArgs(BaseModel):
    account_key: str = Field(description="bank:account key, e.g. 952:8139F54E0")


class WindowArgs(AccountArgs):
    start: str = Field(description="ISO date YYYY-MM-DD")
    end: str = Field(description="ISO date YYYY-MM-DD (window at most 14 days)")


class TransactionsArgs(WindowArgs):
    direction: Literal["in", "out", "both"] = "both"
    limit: int = Field(default=20, ge=1, le=50)


class FlowArgs(WindowArgs):
    direction: Literal["in", "out"] = "out"
    hops: int = Field(default=2, ge=1, le=3)


class SkillArgs(BaseModel):
    name: str


ARG_MODELS: dict[str, type[BaseModel]] = {
    "get_account_profile": AccountArgs,
    "get_transactions": TransactionsArgs,
    "get_counterparties": WindowArgs,
    "trace_flows": FlowArgs,
    "detect_patterns": WindowArgs,
    "screen_sanctions": AccountArgs,
    "request_media_check": AccountArgs,
    "load_skill": SkillArgs,
    "submit_case": CaseFile,
}
DESCRIPTIONS = {
    "get_account_profile": "Owner entity (type, country, name), its other accounts, activity.",
    "get_transactions": "Transactions of an account in a window with exact totals (largest first).",
    "get_counterparties": "Counterparties by volume in a window, with counts and totals.",
    "trace_flows": "Multi-hop money flow out of or into an account (max 3 hops, time-ordered).",
    "detect_patterns": "Code-computed indicators: fan-in/out, cycles, flow-through, structuring.",
    "screen_sanctions": "Screen the account owner against OFAC SDN. Status is decided by code.",
    "request_media_check": "Adverse-media check of the account owner (only test entities allowed).",
    "load_skill": "Load an investigation playbook by name (see the skills list).",
    "submit_case": "Submit the final case file. Call exactly once, at the end.",
}


def tool_specs(names: list[str]) -> list[dict[str, Any]]:
    specs = []
    for name in names:
        spec = tool_spec(name, ARG_MODELS[name])
        spec["function"]["description"] = DESCRIPTIONS[name]
        specs.append(spec)
    return specs


@dataclass
class InvestigatorDeps:
    ctx: inv.ToolContext
    skills: SkillLibrary
    audit: AuditLog
    run_id: str
    alert_id: str
    screen: Callable[[str], Evidence]
    media_check: Callable[[str], Evidence | str]
    on_llm_call: Callable[[AIMessage | None, float, str | None], None]
    config: RunnableConfig | None = None
    tool_budget: int = DEFAULT_TOOL_BUDGET


@dataclass
class InvestigatorResult:
    messages: list[BaseMessage]
    draft: CaseFile | None
    tool_calls_used: int
    skills_loaded: list[str] = field(default_factory=list)
    fallback_reason: str | None = None


def _audit(deps: InvestigatorDeps, event_type: Any, payload: dict[str, Any]) -> None:
    deps.audit.append(
        NewEvent(
            event_type, "agent:investigator", payload, run_id=deps.run_id, alert_id=deps.alert_id
        )
    )


def _execute(name: str, args: dict[str, Any], deps: InvestigatorDeps, skills: list[str]) -> str:
    if name not in ARG_MODELS or name == "submit_case":
        _audit(deps, "TOOL_CALL", {"tool": name, "args": args, "rejected": "not on allowlist"})
        return f"ERROR: tool {name!r} is not available"
    try:
        parsed = ARG_MODELS[name].model_validate(args)
    except ValidationError as error:
        return f"ERROR: invalid arguments for {name}: {error.errors()[:3]}"
    a = parsed.model_dump()
    try:
        if name == "load_skill":
            skill = deps.skills.load(a["name"])
            skills.append(skill.name)
            _audit(deps, "SKILL_LOADED", {"skill": skill.name})
            return f"# Skill: {skill.name}\n{skill.body}"
        evidence: Evidence | str
        if name == "screen_sanctions":
            evidence = deps.screen(a["account_key"])
        elif name == "request_media_check":
            evidence = deps.media_check(a["account_key"])
        else:
            evidence = getattr(inv, name)(deps.ctx, **a)
    except (inv.ToolError, UnknownSkillError) as error:
        _audit(deps, "TOOL_CALL", {"tool": name, "args": a, "error": str(error)})
        return f"ERROR: {error}"
    if isinstance(evidence, str):
        _audit(deps, "TOOL_CALL", {"tool": name, "args": a, "result": evidence})
        return evidence
    _audit(
        deps,
        "TOOL_CALL",
        {
            "tool": name,
            "args": a,
            "evidence_id": evidence.evidence_id,
            "injection_hits": len(evidence.injection_hits),
        },
    )
    return f"[{evidence.evidence_id}] {evidence.summary}"


def _submit(args: dict[str, Any], alert_id: str) -> CaseFile | str:
    try:
        return CaseFile.model_validate({**args, "alert_id": alert_id})
    except ValidationError as error:
        return f"ERROR: case file invalid, fix and call submit_case again: {error.errors()[:5]}"


def run_investigator(
    model: BaseChatModel, messages: list[BaseMessage], deps: InvestigatorDeps, used: int = 0
) -> InvestigatorResult:
    """Continue (or start) the investigation from ``messages`` until a valid case is submitted."""
    all_tools = list(ARG_MODELS)
    skills: list[str] = []
    submit_attempts = 0
    budget_notice_sent = False
    while True:
        exhausted = used >= deps.tool_budget
        if exhausted and not budget_notice_sent:
            messages.append(
                HumanMessage(
                    "Tool budget exhausted. Call submit_case now. If material questions "
                    "remain, use NEEDS_MORE_INFO and list them in open_questions."
                )
            )
            budget_notice_sent = True
        bound = model.bind_tools(tool_specs(["submit_case"] if exhausted else all_tools))
        started = time.perf_counter()
        try:
            response = bound.invoke(messages, config=deps.config)
        except Exception as error:
            deps.on_llm_call(None, started, f"{type(error).__name__}: {error}")
            return InvestigatorResult(messages, None, used, skills, f"LLM unavailable: {error}")
        assert isinstance(response, AIMessage)  # noqa: S101 - chat models return AIMessage
        deps.on_llm_call(response, started, None)
        messages.append(response)
        if not response.tool_calls:
            submit_attempts += 1
            if submit_attempts >= MAX_SUBMIT_ATTEMPTS:
                return InvestigatorResult(messages, None, used, skills, "no case submitted")
            messages.append(HumanMessage("Continue with tool calls, or call submit_case."))
            continue
        draft: CaseFile | None = None
        for call in response.tool_calls:
            call_id = call["id"] or call["name"]
            if call["name"] == "submit_case":
                result = _submit(call["args"], deps.alert_id)
                if isinstance(result, CaseFile):
                    draft = result
                    messages.append(ToolMessage("case received", tool_call_id=call_id))
                else:
                    submit_attempts += 1
                    messages.append(ToolMessage(result, tool_call_id=call_id))
                continue
            if used >= deps.tool_budget:
                messages.append(ToolMessage("ERROR: tool budget exhausted", tool_call_id=call_id))
                continue
            used += 1
            messages.append(
                ToolMessage(
                    _execute(call["name"], call["args"], deps, skills), tool_call_id=call_id
                )
            )
        if draft is not None:
            return InvestigatorResult(messages, draft, used, skills)
        if submit_attempts >= MAX_SUBMIT_ATTEMPTS:
            return InvestigatorResult(messages, None, used, skills, "case file never validated")


def fallback_case(alert_id: str, evidence: list[Evidence], reason: str) -> CaseFile:
    """Deterministic case when the LLM is unavailable or never produced a valid case file."""
    return CaseFile(
        alert_id=alert_id,
        summary=(
            "Automated investigation did not complete; the evidence gathered so far is attached "
            "for manual review. No disposition was reached by the agent."
        ),
        disposition="NEEDS_MORE_INFO",
        confidence="LOW",
        open_questions=[
            f"Agent stopped: {reason}"[:300],
            f"Evidence collected: {', '.join(e.evidence_id for e in evidence) or 'none'}",
        ],
    )


def evidence_digest(evidence: list[Evidence], max_chars: int = 1500) -> str:
    """Compact view of all evidence for the reviewer (bounded per item)."""
    parts = [
        f"[{e.evidence_id}] {e.tool} {json.dumps(e.args, default=str)}: {e.summary[:max_chars]}"
        for e in evidence
    ]
    return "\n".join(parts)
