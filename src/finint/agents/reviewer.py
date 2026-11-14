"""Reviewer agent (task 3.16 · R3.6): adversarial critique of a grounded draft."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from finint.core.models import CaseFile


class ReviewVerdict(BaseModel):
    verdict: Literal["ACCEPT", "REVISE"]
    objections: list[str] = Field(default_factory=list)
    evidence_requests: list[str] = Field(default_factory=list)


def review_case(
    model: BaseChatModel,
    system_prompt: str,
    case: CaseFile,
    evidence_digest: str,
    *,
    on_llm_call: Callable[[AIMessage | None, float, str | None], None],
    config: RunnableConfig | None = None,
) -> ReviewVerdict:
    """One structured call. If the reviewer fails, the draft goes on with that noted."""
    structured = model.with_structured_output(ReviewVerdict, include_raw=True)
    messages = [
        SystemMessage(system_prompt),
        HumanMessage(
            "Draft case file:\n"
            f"{json.dumps(case.model_dump(mode='json'), indent=1)}\n\n"
            f"Evidence available to the investigator:\n{evidence_digest}"
        ),
    ]
    started = time.perf_counter()
    try:
        out: Any = structured.invoke(messages, config=config)
    except Exception as error:
        on_llm_call(None, started, f"{type(error).__name__}: {error}")
        return ReviewVerdict(verdict="ACCEPT", objections=[f"reviewer unavailable: {error}"])
    raw = out.get("raw") if isinstance(out, dict) else None
    on_llm_call(raw if isinstance(raw, AIMessage) else None, started, None)
    parsed = out.get("parsed") if isinstance(out, dict) else out
    if isinstance(parsed, ReviewVerdict):
        return parsed
    if isinstance(parsed, dict):
        return ReviewVerdict.model_validate(parsed)
    return ReviewVerdict(verdict="ACCEPT", objections=["reviewer output could not be parsed"])
