"""Quarantined adverse-media agent (task 3.14 · R3.3 · ADR-03.3).

It decides its own queries (max 3), which documents to read, whether each is about the same
entity, and how to classify it. It receives only the entity's name, country and type: no
transaction tools, no case context. News text is the main injection vector, so isolation
means a successful injection can at most corrupt media findings, and every finding is then
checked in code: its quote must be a verbatim substring of a document fetched in this run.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field, ValidationError

from finint.core.injection import scan
from finint.core.minimize import wrap_untrusted
from finint.tools.media import MediaProvider, MediaText

MAX_QUERIES = 3
MAX_STEPS = 10
MIN_QUOTE_CHARS = 20
Category = Literal[
    "ALLEGATION", "CHARGE", "CONVICTION", "SANCTION", "IRRELEVANT", "SAME_NAME_DIFFERENT_ENTITY"
]


class SearchMedia(BaseModel):
    """Search news about the entity. Short query; at most 3 searches per check."""

    query: str = Field(max_length=120)


class FetchDocument(BaseModel):
    """Read the full text of a document returned by a search."""

    url: str


class MediaFinding(BaseModel):
    url: str
    quote: str = Field(description="exact text copied from the document")
    category: Category
    same_entity: bool
    reason: str = Field(max_length=300)


class SubmitFindings(BaseModel):
    """Submit the classified findings (call once, at the end)."""

    findings: list[MediaFinding]


def tool_spec(name: str, model: type[BaseModel]) -> dict[str, Any]:
    spec: dict[str, Any] = convert_to_openai_tool(model)
    spec["function"]["name"] = name
    return spec


TOOLS: dict[str, type[BaseModel]] = {
    "search_media": SearchMedia,
    "fetch_document": FetchDocument,
    "submit_findings": SubmitFindings,
}


@dataclass
class MediaCheck:
    queries: list[str] = field(default_factory=list)
    fetched: dict[str, MediaText] = field(default_factory=dict)
    verified: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    injection_hits: list[dict[str, Any]] = field(default_factory=list)
    submitted: bool = False


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def verify(
    findings: list[MediaFinding], fetched: dict[str, MediaText]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep findings whose quote is verbatim (whitespace-insensitive) in a fetched document."""
    verified, dropped = [], []
    for f in findings:
        doc = fetched.get(f.url)
        if doc is None:
            dropped.append(
                {**f.model_dump(), "drop_reason": "document was not fetched in this check"}
            )
        elif len(f.quote.strip()) < MIN_QUOTE_CHARS:
            dropped.append({**f.model_dump(), "drop_reason": "quote too short to verify"})
        elif _norm(f.quote) not in _norm(doc.text):
            dropped.append(
                {**f.model_dump(), "drop_reason": "quote not found verbatim in document"}
            )
        else:
            verified.append(
                {
                    **f.model_dump(),
                    "title": doc.title,
                    "source": doc.source,
                    "published": doc.published,
                    "retrieved": doc.retrieved,
                }
            )
    return verified, dropped


def run_media_agent(
    model: BaseChatModel,
    provider: MediaProvider,
    entity: dict[str, Any],
    system_prompt: str,
    *,
    on_llm_call: Callable[[AIMessage | None, float, str | None], None],
    config: RunnableConfig | None = None,
) -> MediaCheck:
    bound = model.bind_tools([tool_spec(name, schema) for name, schema in TOOLS.items()])
    check = MediaCheck()
    messages: list[BaseMessage] = [
        SystemMessage(system_prompt),
        HumanMessage(
            "Entity to check:\n"
            f"- name: {wrap_untrusted(str(entity['name']))}\n"
            f"- country: {entity.get('country')}\n- type: {entity.get('kind')}"
        ),
    ]
    for _ in range(MAX_STEPS):
        started = time.perf_counter()
        try:
            response = bound.invoke(messages, config=config)
        except Exception as error:
            on_llm_call(None, started, f"{type(error).__name__}: {error}")
            return check
        assert isinstance(response, AIMessage)  # noqa: S101 - chat models return AIMessage
        on_llm_call(response, started, None)
        messages.append(response)
        if not response.tool_calls:
            messages.append(HumanMessage("Call submit_findings now (an empty list is allowed)."))
            continue
        for call in response.tool_calls:
            content = _execute(call["name"], call["args"], provider, check)
            messages.append(ToolMessage(content, tool_call_id=call["id"] or call["name"]))
            if check.submitted:
                return check
    return check


def _execute(name: str, args: dict[str, Any], provider: MediaProvider, check: MediaCheck) -> str:
    if name not in TOOLS:
        return f"ERROR: tool {name!r} is not available to you"
    try:
        parsed = TOOLS[name].model_validate(args)
    except ValidationError as error:
        return f"ERROR: invalid arguments: {error.errors()[:3]}"
    if isinstance(parsed, SearchMedia):
        if len(check.queries) >= MAX_QUERIES:
            return "ERROR: search limit reached; fetch documents or submit findings"
        check.queries.append(parsed.query)
        hits = provider.search(parsed.query)
        return json.dumps(
            [
                {
                    "url": h.url,
                    "title": wrap_untrusted(h.title),
                    "source": h.source,
                    "published": str(h.published),
                }
                for h in hits
            ]
        )
    if isinstance(parsed, FetchDocument):
        try:
            doc = provider.fetch(parsed.url)
        except FileNotFoundError as error:
            return f"ERROR: {error}"
        check.fetched[doc.url] = doc
        result = scan(doc.text)
        if result.is_attempt:
            check.injection_hits.append({"url": doc.url, "rules": result.rules})
        return json.dumps(
            {
                "url": doc.url,
                "title": wrap_untrusted(doc.title),
                "source": doc.source,
                "published": str(doc.published),
                "text": wrap_untrusted(doc.text),
            }
        )
    assert isinstance(parsed, SubmitFindings)  # noqa: S101 - exhaustive over TOOLS
    check.verified, check.dropped = verify(parsed.findings, check.fetched)
    check.submitted = True
    return f"received: {len(check.verified)} verified, {len(check.dropped)} dropped"
