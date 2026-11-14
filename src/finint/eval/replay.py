"""Record/replay chat model for the CI evaluation (task 4.4 · R4.4 · ADR-04.2).

``record`` wraps a live model and stores every response under a key derived from the
request (role, message contents, tool calls, bound tools). ``replay`` returns the stored
response for the same request and fails loudly on any request it has not seen, so a
change in prompts, tools or evidence formatting shows up as a CI failure. No provider key
is needed to replay.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, messages_from_dict, messages_to_dict
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field


class CassetteMissError(KeyError):
    """The replayed run made a request that was never recorded."""


def _key(role: str, messages: list[BaseMessage], tools: list[str]) -> str:
    parts = [
        {
            "type": m.type,
            "content": m.content,
            "tool_calls": [(c["name"], c["args"]) for c in getattr(m, "tool_calls", [])],
        }
        for m in messages
    ]
    material = json.dumps(
        {"role": role, "messages": parts, "tools": tools}, sort_keys=True, default=str
    )
    return hashlib.sha256(material.encode()).hexdigest()[:24]


class Cassette:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, Any] = json.loads(path.read_text("utf-8")) if path.is_file() else {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=1, sort_keys=True), "utf-8")


class RecordReplayChatModel(BaseChatModel):
    role: str
    cassette: Any  # Cassette (kept Any for pydantic)
    inner: Any = None  # live model in record mode; None in replay mode
    bound_tools: list[str] = Field(default_factory=list)
    tool_specs: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "record-replay"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        key = _key(self.role, messages, self.bound_tools)
        if self.inner is not None:
            model = self.inner.bind_tools(self.tool_specs) if self.tool_specs else self.inner
            response = model.invoke(messages)
            self.cassette.entries[key] = messages_to_dict([response])[0]
        elif key not in self.cassette.entries:
            raise CassetteMissError(
                f"{self.role}: request {key} not in cassette {self.cassette.path}"
            )
        message = messages_from_dict([self.cassette.entries[key]])[0]
        assert isinstance(message, AIMessage)  # noqa: S101 - cassettes only store AI messages
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        names = [t["function"]["name"] for t in tools]
        return self.model_copy(update={"bound_tools": names, "tool_specs": list(tools)})

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Any:
        def respond(messages: Any) -> Any:
            msgs = messages if isinstance(messages, list) else messages.to_messages()
            key = _key(self.role + ":structured", msgs, [schema.__name__])
            if self.inner is not None:
                out = self.inner.with_structured_output(schema, include_raw=True).invoke(msgs)
                parsed = out["parsed"]
                self.cassette.entries[key] = {
                    "raw": messages_to_dict([out["raw"]])[0],
                    "parsed": parsed.model_dump() if parsed is not None else None,
                }
            elif key not in self.cassette.entries:
                raise CassetteMissError(f"{self.role}: structured request {key} not in cassette")
            entry = self.cassette.entries[key]
            raw = messages_from_dict([entry["raw"]])[0]
            parsed = schema.model_validate(entry["parsed"]) if entry["parsed"] is not None else None
            return {"raw": raw, "parsed": parsed, "parsing_error": None} if include_raw else parsed

        return RunnableLambda(respond)
