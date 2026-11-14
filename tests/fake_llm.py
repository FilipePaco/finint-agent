"""A scripted chat model for testing agents and the graph without an LLM provider."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

Step = AIMessage | Callable[[list[BaseMessage]], AIMessage] | Exception


def call(name: str, args: dict[str, Any], call_id: str | None = None) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id or f"call-{name}"}


def ai(*tool_calls: dict[str, Any], content: str = "") -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=list(tool_calls),
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    )


class ScriptedChatModel(BaseChatModel):
    """Returns scripted steps in order; a step may inspect the conversation or raise."""

    script: list[Any]
    bound_tool_names: list[list[str]] = Field(default_factory=list)
    verdicts: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.script:
            raise RuntimeError("script exhausted")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        message = step(list(messages)) if callable(step) else step
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        self.bound_tool_names.append([t["function"]["name"] for t in tools])
        return self

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Any:
        def respond(_messages: Any) -> Any:
            verdict = self.verdicts.pop(0) if self.verdicts else {"verdict": "ACCEPT"}
            parsed = schema.model_validate(verdict)
            raw = ai(content="review")
            return {"raw": raw, "parsed": parsed, "parsing_error": None} if include_raw else parsed

        return RunnableLambda(respond)
