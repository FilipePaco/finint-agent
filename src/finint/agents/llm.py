"""Chat models per agent role, and LLM-call accounting (tasks 3.12, 3.13 · R3.12-R3.14).

Every model comes from ``init_chat_model`` with provider and pinned model id from settings,
so changing provider is configuration. Every call is recorded as an ``LLM_CALL`` audit event
with role, model, prompt version, tokens (LangChain ``usage_metadata``), latency and an
estimated cost from ``genai-prices`` (no hand-maintained price table).
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from finint.config import PROJECT_ROOT, ModelRole, Settings
from finint.core.audit import AuditLog, NewEvent

Role = Literal["investigator", "reviewer", "media"]
PROMPTS_DIR = PROJECT_ROOT / "prompts"


def make_model(spec: ModelRole, settings: Settings) -> BaseChatModel:
    if spec.provider == "google_genai" and settings.google_api_key is not None:
        os.environ.setdefault("GOOGLE_API_KEY", settings.google_api_key.get_secret_value())
    # Retries with backoff cover rate limits (free tier) and transient provider errors.
    model: BaseChatModel = init_chat_model(
        spec.model, model_provider=spec.provider, temperature=spec.temperature, max_retries=6
    )
    return model


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str
    version: str  # first 12 hex chars of the file's SHA-256


@lru_cache(maxsize=16)
def load_prompt(name: str, directory: Path = PROMPTS_DIR) -> Prompt:
    text = (directory / f"{name}.md").read_text(encoding="utf-8")
    return Prompt(name, text, hashlib.sha256(text.encode()).hexdigest()[:12])


@lru_cache(maxsize=64)
def _unit_cost(model: str) -> tuple[Decimal, Decimal] | None:
    """USD per input and per output token, or None if the model is not priced."""
    try:
        from genai_prices import Usage, calc_price

        per_in = calc_price(
            Usage(input_tokens=1_000_000, output_tokens=0), model_ref=model, provider_id="google"
        ).total_price
        per_out = calc_price(
            Usage(input_tokens=0, output_tokens=1_000_000), model_ref=model, provider_id="google"
        ).total_price
    except Exception:
        return None
    return Decimal(per_in) / 1_000_000, Decimal(per_out) / 1_000_000


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    unit = _unit_cost(model)
    if unit is None:
        return None
    return float(unit[0] * input_tokens + unit[1] * output_tokens)


@dataclass
class UsageLedger:
    """Per-run totals, by model. Feeds RUN_COMPLETED and the evaluation."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cost_known: bool = True
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, model: str, tokens_in: int, tokens_out: int, cost: float | None) -> None:
        self.calls += 1
        self.input_tokens += tokens_in
        self.output_tokens += tokens_out
        if cost is None:
            self.cost_known = False
        else:
            self.cost_usd += cost
        row = self.by_model.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        row["calls"] += 1
        row["input_tokens"] += tokens_in
        row["output_tokens"] += tokens_out

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6) if self.cost_known else None,
            "by_model": self.by_model,
        }


def record_llm_call(
    audit: AuditLog,
    ledger: UsageLedger | None,
    *,
    run_id: str,
    alert_id: str,
    role: Role,
    model: str,
    prompt: Prompt,
    response: AIMessage | None,
    started: float,
    error: str | None = None,
) -> None:
    usage: dict[str, Any] = dict((response.usage_metadata if response is not None else None) or {})
    tokens_in = int(usage.get("input_tokens", 0) or 0)
    tokens_out = int(usage.get("output_tokens", 0) or 0)
    cost = estimate_cost(model, tokens_in, tokens_out)
    if ledger is not None:
        ledger.add(model, tokens_in, tokens_out, cost)
    audit.append(
        NewEvent(
            "LLM_CALL",
            f"agent:{role}",
            {
                "model": model,
                "prompt": prompt.name,
                "prompt_version": prompt.version,
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "cost_usd": cost,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "n_tool_calls": len(response.tool_calls) if response is not None else 0,
                "error": error,
            },
            run_id=run_id,
            alert_id=alert_id,
        )
    )


class UnavailableChatModel(BaseChatModel):
    """Stand-in when no provider key is configured (e.g. the hosted demo).

    Any call raises, so the graph degrades exactly as during an outage: the investigator
    returns a deterministic NEEDS_MORE_INFO evidence pack and the analyst still decides.
    """

    reason: str = "no LLM provider configured"

    @property
    def _llm_type(self) -> str:
        return "unavailable"

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> Any:
        raise RuntimeError(self.reason)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        from langchain_core.runnables import RunnableLambda

        def fail(_: Any) -> Any:
            raise RuntimeError(self.reason)

        return RunnableLambda(fail)
