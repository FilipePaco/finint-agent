"""Optional tracing with Langfuse (R3.14 · ADR-03.6). The only module that knows Langfuse.

Degradation contract: no credentials, missing SDK or an unreachable collector means
``callbacks()`` returns ``[]`` and everything runs unchanged. The audit trail never depends
on this module; ``run_id`` is used as the Langfuse trace id to link the two.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from finint.config import Settings

logger = logging.getLogger(__name__)


def callbacks(settings: Settings, run_id: str, alert_id: str) -> list[Any]:
    if not settings.langfuse_enabled:
        return []
    try:
        from langfuse.langchain import CallbackHandler

        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key or "")
        if settings.langfuse_secret_key is not None:
            os.environ.setdefault(
                "LANGFUSE_SECRET_KEY", settings.langfuse_secret_key.get_secret_value()
            )
        if settings.langfuse_host:
            os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
        return [CallbackHandler(trace_context={"trace_id": run_id})]
    except Exception as error:
        logger.warning("Langfuse disabled for run %s (%s): %s", run_id, alert_id, error)
        return []


def run_config(settings: Settings, run_id: str, alert_id: str) -> dict[str, Any]:
    """LangChain config for every model call of a run (tags + optional tracing)."""
    return {
        "callbacks": callbacks(settings, run_id, alert_id),
        "metadata": {"run_id": run_id, "alert_id": alert_id},
        "tags": ["finint", f"alert:{alert_id}"],
    }
