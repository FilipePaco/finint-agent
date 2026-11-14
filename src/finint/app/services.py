"""Analyst-console services (task 3.19 · R3.8, R3.11).

Every function that reads or changes a case requires an authenticated :class:`Analyst`.
The console (``app/console.py``) is the only consumer; the project has no customer-facing
surface at all, and a test asserts that case data cannot be reached without the role.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal

import duckdb

from finint.agents.graph import build_graph, resume, sqlite_checkpointer, start
from finint.agents.runtime import build_runtime, llm_available
from finint.config import Settings, load_yaml
from finint.core.audit import AuditLog, StoredEvent
from finint.core.models import AnalystDecision

Role = Literal["analyst", "supervisor"]
CaseStatus = Literal["NEW", "IN_PROGRESS", "AWAITING_ANALYST", "CLOSED"]


@dataclass(frozen=True)
class Analyst:
    analyst_id: str
    name: str
    role: Role


def authenticate(analyst_id: str, settings: Settings) -> Analyst:
    for entry in load_yaml("analysts", settings.config_dir)["analysts"]:
        if entry["id"] == analyst_id and entry["role"] in ("analyst", "supervisor"):
            return Analyst(entry["id"], entry["name"], entry["role"])
    raise PermissionError(f"{analyst_id!r} is not an authorised FinCrime analyst")


def _require(analyst: Analyst | None) -> Analyst:
    if not isinstance(analyst, Analyst) or analyst.role not in ("analyst", "supervisor"):
        raise PermissionError("case data requires an authenticated analyst")
    return analyst


class CaseService:
    def __init__(self, settings: Settings, runtime: Any | None = None) -> None:
        """``runtime`` overrides the live one (the hosted demo replays recorded responses)."""
        self.settings = settings
        self.audit = AuditLog(settings.audit_db_path)
        self._runtime = runtime

    @property
    def llm_available(self) -> bool:
        return self._runtime is not None or llm_available(self.settings)

    @property
    def mode(self) -> str:
        if self._runtime is not None:
            return "demo: synthetic data, replayed Gemini responses"
        return "live LLM (Gemini)" if llm_available(self.settings) else "offline (no LLM key)"

    @cached_property
    def graph(self) -> Any:
        runtime = self._runtime or build_runtime(self.settings)
        return build_graph(
            runtime, sqlite_checkpointer(self.settings.data_dir / "checkpoints.sqlite")
        )

    # --- queue ------------------------------------------------------------------------------

    def runs_by_alert(self) -> dict[str, list[StoredEvent]]:
        runs: dict[str, list[StoredEvent]] = {}
        for event in self.audit.events():
            if event.run_id and event.alert_id and not event.run_id.startswith("alerts-"):
                runs.setdefault(event.alert_id, []).append(event)
        return runs

    @staticmethod
    def status_of(events: list[StoredEvent]) -> CaseStatus:
        types = {e.event_type for e in events}
        if "ANALYST_DECISION" in types:
            return "CLOSED"
        if "CASE_DRAFTED" in types or any(
            e.event_type == "GROUNDING_RESULT"
            and not e.payload.get("passed")
            and e.payload.get("attempt", 0) >= 3
            for e in events
        ):
            return "AWAITING_ANALYST"
        return "IN_PROGRESS" if types else "NEW"

    def queue(
        self, analyst: Analyst | None, split: str = "test", limit: int = 300
    ) -> list[dict[str, Any]]:
        _require(analyst)
        with duckdb.connect(str(self.settings.duckdb_path), read_only=True) as con:
            cursor = con.execute(
                "SELECT alert_id, alert_date, account_key, round(score, 4) AS score, priority_rank "
                "FROM alerts WHERE split = $split ORDER BY alert_date, priority_rank LIMIT $limit",
                {"split": split, "limit": limit},
            )
            names = [d[0] for d in cursor.description or []]
            rows = [dict(zip(names, r, strict=True)) for r in cursor.fetchall()]
        runs = self.runs_by_alert()
        for row in rows:
            events = runs.get(row["alert_id"], [])
            row["status"] = self.status_of(events)
            row["latest_run"] = events[-1].run_id if events else None
        return rows

    # --- cases ------------------------------------------------------------------------------

    def investigate(self, analyst: Analyst | None, alert_id: str) -> str:
        _require(analyst)
        run_id, _ = start(self.graph, alert_id)
        return run_id

    def case(self, analyst: Analyst | None, run_id: str) -> dict[str, Any]:
        """Everything the analyst needs: interrupt payload, evidence, audit, cost."""
        _require(analyst)
        snapshot = self.graph.get_state({"configurable": {"thread_id": run_id}})
        values = snapshot.values or {}
        payload: dict[str, Any] = {}
        for task in snapshot.tasks or ():
            for item in getattr(task, "interrupts", ()) or ():
                payload = dict(item.value)
        events = self.audit.events(run_id=run_id)
        llm = [e for e in events if e.event_type == "LLM_CALL"]
        costs = [e.payload.get("cost_usd") for e in llm]
        return {
            "run_id": run_id,
            "alert_id": values.get("alert_id"),
            "status": "CLOSED" if values.get("decision") else values.get("status"),
            "payload": payload,
            "evidence": {
                k: v.model_dump(mode="json") for k, v in (values.get("evidence") or {}).items()
            },
            "decision": values["decision"].model_dump() if values.get("decision") else None,
            "events": events,
            "usage": {
                "llm_calls": len(llm),
                "input_tokens": sum(int(e.payload.get("input_tokens", 0)) for e in llm),
                "output_tokens": sum(int(e.payload.get("output_tokens", 0)) for e in llm),
                "cost_usd": round(sum(c for c in costs if c is not None), 6),
                "tool_calls": values.get("tool_calls_used", 0),
                "gathering_seconds": round(values.get("gathering_seconds", 0.0), 1),
            },
        }

    def decide(
        self,
        analyst: Analyst | None,
        run_id: str,
        action: str,
        final_disposition: str,
        reason: str = "",
        edited_summary: str | None = None,
    ) -> AnalystDecision:
        who = _require(analyst)
        decision = AnalystDecision.model_validate(
            {
                "action": action,
                "analyst_id": who.analyst_id,
                "reason": reason,
                "final_disposition": final_disposition,
                "edited_summary": edited_summary or None,
            }
        )
        resume(self.graph, run_id, decision)
        return decision

    def verify_audit(self, analyst: Analyst | None) -> int:
        _require(analyst)
        return self.audit.verify()
