"""Append-only, hash-chained audit trail (R2.8, R3.12 · constitution VI).

The compliance system of record: every alert, agent step and analyst decision is an event.
Each event stores the SHA-256 of its canonical content plus the previous event's hash, so
editing or deleting any stored event breaks :meth:`AuditLog.verify`. Lives in its own
SQLite file, apart from the analytical DuckDB store (ADR-01.1).

Ported from the pig_finserv project and generalized to typed agent events.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, get_args
from uuid import uuid4

EventType = Literal[
    "ALERT_CREATED",
    "RUN_STARTED",
    "LLM_CALL",
    "TOOL_CALL",
    "SKILL_LOADED",
    "MEDIA_CLAIM_DROPPED",
    "GROUNDING_RESULT",
    "REVIEW_VERDICT",
    "POLICY_OVERRIDE",
    "CASE_DRAFTED",
    "ANALYST_DECISION",
    "RUN_COMPLETED",
]
EVENT_TYPES: frozenset[str] = frozenset(get_args(EventType))


class AuditIntegrityError(RuntimeError):
    """Raised when the hash chain is broken (an event was edited, removed or reordered)."""


@dataclass(frozen=True, slots=True)
class NewEvent:
    event_type: EventType
    actor: str  # e.g. "ml:m1-abc1234-20261113", "agent:investigator", "analyst:a.silva"
    payload: Mapping[str, Any]
    run_id: str | None = None
    alert_id: str | None = None


@dataclass(frozen=True, slots=True)
class StoredEvent:
    event_id: str
    seq: int
    occurred_at: str
    event_type: str
    actor: str
    run_id: str | None
    alert_id: str | None
    payload: dict[str, Any]
    prev_hash: str | None
    hash: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    seq INTEGER PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE,
    occurred_at TEXT NOT NULL,
    event_type TEXT NOT NULL,
    actor TEXT NOT NULL,
    run_id TEXT,
    alert_id TEXT,
    payload_json TEXT NOT NULL,
    prev_hash TEXT,
    hash TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS audit_by_alert ON audit_events(alert_id);
CREATE INDEX IF NOT EXISTS audit_by_run ON audit_events(run_id);
"""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _hash(fields: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(fields).encode()).hexdigest()


class AuditLog:
    """SQLite event store with append-only semantics enforced by the application."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self._path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def append(self, event: NewEvent) -> StoredEvent:
        return self.append_many([event])[0]

    def append_many(self, events: Sequence[NewEvent]) -> list[StoredEvent]:
        """Append events atomically, in order, each linked to the previous hash."""
        for event in events:
            if event.event_type not in EVENT_TYPES:
                raise ValueError(f"unknown event type {event.event_type!r}")
            if not event.actor or len(event.actor) > 128:
                raise ValueError("actor must be a non-empty token of at most 128 chars")
        stored: list[StoredEvent] = []
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            last = con.execute("SELECT seq, hash FROM audit_events ORDER BY seq DESC LIMIT 1")
            row = last.fetchone()
            seq, prev = (row[0], row[1]) if row else (0, None)
            for event in events:
                seq += 1
                fields: dict[str, Any] = {
                    "event_id": f"evt_{uuid4().hex}",
                    "seq": seq,
                    "occurred_at": datetime.now(UTC).isoformat(),
                    "event_type": event.event_type,
                    "actor": event.actor,
                    "run_id": event.run_id,
                    "alert_id": event.alert_id,
                    "payload": json.loads(_canonical(dict(event.payload))),
                    "prev_hash": prev,
                }
                digest = _hash(fields)
                con.execute(
                    "INSERT INTO audit_events (seq, event_id, occurred_at, event_type, actor, "
                    "run_id, alert_id, payload_json, prev_hash, hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        seq,
                        fields["event_id"],
                        fields["occurred_at"],
                        event.event_type,
                        event.actor,
                        event.run_id,
                        event.alert_id,
                        _canonical(fields["payload"]),
                        prev,
                        digest,
                    ),
                )
                stored.append(StoredEvent(hash=digest, **fields))
                prev = digest
        return stored

    def events(
        self, *, alert_id: str | None = None, run_id: str | None = None
    ) -> list[StoredEvent]:
        sql = (
            "SELECT event_id, seq, occurred_at, event_type, actor, run_id, alert_id, "
            "payload_json, prev_hash, hash FROM audit_events"
        )
        clauses, params = [], []
        if alert_id is not None:
            clauses.append("alert_id = ?")
            params.append(alert_id)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self._connect() as con:
            rows = con.execute(sql + " ORDER BY seq", params).fetchall()
        return [
            StoredEvent(
                event_id=r[0],
                seq=r[1],
                occurred_at=r[2],
                event_type=r[3],
                actor=r[4],
                run_id=r[5],
                alert_id=r[6],
                payload=json.loads(r[7]),
                prev_hash=r[8],
                hash=r[9],
            )
            for r in rows
        ]

    def verify(self) -> int:
        """Recompute the whole chain. Returns the number of events verified."""
        prev: str | None = None
        events = self.events()
        for expected_seq, event in enumerate(events, start=1):
            if event.seq != expected_seq or event.prev_hash != prev:
                raise AuditIntegrityError(f"chain broken at seq {event.seq} ({event.event_id})")
            fields = {
                "event_id": event.event_id,
                "seq": event.seq,
                "occurred_at": event.occurred_at,
                "event_type": event.event_type,
                "actor": event.actor,
                "run_id": event.run_id,
                "alert_id": event.alert_id,
                "payload": event.payload,
                "prev_hash": event.prev_hash,
            }
            if _hash(fields) != event.hash:
                raise AuditIntegrityError(f"hash mismatch at seq {event.seq} ({event.event_id})")
            prev = event.hash
        return len(events)
