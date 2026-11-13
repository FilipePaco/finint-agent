"""Hash-chained audit trail (task 3.2 · R2.8, R3.12)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from finint.core.audit import AuditIntegrityError, AuditLog, NewEvent


def _event(i: int, alert: str = "AL-1") -> NewEvent:
    return NewEvent("ALERT_CREATED", "ml:m1-test", {"i": i, "score": 0.9}, alert_id=alert)


def test_append_links_hashes_and_verifies(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.sqlite")
    first = log.append(_event(1))
    second, third = log.append_many([_event(2), _event(3, "AL-2")])
    assert first.prev_hash is None
    assert second.prev_hash == first.hash
    assert third.prev_hash == second.hash
    assert [e.seq for e in log.events()] == [1, 2, 3]
    assert log.verify() == 3
    assert [e.payload["i"] for e in log.events(alert_id="AL-1")] == [1, 2]


def test_reopening_continues_the_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.sqlite"
    AuditLog(path).append(_event(1))
    again = AuditLog(path).append(_event(2))
    assert again.seq == 2
    assert AuditLog(path).verify() == 2


@pytest.mark.parametrize(
    "tamper",
    [
        'UPDATE audit_events SET payload_json = \'{"i":99,"score":0.1}\' WHERE seq = 2',
        "DELETE FROM audit_events WHERE seq = 2",
        "UPDATE audit_events SET actor = 'analyst:someone' WHERE seq = 1",
    ],
)
def test_tampering_is_detected(tmp_path: Path, tamper: str) -> None:
    path = tmp_path / "audit.sqlite"
    log = AuditLog(path)
    log.append_many([_event(1), _event(2), _event(3)])
    with sqlite3.connect(path) as con:
        con.execute(tamper)
    with pytest.raises(AuditIntegrityError):
        log.verify()


def test_rejects_unknown_event_type_and_empty_actor(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.sqlite")
    with pytest.raises(ValueError, match="unknown event type"):
        log.append(NewEvent("CASE_CLOSED", "agent:x", {}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="actor"):
        log.append(NewEvent("ALERT_CREATED", "", {}))
    assert log.verify() == 0
