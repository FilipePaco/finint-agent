"""CI evaluation by replay (task 4.4 · R4.4 · ADR-04.2).

Rebuilds the deterministic synthetic world and replays recorded Gemini responses through the
full graph: tools, grounding, review, policy, analyst interrupt and audit. A changed prompt,
tool or evidence format produces an unseen request and fails loudly. No key, no cost.
"""

from __future__ import annotations

from typing import Any

import pytest

from finint.data.synthetic import build_world
from finint.eval.replay import Cassette
from finint.eval.replay_run import CASSETTE, replay_runtime, run_alerts


@pytest.fixture(scope="module")
def replayed(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, list[dict[str, Any]]]:
    if not CASSETTE.is_file():
        pytest.skip("no cassette recorded yet (run: finint eval record)")
    root = tmp_path_factory.mktemp("replay")
    world = build_world(root / "data", root / "reports")
    rt = replay_runtime(world["settings"], Cassette(CASSETTE), None)
    return rt, run_alerts(rt, world["alerts"])


def test_replayed_runs_complete_and_are_grounded(
    replayed: tuple[Any, list[dict[str, Any]]],
) -> None:
    _, runs = replayed
    assert len(runs) == 2
    for run in runs:
        payload = run["payload"]
        assert payload["status"] in ("GROUNDED", "BLOCKED_UNGROUNDED", "FALLBACK")
        if payload["status"] == "GROUNDED":
            assert payload["grounding"]["passed"]
            assert payload["policy"]["disposition"] in (
                "ESCALATE",
                "NEEDS_MORE_INFO",
                "CLOSE_FALSE_POSITIVE",
            )


def test_replayed_runs_are_fully_audited(replayed: tuple[Any, list[dict[str, Any]]]) -> None:
    rt, runs = replayed
    for run in runs:
        types = [e.event_type for e in rt.audit.events(run_id=run["run_id"])]
        for required in (
            "RUN_STARTED",
            "LLM_CALL",
            "TOOL_CALL",
            "ANALYST_DECISION",
            "RUN_COMPLETED",
        ):
            assert required in types, required
    assert rt.audit.verify() == len(rt.audit.events())


def test_injection_memo_never_closes_the_case(replayed: tuple[Any, list[dict[str, Any]]]) -> None:
    _, runs = replayed
    for run in runs:
        policy = run["payload"].get("policy") or {}
        assert policy.get("disposition") != "CLOSE_FALSE_POSITIVE"
