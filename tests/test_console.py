"""Analyst console services and UI smoke test (tasks 3.19-3.20 · R3.8, R3.11)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest

from finint.app.services import Analyst, CaseService, authenticate
from finint.config import Settings
from finint.data.build import build
from synthetic_amlworld import write_dataset

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def svc(tmp_path_factory: pytest.TempPathFactory) -> CaseService:
    root = tmp_path_factory.mktemp("console")
    settings = Settings(data_dir=root / "data", reports_dir=root / "reports")
    write_dataset(settings.raw_dir)
    report = build(settings, offline=True, log=lambda _m: None)
    txn = report.injected_prompts["INJ-02"]
    with duckdb.connect(str(settings.duckdb_path)) as con:
        account, day = con.execute(
            "SELECT from_key, ts::DATE FROM transactions WHERE txn_id = ?", [txn]
        ).fetchone()
        con.execute(
            "CREATE TABLE alerts AS SELECT 'AL-C-0001' AS alert_id, ? AS account_key, "
            "CAST(? AS DATE) AS alert_date, 'test' AS split, 0.8 AS score, 0.5 AS score_raw, "
            "0.45 AS threshold, 'm1-test' AS model_version, 1 AS priority_rank, "
            "CAST(? AS BIGINT) AS top_txn_id, CAST(? AS BIGINT[]) AS contributing_txn_ids, "
            "1 AS n_txn, '{}' AS features_json, ? AS shap_top5_json",
            [account, day, txn, [txn], json.dumps([])],
        )
    return CaseService(settings)


ANALYST = Analyst("analyst-1", "Analyst One", "analyst")


def test_authentication_is_an_allowlist() -> None:
    settings = Settings()
    assert authenticate("analyst-1", settings).role == "analyst"
    with pytest.raises(PermissionError):
        authenticate("customer-42", settings)


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("queue", ()),
        ("investigate", ("AL-C-0001",)),
        ("case", ("x",)),
        ("decide", ("x", "APPROVE", "ESCALATE")),
        ("verify_audit", ()),
    ],
)
def test_customer_surface_has_no_case_access(
    svc: CaseService, method: str, args: tuple[Any, ...]
) -> None:
    with pytest.raises(PermissionError):
        getattr(svc, method)(None, *args)


def test_offline_demo_flow_degrades_then_analyst_decides(svc: CaseService) -> None:
    assert svc.llm_available is False  # tests never see a provider key
    assert svc.queue(ANALYST)[0]["status"] == "NEW"
    run_id = svc.investigate(ANALYST, "AL-C-0001")
    case = svc.case(ANALYST, run_id)
    assert case["payload"]["status"] == "FALLBACK"
    assert case["payload"]["case"]["disposition"] == "NEEDS_MORE_INFO"
    assert "ev-001" in case["evidence"]
    assert svc.queue(ANALYST)[0]["status"] == "AWAITING_ANALYST"
    with pytest.raises(ValueError, match="requires a reason"):
        svc.decide(ANALYST, run_id, "REJECT", "ESCALATE")
    svc.decide(ANALYST, run_id, "EDIT", "ESCALATE", reason="memo tried to instruct the agent")
    assert svc.queue(ANALYST)[0]["status"] == "CLOSED"
    closed = svc.case(ANALYST, run_id)
    assert closed["decision"]["final_disposition"] == "ESCALATE"
    assert svc.verify_audit(ANALYST) == len(svc.audit.events())


def test_console_renders_sign_in_without_touching_case_data() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "src" / "finint" / "app" / "console.py"), default_timeout=60)
    app.run()
    assert not app.exception
    assert any("Sign in" in b.label for b in app.sidebar.button)
