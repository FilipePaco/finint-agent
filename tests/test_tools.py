"""Investigation tools, sanctions tool, media providers and skills (tasks 3.8-3.11, 2.11)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pytest

from finint.config import Settings, load_yaml
from finint.data.build import build
from finint.tools import investigation as tools
from finint.tools.media import FixtureProvider, SearchNotPermittedError, check_permitted
from finint.tools.sanctions import screen_party
from finint.tools.skills import SkillLibrary, UnknownSkillError
from synthetic_amlworld import write_dataset

ROOT = Path(__file__).parents[1]
LABELS = {"is_laundering", "typology", "attempt_id", "is_injected", "injection_case_id"}


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("tools")
    settings = Settings(data_dir=root / "data", reports_dir=root / "reports")
    write_dataset(settings.raw_dir)
    report = build(settings, offline=True, log=lambda _m: None)
    db = settings.duckdb_path
    san01 = report.injected_sanctions["SAN-01"]
    inj = report.injected_prompts["INJ-01"]
    with duckdb.connect(str(db)) as con:
        account = con.execute(
            "SELECT account_key FROM accounts WHERE entity_id = ?", [san01]
        ).fetchone()[0]
        inj_account, inj_day = con.execute(
            "SELECT from_key, ts::DATE FROM transactions WHERE txn_id = ?", [inj]
        ).fetchone()
        others = con.execute(
            "SELECT txn_id FROM transactions WHERE from_key = ? AND txn_id <> ? "
            "ORDER BY ts DESC LIMIT 4",
            [inj_account, inj],
        ).fetchall()
        txn_ids = [inj] + [r[0] for r in others]
        con.execute(
            "CREATE TABLE alerts AS SELECT 'AL-TEST-0001' AS alert_id, ? AS account_key, "
            "CAST(? AS DATE) AS alert_date, 'test' AS split, 0.91 AS score, 0.5 AS score_raw, "
            "0.45 AS threshold, 'm1-test' AS model_version, 1 AS priority_rank, "
            "CAST(? AS BIGINT) AS top_txn_id, CAST(? AS BIGINT[]) AS contributing_txn_ids, "
            "5 AS n_txn, "
            "'{}' AS features_json, ? AS shap_top5_json",
            [
                inj_account,
                inj_day,
                txn_ids[0],
                txn_ids,
                json.dumps([{"feature": "out_sum_1h", "value": 1.0, "contribution": 0.4}]),
            ],
        )
    ctx = tools.ToolContext(db, load_yaml("data_classification", ROOT / "config")["tables"])
    day = inj_day.isoformat()
    week_before = (inj_day - timedelta(days=7)).isoformat()
    return {
        "ctx": ctx,
        "db": db,
        "account": account,
        "inj_account": inj_account,
        "day": day,
        "start": week_before,
        "report": report,
    }


def _keys(node: Any) -> set[str]:
    if isinstance(node, dict):
        return set(node) | set().union(*(_keys(v) for v in node.values())) if node else set()
    if isinstance(node, list):
        return set().union(*(_keys(v) for v in node)) if node else set()
    return set()


def test_get_alert_explains_score_and_flags_injection(env: dict[str, Any]) -> None:
    ev = tools.get_alert(env["ctx"], "AL-TEST-0001")
    assert ev.evidence_id.startswith("ev-")
    assert ev.data["score_drivers_shap_top5"][0]["feature"] == "out_sum_1h"
    assert ev.injection_hits, "the INJ-01 memo must be flagged"
    memos = [t["reference"] for t in ev.data["top_transactions"]]
    assert all(m.startswith("<untrusted_data>") for m in memos)
    assert not _keys(ev.data) & LABELS
    with pytest.raises(tools.ToolError, match="unknown alert"):
        tools.get_alert(env["ctx"], "AL-NOPE")


def test_get_transactions_totals_limits_and_window(env: dict[str, Any]) -> None:
    ev = tools.get_transactions(
        env["ctx"], env["inj_account"], env["start"], env["day"], "out", limit=500
    )
    assert ev.data["shown"] <= tools.MAX_ROWS
    assert ev.data["totals"]["n_transactions"] >= ev.data["shown"]
    assert not _keys(ev.data) & LABELS
    assert ev.values.amounts, "amounts must be extracted for grounding"
    far = (date.fromisoformat(env["day"]) + timedelta(days=30)).isoformat()
    with pytest.raises(tools.ToolError, match="limited"):
        tools.get_transactions(env["ctx"], env["inj_account"], env["start"], far)
    with pytest.raises(tools.ToolError, match="direction"):
        tools.get_transactions(env["ctx"], env["inj_account"], env["start"], env["day"], "sideways")  # type: ignore[arg-type]


def test_detect_patterns_sees_fan_out(env: dict[str, Any]) -> None:
    ev = tools.detect_patterns(env["ctx"], env["inj_account"], env["start"], env["day"])
    assert ev.data["fan"]["distinct_receivers"] >= 5
    assert set(ev.data["cycles_through_account"]) == {"cycles_len2", "cycles_len3", "cycles_len4"}


def test_trace_flows_and_counterparties(env: dict[str, Any]) -> None:
    flow = tools.trace_flows(
        env["ctx"], env["inj_account"], env["start"], env["day"], "out", hops=5
    )
    assert flow.data["hops"] == tools.MAX_HOPS
    assert all(edge["hop"] >= 1 for edge in flow.data["edges"])
    cps = tools.get_counterparties(env["ctx"], env["inj_account"], env["start"], env["day"])
    assert cps.data["counterparties"]


def test_account_profile_masks_dob_and_wraps_name(env: dict[str, Any]) -> None:
    ev = tools.get_account_profile(env["ctx"], env["account"])
    owner = ev.data["owner"]
    assert owner["name"].startswith("<untrusted_data>")
    assert owner["date_of_birth"] == 1955  # SAN-01: masked to the year
    assert not _keys(ev.data) & LABELS


def test_screen_party_finds_injected_true_match(env: dict[str, Any]) -> None:
    ev = screen_party(env["ctx"], env["account"], load_yaml("screening", ROOT / "config"))
    assert ev.data["status"] == "MATCH"
    assert ev.data["screened_birth_year"] == 1955
    assert ev.data["candidates"][0]["ent_num"] == 30879


def test_evidence_ids_are_sequential(env: dict[str, Any]) -> None:
    ids = [e.evidence_id for e in env["ctx"].store]
    assert ids == [f"ev-{i:03d}" for i in range(1, len(ids) + 1)]


def test_media_gate_and_fixtures_hide_ground_truth() -> None:
    with pytest.raises(SearchNotPermittedError):
        check_permitted(False, "Cynthia Webb")
    check_permitted(True, "Velmora Freight Ltd")
    provider = FixtureProvider(
        ROOT / "data" / "fixtures" / "media" / "documents.yaml", date(2026, 11, 13)
    )
    hits = provider.search("Velmora Freight Ltd")
    assert {h.url for h in hits} >= {"https://news.example/ledger/2026/09/velmora-shell-network"}
    doc = provider.fetch(hits[0].url)
    assert not {"relevance", "category", "contains_injection"} & set(vars(doc))


def test_skills_index_and_lookup() -> None:
    library = SkillLibrary(ROOT / "skills")
    assert len(library.names()) == 9
    assert "fan-out:" in library.index()
    assert "Test it with" in library.load("cycle").body
    for bad in ("nope", "../config", "cycle/../../etc"):
        with pytest.raises(UnknownSkillError):
            library.load(bad)
