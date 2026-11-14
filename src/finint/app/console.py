"""Streamlit analyst console (task 3.19 · R3.8, R3.11). Run with ``make app``.

Internal tool for FinCrime analysts only. Every data access goes through
:class:`finint.app.services.CaseService`, which requires an authenticated analyst.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if "finint" not in sys.modules:  # hosted demo: run from a checkout without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from finint.app.services import Analyst, CaseService, authenticate
from finint.config import get_settings, load_yaml

DISPOSITIONS = ["ESCALATE", "NEEDS_MORE_INFO", "CLOSE_FALSE_POSITIVE"]
STATUS_LABEL = {
    "NEW": "New",
    "IN_PROGRESS": "In progress",
    "AWAITING_ANALYST": "Awaiting decision",
    "CLOSED": "Closed",
}

st.set_page_config(page_title="finint · analyst console", layout="wide")


@st.cache_resource(show_spinner="Building the synthetic demo world (about 30 seconds)…")
def service() -> CaseService:
    settings = get_settings()
    # Demo when asked for, or when there is no local database (e.g. a fresh hosted checkout).
    if not settings.demo and settings.duckdb_path.is_file():
        return CaseService(settings)
    import tempfile

    from finint.data.synthetic import build_world
    from finint.eval.replay import Cassette
    from finint.eval.replay_run import CASSETTE, replay_runtime

    root = Path(tempfile.mkdtemp(prefix="finint-demo-"))
    world = build_world(root / "data", root / "reports")
    runtime = replay_runtime(world["settings"], Cassette(CASSETTE), None)
    return CaseService(world["settings"], runtime=runtime)


def sign_in() -> Analyst | None:
    settings = get_settings()
    st.sidebar.title("finint · analyst console")
    st.sidebar.caption("Internal FinCrime tool. Case files never reach customer-facing systems.")
    analyst: Analyst | None = st.session_state.get("analyst")
    if analyst is None:
        ids = [a["id"] for a in load_yaml("analysts", settings.config_dir)["analysts"]]
        choice = st.sidebar.selectbox("Analyst", ids)
        if st.sidebar.button("Sign in", type="primary"):
            st.session_state["analyst"] = authenticate(choice, settings)
            st.rerun()
        return None
    st.sidebar.success(f"Signed in: {analyst.name} ({analyst.role})")
    if st.sidebar.button("Sign out"):
        st.session_state.clear()
        st.rerun()
    st.sidebar.caption(f"Mode: {service().mode}")
    if st.sidebar.button("Verify audit chain"):
        try:
            st.sidebar.success(f"Chain intact: {service().verify_audit(analyst):,} events")
        except Exception as error:
            st.sidebar.error(f"Audit chain broken: {error}")
    return analyst


def queue_tab(analyst: Analyst) -> None:
    rows = service().queue(analyst)
    if not rows:
        st.info("No alerts. Run `make alerts` first (or start the demo with FININT_DEMO=1).")
        return
    frame = pd.DataFrame(rows)
    frame["status"] = frame["status"].map(STATUS_LABEL)
    counts = frame["status"].value_counts()
    cols = st.columns(4)
    for col, label in zip(cols, STATUS_LABEL.values(), strict=True):
        col.metric(label, int(counts.get(label, 0)))
    status_filter = st.multiselect("Status", list(STATUS_LABEL.values()), default=[])
    shown = frame[frame["status"].isin(status_filter)] if status_filter else frame
    st.dataframe(shown.drop(columns=["latest_run"]), hide_index=True, width="stretch")
    alert_id = st.selectbox("Alert", shown["alert_id"].tolist())
    row = frame.set_index("alert_id").loc[alert_id]
    left, right = st.columns(2)
    if left.button(
        "Investigate with agents",
        disabled=not service().llm_available,
        help="Runs the multi-agent investigation and stops at your decision.",
    ):
        with st.spinner("Agents are gathering evidence…"):
            st.session_state["run_id"] = service().investigate(analyst, alert_id)
        st.success("Case ready for review in the Case review tab.")
    if row["latest_run"] and right.button("Open latest run"):
        st.session_state["run_id"] = row["latest_run"]
        st.success("Opened in the Case review tab.")


def _evidence_block(evidence: dict[str, Any], ids: list[str]) -> None:
    for ev_id in ids:
        item = evidence.get(ev_id)
        if item is None:
            st.warning(f"{ev_id}: not found")
            continue
        with st.expander(f"{ev_id} · {item['tool']} {json.dumps(item['args'])}"):
            st.json(item["data"], expanded=False)


def case_tab(analyst: Analyst) -> None:
    run_id = st.session_state.get("run_id")
    if not run_id:
        st.info("Pick an alert in the queue and investigate it, or open its latest run.")
        return
    case = service().case(analyst, run_id)
    payload, evidence = case["payload"], case["evidence"]
    usage = case["usage"]
    st.subheader(f"Alert {case['alert_id']}")
    st.caption(f"run `{run_id}` · status {case['status']}")
    m = st.columns(5)
    m[0].metric("Evidence items", len(evidence))
    m[1].metric("Tool calls", usage["tool_calls"])
    m[2].metric("LLM calls", usage["llm_calls"])
    m[3].metric("Cost (USD)", f"{usage['cost_usd']:.4f}")
    m[4].metric("Gathering time (s)", usage["gathering_seconds"])

    draft = payload.get("case") or {}
    policy = payload.get("policy")
    grounding = payload.get("grounding") or {}
    if payload.get("status") == "BLOCKED_UNGROUNDED":
        st.error(
            "Blocked: the agent's case file failed the grounding check. No recommendation "
            "is shown; review the evidence and decide manually."
        )
    elif policy:
        overrides = policy.get("overrides", [])
        conf = draft.get("confidence")
        line = f"Agent recommends **{draft.get('disposition')}** ({conf} confidence)"
        if overrides:
            line += f" → policy guard: **{policy['disposition']}**"
        st.markdown(line)
        for o in overrides:
            st.warning(
                f"{o['rule_id']}: {o['reason']} ({o['from_disposition']} → {o['to_disposition']})"
            )
        for flag in policy.get("flags", []):
            st.error(f"Flag: {flag}")
    if case["decision"]:
        d = case["decision"]
        st.success(f"Decided by {d['analyst_id']}: {d['action']} → {d['final_disposition']}")

    if draft and payload.get("status") != "BLOCKED_UNGROUNDED":
        st.markdown("#### Summary")
        st.write(draft.get("summary", ""))
        st.markdown("#### Findings")
        for i, f in enumerate(draft.get("findings", []), 1):
            st.markdown(
                f"**{i}. {f['claim']}**  \n_typology: {f.get('typology') or '-'}_ · "
                f"cites {', '.join(f['evidence_ids'])}"
            )
            _evidence_block(evidence, f["evidence_ids"])
        if draft.get("hypotheses"):
            st.markdown("#### Hypotheses tested")
            st.dataframe(pd.DataFrame(draft["hypotheses"]), hide_index=True, width="stretch")
        cols = st.columns(2)
        cols[0].markdown("**Benign explanations considered**")
        cols[0].write(draft.get("benign_explanations_considered") or ["-"])
        cols[1].markdown("**Open questions**")
        cols[1].write(draft.get("open_questions") or ["-"])

    st.markdown("#### Sanctions screening (decided by code)")
    screens = [e for e in evidence.values() if e["tool"] == "screen_sanctions"]
    if not screens:
        st.caption("No screening was run.")
    for s in screens:
        data = s["data"]
        st.markdown(f"`{data['account_key']}` → **{data['status']}**")
        if data["candidates"]:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "SDN entry": c["sdn_name"],
                            "matched name": c["matched_name"],
                            "on": c["matched_on"],
                            "score": c["combined_score"],
                            "matched": ", ".join(c["matched_fields"]),
                            "mismatched": ", ".join(c["mismatched_fields"]),
                            "status": c["match_status"],
                        }
                        for c in data["candidates"]
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

    media = [e for e in evidence.values() if e["tool"] == "request_media_check"]
    if media:
        st.markdown("#### Adverse media (quotes verified verbatim)")
        for m_ in media:
            for f in m_["data"]["findings"]:
                st.markdown(
                    f"- **{f['category']}** "
                    f"({'same entity' if f['same_entity'] else 'other entity'}) "
                    f"[{f['source']}]({f['url']}), {f['published']}: “{f['quote']}”"
                )
            if m_["data"]["n_findings_dropped"]:
                st.caption(f"{m_['data']['n_findings_dropped']} unverifiable claim(s) dropped.")

    review = payload.get("review")
    if review:
        st.markdown(f"#### Reviewer: {review['verdict']}")
        for o in review.get("objections", []):
            st.write(f"- {o}")
    with st.expander(
        f"Grounding check: {'passed' if grounding.get('passed') else 'failed'} "
        f"({grounding.get('n_values_checked', 0)} values)"
    ):
        st.json(grounding, expanded=False)
    with st.expander("All evidence"):
        _evidence_block(evidence, sorted(evidence))

    if case["decision"] or not payload:
        return
    st.markdown("### Your decision")
    with st.form("decision"):
        action = st.radio("Action", ["APPROVE", "EDIT", "REJECT"], horizontal=True)
        default = (policy or {}).get("disposition") or "NEEDS_MORE_INFO"
        final = st.selectbox("Final disposition", DISPOSITIONS, index=DISPOSITIONS.index(default))
        reason = st.text_area("Reason (required for EDIT and REJECT)")
        summary = st.text_area("Edited summary (optional)")
        if st.form_submit_button("Record decision", type="primary"):
            try:
                service().decide(analyst, run_id, action, final, reason, summary or None)
            except ValueError as error:
                st.error(str(error))
            else:
                st.success("Decision recorded in the audit trail.")
                st.rerun()


def audit_tab(analyst: Analyst) -> None:
    run_id = st.session_state.get("run_id")
    if not run_id:
        st.info("Open a case to see its audit trail.")
        return
    events = service().case(analyst, run_id)["events"]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "seq": e.seq,
                    "time (UTC)": e.occurred_at[:19],
                    "event": e.event_type,
                    "actor": e.actor,
                    "details": json.dumps(e.payload, default=str)[:160],
                    "hash": e.hash[:12],
                }
                for e in events
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def main() -> None:
    analyst = sign_in()
    if analyst is None:
        st.title("finint · analyst console")
        st.write("Sign in as an analyst to review AML alerts. The agents recommend; you decide.")
        return
    queue, case, audit = st.tabs(["Alert queue", "Case review", "Audit trail"])
    with queue:
        queue_tab(analyst)
    with case:
        case_tab(analyst)
    with audit:
        audit_tab(analyst)


main()
