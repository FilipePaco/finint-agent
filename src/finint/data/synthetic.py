"""A small AMLworld-shaped synthetic dataset (tests, CI replay and the hosted demo).

Big enough for every injected case to find a candidate in the test window: laundering
subjects (persons and companies), "mule" receivers, and clean counterparties that receive
non-laundering payments from the subjects.
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering"
)
BANKS = [
    "Germany Bank #7",
    "UK Bank #3",
    "Bank of Springfield",
    "France Bank #12",
    "Crytpo Bank #1",
]
FORMATS = ["Cheque", "ACH", "Wire", "Credit Card", "Cash"]


def write_dataset(raw_dir: Path, seed: int = 0) -> None:
    rng = random.Random(seed)  # noqa: S311 - seeded synthetic data, not security
    raw_dir.mkdir(parents=True, exist_ok=True)
    persons = [f"8{i:08X}"[:9] for i in range(1, 201)]
    companies = [f"9{i:08X}"[:9] for i in range(1, 101)]
    accounts: dict[str, tuple[str, str, str]] = {}  # account -> (bank_id, bank_name, entity)
    with (raw_dir / "HI-Small_accounts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Bank Name", "Bank ID", "Account Number", "Entity ID", "Entity Name"])
        for idx, entity in enumerate(persons + companies):
            bank_idx = idx % len(BANKS)
            bank_id = f"{bank_idx + 10:03d}"
            if entity in persons:
                label = f"{'Individual' if idx % 5 == 0 else 'Sole Proprietorship'} #{idx}"
            else:
                label = f"{'Partnership' if idx % 2 else 'Corporation'} #{idx}"
            account = "A" + entity[1:]
            accounts[account] = (bank_id, BANKS[bank_idx], entity)
            writer.writerow([BANKS[bank_idx], bank_id, account, entity, label])

    acct_of = {entity: "A" + entity[1:] for entity in persons + companies}
    launder_persons, mules, clean_persons = persons[:40], persons[40:100], persons[100:]
    launder_companies, clean_companies = companies[:15], companies[60:]
    rows: list[tuple[datetime, str, str, float, str, int]] = []
    start = datetime(2022, 9, 1)
    everyone = persons + companies
    for i in range(2400):  # background, days 1-8
        a, b = rng.sample(everyone, 2)
        ts = start + timedelta(minutes=i * 4)
        rows.append(
            (ts, acct_of[a], acct_of[b], round(rng.uniform(5, 5000), 2), rng.choice(FORMATS), 0)
        )
    late = start + timedelta(days=9)
    attempts: list[tuple[str, list[tuple[datetime, str, str, float, str, int]]]] = []
    for j, subject in enumerate(launder_persons + launder_companies):
        legs = []
        for k in range(3):
            ts = late + timedelta(minutes=j * 10 + k)
            legs.append((ts, acct_of[subject], acct_of[mules[(j + k) % len(mules)]],
                         round(rng.uniform(8000, 9900), 2), "Wire", 1))  # fmt: skip
        rows.extend(legs)
        attempts.append(("FAN-OUT" if j % 2 else "STACK", legs))
        clean = clean_persons if j % 2 == 0 else clean_companies
        for k in range(4):
            ts = late + timedelta(minutes=j * 10 + 5 + k)
            target = clean[(j * 4 + k) % len(clean)]
            rows.append(
                (ts, acct_of[subject], acct_of[target], round(rng.uniform(20, 900), 2), "ACH", 0)
            )
    rows.sort(key=lambda r: r[0])

    def line(r: tuple[datetime, str, str, float, str, int]) -> str:
        ts, src, dst, amount, fmt, label = r
        return (
            f"{ts:%Y/%m/%d %H:%M},{accounts[src][0]},{src},{accounts[dst][0]},{dst},"
            f"{amount:.2f},US Dollar,{amount:.2f},US Dollar,{fmt},{label}"
        )

    (raw_dir / "HI-Small_Trans.csv").write_text(
        HEADER + "\n" + "\n".join(line(r) for r in rows) + "\n", encoding="utf-8"
    )
    blocks = []
    for typology, legs in attempts[:-5]:  # leave a few laundering rows unlisted, like the real data
        blocks.append(f"BEGIN LAUNDERING ATTEMPT - {typology}:  synthetic")
        blocks.extend(line(leg) for leg in legs)
        blocks.append(f"END LAUNDERING ATTEMPT - {typology}\n")
    (raw_dir / "HI-Small_Patterns.txt").write_text("\n".join(blocks), encoding="utf-8")


def build_world(data_dir: Path, reports_dir: Path) -> dict[str, Any]:
    """Synthetic dataset -> full data build (offline OFAC) -> two alerts the agent can investigate.

    Deterministic: the same inputs produce the same database, so recorded LLM responses can be
    replayed against it in CI.
    """
    import json

    import duckdb

    from finint.config import Settings
    from finint.data.build import build

    settings = Settings(data_dir=data_dir, reports_dir=reports_dir)
    write_dataset(settings.raw_dir)
    report = build(settings, offline=True, log=lambda _m: None)
    alerts = []
    with duckdb.connect(str(settings.duckdb_path)) as con:
        con.execute(
            "CREATE OR REPLACE TABLE alerts (alert_id VARCHAR PRIMARY KEY, account_key VARCHAR, "
            "alert_date DATE, split VARCHAR, score DOUBLE, score_raw DOUBLE, threshold DOUBLE, "
            "model_version VARCHAR, priority_rank INTEGER, top_txn_id BIGINT, "
            "contributing_txn_ids BIGINT[], n_txn INTEGER, features_json VARCHAR, "
            "shap_top5_json VARCHAR)"
        )
        con.execute(
            "CREATE OR REPLACE TABLE alert_labels (alert_id VARCHAR PRIMARY KEY, "
            "is_laundering BOOLEAN, n_laundering_txn INTEGER)"
        )
        for i, case in enumerate(("INJ-01", "INJ-05"), start=1):
            txn = report.injected_prompts[case]
            account, day = con.execute(
                "SELECT from_key, ts::DATE FROM transactions WHERE txn_id = ?", [txn]
            ).fetchone()  # type: ignore[misc]
            txns = [
                r[0]
                for r in con.execute(
                    "SELECT txn_id FROM transactions WHERE from_key = ? AND ts::DATE = ? "
                    "ORDER BY amount_usd DESC LIMIT 10",
                    [account, day],
                ).fetchall()
            ]
            alert_id = f"AL-SYN-{i:04d}"
            shap = [{"feature": "out_sum_1h", "value": 1.0, "contribution": 0.4}]
            con.execute(
                "INSERT INTO alerts VALUES (?, ?, ?, 'test', 0.9, 0.6, 0.45, 'synthetic', ?, ?, ?, "
                "?, '{}', ?)",
                [alert_id, account, day, i, txn, txns, len(txns), json.dumps(shap)],
            )
            con.execute("INSERT INTO alert_labels VALUES (?, true, 3)", [alert_id])
            alerts.append(alert_id)
    return {"settings": settings, "alerts": alerts, "report": report}
