"""Generate a small AMLworld-shaped dataset for end-to-end tests of ``finint data build``.

Big enough for every injected case to find a candidate in the test window: laundering
subjects (persons and companies), "mule" receivers, and clean counterparties that receive
non-laundering payments from the subjects.
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

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
    rng = random.Random(seed)
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
