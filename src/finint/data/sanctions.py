"""OFAC SDN list loader (task 1.6 · R1.4).

Downloads ``SDN.CSV`` (primary names), ``ALT.CSV`` (aliases) and ``ADD.CSV`` (addresses)
from the OFAC Sanctions List Service. If a download fails, it uses the pinned gzip
snapshot in ``data/fixtures/sdn_snapshot/``. Provenance (URL, retrieval time, SHA-256,
live or snapshot) goes to ``data_sources``.

The files are headerless CSV, use ``-0-`` for null and end with a ``0x1A`` byte, so they
are parsed with :mod:`csv` rather than DuckDB's sniffer.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import re
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

import duckdb
from pydantic import BaseModel

OFAC_BASE_URL = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports"
SDN_FILES = ("SDN.CSV", "ALT.CSV", "ADD.CSV")
_NULL = "-0-"
_DOB = re.compile(r"\bDOB\s+([^;.]+)")
_NATIONALITY = re.compile(r"\bnationality\s+([^;.]+)", re.IGNORECASE)
_SDN_TYPES = {"individual": "individual", "vessel": "vessel", "aircraft": "aircraft"}

Opener = Callable[[str], BinaryIO]


class SanctionsReport(BaseModel):
    source: str  # "live" or "snapshot"
    fallback_reason: str | None = None
    n_entries: int
    n_individuals: int
    n_aliases: int
    n_addresses: int
    files: dict[str, str]  # file name -> sha256


@dataclass(frozen=True, slots=True)
class _Fetched:
    path: Path
    url: str
    source: str
    fallback_reason: str | None = None


def fetch_sdn(raw_dir: Path, snapshot_dir: Path, opener: Opener) -> list[_Fetched]:
    """Download all three files, or fall back to the snapshot for every file (never mixed)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        fetched = []
        for name in SDN_FILES:
            url = f"{OFAC_BASE_URL}/{name}"
            target = raw_dir / name
            partial = target.with_name(name + ".part")
            with opener(url) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
            partial.replace(target)
            fetched.append(_Fetched(target, url, "live"))
        return fetched
    except OSError as error:
        reason = f"{type(error).__name__}: {error}"
        return [
            _Fetched(snapshot_dir / f"{name}.gz", f"snapshot:{name}.gz", "snapshot", reason)
            for name in SDN_FILES
        ]


def _read_rows(path: Path) -> Iterator[list[str | None]]:
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    text = raw.decode("utf-8", errors="replace").replace("\x1a", "")
    for row in csv.reader(io.StringIO(text)):
        if not row or not row[0].strip().isdigit():
            continue
        yield [None if cell.strip() == _NULL else cell.strip() for cell in row]


def _int(cell: str | None) -> int:
    if cell is None:
        raise ValueError("expected a numeric id, got null")
    return int(cell)


def _extract(pattern: re.Pattern[str], remarks: str | None) -> str | None:
    if not remarks:
        return None
    match = pattern.search(remarks)
    return match.group(1).strip() if match else None


def load_sdn(db_path: Path, fetched: list[_Fetched]) -> SanctionsReport:
    by_name = {f.path.name.removesuffix(".gz"): f for f in fetched}
    entries = [
        (
            _int(r[0]), r[1], _SDN_TYPES.get((r[2] or "").lower(), "entity"), r[3], r[11],
            _extract(_DOB, r[11]), _extract(_NATIONALITY, r[11]),
        )
        for r in _read_rows(by_name["SDN.CSV"].path)
    ]  # fmt: skip
    aliases = [(_int(r[0]), _int(r[1]), r[2], r[3]) for r in _read_rows(by_name["ALT.CSV"].path)]
    addresses = [
        (_int(r[0]), _int(r[1]), r[2], r[3], r[4]) for r in _read_rows(by_name["ADD.CSV"].path)
    ]
    hashes = {name: hashlib.sha256(f.path.read_bytes()).hexdigest() for name, f in by_name.items()}
    retrieved = datetime.now(UTC)
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            "CREATE OR REPLACE TABLE sdn_entries (ent_num INTEGER PRIMARY KEY, name VARCHAR, "
            "sdn_type VARCHAR, programs VARCHAR, remarks VARCHAR, dob_text VARCHAR, "
            "nationality VARCHAR)"
        )
        con.executemany("INSERT INTO sdn_entries VALUES (?, ?, ?, ?, ?, ?, ?)", entries)
        con.execute(
            "CREATE OR REPLACE TABLE sdn_aliases "
            "(ent_num INTEGER, alt_num INTEGER, alt_type VARCHAR, alt_name VARCHAR)"
        )
        con.executemany("INSERT INTO sdn_aliases VALUES (?, ?, ?, ?)", aliases)
        con.execute(
            "CREATE OR REPLACE TABLE sdn_addresses (ent_num INTEGER, add_num INTEGER, "
            "address VARCHAR, city VARCHAR, country VARCHAR)"
        )
        con.executemany("INSERT INTO sdn_addresses VALUES (?, ?, ?, ?, ?)", addresses)
        con.execute(
            "CREATE TABLE IF NOT EXISTS data_sources (source VARCHAR, url VARCHAR, "
            "origin VARCHAR, retrieved_at TIMESTAMPTZ, sha256 VARCHAR)"
        )
        con.execute("DELETE FROM data_sources WHERE source LIKE 'ofac:%'")
        con.executemany(
            "INSERT INTO data_sources VALUES (?, ?, ?, ?, ?)",
            [(f"ofac:{n}", f.url, f.source, retrieved, hashes[n]) for n, f in by_name.items()],
        )
    return SanctionsReport(
        source=fetched[0].source,
        fallback_reason=fetched[0].fallback_reason,
        n_entries=len(entries),
        n_individuals=sum(1 for e in entries if e[2] == "individual"),
        n_aliases=len(aliases),
        n_addresses=len(addresses),
        files=hashes,
    )
