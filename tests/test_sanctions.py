"""OFAC SDN loader (task 1.6 · R1.4)."""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import BinaryIO

import duckdb
import pytest

from finint.data.sanctions import SDN_FILES, _read_rows, fetch_sdn, load_sdn

SNAPSHOT = Path(__file__).parents[1] / "data" / "fixtures" / "sdn_snapshot"

SDN = (
    '36,"AEROCARIBBEAN AIRLINES",-0- ,"CUBA",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- \r\n'
    '9001,"DOE, John Quincy","individual","SDGT",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,'
    '"DOB 01 Jan 1960; POB Somewhere; nationality Freedonia; Gender Male."\r\n'
    "\x1a"
)
ALT = '9001,1,"aka","DOE, Johnny"\r\n\x1a'
ADD = '9001,7,-0- ,"Capital","Freedonia",-0- \r\n\x1a'


@pytest.fixture
def snapshot_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "snapshot"
    directory.mkdir()
    for name, content in (("SDN.CSV", SDN), ("ALT.CSV", ALT), ("ADD.CSV", ADD)):
        (directory / f"{name}.gz").write_bytes(gzip.compress(content.encode()))
    return directory


def _failing_opener(url: str) -> BinaryIO:
    raise OSError("network down")


def test_rows_skip_eof_marker_and_map_nulls(snapshot_dir: Path) -> None:
    rows = list(_read_rows(snapshot_dir / "SDN.CSV.gz"))
    assert len(rows) == 2
    assert rows[0][2] is None  # "-0- " -> None


def test_falls_back_to_snapshot_when_download_fails(tmp_path: Path, snapshot_dir: Path) -> None:
    fetched = fetch_sdn(tmp_path / "raw", snapshot_dir, _failing_opener)
    assert {f.source for f in fetched} == {"snapshot"}
    assert fetched[0].fallback_reason == "OSError: network down"


def test_live_download_writes_raw_files(tmp_path: Path, snapshot_dir: Path) -> None:
    payloads = {"SDN.CSV": SDN, "ALT.CSV": ALT, "ADD.CSV": ADD}

    def opener(url: str) -> BinaryIO:
        return io.BytesIO(payloads[url.rsplit("/", 1)[-1]].encode())

    fetched = fetch_sdn(tmp_path / "raw", snapshot_dir, opener)
    assert {f.source for f in fetched} == {"live"}
    assert all((tmp_path / "raw" / name).is_file() for name in SDN_FILES)


def test_load_extracts_type_dob_nationality_and_provenance(
    tmp_path: Path, snapshot_dir: Path
) -> None:
    db = tmp_path / "finint.duckdb"
    report = load_sdn(db, fetch_sdn(tmp_path / "raw", snapshot_dir, _failing_opener))
    assert (report.n_entries, report.n_individuals, report.n_aliases) == (2, 1, 1)
    with duckdb.connect(str(db), read_only=True) as con:
        person = con.execute(
            "SELECT sdn_type, dob_text, nationality FROM sdn_entries WHERE ent_num = 9001"
        ).fetchone()
        airline = con.execute("SELECT sdn_type FROM sdn_entries WHERE ent_num = 36").fetchone()
        sources = con.execute("SELECT origin, sha256 FROM data_sources ORDER BY 1").fetchall()
    assert person == ("individual", "01 Jan 1960", "Freedonia")
    assert airline == ("entity",)
    assert len(sources) == 3
    assert all(origin == "snapshot" and len(sha) == 64 for origin, sha in sources)


def test_pinned_snapshot_is_complete() -> None:
    for name in SDN_FILES:
        assert (SNAPSHOT / f"{name}.gz").is_file()
    assert sum(1 for _ in _read_rows(SNAPSHOT / "SDN.CSV.gz")) > 10_000
