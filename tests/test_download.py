"""Raw-data download (task 1.4 · R1.1)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import pytest

from finint.data.download import _https_open, download, file_url
from finint.data.sources import AmlWorldFiles


class _FakeOpener:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def __call__(self, url: str) -> BinaryIO:
        self.urls.append(url)
        return io.BytesIO(f"content of {url.rsplit('/', 1)[-1]}".encode())


def test_downloads_only_missing_files(tmp_path: Path) -> None:
    files = AmlWorldFiles.in_dir(tmp_path)
    files.accounts.write_text("already here", encoding="utf-8")
    opener = _FakeOpener()
    fetched = download(files, opener=opener)
    assert fetched == [files.transactions, files.patterns]
    assert opener.urls == [file_url("HI-Small_Trans.csv"), file_url("HI-Small_Patterns.txt")]
    assert files.accounts.read_text(encoding="utf-8") == "already here"
    assert files.patterns.read_text(encoding="utf-8") == "content of HI-Small_Patterns.txt"
    assert not list(tmp_path.glob("*.part"))


def test_force_redownloads_everything(tmp_path: Path) -> None:
    files = AmlWorldFiles.in_dir(tmp_path)
    for path in (files.transactions, files.accounts, files.patterns):
        path.write_text("old", encoding="utf-8")
    assert len(download(files, force=True, opener=_FakeOpener())) == 3


def test_url_points_to_public_kaggle_endpoint() -> None:
    assert file_url("HI-Small_Trans.csv") == (
        "https://www.kaggle.com/api/v1/datasets/download/"
        "ealtman2019/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans.csv"
    )


def test_refuses_non_https() -> None:
    with pytest.raises(ValueError, match="non-HTTPS"):
        _https_open("http://example.com/file.csv")
