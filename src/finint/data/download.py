"""Download AMLworld raw files from Kaggle's public endpoint (no account needed)."""

from __future__ import annotations

import shutil
import ssl
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import truststore

from finint.data.sources import KAGGLE_DATASET, AmlWorldFiles

KAGGLE_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/" + KAGGLE_DATASET

Opener = Callable[[str], BinaryIO]


def file_url(name: str) -> str:
    return f"{KAGGLE_DOWNLOAD_URL}/{name}"


def tls_context() -> ssl.SSLContext:
    """TLS verified against the OS trust store (works with python.org builds and TLS
    inspection proxies). Shared by every outbound HTTPS call in the project."""
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def https_open(url: str) -> BinaryIO:
    """Open an HTTPS URL verifying certificates against the OS trust store (``truststore``).

    Python builds that ship without system CA certificates, and networks with TLS
    inspection, otherwise fail verification even though the system trusts the chain.
    """
    if not url.startswith("https://"):
        raise ValueError(f"Refusing non-HTTPS download: {url}")
    context = tls_context()
    request = urllib.request.Request(url, headers={"User-Agent": "finint-agent"})  # noqa: S310
    response: BinaryIO = urllib.request.urlopen(request, timeout=60, context=context)  # noqa: S310
    return response


def download(
    files: AmlWorldFiles, *, force: bool = False, opener: Opener = https_open
) -> list[Path]:
    """Fetch missing files (or all with ``force``). Writes to ``*.part`` then renames atomically."""
    targets = [files.transactions, files.accounts, files.patterns]
    fetched: list[Path] = []
    for target in targets:
        if target.is_file() and not force:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        with opener(file_url(target.name)) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(target)
        fetched.append(target)
    return fetched
