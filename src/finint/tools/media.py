"""Adverse-media providers (task 3.11 · R3.3).

* :class:`FixtureProvider` (default): the fictional documents in
  ``data/fixtures/media/documents.yaml``. Offline, deterministic, free.
* :class:`TavilyProvider` (opt-in): live web search, only behind the permission gate.

The permission gate refuses any search for an entity that is not an injected test entity:
web search never runs against a real private person (constitution IX). Ground-truth fields
of fixture documents (relevance, category, contains_injection) never leave this module.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from finint.data.download import tls_context
from finint.data.media import MediaDocument, load_media_documents

TAVILY_URL = "https://api.tavily.com/search"


@dataclass(frozen=True)
class MediaHit:
    url: str
    title: str
    source: str
    published: date | None
    snippet: str


@dataclass(frozen=True)
class MediaText:
    url: str
    title: str
    source: str
    published: date | None
    text: str
    retrieved: date


class MediaProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int = 5) -> list[MediaHit]: ...

    def fetch(self, url: str) -> MediaText: ...


class SearchNotPermittedError(PermissionError):
    pass


def check_permitted(is_injected: bool, entity_label: str) -> None:
    if not is_injected:
        raise SearchNotPermittedError(
            f"adverse-media search refused for {entity_label!r}: only injected synthetic test "
            "entities may be searched (no queries about real people)"
        )


class FixtureProvider:
    name = "fixtures"

    def __init__(self, path: Path, today: date) -> None:
        self._docs: dict[str, MediaDocument] = {str(d.url): d for d in load_media_documents(path)}
        self._today = today

    def search(self, query: str, max_results: int = 5) -> list[MediaHit]:
        tokens = [t for t in query.casefold().split() if len(t) > 2]
        scored = []
        for doc in self._docs.values():
            haystack = f"{doc.title} {doc.body}".casefold()
            hits = sum(t in haystack for t in tokens)
            if tokens and hits >= max(1, len(tokens) - 1):
                scored.append((hits, doc))
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [
            MediaHit(str(d.url), d.title, d.source, d.published, d.body[:200])
            for _, d in scored[:max_results]
        ]

    def fetch(self, url: str) -> MediaText:
        doc = self._docs.get(url)
        if doc is None:
            raise FileNotFoundError(f"no fixture document at {url}")
        return MediaText(str(doc.url), doc.title, doc.source, doc.published, doc.body, self._today)


class TavilyProvider:
    """Live search through Tavily's REST API. Only used when a key is configured."""

    name = "tavily"

    def __init__(self, api_key: str, today: date) -> None:
        self._key = api_key
        self._today = today
        self._cache: dict[str, MediaText] = {}

    def search(self, query: str, max_results: int = 5) -> list[MediaHit]:
        body = json.dumps(
            {
                "query": query,
                "max_results": max_results,
                "include_raw_content": True,
                "topic": "news",
            }
        ).encode()
        request = urllib.request.Request(
            TAVILY_URL,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"},
        )
        with urllib.request.urlopen(request, timeout=30, context=tls_context()) as response:  # noqa: S310
            payload = json.load(response)
        hits = []
        for item in payload.get("results", [])[:max_results]:
            text = item.get("raw_content") or item.get("content") or ""
            published = _parse_date(item.get("published_date"))
            hit = MediaHit(
                item["url"],
                item.get("title", ""),
                _domain(item["url"]),
                published,
                (item.get("content") or "")[:200],
            )
            self._cache[hit.url] = MediaText(
                hit.url, hit.title, hit.source, published, text, self._today
            )
            hits.append(hit)
        return hits

    def fetch(self, url: str) -> MediaText:
        if url not in self._cache:
            raise FileNotFoundError("only URLs returned by a search can be fetched")
        return self._cache[url]


def _domain(url: str) -> str:
    return url.split("/")[2] if "//" in url else url


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None
