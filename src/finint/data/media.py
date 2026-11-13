"""Fictional adverse-media fixture documents (task 1.8 · R1.6).

The fixture provider in spec 03 serves these documents offline. Ground-truth fields
(``relevance``, ``category``, ``contains_injection``) are for evaluation only and are
never shown to an agent.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, HttpUrl, field_validator

Relevance = Literal["SAME_ENTITY", "SAME_NAME_DIFFERENT_ENTITY", "IRRELEVANT"]
Category = Literal["ALLEGATION", "CHARGE", "CONVICTION", "SANCTION", "NONE"]


class MediaDocument(BaseModel):
    id: str
    case_id: str
    url: HttpUrl
    source: str
    published: date
    title: str
    body: str
    relevance: Relevance
    category: Category
    contains_injection: bool

    @field_validator("url")
    @classmethod
    def _reserved_domain_only(cls, url: HttpUrl) -> HttpUrl:
        if url.host != "news.example":
            raise ValueError("fixture documents must use the reserved news.example domain")
        return url


def load_media_documents(path: Path) -> list[MediaDocument]:
    with path.open(encoding="utf-8") as handle:
        docs = [MediaDocument.model_validate(d) for d in yaml.safe_load(handle)["documents"]]
    if len({d.id for d in docs}) != len(docs):
        raise ValueError("duplicate media document ids")
    return docs
