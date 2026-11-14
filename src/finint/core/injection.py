"""Heuristic prompt-injection detector for untrusted text (task 3.7 · R3.10).

Detection is one layer, not the defence: tools are read-only, tool allowlists are enforced
by the graph, and the policy guard and grounding validator run after every LLM step
(ADR-03.3). A hit is registered as evidence and shown to the analyst, because an attempt
to steer the investigation is itself a red flag.
"""

from __future__ import annotations

import base64
import binascii
import re

from pydantic import BaseModel

RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (rule_id, re.compile(pattern, re.IGNORECASE))
    for rule_id, pattern in (
        (
            "override",
            r"\b(ignore|disregard|forget)\b.{0,40}\b(instructions?|rules?|above|prompt)\b",
        ),
        ("role_token", r"(^|\s|<)/?(system|assistant|developer)\s*[:>]|\byou are now\b"),
        ("delimiter", r"</?\s*(untrusted|untrusted_data|system|instructions?)\s*>"),
        ("tool_call", r"\b(call|invoke|run|execute)\b.{0,30}\b[a-z]+_[a-z_]+\s*\("),
        (
            "action_verbs",
            r"\b(close|block|unblock|whitelist|approve)\s+(this\s+)?(alert|account|case)\b",
        ),
        (
            "disposition",
            r"\b(close_false_positive|mark (it )?as (a )?false positive"
            r"|escalation is not allowed)\b",
        ),
        (
            "authority",
            r"\b(compliance note|approved by|no further review|system override"
            r"|cleared by (regulators|ofac))\b",
        ),
        (
            "ai_address",
            r"\b(note (to|for) (any )?(ai|the investigator|assistant)|ai system"
            r"|virtual analyst|analista virtual)\b",
        ),
        ("exfiltration", r"\b(system prompt|other customers|email (it|this) to)\b"),
        ("html_comment", r"<!--.*?-->"),
        (
            "fake_tool_output",
            r"\b(match_status|screen_sanctions|search_media)\b\s*(result)?\s*[:=]",
        ),
        ("multilingual", r"\b(encerre|ignore as|instru[cç][aã]o ao|ignora las|cierra la alerta)\b"),
        ("suppression", r"\bdo not (mention|report|include)\b"),
    )
)
_B64 = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}")


class InjectionScan(BaseModel):
    is_attempt: bool
    rules: list[str]
    decoded: str | None = None


def _match(text: str) -> list[str]:
    return [rule_id for rule_id, pattern in RULES if pattern.search(text)]


def scan(text: str) -> InjectionScan:
    hits = _match(text)
    decoded_text = None
    for blob in _B64.findall(text):
        try:
            decoded = base64.b64decode(blob, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        inner = _match(decoded)
        if inner:
            hits.extend(f"encoded:{r}" for r in inner)
            decoded_text = decoded
    unique = sorted(set(hits))
    return InjectionScan(is_attempt=bool(unique), rules=unique, decoded=decoded_text)
