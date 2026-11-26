"""Controlled typology vocabulary for hypotheses and its deterministic normaliser (R3.1, R4.2).

The canonical names are the typology skills (``skills/<name>``), which are also the names the
evaluation ground truth uses. ``other`` holds hypotheses that are not a laundering typology
(e.g. a sanctions concern, which belongs in ``sanctions_evidence_ids``) or that name more than
one typology at once. Framework-free (constitution X).
"""

from __future__ import annotations

import re
from typing import Literal, get_args

Typology = Literal[
    "fan-in", "fan-out", "cycle", "scatter-gather", "stack", "bipartite", "structuring", "other"
]
TYPOLOGIES: tuple[str, ...] = get_args(Typology)
OTHER: Typology = "other"

# Aliases come from the skill descriptions (round-tripping, layering, smurfing...) and from the
# dataset's own names (GATHER-SCATTER, RANDOM), so recorded free-text labels map back.
_ALIASES: dict[str, Typology] = {
    **{t: t for t in TYPOLOGIES},  # type: ignore[misc]
    "fanin": "fan-in",
    "aggregation": "fan-in",
    "fanout": "fan-out",
    "dispersal": "fan-out",
    "cycles": "cycle",
    "cyclic": "cycle",
    "round-tripping": "cycle",
    "round-trip": "cycle",
    "roundtripping": "cycle",
    "gather-scatter": "scatter-gather",
    "scatter-and-gather": "scatter-gather",
    "gather-and-scatter": "scatter-gather",
    "stacking": "stack",
    "layering": "stack",
    "random": "stack",
    "smurfing": "structuring",
    "threshold-avoidance": "structuring",
    "none": "other",
}
# Separators of compound labels such as "Structuring / Layering" or "Cycle (round-tripping)".
_SPLIT = re.compile(r"\s*(?:/|\(|\)|,|;|&|\+|\band\b|\bor\b|-and-|-or-)\s*")


def _key(text: str) -> str:
    return re.sub(r"[\s_]+", "-", text.strip().lower()).strip("-")


def normalize_typology(raw: str | None) -> Typology:
    """Map a free-text typology label to the controlled vocabulary.

    Exact name or alias first; otherwise the label is split on separators and mapped only if
    its parts name exactly one typology ("Fan-Out / Dispersal" -> fan-out). Labels naming two
    different typologies ("Structuring / Layering") or none ("Sanctions Match") are ``other``.
    """
    if not raw:
        return OTHER
    whole = _key(raw.replace("/", "-"))
    if whole in _ALIASES:
        return _ALIASES[whole]
    found = {_ALIASES[k] for k in (_key(p) for p in _SPLIT.split(raw.lower())) if k in _ALIASES}
    found.discard(OTHER)
    return found.pop() if len(found) == 1 else OTHER
