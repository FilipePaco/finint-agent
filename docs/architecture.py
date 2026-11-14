"""Render the conceptual architecture diagram (task 4.9 · R4.6): docs/architecture.{pdf,png}.

    uv run python docs/architecture.py

Drawn with matplotlib so it renders anywhere without a browser or Graphviz. Colours encode
who decides: LLM agents (orange), deterministic code (blue), data (grey), the human (green).
Dashed zones are trust boundaries.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

OUT = Path(__file__).parent
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
LLM, CODE, DATA, HUMAN = "#fbe3d6", "#dbe8f9", "#ecebe7", "#d8f0e3"
LLM_EDGE, CODE_EDGE, DATA_EDGE, HUMAN_EDGE = "#eb6834", "#2a78d6", "#8f8d87", "#1baf7a"


def box(ax, x, y, w, h, title, body="", fill=CODE, edge=CODE_EDGE, size=8.5):  # type: ignore[no-untyped-def]
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc=fill, ec=edge, lw=1.4
        )
    )
    ax.text(
        x + w / 2,
        y + h - 1.0,
        title,
        ha="center",
        va="top",
        fontsize=size + 0.5,
        fontweight="bold",
        color=INK,
    )
    if body:
        ax.text(
            x + w / 2,
            y + h - 3.0,
            body,
            ha="center",
            va="top",
            fontsize=size - 1,
            color=INK_2,
            linespacing=1.35,
        )
    return (x, y, w, h)


def zone(ax, x, y, w, h, label):  # type: ignore[no-untyped-def]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.2,rounding_size=1.2",
            fc="none",
            ec="#b5b3ad",
            lw=1.2,
            ls=(0, (5, 4)),
        )
    )
    ax.text(
        x + 0.8, y + h - 0.6, label, ha="left", va="top", fontsize=9, color=INK_2, style="italic"
    )


def arrow(ax, a, b, label="", both=False, dashed=False, rad=0.0, lx=0.0, ly=0.0):  # type: ignore[no-untyped-def]
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="<|-|>" if both else "-|>",
            mutation_scale=11,
            color=INK_2,
            lw=1.1,
            ls="--" if dashed else "-",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        )
    )
    if label:
        mx, my = (a[0] + b[0]) / 2 + lx, (a[1] + b[1]) / 2 + ly
        ax.text(
            mx,
            my,
            label,
            fontsize=7,
            color=INK_2,
            ha="center",
            va="center",
            bbox={"fc": SURFACE, "ec": "none", "pad": 0.6},
        )


def path(ax, points, label="", dashed=False, both=False, label_at=None):  # type: ignore[no-untyped-def]
    """Orthogonal connector through the gutters between boxes (no box crossings)."""
    xs, ys = zip(*points, strict=True)
    ax.plot(
        (*xs[:-1], xs[-1]),
        (*ys[:-1], ys[-1]),
        color=INK_2,
        lw=1.1,
        ls="--" if dashed else "-",
        solid_capstyle="butt",
    )
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=11,
            color=INK_2,
            lw=1.1,
            ls="--" if dashed else "-",
            shrinkB=1,
        )
    )
    if both:
        ax.add_patch(
            FancyArrowPatch(
                points[1],
                points[0],
                arrowstyle="-|>",
                mutation_scale=11,
                color=INK_2,
                lw=1.1,
                shrinkB=1,
            )
        )
    if label:
        x, y = label_at or points[len(points) // 2]
        ax.text(
            x,
            y,
            label,
            fontsize=7,
            color=INK_2,
            ha="center",
            va="center",
            bbox={"fc": SURFACE, "ec": "none", "pad": 0.6},
        )


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 10.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")
    ax.text(
        1,
        65.3,
        "finint-agent: agentic AML alert investigation with ML scoring",
        fontsize=15,
        fontweight="bold",
        color=INK,
        va="top",
    )
    ax.text(
        1,
        62.9,
        "The agents recommend; deterministic code enforces grounding, sanctions status "
        "and policy; a human analyst decides. Every step is written to a hash-chained audit log.",
        fontsize=9.5,
        color=INK_2,
        va="top",
    )

    zone(ax, 0.5, 1, 19, 59, "External sources (untrusted)")
    zone(ax, 21, 1, 58.5, 59, "Bank internal: investigation environment")
    zone(ax, 80.5, 27, 19, 33, "LLM provider")
    zone(ax, 80.5, 1, 19, 24, "Observability (optional)")

    box(
        ax,
        2,
        45,
        16,
        11,
        "IBM AMLworld HI-Small",
        "5.08M synthetic transfers\nlabels + laundering\npatterns (Kaggle)",
        DATA,
        DATA_EDGE,
    )
    box(
        ax,
        2,
        31,
        16,
        11,
        "OFAC SDN list",
        "SDN + aliases + addresses\n(Sanctions List Service;\npinned snapshot fallback)",
        DATA,
        DATA_EDGE,
    )
    box(
        ax,
        2,
        15.5,
        16,
        12,
        "News sources",
        "fictional fixtures (default)\nor Tavily (opt-in); only\nfor designated test\nentities",
        DATA,
        DATA_EDGE,
    )

    box(
        ax,
        22.5,
        43,
        15.5,
        13,
        "Data store (DuckDB)",
        "transactions, entities\n(synthetic names), features,\nalerts, OFAC tables;\ncolumn classification",
        DATA,
        DATA_EDGE,
    )
    box(
        ax,
        22.5,
        24,
        15.5,
        15.5,
        "ML detection",
        "tabular + graph features\nLightGBM champion (M1)\nvs GNN (GraphSAGE)\nisotonic calibration\ncapacity threshold\n(200/day) → alerts",
        CODE,
        CODE_EDGE,
    )
    box(
        ax,
        22.5,
        2.5,
        15.5,
        10.5,
        "Analyst console",
        "Streamlit, role-gated\ncase review · audit trail\nAPPROVE / EDIT / REJECT",
        HUMAN,
        HUMAN_EDGE,
    )

    ax.add_patch(
        FancyBboxPatch(
            (42.2, 15.8),
            36.6,
            41.2,
            boxstyle="round,pad=0.3,rounding_size=1.0",
            fc="#f6f5f2",
            ec=INK_2,
            lw=1.2,
        )
    )
    ax.text(
        60.5,
        56.5,
        "Orchestrating agent graph (LangGraph, checkpointed)",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )
    box(ax, 44, 45.5, 8.5, 7.5, "intake", "get_alert +\nML explanation", CODE, CODE_EDGE, 8)
    box(
        ax,
        55.5,
        44.5,
        10.5,
        9,
        "Investigator",
        "LLM agent: picks\ntools, skills,\nhypotheses; stops",
        LLM,
        LLM_EDGE,
        8,
    )
    box(ax, 69, 45.5, 9, 7.5, "Grounding", "code: every value\nin evidence", CODE, CODE_EDGE, 8)
    box(ax, 69, 35, 9, 7.5, "Reviewer", "LLM agent:\nACCEPT / REVISE", LLM, LLM_EDGE, 8)
    box(
        ax,
        56,
        35,
        10,
        7.5,
        "Policy guard",
        "code: only raises\nscrutiny (POL-01..05)",
        CODE,
        CODE_EDGE,
        8,
    )
    box(ax, 44, 35, 8.5, 7.5, "Analyst", "interrupt:\nhuman decides", HUMAN, HUMAN_EDGE, 8)
    box(
        ax,
        44,
        17,
        22.5,
        15,
        "Read-only tools (allowlisted, budgeted)",
        "get_alert · explain (ML scorer, SHAP)\naccount profile · transactions (totals)\ncounterparties · trace_flows · detect_patterns\nscreen_sanctions (deterministic screener)\nrequest_media_check · load_skill\nminimized fields · memos as untrusted data",
        CODE,
        CODE_EDGE,
        8,
    )
    box(
        ax,
        68.5,
        24.5,
        10,
        7.5,
        "Media agent",
        "quarantined LLM:\nsearch/fetch only;\nquotes verified",
        LLM,
        LLM_EDGE,
        7.5,
    )
    box(ax, 68.5, 17, 10, 5.5, "Skills", "9 playbooks", DATA, DATA_EDGE, 7.5)

    box(
        ax,
        44,
        2.5,
        34.5,
        9.5,
        "Hash-chained audit log (SQLite, append-only)",
        "ALERT_CREATED · RUN_STARTED · LLM_CALL (tokens, cost, prompt version)\nTOOL_CALL · GROUNDING_RESULT · REVIEW_VERDICT · POLICY_OVERRIDE\nCASE_DRAFTED · ANALYST_DECISION · RUN_COMPLETED",
        DATA,
        DATA_EDGE,
        7.8,
    )

    box(
        ax,
        82,
        41,
        16,
        15,
        "Google Gemini",
        "Flash-Lite: all three\nagents (free tier);\nFlash: optional reviewer\nreceives minimized context\nonly (no labels, DOB→year)",
        LLM,
        LLM_EDGE,
    )
    box(
        ax,
        82,
        29,
        16,
        8.5,
        "Pinned models",
        "init_chat_model:\nprovider is config",
        LLM,
        LLM_EDGE,
        8,
    )
    box(
        ax,
        82,
        3,
        16,
        17,
        "Langfuse traces",
        "trace id = run id\nlatency, tokens, cost\nper agent; degrades\nto no-op",
        DATA,
        DATA_EDGE,
    )

    # data in
    arrow(ax, (18.4, 50.5), (22.1, 50.5), "ingest", ly=1.2)
    path(ax, [(18.4, 36.5), (20.3, 36.5), (20.3, 46), (22.1, 46)], "load", label_at=(20.3, 41))
    arrow(ax, (30.2, 42.6), (30.2, 39.9), "features / alerts", both=True, lx=0)
    # alerts -> intake (gutter x=40.6), read-only SQL (gutter x=39.4)
    arrow(ax, (38.4, 51), (43.6, 51), "alert", ly=1.1)
    path(
        ax,
        [(43.6, 30), (39.6, 30), (39.6, 45), (38.4, 45)],
        "read-only SQL",
        both=True,
        label_at=(40.2, 27.8),
    )
    # graph flow
    arrow(ax, (52.8, 49.2), (55.2, 49.2))
    arrow(ax, (66.3, 49.2), (68.7, 49.2), "draft", ly=1.2)
    arrow(ax, (73.5, 45.2), (73.5, 42.8), "pass", lx=1.6)
    arrow(ax, (68.7, 38.7), (66.3, 38.7))
    arrow(ax, (55.7, 38.7), (52.8, 38.7))
    arrow(ax, (71, 45.4), (64.5, 44.4), "repair ≤ 2", dashed=True, rad=0.35, ly=-1.6)
    path(
        ax, [(55.2, 46.5), (54.2, 46.5), (54.2, 32.3)], "tool calls", both=True, label_at=(54.2, 34)
    )
    arrow(ax, (66.8, 28.3), (68.2, 28.3), both=True)
    # LLM calls (above the graph box / to the right)
    path(
        ax,
        [(62, 53.8), (62, 58.6), (80.0, 58.6), (80.0, 52), (81.7, 52)],
        "prompts / tool calls",
        both=True,
        label_at=(71, 58.6),
    )
    arrow(ax, (78.3, 39.5), (81.7, 44.5), both=True)
    arrow(ax, (78.8, 28.3), (81.7, 42.5), both=True, dashed=True)
    # media search goes out through the gutter under the tools
    path(
        ax,
        [(68.2, 27), (67.5, 27), (67.5, 14.2), (19.0, 14.2), (19.0, 18), (18.4, 18)],
        "search / fetch (permission gate)",
        dashed=True,
        label_at=(30, 14.2),
    )
    # analyst console <-> interrupt (gutter x=43)
    path(
        ax,
        [(43.7, 37.5), (43.0, 37.5), (43.0, 7.8), (38.4, 7.8)],
        "case review / decision",
        both=True,
        label_at=(40.7, 9.6),
    )
    arrow(ax, (61, 15.5), (61, 12.4), "every step", lx=2.6)
    arrow(ax, (78.8, 7), (81.7, 9), "run id", dashed=True, ly=1.2)

    legend = [
        Patch(fc=LLM, ec=LLM_EDGE, label="LLM agent (judgment)"),
        Patch(fc=CODE, ec=CODE_EDGE, label="Deterministic code (rules)"),
        Patch(fc=DATA, ec=DATA_EDGE, label="Data / external source"),
        Patch(fc=HUMAN, ec=HUMAN_EDGE, label="Human analyst (decides)"),
    ]
    ax.legend(
        handles=legend, loc="lower left", bbox_to_anchor=(0.012, 0.025), frameon=False, fontsize=8
    )
    ax.text(23, 21, "logs ALERT_CREATED\nto the audit trail", fontsize=7, color=INK_2, va="top")
    for name in ("architecture.pdf", "architecture.png"):
        fig.savefig(OUT / name, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'architecture.pdf'} and .png")


if __name__ == "__main__":
    main()
