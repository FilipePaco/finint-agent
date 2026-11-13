---
inclusion: always
---

# Structure

```
finint-agent/
├── .github/workflows/       # quality.yml, security.yml
├── config/                  # YAML: thresholds, capacity, FX, model ids per agent role
├── data/                    # gitignored except data/fixtures/ and data/injections/
│   ├── raw/                 # Kaggle HI-Small + OFAC SDN downloads
│   ├── fixtures/            # recorded adverse-media docs (synthetic + public examples)
│   └── injections/          # documented near-match names + prompt-injection payloads
├── prompts/                 # versioned prompt files (investigator, reviewer, media)
├── skills/                  # Agent Skills: one folder per typology, each with SKILL.md
├── src/finint/
│   ├── config.py            # pydantic-settings, single config source
│   ├── cli.py               # Typer entry point
│   ├── observability.py     # the ONLY module that knows Langfuse; returns [] when off
│   ├── core/                # framework-free domain logic (constitution X)
│   │   ├── models.py        # Pydantic: Alert, Evidence, CaseFile, Disposition, AuditEvent
│   │   ├── grounding.py     # faithfulness validator
│   │   ├── screening.py     # sanctions fuzzy matcher
│   │   ├── policy.py        # deterministic disposition overrides
│   │   ├── minimize.py      # field allowlists for LLM context
│   │   └── audit.py         # append-only hash-chained audit log
│   ├── data/                # ingestion, synthetic identities, SDN loader, injections
│   ├── ml/                  # features, graph features, train, calibrate, threshold, gnn
│   ├── tools/               # read-only investigation tools (SQL over DuckDB)
│   ├── agents/              # LangGraph graph, nodes, state, LLM factory
│   ├── eval/                # eval set builder, metrics, runner
│   ├── reporting/           # weekly ops report + charts
│   └── app/                 # Streamlit analyst console
├── tests/                   # mirrors src/finint/
├── reports/                 # generated: ml_report.md, eval_report.md, ops_report_*.md, charts
├── docs/
│   ├── architecture.mmd / architecture.pdf   # conceptual diagram (deliverable)
│   └── audit.md             # how a reviewer reads the audit trail and traces
├── Dockerfile · compose.yaml  # app + optional self-hosted Langfuse profile
├── .kiro/steering/          # this folder
├── .kiro/specs/             # feature specs: requirements → design → tasks
└── .specify/memory/constitution.md
```

## Rules
- Specs are numbered by dependency order: `01-data-foundation` → `02-ml-detection` → `03-investigation-agent` → `04-evaluation-and-reporting`.
- Requirement ids are `R<spec>.<n>` (e.g., `R3.4`). Tasks and tests reference them.
- Each spec folder has `requirements.md`, `design.md` (with **ADRs**: decision, alternatives, why), `tasks.md`, and `test-strategy.md` where guardrails are involved.
- `tools/` functions are read-only, return Pydantic models and an `evidence_id`, and never call an LLM.
- Generated outputs go in `reports/`. Committed reports are the ones quoted in the README.
