---
inclusion: always
---

# Tech

## Stack
| Concern | Choice | Why |
|---|---|---|
| Language / packaging | Python 3.12, **uv** + hatchling, `src/` layout, **dependency groups** (`dev`, `ml`, `gnn`, `agent`, `ui`) | the target role's core language; reproducible installs; CI and demo install only what they need (torch/PyG stay out of the demo) |
| Config | **pydantic-settings** (`finint.config`) reading `.env` + `config/*.yaml`; no scattered `os.getenv` | one typed source of truth for thresholds, capacity, model ids |
| CLI | Typer (`finint data`, `finint train`, `finint investigate`…), wrapped by `make` | |
| Storage / query | **DuckDB** (`data/finint.duckdb`) + SQL | analytical SQL over 5M rows on a laptop; tools query with SQL |
| Audit store | separate SQLite file, hash-chained, append-only | compliance system of record, isolated from analytical data |
| Tabular ML | LightGBM | strong imbalanced-tabular baseline, fast |
| Graph features | NetworkX / igraph on time-windowed graphs | degree, fan-in/out, PageRank, bounded cycles |
| GNN | PyTorch + PyTorch Geometric (GraphSAGE, edge classification) | role names PyTorch; deep-learning fundamentals |
| Calibration / metrics | scikit-learn (isotonic), custom precision@k | |
| Agents | **LangGraph** (thin): explicit graph, `interrupt` for HITL, SQLite checkpointer | the graph is visible; critical rules stay in plain Python |
| LLM | **Gemini** via `init_chat_model`, version-pinned per agent role (Flash-Lite / Flash). PoC runs on the **free tier** with fully synthetic data; production would use a paid tier or Vertex AI (prompts not used for training) | provider-agnostic: an open-weight model (Ollama) is a config change; not run here (laptop disk) |
| Structured output | Pydantic v2 models for every LLM output | validation, retries, schema-first |
| Observability | **Langfuse** (LangChain `CallbackHandler`; self-hosted via docker compose or Langfuse Cloud) for traces, latency, token and cost per trace/agent, eval scores. Token counts in-process come from LangChain's built-in `UsageMetadataCallbackHandler`, and USD comes from Langfuse's model pricing (or the `genai-prices` package when Langfuse is off) | use maintained tools instead of hand-rolled trackers; optional by design (constitution VI) |
| Name screening | `rapidfuzz` + `unidecode` | deterministic, explainable scores |
| Adverse media | `MediaProvider` interface: `FixtureProvider` (default, offline) and `TavilyProvider` (opt-in) | reproducible eval, controlled cost, no real-person queries |
| Analyst console / demo | Streamlit, in Docker; deploy on Railway (or Streamlit Cloud) | fastest HITL UI; hosted demo |
| Reporting | pandas + matplotlib → Markdown/HTML + PNG; Mermaid → `mmdc` → PDF for the architecture diagram | |
| Quality | **ruff** (lint + format), mypy (strict on `core/`), pytest | |
| CI (GitHub Actions) | `quality.yml`: ruff check, ruff format --check, mypy, pytest, **replay eval**. `security.yml`: bandit (via ruff `S`) + pip-audit | "clean code" and "sensitive data" are verified, not claimed |

## Ruff
```toml
[tool.ruff]
line-length = 100
target-version = "py312"
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "S", "SIM", "RUF", "PT"]
[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101"]
```
`S` matters here because tools build SQL, so every query must be parameterized.

## Conventions
- Secrets only in `.env` (`.env.*` ignored, `.env.example` committed).
- Every LLM prompt lives in `prompts/` as a versioned file. The prompt version (file hash) goes into audit events and Langfuse trace metadata.
- `run_id` (uuid4 hex) is both the audit run id and the Langfuse `trace_id`, so one id links compliance records to traces.
- Model artifacts carry `model_version = {name}-{git_sha}-{train_date}` and a **model card**.
- All randomness is seeded (`seed: 42`).
- Modules in `core/` must not import LangChain/LangGraph (constitution X).
- If the LLM is unavailable, the graph degrades to a deterministic evidence pack with disposition `NEEDS_MORE_INFO`. It never crashes the analyst's queue.

## Reused from previous projects
| From | What |
|---|---|
| `HealthCareAI` | ADR format in design docs; pydantic-settings config; `init_chat_model` factory; tool-calling loop with `response_format="content_and_artifact"` + max iterations + deterministic fallback; two-layer audit/observability idea with Langfuse; `TEST_PLAN` (test ↔ requirement, one rejection test per guardrail); `docs/audit.md` for reviewers; Railway deploy notes |
| `pig_finserv` | `pyproject.toml` layout (hatchling, uv, dependency groups, Typer entry point); hash-chained `audit.py` with `AuditIntegrityError`; `modeling/` split, calibration, versioned artifact, model card; Dockerfile with `uv sync --locked`; `.gitignore`/`.env.example` |
| `CantadasProject` | GitHub Actions security workflow (SAST + pip-audit) |

## Commands
`make data` · `make train` · `make alerts` · `make eval` · `make eval-record` · `make report` · `make app` · `make docs` · `make test` · `make lint` · `make audit-verify`
