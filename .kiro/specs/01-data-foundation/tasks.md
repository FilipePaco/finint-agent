# 01 · Data Foundation: Tasks

- [x] 1.1 Project scaffold (**port `pyproject.toml`, Dockerfile, `.gitignore`, `.env.example` from `pig_finserv`**): uv + hatchling, dependency groups (`dev`, `ml`, `gnn`, `agent`, `ui`), Typer CLI, ruff rules from `tech.md`, mypy, pytest, Makefile · _R1.1_
- [x] 1.1b `finint/config.py` with pydantic-settings (**pattern from HealthCareAI `config.py`**) · _R1.1_
- [x] 1.1c GitHub Actions: `quality.yml` (ruff check, ruff format --check, mypy, pytest) and `security.yml` (pip-audit; **adapted from CantadasProject**) · _R1.1_
- [x] 1.2 `data/ingest.py`: load HI-Small transactions + accounts into DuckDB with typed columns and `txn_id`; missing-file message with the Kaggle command · _R1.1_
- [x] 1.3 Patterns parser + linker → `attempts`, `transactions.attempt_id/typology`; unmatched lines go to the data-quality report · _R1.1_
- [x] 1.4 FX table + `amount_usd`, `cross_currency`, `cross_bank`, `ts_shifted` · _R1.2_
- [x] 1.5 Synthetic identities (seeded Faker) · _R1.3_
- [x] 1.6 OFAC SDN loader with snapshot fallback + `data_sources` provenance · _R1.4_
- [x] 1.7 `sanctions_cases.yaml` (≥ 20 cases incl. ≥ 5 decoys) + injector · _R1.5_
- [x] 1.8 Synthetic `reference` memos + `prompt_injections.yaml` + adverse-media fixtures · _R1.6_
- [ ] 1.9 `data_classification.yaml` · _R1.7_
- [ ] 1.10 `reports/data_quality.md` generator
- [ ] 1.11 Tests: ingestion row counts on a tiny fixture CSV, linker matching, deterministic identities, injection idempotency
