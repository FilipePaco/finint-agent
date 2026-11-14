# Single entry points (steering/tech.md). Targets are added as specs are implemented.
.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install lint format typecheck test test-gnn check data features train alerts report-ml audit-verify app eval-build eval eval-report report docs

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## Create the venv with dev tools
	$(UV) sync --group dev --group data --group ml --group agent --group ui

lint: ## Ruff lint + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Auto-format and fix lint where safe
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck: ## mypy (strict on core/ and config)
	$(UV) run mypy

test: ## Unit tests (no network, no LLM)
	$(UV) run pytest -m "not slow and not live and not gnn"

test-gnn: ## GNN tests in their own process (needs: uv sync --group gnn)
	$(UV) run pytest -m gnn tests/test_gnn.py  # only this file: collecting others loads LightGBM

check: lint typecheck test ## Everything CI runs

data: ## Build the data foundation in data/finint.duckdb (spec 01)
	$(UV) run finint data download
	$(UV) run finint data build

features: ## Build tabular and graph features (spec 02)
	$(UV) run finint ml features
	$(UV) run finint ml graph

train: features ## Train and evaluate the detection models (spec 02)
	$(UV) run finint ml train --model m0
	$(UV) run finint ml train --model m1

alerts: ## Champion, calibration, capacity threshold, alerts + audit events (spec 02)
	$(UV) run finint ml alerts

audit-verify: ## Recompute the audit hash chain
	$(UV) run finint audit verify

report-ml: ## ML report and figures from saved artifacts (spec 02)
	$(UV) run finint ml report

app: ## Analyst console (Streamlit) on http://localhost:8501
	$(UV) run streamlit run src/finint/app/console.py

eval-build: ## Build the evaluation set (spec 04)
	$(UV) run finint eval build

eval: ## Run the agent evaluation (needs GOOGLE_API_KEY; resumable)
	$(UV) run finint eval run --config main

eval-report: ## Metrics and reports/eval_report.md from eval/results
	$(UV) run finint eval report

report: ## Weekly operations report with the two charts (spec 04)
	$(UV) run finint report weekly

docs: ## Render docs/architecture.pdf and .png
	$(UV) run python docs/architecture.py
