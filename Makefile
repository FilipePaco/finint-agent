# Single entry points (steering/tech.md). Targets are added as specs are implemented.
.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install lint format typecheck test check data features train

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## Create the venv with dev tools
	$(UV) sync --group dev --group data --group ml

lint: ## Ruff lint + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Auto-format and fix lint where safe
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck: ## mypy (strict on core/ and config)
	$(UV) run mypy

test: ## Unit tests (no network, no LLM)
	$(UV) run pytest -m "not slow and not live"

check: lint typecheck test ## Everything CI runs

data: ## Build the data foundation in data/finint.duckdb (spec 01)
	$(UV) run finint data download
	$(UV) run finint data build

features: ## Build per-transaction features (spec 02)
	$(UV) run finint ml features

train: features ## Train and evaluate the detection models (spec 02)
	$(UV) run finint ml train --model m0
