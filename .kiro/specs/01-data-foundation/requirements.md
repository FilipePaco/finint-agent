# 01 · Data Foundation: Requirements

## Introduction
Load IBM AMLworld **HI-Small** into DuckDB, normalize it, attach synthetic identities, load the OFAC SDN list, and inject a documented set of sanctions near-matches and prompt-injection payloads. Every later spec depends on this.

## Glossary
- **Transaction**: one row of `HI-Small_Trans.csv`.
- **Account**: `(bank_id, account_number)`. **Entity**: owner of one or more accounts (from `HI-Small_accounts.csv`).
- **Attempt**: one laundering attempt from `HI-Small_Patterns.txt`, with a typology (FAN-OUT, FAN-IN, CYCLE, GATHER-SCATTER, SCATTER-GATHER, STACK, BIPARTITE, RANDOM).
- **Injected entity**: synthetic entity whose name is a deliberate near-match to an SDN entry, or which carries a prompt-injection payload.

## Requirements

### R1.1 Ingestion
**User story:** As an engineer, I want the raw dataset in a queryable store so that features and tools use SQL.
- WHEN `make data` runs THE SYSTEM SHALL load transactions, accounts and patterns into `data/finint.duckdb` with typed columns.
- THE SYSTEM SHALL assign a stable `txn_id` to every transaction.
- THE SYSTEM SHALL link each laundering transaction in the patterns file to its `attempt_id` and `typology`. Unmatched lines are counted and reported, not dropped silently.
- IF raw files are missing THEN THE SYSTEM SHALL print the exact Kaggle download command and exit non-zero.

### R1.2 Normalization
- THE SYSTEM SHALL store `amount_usd` using a documented static FX table in `config/fx.yaml`.
- THE SYSTEM SHALL flag `cross_currency` and `cross_bank` per transaction.
- THE SYSTEM SHALL expose a `timestamp_shifted` column that maps the dataset's date range so it ends on a configured "as-of" date (needed for "last 30 days" reports).

### R1.3 Synthetic identities
- THE SYSTEM SHALL generate a deterministic (seeded) synthetic name, country and entity type for every entity.
- THE SYSTEM SHALL never derive names from, or attempt to link them to, real persons.

### R1.4 Sanctions list
- WHEN `make data` runs THE SYSTEM SHALL load the OFAC SDN primary and alternate-name CSVs into `sdn_entries` and `sdn_aliases`, recording the download date and file hash.
- IF the download fails THEN THE SYSTEM SHALL fall back to a pinned snapshot in `data/fixtures/sdn_snapshot/` and log which source was used.

### R1.5 Injected near-matches
- THE SYSTEM SHALL inject ≥ 20 synthetic entities whose names are near-matches to SDN entries. The set covers transliteration, missing middle name, swapped word order, diacritics/spacing, and ≥ 5 **decoys** (similar but not the same person, e.g. different date of birth or country).
- Each injected entity SHALL be documented in `data/injections/sanctions_cases.yaml` with its intended outcome (`TRUE_MATCH`, `FALSE_MATCH`).
- At least half the injected entities SHALL be attached to accounts that appear in eval-set alerts.

### R1.6 Injected untrusted text
- Because the dataset has no free-text field, THE SYSTEM SHALL add a synthetic `reference` column (payment memo). Most rows get benign memos; ≥ 10 rows in eval-set alerts carry documented prompt-injection payloads (`data/injections/prompt_injections.yaml`).
- THE SYSTEM SHALL create adverse-media fixture documents for injected entities: relevant hits, irrelevant same-name hits, and ≥ 3 documents carrying injection payloads.

### R1.7 Data minimization map
- THE SYSTEM SHALL keep a field classification (`config/data_classification.yaml`): each column marked `allowed`, `untrusted` (sent only inside a delimited untrusted-data block), `masked` or `forbidden`. Spec 03 enforces it.
- WHEN `finint data build` runs THE SYSTEM SHALL fail if any column of any table is unclassified.

## Open questions [NEEDS CLARIFICATION]
- Q1 (resolved: simulated history, labeled): HI-Small covers Sep 1–18, 2022, with volume only on Sep 1–10. For the 12-month monthly chart we either (a) replay/bootstrap daily alert volumes into a synthetic 12-month history, clearly labeled as simulated, or (b) also load LI-Small/HI-Medium for more days. **Proposal: (a)**. It is cheap and transparent.
- Q2: Do we keep all 5M rows or downsample non-laundering rows for dev speed? **Proposal:** use the full set for features and evaluation, and an optional `--sample` flag for development.
