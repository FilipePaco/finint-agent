---
inclusion: always
---

# Product

## Problem
A digital bank's FinCrime team gets thousands of transaction-monitoring alerts a week. Most are false positives, and analysts spend most of each case gathering the same evidence by hand: account history, counterparties, money flows, sanctions hits and news.

## What we build (PoC)
1. **Detection.** An ML model scores transactions for laundering risk. Scores roll up to account-day alerts, and the analyst team's capacity sets how many become alerts.
2. **Investigation.** For a selected alert, a multi-agent system gathers evidence with tools, identifies laundering typologies, screens counterparties, checks adverse media and drafts a grounded case file with a recommended disposition: `CLOSE_FALSE_POSITIVE`, `ESCALATE` or `NEEDS_MORE_INFO`.
3. **Decision.** An analyst approves, edits or rejects the case file in an analyst console. Every step is audited.
4. **Operations.** A weekly report covers alert volume, precision, time saved, agent–analyst agreement and cost per alert.

## Users
- **FinCrime analyst**: reviews case files and makes the decision.
- **FinCrime ops lead**: reads the weekly report and tunes capacity and thresholds.
- **Model risk / audit**: replays any decision from the audit trail.

## Success metrics (PoC targets, measured by `make eval`)
| Metric | Target |
|---|---|
| Precision @ alert budget (test split) | beats the tabular baseline; graph features add lift |
| Disposition accuracy vs ground truth (≥50 alerts) | ≥ 80% |
| Faithfulness (unsupported values in released case files) | 0 |
| Pattern identification recall | ≥ 70% |
| Prompt-injection success rate | 0% |
| Evidence-gathering time per alert | report p50/p95 vs the documented manual baseline |
| Cost per investigated alert | reported per model; hosted vs open-weight compared |

## Non-goals
Real customer data, real SAR filing, account blocking, customer communication, production-grade auth, and a real-time streaming pipeline.

## Data ethics
All account holders are synthetic. Adverse-media search runs only against injected synthetic test entities and public enforcement examples. It never runs against real private individuals.
