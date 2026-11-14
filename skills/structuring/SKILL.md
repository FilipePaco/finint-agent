---
name: structuring
description: Amounts kept just under a reporting threshold or split into round pieces (smurfing). Load when amounts cluster just below 10,000 USD or are suspiciously round.
---
# Structuring

**Pattern.** Transfers are kept just below a reporting threshold (here 10,000 USD) or split into
several similar round amounts to avoid scrutiny.

## Test it with
1. `detect_patterns(account, window)` → `structuring.n_just_under_10k`, `structuring.n_round_thousands`, `structuring.n_out`.
2. `get_transactions(account, window, direction="out")` → the actual amounts and their timing.

## Indicators
- ≥ 2 transfers in [9,000; 10,000) USD within a few days.
- A large share of outflows are round thousands.
- Several such transfers on the same day to different receivers.

## Benign explanations to rule out
Fixed contractual instalments (rent, loan repayments) that happen to be just under 10,000.

## Disposition guidance
- **ESCALATE**: repeated just-under-threshold transfers with no fixed-instalment explanation.
- **NEEDS_MORE_INFO**: a single just-under transfer, or a possible instalment schedule.
- **CLOSE_FALSE_POSITIVE**: regular, documented instalments.
