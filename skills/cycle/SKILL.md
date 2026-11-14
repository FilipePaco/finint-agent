---
name: cycle
description: Funds leave an account and return to it through a chain of accounts (round-tripping). Load when cycles are detected or flows return to the origin.
---
# Cycle (round-tripping)

**Pattern.** Money travels A → B → C → … → A, often across banks and currencies, to create
the appearance of legitimate turnover.

## Test it with
1. `detect_patterns(account, window)` → `cycles_through_account.cycles_len2/3/4`.
2. `trace_flows(account, window, direction="out", hops=3)` → `returns_to_origin`, the path, amounts per hop.
3. `get_transactions` on intermediate accounts if needed (amounts shrinking slightly each hop = fees skimmed).

## Indicators
- Any directed cycle of length 2–4 within the window, especially with similar amounts.
- Cross-bank or cross-currency hops; short time between hops.

## Benign explanations to rule out
Loan and repayment between the same two parties (length-2 cycle, documented terms), refunds,
transfers between accounts of the same owner (check `get_account_profile.other_accounts_of_owner`).

## Disposition guidance
- **ESCALATE**: cycle of length ≥ 3 with similar amounts and no common owner.
- **NEEDS_MORE_INFO**: length-2 cycle that could be a loan/refund.
- **CLOSE_FALSE_POSITIVE**: the cycle runs only through accounts of the same owner with a clear reason.
