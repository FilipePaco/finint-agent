---
name: fan-in
description: Many accounts pay into one account (aggregation of funds). Load when an account has many distinct senders or a high fan-in ratio.
---
# Fan-in

**Pattern.** Many senders, often new and unrelated, pay one collector account, which may then
move the total onward. Typical of the placement/aggregation stage.

## Test it with
1. `detect_patterns(account, window)` → `fan.distinct_senders`, `fan.fan_in_ratio` (the account is
   the collector), and `multi_account.fan_in_feeder` (the account is one of many senders to a
   collector: `receiver`, `receiver_distinct_senders`; code flags it at ≥ 11).
2. `get_counterparties(account, window)` → number of senders, their banks, amounts.
3. `trace_flows(account, window, direction="out")` → does the aggregated money leave quickly?

## Indicators
- ≥ 5 distinct senders in a few days, or `fan_in_ratio` ≥ 3 (collector).
- The account pays a collector with ≥ 11 distinct senders in the window (feeder).
- Senders at many banks; similar or round amounts; new pairs.
- The collector pays out most of what came in within 24 h.

## Benign explanations to rule out
Merchant receiving card or invoice payments, rent collection, crowdfunding.

## Disposition guidance
- **ESCALATE**: aggregation from many new senders followed by rapid onward transfer.
- **NEEDS_MORE_INFO**: plausible merchant activity but entity type or history does not fit.
- **CLOSE_FALSE_POSITIVE**: consistent merchant/collection pattern, funds stay or move regularly.
