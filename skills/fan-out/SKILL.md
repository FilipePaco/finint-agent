---
name: fan-out
description: One account quickly pays many different accounts (dispersal of funds). Load when an alerted account has many distinct receivers or a high fan-out ratio.
---
# Fan-out

**Pattern.** A single source splits funds across many receivers in a short time, often just
before or after receiving a large inflow. Typical of the dispersal stage of laundering.

## Test it with
1. `detect_patterns(account, window)` → `fan.distinct_receivers`, `fan.max_distinct_receivers_per_day`, `fan.fan_out_ratio`.
2. `get_counterparties(account, window)` → are the receivers new, many, similar amounts?
3. `get_transactions(account, window, direction="out")` → timing (same day / same hour), amounts, formats.
4. `trace_flows(account, window, direction="in")` → where the money came from just before.

## Indicators (quantitative)
- ≥ 5 distinct receivers in one day, or `fan_out_ratio` ≥ 3.
- Outflows within 24 h of a large inflow (`flow_through.out_to_in_amount_ratio` close to 1).
- Similar amounts across receivers; receivers at several banks.

## Benign explanations to rule out
Payroll (regular, same day each month, `ACH`), supplier runs of a business, family support.
A business entity with a long activity history weakens the suspicion.

## Disposition guidance
- **ESCALATE**: fan-out funded by a fresh inflow, new receivers, no business pattern.
- **NEEDS_MORE_INFO**: high fan-out but plausible payroll/supplier pattern you cannot confirm.
- **CLOSE_FALSE_POSITIVE**: regular, recurring, low-value payments consistent with the entity type.
