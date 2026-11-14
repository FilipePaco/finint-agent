---
name: scatter-gather
description: Funds are split across intermediaries and then re-collected (scatter-gather), or collected from many and then dispersed (gather-scatter). Load for hub-like or two-stage flows.
---
# Scatter-gather and gather-scatter

**Pattern.**
- *Scatter-gather*: source → many intermediaries → one destination (classic layering).
- *Gather-scatter*: many sources → one hub → many destinations.

## Test it with
1. `detect_patterns(account, window)` → both `distinct_senders` and `distinct_receivers` high (hub), or high fan-out.
2. `trace_flows(account, window, direction="out", hops=2)` → do intermediaries converge on one account?
3. `trace_flows(account, window, direction="in", hops=2)` → for a hub, where did the gathered money come from?

## Indicators
- Hub: ≥ 3 senders and ≥ 3 receivers in the window with `out_to_in_amount_ratio` near 1.
- Scatter-gather: two-hop paths from the source that end at the same destination.

## Benign explanations to rule out
Treasury or payment-processor accounts, marketplaces settling merchants.

## Disposition guidance
- **ESCALATE**: two-stage flow through new intermediaries with pass-through amounts.
- **NEEDS_MORE_INFO**: hub behaviour by an entity type that could be a processor.
- **CLOSE_FALSE_POSITIVE**: documented processor/marketplace pattern with stable counterparties.
