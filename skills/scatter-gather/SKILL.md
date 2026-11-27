---
name: scatter-gather
description: Funds are split across intermediaries and then re-collected (scatter-gather), or collected from many and then dispersed (gather-scatter). Load for hub-like or two-stage flows.
---
# Scatter-gather and gather-scatter

**Pattern.**
- *Scatter-gather*: source → many intermediaries → one destination (classic layering).
- *Gather-scatter*: many sources → one hub → many destinations.

The alerted account is often **not** the hub: it can be one intermediary (1 sender, 1
receiver: it looks ordinary on its own) or one feeder paying the hub. Single-account fan
numbers miss this; the two-hop numbers in `detect_patterns.multi_account` do not.

## Test it with
1. `detect_patterns(account, window)` with a window from 7 days before to 3 days after the alert
   date (the gather leg often happens after it). Read `multi_account`:
   - `scatter_gather_intermediary.parallel_intermediaries`: accounts that, like this one,
     received from the same `source` and paid the same `destination` (this account included);
   - `scatter_gather_source.converging_intermediaries`: this account's receivers that all pay
     one `destination`, and `share_of_outflow_converging`;
   - `gather_scatter_hub.hub_degree`: min(distinct senders, distinct receivers) of the account;
   - `gather_scatter_feeder.hub_degree`: the same for the `hub` this account pays.
2. `trace_flows(account, window, direction="out", hops=2)` → the converging paths and amounts.
3. For a hub or feeder: `get_counterparties(hub, window)` → who sends and who receives.

## Indicators
- Intermediary: `parallel_intermediaries` ≥ 5 (code flags it at this threshold).
- Source: `converging_intermediaries` ≥ 2 with a large `share_of_outflow_converging`.
- Gather-scatter: `hub_degree` ≥ 7 for the account or for the hub it pays.
- Amounts in and out of each intermediary are similar; hops follow within a few days.

## Benign explanations to rule out
Treasury or payment-processor accounts, marketplaces settling merchants, payroll bureaus
(many employers → bureau → many employees).

## Disposition guidance
- **CONFIRMED**: the indicators above, with the counterpart account (`source`, `destination`
  or `hub`) cited from the evidence.
- **REJECTED**: cite the numbers that are below the indicators, or the benign explanation the
  evidence supports.
- **ESCALATE**: two-stage flow through new intermediaries with pass-through amounts.
- **NEEDS_MORE_INFO**: hub behaviour by an entity type that could be a processor.
- **CLOSE_FALSE_POSITIVE**: documented processor/marketplace pattern with stable counterparties.
