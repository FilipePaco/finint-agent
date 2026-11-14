---
name: bipartite
description: A set of senders pays a set of receivers in a dense many-to-many pattern. Load when several related accounts send to several shared receivers.
---
# Bipartite

**Pattern.** A group of source accounts each pays several accounts of a receiving group: many
edges between two sets, rarely within a set. Used to spread and mix funds.

## Test it with
1. `get_counterparties(account, window)` → receivers of the alerted account.
2. `get_counterparties` on one or two receivers (direction in) → do they share senders with the alerted account?
3. `detect_patterns` → moderate fan-out together with receivers that have fan-in.

## Indicators
- ≥ 2 receivers that also receive from ≥ 2 of the same senders in the window.
- Similar amounts and timing across the group.

## Benign explanations to rule out
Group of companies paying common suppliers, payroll from several employer accounts.

## Disposition guidance
- **ESCALATE**: dense shared-counterparty structure among new accounts.
- **NEEDS_MORE_INFO**: structure visible but group relationships unknown.
- **CLOSE_FALSE_POSITIVE**: shared suppliers of a documented corporate group.
