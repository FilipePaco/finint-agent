---
name: stack
description: Layering through several stacked layers of intermediaries (stack), including irregular multi-hop paths (random). Load when money passes through chains of accounts over several hops.
---
# Stack (layered intermediaries) and random paths

**Pattern.** Funds move through two or more layers of intermediary accounts, each layer paying
the next, to distance the money from its source. AMLworld also has *random* multi-hop paths
with the same purpose but no regular shape.

## Test it with
1. `trace_flows(account, window, direction="out", hops=3)` → layers of accounts, timing per hop.
2. `trace_flows(account, window, direction="in", hops=3)` → layers before the account.
3. `detect_patterns` on the account → `multi_account.amount_matched_chain.forward_depth` (layers
   after the account that forward a similar amount within 72 h) and `backward_depth` (layers
   before it); code flags forward ≥ 2 or backward ≥ 4. Also `flow_through`.

## Indicators
- Intermediaries that pay out most of what they receive within 24 h.
- Each hop happens soon after the previous one; amounts slightly decrease.
- Intermediaries with little other activity.

## Benign explanations to rule out
Correspondent-banking chains, intra-group transfers, payment aggregators.

## Disposition guidance
- **ESCALATE**: ≥ 2 pass-through layers with short delays and no business rationale.
- **NEEDS_MORE_INFO**: layering-like path whose intermediaries you could not profile.
- **CLOSE_FALSE_POSITIVE**: the chain is a known intra-group or settlement route.
