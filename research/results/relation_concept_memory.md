# Relation Concept Memory Feasibility

This report compares exact pair-overlap aliasing with `dax`-style relation
concept memory prototypes over synthetic disjoint-relation data.

## Setup

- `dim=2048`
- `seeds=[0, 1, 2, 3, 4]`
- `train_per_surface=12`
- `eval_per_alias=8`
- `families=5`
- Scenarios:
  - `pair_reuse`: unknown alias reuses subject/object pairs seen under the canonical relation.
  - `disjoint_entities`: unknown alias uses new entities but the same relation-role patterns.
- Methods:
  - `pair_overlap`: current exact subject/object overlap heuristic.
  - `identity_memory`: HRR relation memory using only entity identities.
  - `typed_memory`: HRR relation memory using subject role, object role, domain, and cue context.
  - `hybrid_memory`: identity plus typed context.

## Aggregate Results

| scenario | support | method | accuracy | unresolved | mean_score | mean_margin |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| disjoint_entities | 1 | hybrid_memory | 1.000 | 0.000 | 0.573 | 0.462 |
| disjoint_entities | 1 | identity_memory | 0.133 | 0.000 | 0.023 | 0.015 |
| disjoint_entities | 1 | pair_overlap | 0.000 | 1.000 | 0.000 | 0.000 |
| disjoint_entities | 1 | typed_memory | 1.000 | 0.000 | 0.703 | 0.573 |
| disjoint_entities | 2 | hybrid_memory | 1.000 | 0.000 | 0.758 | 0.652 |
| disjoint_entities | 2 | identity_memory | 0.067 | 0.000 | 0.025 | 0.016 |
| disjoint_entities | 2 | pair_overlap | 0.000 | 1.000 | 0.000 | 0.000 |
| disjoint_entities | 2 | typed_memory | 1.000 | 0.000 | 0.908 | 0.790 |
| disjoint_entities | 4 | hybrid_memory | 1.000 | 0.000 | 0.862 | 0.748 |
| disjoint_entities | 4 | identity_memory | 0.133 | 0.000 | 0.026 | 0.015 |
| disjoint_entities | 4 | pair_overlap | 0.000 | 1.000 | 0.000 | 0.000 |
| disjoint_entities | 4 | typed_memory | 1.000 | 0.000 | 0.974 | 0.855 |
| disjoint_entities | 8 | hybrid_memory | 1.000 | 0.000 | 0.916 | 0.797 |
| disjoint_entities | 8 | identity_memory | 0.233 | 0.000 | 0.023 | 0.013 |
| disjoint_entities | 8 | pair_overlap | 0.000 | 1.000 | 0.000 | 0.000 |
| disjoint_entities | 8 | typed_memory | 1.000 | 0.000 | 0.993 | 0.872 |
| pair_reuse | 1 | hybrid_memory | 1.000 | 0.000 | 0.609 | 0.493 |
| pair_reuse | 1 | identity_memory | 1.000 | 0.000 | 0.286 | 0.266 |
| pair_reuse | 1 | pair_overlap | 1.000 | 0.000 | 1.000 | 1.000 |
| pair_reuse | 1 | typed_memory | 1.000 | 0.000 | 0.702 | 0.563 |
| pair_reuse | 2 | hybrid_memory | 1.000 | 0.000 | 0.810 | 0.705 |
| pair_reuse | 2 | identity_memory | 1.000 | 0.000 | 0.404 | 0.387 |
| pair_reuse | 2 | pair_overlap | 1.000 | 0.000 | 2.000 | 2.000 |
| pair_reuse | 2 | typed_memory | 1.000 | 0.000 | 0.908 | 0.787 |
| pair_reuse | 4 | hybrid_memory | 1.000 | 0.000 | 0.920 | 0.802 |
| pair_reuse | 4 | identity_memory | 1.000 | 0.000 | 0.571 | 0.553 |
| pair_reuse | 4 | pair_overlap | 1.000 | 0.000 | 4.000 | 4.000 |
| pair_reuse | 4 | typed_memory | 1.000 | 0.000 | 0.974 | 0.852 |
| pair_reuse | 8 | hybrid_memory | 1.000 | 0.000 | 0.978 | 0.858 |
| pair_reuse | 8 | identity_memory | 1.000 | 0.000 | 0.816 | 0.797 |
| pair_reuse | 8 | pair_overlap | 1.000 | 0.000 | 8.000 | 8.000 |
| pair_reuse | 8 | typed_memory | 1.000 | 0.000 | 0.993 | 0.872 |

## Interpretation

- Best disjoint-setting result: `typed_memory` at support `8` reached
  `accuracy=1.000` with `unresolved_rate=0.000`.
- At the same support, exact pair-overlap reached `accuracy=0.000`
  with `unresolved_rate=1.000`.
- If typed or hybrid relation memory clearly beats pair overlap on disjoint entities,
  then a similarity-based relation-concept subsystem is worth prototyping further.
- If identity-only memory stays weak on disjoint entities, that is evidence that raw
  triple identities alone are not enough; the implementation would need richer context
  features such as role abstractions, graph neighborhoods, or excerpt cues.
