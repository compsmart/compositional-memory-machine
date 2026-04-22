# Projected SDM N-Gram Prediction

Generated: 2026-04-22

## Purpose

Test whether the D-2829 next-token primitive survives projected SDM retrieval
using the stronger configuration discovered in earlier sweeps:

```text
write_mode=sum
write_k=8
read_k=128
n_locations=2048
addr_dim=512
```

## Command

```powershell
python experiments/exp_projected_sdm_ngram.py
```

## Result

| seed | seen_em | familiar_em | novel_hit_rate | calibrated_familiar_em | calibrated_novel_hit_rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.00 | 0.75 | 1.00 | 0.50 | 0.50 |
| 1 | 1.00 | 0.25 | 1.00 | 0.25 | 0.25 |
| 2 | 1.00 | 1.00 | 1.00 | 0.50 | 0.25 |

Mean scores:

| seed | seen_score | familiar_score | novel_score |
| ---: | ---: | ---: | ---: |
| 0 | 0.8391 | 0.6673 | 0.6883 |
| 1 | 0.8392 | 0.7305 | 0.5862 |
| 2 | 0.8047 | 0.7042 | 0.6600 |

## Interpretation

This is a mixed boundary result.

Projected SDM retrieves seen n-gram contexts perfectly, so the memory backend can
store and recover learned sequence keys. However, familiar-context
generalization is unstable and novel contexts over-trigger. The familiar and
novel score distributions overlap, so a simple confidence threshold cannot
cleanly separate "known pattern with altered context" from "unseen bigram."

This differs from the full-vector AMM D-2829-style experiment, where seen and
familiar contexts both pass and novel contexts fail honestly. Projected SDM
needs an additional discriminator or context representation before it can match
that behavior.

## Next Step

Try one or more of:

- trigram keys with explicit noise/filler role, matching D-2829 more closely;
- separate value memory for next-token embeddings instead of payload-only labels;
- margin-based novelty detection using top-1 vs top-2 cleanup scores;
- relation between readout width and false-positive novel hits;
- domain-separated address projections for sequence memory.
