# Projected SDM Trigram Context Prediction

Generated: 2026-04-22

## Purpose

The projected bigram n-gram experiment retrieved seen contexts but over-triggered
on novel contexts. This experiment moves closer to the D-2829 mechanism by using
trigram-style context keys:

```text
(left, right, filler) -> next_token
```

Seen queries reuse a trained filler. Familiar queries keep the same `(left,
right)` bigram but use a new filler. Novel queries use unseen bigrams.

## Command

```powershell
python experiments/exp_projected_sdm_trigram.py
```

Default protocol:

```text
dim=2048
addr_dim=512
n_locations=2048
write_k=8
read_k=16
cycles=5
seeds={0,1,2}
```

## Results

| seed | seen_em | familiar_em | novel_hit_rate | score_cal_familiar | score_cal_novel |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | 0.875 | 1.000 | 0.750 | 0.250 |
| 1 | 1.000 | 0.875 | 0.750 | 0.750 | 0.500 |
| 2 | 1.000 | 0.875 | 0.750 | 0.875 | 0.500 |

Mean scores:

| seed | seen_score | familiar_score | novel_score |
| ---: | ---: | ---: | ---: |
| 0 | 0.9726 | 0.8809 | 0.7622 |
| 1 | 0.9790 | 0.8694 | 0.6300 |
| 2 | 0.9741 | 0.9373 | 0.6465 |

## Interpretation

This is a stronger projected-SDM sequence result than the bigram version.

The explicit filler/noise role creates the intended partial-match structure:
familiar contexts share two of three bound terms with training examples, while
novel contexts change the core bigram. As a result:

- seen contexts are perfect;
- familiar contexts are stable at 0.875 EM;
- score-only calibration keeps 0.75-0.875 familiar EM while reducing novel hits
  to 0.25-0.50.

Margin-based calibration is not yet reliable. Novel contexts sometimes have
larger top-1/top-2 separation than familiar contexts, so margin alone is the
wrong novelty signal in this setup.

## Current Boundary

Projected SDM now has a credible sequence-memory path, but it still does not
fully match full-vector AMM D-2829:

- familiar EM is below 1.0;
- novel hit rate is not zero after calibration;
- novelty detection needs a better signal than score or margin alone.

## Next Step

Use value embeddings and cluster cleanup instead of token labels only. The
system should retrieve a next-token vector, compare it to the vocabulary, and
use score plus vocabulary margin for novelty detection.
