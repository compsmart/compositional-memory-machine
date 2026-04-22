# Projected Address Dimension Sweep

Generated: 2026-04-22

## Purpose

This experiment starts the D-2831/D-2832 critical path inside this repository:
test whether a projected SDM-style address layer behaves differently from the
repo's full-vector AMM.

The new `ProjectedSDM` is intentionally separate from `AMM`. It projects HRR
vectors into a binary address space, writes to shared hard locations, and uses a
cleanup step against stored records.

## Command

```powershell
python experiments/exp_addr_dim_sweep.py
```

Default protocol:

```text
hrr_dim=2048
addr_dim in {64, 128, 256, 512, 1024, 2048}
domains=5
facts_per_domain=40
n_locations=512
k=8 hard locations per write/read
write_mode=overwrite
seeds={0,1,2}
```

## Result

Mean over 3 seeds:

| addr_dim | d1_after_first | d1_final | forgetting | all_final |
| --- | ---: | ---: | ---: | ---: |
| 64 | 0.9750 | 0.1083 | 0.8667 | 0.5383 |
| 128 | 0.9917 | 0.0833 | 0.9083 | 0.5400 |
| 256 | 0.9917 | 0.1250 | 0.8667 | 0.5533 |
| 512 | 0.9833 | 0.1500 | 0.8333 | 0.5417 |
| 1024 | 0.9917 | 0.1000 | 0.8917 | 0.5650 |
| 2048 | 1.0000 | 0.1667 | 0.8333 | 0.5650 |

## Interpretation

This is a negative result, and it is useful.

Under fixed `n_locations=512`, `k=8`, and overwrite-style hard-location writes,
raising `addr_dim` alone does not recover continual retention. Domain 1 is
learned initially, then mostly erased by later domains. This mirrors the
qualitative D-2831/D-2832 warning: compressed/shared address spaces are the
critical risk, and full-vector AMM success should not be confused with projected
SDM success.

The result also sharpens the next question. In this simplified implementation,
failure appears dominated by hard-location load and overwrite policy, not only
the binary address dimensionality. The next sweep should vary:

- `n_locations`
- `k`
- write mode: overwrite vs sum/accumulate
- cleanup strategy
- address radius/candidate threshold

## Contrast With Full-Vector AMM

The existing full-vector AMM experiments remain positive because each key keeps
its own vector record and retrieval scans by cosine similarity. `ProjectedSDM`
is deliberately more collision-prone: many keys write into shared locations.

That distinction is now explicit in code and reports.
