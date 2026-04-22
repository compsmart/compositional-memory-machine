# Projected SDM Capacity And Write-Policy Sweep

Generated: 2026-04-22

## Purpose

The first projected-address sweep showed that raising `addr_dim` alone did not
fix forgetting under hard-location overwrite. This sweep tests the next likely
variables:

- hard-location count (`n_locations`)
- number of selected locations per key (`k`)
- write policy (`overwrite` vs `sum`)

## Command

```powershell
python experiments/exp_projected_sdm_capacity.py
```

Default protocol:

```text
hrr_dim=2048
addr_dim=512
domains=5
facts_per_domain=40
n_locations in {512, 1024, 2048}
k in {1, 4, 8, 16}
write_mode in {overwrite, sum}
seeds={0,1,2}
```

## Key Results

### Overwrite Writes

| n_locations | k | mean_d1_final | mean_forgetting | mean_all_final |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | 0.7250 | 0.2167 | 0.8350 |
| 512 | 4 | 0.3583 | 0.5917 | 0.6650 |
| 512 | 8 | 0.1500 | 0.8333 | 0.5417 |
| 512 | 16 | 0.0083 | 0.9417 | 0.3617 |
| 1024 | 1 | 0.8167 | 0.1500 | 0.8950 |
| 1024 | 4 | 0.7333 | 0.2667 | 0.8767 |
| 1024 | 8 | 0.6250 | 0.3750 | 0.8733 |
| 1024 | 16 | 0.2667 | 0.7333 | 0.6883 |
| 2048 | 1 | 0.8917 | 0.1000 | 0.9417 |
| 2048 | 4 | 0.9250 | 0.0750 | 0.9767 |
| 2048 | 8 | 0.9583 | 0.0417 | 0.9867 |
| 2048 | 16 | 0.9250 | 0.0750 | 0.9700 |

### Sum / Accumulation Writes

| n_locations | k | mean_d1_final | mean_forgetting | mean_all_final |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | 0.8583 | 0.0833 | 0.8350 |
| 512 | 4 | 1.0000 | 0.0000 | 1.0000 |
| 512 | 8 | 1.0000 | 0.0000 | 1.0000 |
| 512 | 16 | 1.0000 | 0.0000 | 1.0000 |
| 1024 | 1 | 0.9333 | 0.0333 | 0.8950 |
| 1024 | 4 | 1.0000 | 0.0000 | 1.0000 |
| 1024 | 8 | 1.0000 | 0.0000 | 1.0000 |
| 1024 | 16 | 1.0000 | 0.0000 | 1.0000 |
| 2048 | 1 | 0.9750 | 0.0167 | 0.9417 |
| 2048 | 4 | 1.0000 | 0.0000 | 1.0000 |
| 2048 | 8 | 1.0000 | 0.0000 | 1.0000 |
| 2048 | 16 | 1.0000 | 0.0000 | 1.0000 |

## Interpretation

This is the first positive projected-SDM result in the repo.

The main variable is not `addr_dim` by itself. The memory succeeds or fails
based on the interaction between:

- hard-location capacity,
- number of locations touched per key,
- and whether writes overwrite or accumulate.

With overwrite writes, larger `k` can increase damage because every write
clobbers more shared locations. Increasing `n_locations` helps substantially:
`n_locations=2048, k=8` reaches mean forgetting 0.0417 in this PoC.

With sum/accumulation writes, the simplified projected SDM becomes highly
stable: `k>=4` achieves 0.0 forgetting across all tested location counts. This
matches the intuition behind SDM-style superposition: shared locations can
retain multiple memories if writes accumulate and cleanup can separate them.

## Next Step

The next experiment should combine accumulation with more realistic stress:

- more domains,
- more facts per domain,
- noisy queries,
- held-out/familiar n-gram contexts,
- and a cleanup threshold rather than cleanup against every known record.

That will tell us whether the positive `sum` regime is robust or just an easy
cleanup-memory artifact.
