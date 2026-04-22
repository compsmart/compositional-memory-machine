# Projected SDM Accumulation Stress

Generated: 2026-04-22

## Purpose

The capacity sweep found that projected SDM can retain facts when writes
accumulate instead of overwrite. This experiment stresses that positive regime
with:

- 8 domains
- 100 facts per domain
- 800 total facts
- noisy queries
- global cleanup vs address-gated cleanup

## Command

```powershell
python experiments/exp_projected_sdm_stress.py
```

Default protocol:

```text
hrr_dim=2048
addr_dim=512
n_locations=2048
k=8
write_mode=sum
cleanup in {global, address}
noise in {0.0, 0.25, 0.5, 0.85}
seeds={0,1,2}
```

## Mean Results

Global cleanup:

| noise | mean_d1_final | mean_all_final | mean_forgetting |
| ---: | ---: | ---: | ---: |
| 0.00 | 1.0000 | 1.0000 | 0.0000 |
| 0.25 | 0.8933 | 0.8571 | 0.1067 |
| 0.50 | 0.5000 | 0.5250 | 0.5000 |
| 0.85 | 0.2000 | 0.2279 | 0.8000 |

Address-gated cleanup:

| noise | mean_d1_final | mean_all_final | mean_forgetting |
| ---: | ---: | ---: | ---: |
| 0.00 | 1.0000 | 1.0000 | 0.0000 |
| 0.25 | 0.8967 | 0.8575 | 0.1033 |
| 0.50 | 0.5000 | 0.5254 | 0.5000 |
| 0.85 | 0.2000 | 0.2283 | 0.8000 |

## Interpretation

The accumulation regime remains strong at 800 facts with clean queries:
zero forgetting and perfect final retrieval for both global and address-gated
cleanup.

The limiting factor in this stress test is query noise, not cleanup scope.
Address-gated cleanup and global cleanup are nearly identical, which means the
query address still surfaces the correct candidate set at this load. Accuracy
drops as noisy vectors move away from the original HRR address and content
direction.

## Consequence

The next research step is not another clean-query retention test. The next step
should be robustness:

- denoising / cleanup memories for noisy HRR queries,
- larger `k` or radius-style read candidates for noisy address recovery,
- relation-aware query reconstruction,
- and n-gram familiar-context stress under projected SDM.
