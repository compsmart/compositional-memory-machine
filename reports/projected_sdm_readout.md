# Projected SDM Readout Width Sweep

Generated: 2026-04-22

## Purpose

The projected SDM stress test showed that accumulation writes preserve clean
queries but noisy query vectors degrade retrieval. This experiment tests whether
reading from more hard locations than were written can recover noisy queries.

## Command

```powershell
python experiments/exp_projected_sdm_readout.py
```

Default protocol:

```text
hrr_dim=2048
addr_dim=512
domains=5
facts_per_domain=60
facts=300
n_locations=2048
write_k=8
read_k in {8, 32, 128}
noise in {0.25, 0.5, 0.85}
cleanup=address
seeds={0,1}
```

## Mean Results

| read_k | noise | mean_top1 |
| ---: | ---: | ---: |
| 8 | 0.25 | 0.8967 |
| 8 | 0.50 | 0.5783 |
| 8 | 0.85 | 0.2683 |
| 32 | 0.25 | 0.9817 |
| 32 | 0.50 | 0.8400 |
| 32 | 0.85 | 0.5433 |
| 128 | 0.25 | 1.0000 |
| 128 | 0.50 | 0.9383 |
| 128 | 0.85 | 0.6517 |

## Interpretation

Wider readout is an effective denoising mechanism in this simplified projected
SDM.

The memory writes each fact to `write_k=8` hard locations, but querying with a
larger `read_k` recovers more of the relevant superposed signal when noise
moves the query away from the original address. At `read_k=128`, moderate noise
(`0.5`) still reaches about 94% top-1 accuracy.

This is the first strong robustness result after the D-2831/D-2832 bottleneck:
the failure was not inherent to projected addressing. The read path needs enough
location coverage to tolerate noisy HRR keys.

## Next Step

The next experiment should apply this readout configuration to projected-SDM
n-gram familiar-context prediction:

- train HRR context keys with `write_k=8`,
- query familiar/noisy contexts with `read_k=128`,
- compare seen, familiar, novel, and forgetting metrics.
