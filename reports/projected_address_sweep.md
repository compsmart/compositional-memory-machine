# Projected Address Key-Family Sweep

Generated on 2026-04-22.

## Command

```powershell
python experiments/exp_projected_address_sweep.py --dim 512 --addr-dims 64 128 256 512 --seeds 0 1 2 --items 300 --probes 120 --noise 0.5
python experiments/exp_projected_address_sweep.py --dim 512 --addr-dims 64 128 256 512 --seeds 0 1 2 --items 300 --probes 120 --noise 1.0
```

This is a bounded repo run, not the full roadmap sweep. The full sweep should
use `dim=2048`, `addr_dim={64,128,256,512,1024,2048}`, larger item counts, and
more stress conditions.

## Result Summary

At `noise=0.5`, all four key families reached 1.0 exact top-1 and 1.0 noisy
top-1 across the tested seeds and address dimensions. The useful signal was
candidate-pool quality:

- One-hot keys cleaned up rapidly as `addr_dim` increased: mean candidates fell
  from about 3.6 at `addr_dim=64` to 1.0 by `addr_dim=256`.
- HRR SVO keys also improved with address dimension: mean candidates fell from
  about 5.6 at `addr_dim=64` to about 1.2 at `addr_dim=512`.
- HRR n-gram keys retained more candidate ambiguity than SVO keys, with mean
  candidates around 2.3-2.5 even at higher address dimensions in this bounded
  setup.
- Continuous context keys had the dirtiest candidate pools: mean candidates
  stayed around 9.6-12.2 and stale contamination stayed near 0.90.

At `noise=1.0`, top-1 degradation first appeared at low address dimensions:

- one-hot `addr_dim=64`: noisy top-1 about 0.97-0.99;
- HRR SVO `addr_dim=64`: noisy top-1 about 0.96-0.98;
- HRR n-gram `addr_dim=64`: noisy top-1 about 0.96-0.98;
- continuous `addr_dim=64`: noisy top-1 about 0.95-0.98.

By `addr_dim=256`, all families returned to 1.0 noisy top-1 in this bounded
run, but continuous keys still carried large candidate pools.

## Interpretation

The result matches the D-2837 caveat: one-hot projected addressing is the easy
case and should not be merged with HRR or continuous embedding claims. The
repo-level harness now exposes this explicitly.

The next full experiment should increase `dim`, item count, and stress level,
then report aggregate means and confidence intervals. The most important metric
to preserve is not only top-1 accuracy but candidate-pool contamination, because
continuous keys can look correct at top-1 while still routing through many stale
near-candidates.

