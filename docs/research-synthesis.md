# Research Synthesis

Generated: 2026-04-22

This document tracks roadmap and research iterations on `main` so experiment
work stays cumulative, commit-sized, and easy to review.

## Current Position

- Full-vector HRR+AMM behavior remains strong in the repo PoC.
- The projected-address harness is the active critical-path experiment because
  projected routing, not raw HRR width alone, is where the repo needs sharper
  evidence before making stronger SDM-style claims.
- The bounded sweep showed top-1 retrieval remains strong at small scale, but
  stale candidate contamination clearly separates one-hot, HRR SVO, HRR n-gram,
  and continuous keys.

## Iteration Log

### 2026-04-22 - Projected-address serious-run harness

Scope:

- Extended `experiments/exp_projected_address_sweep.py` to sweep multiple item
  counts and noise levels in a single run.
- Added aggregate summaries with mean, minimum, and 95% confidence interval
  reporting across seeds.
- Added markdown report generation so a serious run can write a reusable
  artifact directly into `reports/`.
- Added a `roadmap_serious` preset for the planned `dim=2048`,
  `addr_dim={64,128,256,512,1024,2048}`, multi-noise, larger-item sweep.

Verification:

- `python -m pytest tests/test_experiments.py`

Next:

- Execute the `roadmap_serious` preset and inspect where candidate contamination
  remains high even when top-1 is near saturation.
- Use the aggregate report as the reference point for the next roadmap update.
