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
- Generation and sequence-memory evidence are now stronger: the repo has a
  compositional generation benchmark and a sequence-chain prefix benchmark that
  connect recent lab findings to concrete local experiments.

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

### 2026-04-22 - D-2838 and D-2839 benchmark pass

Scope:

- Added `experiments/exp_d2838_compositional_generation.py` to benchmark
  compositional value decoding through two routes: HRR-native unbinding plus
  nearest-neighbour cleanup, and a linear ridge-regression head over retrieved
  HRR value vectors.
- Added `experiments/exp_d2839_sequence_chain.py` to benchmark the minimum HRR
  prefix length needed to disambiguate which rule generated a chained sequence.
- Extended `tests/test_experiments.py` with smoke coverage for both new
  benchmarks.
- Updated the roadmap, README, and synthesis docs so `D-2838` and `D-2839`
  are tracked explicitly as lab-grounded findings and local repo capabilities.

Verification:

- `python -m pytest tests/test_experiments.py`
- `python experiments/exp_d2838_compositional_generation.py --summary`
- `python experiments/exp_d2839_sequence_chain.py --summary`

Observed repo-local behavior:

- The current `D-2838` synthetic benchmark reaches 1.0 exact retrieval, 1.0
  HRR-native EM, and 1.0 linear-head EM across
  `D={64,128,256,512,2048}` for the tested 150-entity setup.
- The current `D-2839` benchmark reproduces the sharp structural transition:
  `K={1,2}` stays at 0.25 EM and `K={3,5,7,10}` reaches 1.0 EM across 3 seeds.

Next:

- Decide whether to connect the `D-2838` decoding path to the existing
  `generation` adapter as a structured evidence-to-surface prototype.
- Expand `D-2839` from rule disambiguation into a harder sequence benchmark with
  partially overlapping prefixes or noisy token corruption.

### 2026-04-22 - D-2838 adapter prototype

Scope:

- Added a shared `generation.compositional` decoder module so the HRR-native and
  linear `D-2838` decode paths are available outside the experiment script.
- Extended `FrozenGeneratorAdapter` to answer from retrieved `value_vector`
  evidence when a compositional decoder is available.
- Updated `demo.py` to show a controlled compositional value answer produced from
  retrieved HRR evidence.
- Added `tests/test_generation.py` to regression-test held-out decoding and the
  adapter surface itself.

Verification:

- `python -m pytest tests/test_generation.py tests/test_experiments.py`
- `python demo.py`

Observed repo-local behavior:

- The adapter now produces a controlled compositional answer:
  `entity_demo has property silver signal.`
- The shared decoder keeps the benchmark and adapter paths aligned, so the
  generation prototype exercises the same HRR-native and linear decode logic as
  the `D-2838` experiment.

Next:

- Add a small query layer for non-SVO value memories so compositional answers can
  be retrieved through the same interface as fact queries.
- Stress the adapter path with noisy value vectors and partial-value prompts to
  see where decode quality drops before any stronger generation claims.

### 2026-04-22 - Web UI demo prototype

Scope:

- Added `web.py` as a local HTTP server with an HHR `WebState` that owns seeded
  AMM memory, FactGraph state, ingestion, structured SVO querying, demo reset,
  and compositional-value demo endpoints.
- Added `web_static/index.html`, `web_static/app.css`, and `web_static/app.js`
  to mirror the `nexus-16` dashboard design and 3D fact visualization while
  swapping in HHR-native controls.
- Added `tests/test_web.py` to cover status, facts, structured query, ingestion,
  compositional demo, and graph-export routes.
- Updated README, roadmap, and results docs so the browser UI is tracked as a
  product-shaped demo surface rather than an undocumented side artifact.

Verification:

- `python -m pytest tests/test_web.py`
- `python -m pytest`

Observed repo-local behavior:

- The browser UI now exposes seeded HHR fact memory through the same dashboard
  pattern used in `nexus-16`, including the 3D fact graph and node inspector.
- Structured SVO queries return template answers plus top nearest-neighbour
  evidence, and the graph view focuses on the retrieved answer path.
- The compositional demo card surfaces the shared `D-2838` decoder logic in the
  browser without exposing raw vectors to the frontend.

Next:

- Add a dedicated value-memory query surface so compositional values are queried
  alongside SVO facts instead of only through a fixed demo card.
- Decide whether to add manual fact authoring and graph revision controls to the
  UI or keep the browser demo narrower and benchmark-focused.
