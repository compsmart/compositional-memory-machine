# Research Synthesis

Generated: 2026-04-23

This is the living research document for the repo. It replaces the old split
between a separate roadmap and synthesis note: completed work moves into the
iteration log below, and active next steps stay at the bottom as a small todo
list.

## Current Position

- Full-vector HRR+AMM behavior remains strong in the repo PoC.
- The serious projected-address sweep is now complete at
  `dim=2048`, `addr_dim={64,128,256,512,1024,2048}`,
  `families={one_hot,hrr_svo,hrr_ngram,continuous}`,
  `items={500,1000,2000}`, and `noise={0.5,1.0}`.
- The sweep resolved the immediate address-dimension question enough to tighten
  claims: one-hot is the easy case, HRR SVO becomes comfortably clean by about
  `addr_dim=512`, HRR n-gram keeps stronger top-1 than candidate-pool purity at
  mid dimensions, and continuous keys still route through large stale candidate
  pools even when noisy top-1 is perfect.
- The next active track is relation identity and provenance: the Gemini path is
  already normalized, and the current implementation work is extending the same
  registry / provenance shape across direct structured writes.
- The episodic benchmark is now grounded in verified conversation findings:
  dialogue turns carry `speaker`, `intent`, introduced-word, assistant-answer,
  and correction facts while still matching the D-2836 / D-2857 success pattern
  in repo-local verification.

## Claim Boundary

Defensible repo-local claim:

> This repo is a working PoC for continual language-shaped memory. HRR+AMM can
> store, retrieve, revise, and recombine structured facts, while preserving a
> clean boundary between transformer extraction or generation and
> non-transformer memory.

Not supported yet:

> This repo has solved full SDM-style projected-address routing for all key
> families, or acts as a standalone open-ended language model.

Why the projected-address caveat still matters:

- One-hot projected addressing should stay separate from HRR and continuous-key
  claims.
- Strong noisy top-1 is not enough by itself; candidate contamination and stale
  routing still matter, especially for continuous keys.
- The repo's core memory is still full-vector nearest-neighbor AMM, not the
  full positive SDM gating recipe from the lab.

## Iteration Log

### 2026-04-23 - Serious projected-address sweep review

Scope:

- Ran the full serious preset in
  `experiments/exp_projected_address_sweep.py`.
- Wrote the aggregate report to
  `reports/projected_address_sweep_full.md`.
- Reviewed the serious-run outcome against the address-dimension critical-path
  success criteria.

Verification:

- `python experiments/exp_projected_address_sweep.py --preset roadmap_serious --output summary --report-file reports/projected_address_sweep_full.md`

Observed repo-local behavior:

- `one_hot`: effectively clean by `addr_dim=256`, with candidate sets collapsing
  to singleton or near-singleton routing.
- `hrr_svo`: perfect noisy top-1 by `addr_dim>=256`, with candidate-pool purity
  becoming clearly comfortable around `addr_dim=512` and improving further at
  `1024-2048`.
- `hrr_ngram`: perfect noisy top-1 by `addr_dim>=256`, but candidate pools stay
  materially dirtier than SVO until about `1024-2048`.
- `continuous`: noisy top-1 also reaches 1.0 by `addr_dim>=256`, but stale
  contamination remains very high across the whole sweep, staying around
  `0.94-0.98` at the larger settings.

Implication:

- The address-dimension critical path is no longer blocked on "run the serious
  sweep."
- The right project message is now "projected top-1 can work, but routing
  cleanliness remains strongly family-dependent," not "projected SDM routing is
  solved."
- This is enough evidence to move to the next roadmap item while keeping the
  claim boundary explicit.

Next:

- Move to relation registry and provenance as the active implementation track.
- Keep projected-address follow-up narrow: candidate-pool purity or fuller
  SDM-gating reproduction, not another broad sweep immediately.

### 2026-04-22 - Projected-address serious-run harness

Scope:

- Extended `experiments/exp_projected_address_sweep.py` to sweep multiple item
  counts and noise levels in a single run.
- Added aggregate summaries with mean, minimum, and 95% confidence interval
  reporting across seeds.
- Added markdown report generation so a serious run can write a reusable
  artifact directly into `reports/`.
- Added a `roadmap_serious` preset for the intended full run.

Verification:

- `python -m pytest tests/test_experiments.py`

Outcome:

- This harness is now validated by the completed 2026-04-23 serious run above.

### 2026-04-22 - D-2838 and D-2839 benchmark pass

Scope:

- Added `experiments/exp_d2838_compositional_generation.py`.
- Added `experiments/exp_d2839_sequence_chain.py`.
- Extended `tests/test_experiments.py` with smoke coverage.

Verification:

- `python -m pytest tests/test_experiments.py`
- `python experiments/exp_d2838_compositional_generation.py --summary`
- `python experiments/exp_d2839_sequence_chain.py --summary`

Observed repo-local behavior:

- `D-2838` reaches 1.0 exact retrieval, 1.0 HRR-native EM, and 1.0 linear-head
  EM across `D={64,128,256,512,2048}` in the current synthetic setup.
- `D-2839` reproduces the sharp prefix threshold: `K={1,2}` stays at chance and
  `K>=3` reaches 1.0 EM across 3 seeds.

### 2026-04-22 - D-2838 adapter prototype

Scope:

- Added the shared `generation.compositional` decoder module.
- Extended `FrozenGeneratorAdapter` to answer from retrieved `value_vector`
  evidence when a compositional decoder is available.
- Updated `demo.py` and added `tests/test_generation.py`.

Verification:

- `python -m pytest tests/test_generation.py tests/test_experiments.py`
- `python demo.py`

Observed repo-local behavior:

- The adapter now produces the controlled compositional answer
  `entity_demo has property silver signal.`

### 2026-04-22 - Web UI demo prototype

Scope:

- Added `web.py` plus `web_static/` as a local browser workbench.
- Added `tests/test_web.py`.

Verification:

- `python -m pytest tests/test_web.py`
- `python -m pytest`

Observed repo-local behavior:

- The UI exposes seeded fact memory, structured query, text ingestion, chunk
  summaries, graph export, and the compositional demo path in one local surface.

### 2026-04-23 - Relation registry pass for episodic memory

Scope:

- Extended `memory/episodic.py` so episodic facts normalize relations through the
  shared registry before writing to the graph, chunk store, and global AMM.
- Added richer episodic payload provenance, including `source`, `source_id`,
  `source_chunk_id`, `excerpt`, character spans, and sentence index when
  available.
- Added focused alias / revision coverage in `tests/test_episodic.py`.

Verification:

- `python -m pytest tests/test_episodic.py tests/test_experiments.py`

Observed repo-local behavior:

- Episodic writes now collapse alias-equivalent relations such as
  `collaborated with` and `worked on with` onto the same canonical graph edge.
- Episodic evidence payloads now carry the same relation/provenance shape as the
  shared ingestion path, while preserving temporal state tokens for `observed`
  and `revised` events.
- Ingestion stats now expose unresolved normalized relation examples directly, so
  alias growth can be driven from observed misses instead of only summary counts.

### 2026-04-23 - D-2836 dialogue-metadata episodic benchmark

Grounding:

- `D-2836` (discovery): multi-turn conversational memory reached 100% EM across
  immediate recall, distant recall, cross-session recall, in-conversation
  revision, and long-term retention at `D=2048`, `3 sessions`, `10 turns`,
  `3 facts`, `3 seeds`.
- `D-2857` (discovery): PerKey-reset-style revision is the certified mechanism
  for perfect revision and retention.
- `D-2830` (discovery): word-teaching scenarios are a validated language-memory
  primitive, which makes introduced-word / assistant-answer dialogue facts a
  reasonable benchmark extension rather than an arbitrary product feature.

Scope:

- Extended `experiments/exp_d2836_episodic_memory.py` from anonymous synthetic
  facts into a scripted dialogue-turn benchmark with:
  - turn-level `speaker` and `intent` metadata,
  - `introduced_word` user facts,
  - assistant `means` answers,
  - in-conversation correction turns that revise earlier answers.
- Kept the original D-2836 metric family intact while adding dialogue-shaped
  probes for metadata, assistant answers, and correction history integrity.

Verification:

- `python -m pytest tests/test_experiments.py tests/test_episodic.py`
- `python experiments/exp_d2836_episodic_memory.py --dim 2048 --seeds 42 123 7 --sessions 3 --turns 10 --facts-per-turn 3`

Observed repo-local behavior:

- All three seeds reached 1.0 on the original D-2836 metrics:
  `immediate_em`, `distant_em`, `cross_session_em`, `revision_em`,
  `retention_em`.
- The richer dialogue probes also stayed perfect:
  `speaker_intent_em=1.0`, `assistant_answer_em=1.0`, `correction_em=1.0`.
- The repo-local benchmark now exercises a more conversational evidence shape
  without changing the verified research envelope it is supposed to mirror.

## Active Todo List

1. Relation registry and provenance
   - Extend the shared registry / payload shape across the remaining ad hoc
     experiment helpers that still write directly to AMM / FactGraph.
   - Grow aliases from benchmark misses without collapsing distinct relation
     families prematurely.
   - Use the new unresolved-relation examples to drive alias expansion from real
     benchmark misses rather than guessed synonym lists.
2. Episodic conversation memory
   - Move beyond scripted dialogue-turn facts toward controller-driven or
     extracted conversational episodes with the same verified metric structure.
   - Add a benchmark slice for pronoun carryover, conversational self-reference,
     or mixed fact-plus-word-teaching turns without losing the D-2836-style
     exact-match discipline.
3. Large-document memory
   - Run a benchmark over larger extracted corpora with contradiction and refusal
     checks.
4. Codebase memory
   - Parse Python code into explicit graph facts and support dependency-style
     questions.
5. Contradiction and temporal revision
   - Distinguish current truth from historical evidence with explicit source
     provenance.
6. Emergent syntax
   - Push beyond the existing prefix benchmark into broader structural
     generalization.
7. MemoryWorkbench
   - Keep evolving the browser and CLI surfaces into a reusable research
     instrument rather than a fixed demo.
