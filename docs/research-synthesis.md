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
- Relation identity and provenance now have a concrete next-step shape: exact
  registry hits stay first, and an experimental typed relation fallback exists
  behind a feature flag for unknown relation surfaces.
- The typed relation research result is now split cleanly into two stages:
  synthetic feasibility is strong, but the first curated real-corpus validation
  is mixed. The fallback looks promising enough to keep exploring, but not yet
  strong enough to treat as rollout-ready behavior.
- The episodic benchmark is now grounded in verified conversation findings:
  dialogue turns carry `speaker`, `intent`, introduced-word, assistant-answer,
  and correction facts while still matching the D-2836 / D-2857 success pattern
  in repo-local verification.
- The repo now has a longitudinal conversation benchmark that can be rerun
  after each roadmap item and splits current implemented strengths from
  frontier challenge probes such as logic, coding, multilingual prompts,
  sentiment, and explanation quality.

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

### 2026-04-23 - Remaining active roadmap implementation pass

Scope:

- Strengthened relation normalization review flow with typed-support proposal
  tracking, richer alias candidate logs, and expanded curated real-corpus
  validation coverage.
- Added explicit current-truth vs historical-evidence support in the graph/query
  stack, plus contradiction-aware truth/provenance experiments.
- Added a larger extracted-corpus benchmark, controller-driven episodic episode
  ingestion, Python codebase ingestion, a broader structural generalization
  suite, and MemoryWorkbench snapshot/scenario surfaces with a headless CLI.
- Re-ran the full test suite, refreshed the curated relation fallback report,
  and re-ran the roadmap serious longitudinal benchmark.

Verification:

- `python -m pytest`
- `python experiments/exp_relation_fallback_real_corpus.py --output summary --json-file research/results/relation_fallback_real_corpus.json --report-file research/results/relation_fallback_real_corpus.md`
- `python experiments/exp_conversation_benchmark.py --preset roadmap_serious --output summary --results-file reports/conversation_benchmark_latest.json --report-file reports/conversation_benchmark_latest.md`

Observed repo-local behavior:

- The repo test suite now passes at `59 passed`.
- The serious conversation benchmark improved to `mean_score=0.768` /
  `pass_rate=0.696`, with the implemented track at `mean_score=0.980` /
  `pass_rate=0.941`.
- The relation fallback validation now covers `8` positive and `4` negative
  corpus-style cases. Positive exact canonical recovery is still only `0.25`,
  but the broader overall pass rate improved to `0.500` with fallback on while
  keeping the negative safety slice clean at `1.0`.
- Current truth, superseded history, competing evidence, large-document
  contradiction/refusal behavior, controller-driven episodic turns, codebase
  dependency queries, structural breadth, and workbench snapshot/scenario flows
  are now all repo-local tested surfaces rather than only roadmap notes.

Implication:

- The active roadmap themes from the previous todo list are now implemented as
  concrete repo capabilities and benchmark surfaces.
- The limiting factor has shifted from missing infrastructure to quality lift on
  the remaining hard slices, especially typed relation recovery on realistic
  corpora and frontier chat behaviors.

### 2026-04-23 - Curated real-corpus validation for typed relation fallback

Scope:

- Added `experiments/exp_relation_fallback_real_corpus.py` plus
  `research/results/relation_fallback_real_corpus.md` to validate the
  experimental typed relation fallback on more realistic corpus-style facts.
- Tested both positive alias-style cases and negative safety cases with the
  fallback off vs on.
- Re-ran the roadmap serious conversation benchmark after the iteration to keep
  the project scorecard current.

Verification:

- `python experiments/exp_relation_fallback_real_corpus.py --output summary --json-file research/results/relation_fallback_real_corpus.json --report-file research/results/relation_fallback_real_corpus.md`
- `python experiments/exp_conversation_benchmark.py --preset roadmap_serious --output summary --results-file reports/conversation_benchmark_latest.json --report-file reports/conversation_benchmark_latest.md`

Observed repo-local behavior:

- With fallback off, the curated positive corpus cases achieved
  `pass_rate=0.0` / `exact_canonical_recovery_rate=0.0`.
- With fallback on, the same positive set improved to
  `pass_rate=0.25` / `exact_canonical_recovery_rate=0.25`, meaning the current
  typed fallback recovered one of the four realistic unseen relation surfaces.
- The negative safety cases stayed clean at `pass_rate=1.0` with fallback on,
  so the current thresholds remained conservative rather than over-collapsing.
- The roadmap serious conversation benchmark stayed unchanged at
  `mean_score=0.737` / `pass_rate=0.684`, so this validation pass did not
  regress the broader repo scorecard.

Implication:

- The typed fallback has cleared the "worth exploring" bar but not the "ready
  for broad rollout" bar.
- The current implementation is best understood as an experimental substrate:
  safe enough to keep behind a feature flag, but not yet strong enough on
  realistic corpora to replace conservative unresolved-relation handling.

Next:

- Improve typed relation features or support accumulation so the real-corpus
  recovery rate rises materially above the current `0.25`.
- Validate against larger extracted corpora and harder benchmark misses before
  treating typed fallback as more than an experimental assist path.

### 2026-04-23 - Typed relation fallback feasibility and feature-flag prototype

Scope:

- Added `experiments/exp_relation_concept_memory.py` plus
  `research/results/relation_concept_memory.md` to test whether a `dax`-style
  relation concept memory is worth implementing.
- Compared exact pair-overlap aliasing against identity-only, typed-context, and
  hybrid HRR relation memories on synthetic unseen-relation data.
- Added an experimental staged hybrid path in the main codebase: exact registry
  lookup first, then typed relation fallback for unresolved surfaces, guarded by
  `HHR_ENABLE_TYPED_RELATION_FALLBACK=1` or
  `TextIngestionPipeline(..., enable_typed_relation_fallback=True)`.
- Added focused ingestion coverage showing the fallback stays off by default and
  can map a disjoint unknown relation surface onto a known canonical family when
  enabled.

Verification:

- `python experiments/exp_relation_concept_memory.py --output summary --json-file research/results/relation_concept_memory.json --report-file research/results/relation_concept_memory.md`
- `python -m pytest tests/test_ingestion.py tests/test_query.py tests/test_experiments.py`

Observed repo-local behavior:

- In the synthetic `pair_reuse` setting, exact pair overlap remains perfect and
  should stay the first-stage path.
- In the harder `disjoint_entities` setting, exact pair overlap collapses to
  `accuracy=0.0` / `unresolved_rate=1.0`, while typed relation memory reaches
  `accuracy=1.0` across supports `1,2,4,8`.
- Identity-only relation memory stays near chance on disjoint entities, which
  argues against a pure entity-overlap fallback.
- The best current architecture is therefore staged rather than monolithic:
  deterministic registry first, typed relation fallback second, cautious alias
  registration last.

Implication:

- A typed relation concept subsystem looks worth prototyping further in the repo
  because it solves the specific failure mode that exact overlap cannot touch:
  new relation surfaces on new entity pairs.
- The current feature-flag implementation should still be treated as
  experimental because the positive result is synthetic and cue-rich, not yet a
  broad extracted-corpus benchmark.

Next:

- Validate the typed fallback on real extracted corpora and conversation
  benchmark misses before considering any broader rollout.
- Keep pair-overlap alias growth as the safe exact path for seen relation pairs,
  with typed fallback acting only on misses.

### 2026-04-23 - Longitudinal conversation benchmark

Scope:

- Added `experiments/conversation_benchmark_cases.py` with deterministic case
  definitions spanning memory, multi-hop, temporal, canonical meanings,
  explanation, logic, puzzles, trick questions, multilingual prompts, coding,
  sentiment, and general context handling.
- Added `experiments/exp_conversation_benchmark.py` with `run(...)`,
  `summarize(...)`, markdown report generation, JSON result export, and
  previous-run delta comparison.
- Added smoke coverage in `tests/test_experiments.py`.
- Wrote the first benchmark artifacts to
  `reports/conversation_benchmark_latest.json` and
  `reports/conversation_benchmark_latest.md`.

Verification:

- `python -m pytest tests/test_experiments.py`
- `python experiments/exp_conversation_benchmark.py --preset roadmap_serious --output summary --results-file reports/conversation_benchmark_latest.json --report-file reports/conversation_benchmark_latest.md`

Observed repo-local behavior:

- Implemented capabilities score perfectly in the first serious benchmark pass:
  `implemented=1.0`, with strong `memory`, `multi_hop`, `temporal`,
  `general_context`, `canonical_meanings`, `language_patterning`, and
  `trick_questions` slices.
- The broader challenge track is intentionally weak today:
  `frontier=0.167`, with `explanation_understanding=0.5`,
  `multilingual=0.5`, and `coding`, `logic`, `puzzles`, plus `sentiment` all
  at `0.0`.
- The combined serious preset currently lands at `mean_score=0.737` and
  `pass_rate=0.684`, which gives the project a stable before/after scorecard
  for future roadmap work.

Implication:

- The repo now has a reusable benchmark that measures whether conversational
  capability is actually broadening over time instead of only checking that
  today's implemented paths still pass focused tests.
- The benchmark sharpens the current product boundary: the system is strong on
  memory-backed structured dialogue, but broader reasoning and open-ended
  assistant behaviors remain roadmap gaps rather than hidden assumptions.

Next:

- Re-run the benchmark after each roadmap item and compare against the previous
  JSON artifact to make regressions and capability gains explicit.
- Use the current lowest-scoring categories (`coding`, `logic`, `puzzles`,
  `sentiment`, richer `explanation`, and multilingual answer quality) as the
  clearest expansion targets for future conversational work.

### 2026-04-23 - Shared structured-write path for chain helpers

Scope:

- Switched the remaining chain-oriented helper paths in
  `tests/test_query.py` and `experiments/exp_chunked_multihop.py` from
  hand-rolled AMM / FactGraph writes to `TextIngestionPipeline.write_structured_fact`.
- Aligned those helpers with the shared relation registry and provenance payload
  shape already used by Gemini ingestion, the web surface, and episodic memory.
- Added a query-chain regression that writes an alias relation through the shared
  path and verifies canonical graph routing plus stored provenance.

Verification:

- `python -m pytest tests/test_query.py tests/test_experiments.py`

Observed repo-local behavior:

- Multi-hop chain helpers now store the same normalized relation fields and
  provenance envelope as other structured-ingest paths instead of bypassing
  them.
- Alias-written edges such as `worked on with` now land under the canonical
  `works_with` graph edge while preserving the raw relation in memory payloads.

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

1. Relation quality lift
   - Push the typed fallback positive-case recovery materially above the current
     `0.25` on realistic corpora before widening rollout beyond the feature
     flag.
   - Turn alias proposal logs into a benchmark-miss review loop so accepted
     aliases come from observed misses rather than manual guesswork.
2. Workbench and corpus ergonomics
   - Add canned scenario fixtures and richer export/report views so the
     MemoryWorkbench can reproduce larger benchmark investigations with less
     manual setup.
3. Frontier benchmark lift
   - Use the stronger structured-memory substrate to improve explanation,
     coding, multilingual, logic, puzzle, and sentiment behaviors without
     blurring the retrieval-vs-generation claim boundary.
