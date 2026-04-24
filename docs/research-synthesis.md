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
- The frontier research gap is now narrower and more explicit: the repo has a
  direct SDM-side `n_locs` harness, a dynamic overwrite scaling artifact, a
  unified sequential-unbinding sweep, and an order-focused temporal probe, so
  the remaining work is less about missing experiments and more about protocol
  alignment and quality lift.

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

### 2026-04-24 - Scaled structured-wikipedia medical bank

Scope:

- Added a medical routing path for `wikimedia/structured-wikipedia` so rows can
  be filtered with `--medical-only` and assigned to concrete subdomains such as
  `medical.pathogen`, `medical.drug`, `medical.disease`, `medical.specialty`,
  `medical.anatomy`, `medical.procedure`, `medical.symptom`, and `medical.core`.
- Added a starter Wikidata QID override map at
  `data/medical_wikidata_qid_map.json` and wired it into the ingest path as a
  reproducible override layer instead of relying only on inline heuristics.
- Fixed the corpus ingest loop so `StructuredFactRecord.domain` survives
  batching, in-memory normalization, JSONL export, and SQLite deduplication
  instead of collapsing back to one dataset-wide default domain.
- Reworked the structured-wikipedia loader in
  `experiments/exp_hf_corpus_ingest.py` to stream the published zip/jsonl shards
  directly. This avoids the mid-run nested-schema cast failure that appeared in
  the generic `datasets` streaming path.
- Generated a large medical archive at
  `reports/hf_ingest_runs/structured_wikipedia_medical_250k/` from 250k source
  rows.

Verification:

- `python -m pytest`
- `python experiments/exp_hf_corpus_ingest.py --dataset "wikimedia/structured-wikipedia" --config-name "20240916.en" --split train --batch-rows 10000 --max-total-rows 250000 --medical-only --output-dir reports/hf_ingest_runs/structured_wikipedia_medical_250k --sqlite-path reports/hf_ingest_runs/structured_wikipedia_medical_250k/ledger.sqlite`
- `python experiments/exp_conversation_benchmark.py --preset roadmap_serious --preload-jsonl reports/hf_ingest_runs/structured_wikipedia_medical_250k/facts.jsonl --preload-limit 5000 --output summary --results-file reports/conversation_benchmark_latest.json --report-file reports/conversation_benchmark_latest.md`

Observed repo-local behavior:

- The repo test suite passes at `79 passed`.
- The scaled medical archive completed with `input_rows=250000`,
  `mapped_facts=93102`, and `written_facts=93013`.
- The resulting ledger domain mix is currently:
  `medical.pathogen=44877`, `medical.drug=10515`, `medical.core=9445`,
  `medical.disease=8515`, `medical.specialty=7311`,
  `medical.anatomy=6088`, `medical.procedure=4491`,
  `medical.symptom=1771`.
- The serious conversation benchmark stayed flat at
  `mean_score=0.9855` / `pass_rate=0.9565`, which is useful: the larger medical
  bank did not regress the current benchmark surface, but it also did not lift
  the existing frontier or coding slices on this preload size.

### 2026-04-24 - Controller calibration and workbench branching pass

Scope:

- Tightened the frontier research artifacts so `D-2846` now reports explicit
  candidate-read rescue vs read-path failure rates, and `D-2872` now reports
  per-grid reset gains plus short conclusions in the saved markdown artifacts.
- Added saved report/JSON artifacts for `exp_sequential_unbinding_scaling.py`
  and `exp_temporal_ordering_frontier.py`, then used the sequential frontier to
  move `query.py` from coarse `D-2873` hop tiers to a fitted per-hop budget.
- Added a hybrid temporal frontier strategy that keeps latest-state and
  pairwise-order metrics measurable together instead of collapsing latest-state
  to near-zero.
- Improved the web controller with:
  - evidence-backed explanation routing,
  - narrow built-in helpers for coding, logic, puzzle, and sentiment prompts,
  - a multilingual fact-recall path for the benchmark Spanish prompt,
  - a clearer capability overview route.
- Expanded the workbench/query surface with:
  - current-truth, history, and provenance-aware branching chain APIs,
  - checked-in scenario fixtures under `scenarios/`,
  - fixture support in `cli/workbench_cli.py`.

Verification:

- `python -m pytest`
- `python experiments/exp_d2846_sdm_nlocs.py --n-locs 8 16 32 64 128 --gate-betas -3.0 -2.0 -1.0 --route-top-ks 1 3 --json-file research/results/d2846_sdm_nlocs.json --report-file research/results/d2846_sdm_nlocs.md`
- `python experiments/exp_d2872_dynamic_overwrite_scaling.py --json-file research/results/d2872_dynamic_overwrite_scaling.json --report-file research/results/d2872_dynamic_overwrite_scaling.md`
- `python experiments/exp_sequential_unbinding_scaling.py --dims 256 1024 2048 --hop-depths 1 2 3 --syntax-depths 1 2 3 --json-file research/results/sequential_unbinding_scaling.json --report-file research/results/sequential_unbinding_scaling.md`
- `python experiments/exp_temporal_ordering_frontier.py --dims 1024 2048 4096 --events 50 100 200 --json-file research/results/temporal_ordering_frontier.json --report-file research/results/temporal_ordering_frontier.md`
- `python experiments/exp_conversation_benchmark.py --preset roadmap_serious --output summary --results-file reports/conversation_benchmark_latest.json --report-file reports/conversation_benchmark_latest.md`

Observed repo-local behavior:

- The repo test suite now passes at `75 passed`.
- The `D-2846` report now makes the router/read-path split explicit via
  `candidate_read_rescue_rate` and `read_path_failure_rate`, which makes the
  residual mismatch much easier to describe numerically.
- The `D-2872` report now makes reset-vs-no-reset gains explicit per grid,
  with `perkey_reset` staying better throughout the current sweep even though
  the absolute surface remains much harsher than the lab-positive result.
- The sequential-unbinding artifact now exports fitted hop bases directly:
  about `0.444` at `d=256`, `0.913` at `d=1024`, and `1.0` at `d=2048`, and
  `query.py` now uses that fitted story instead of only broad dimension tiers.
- The temporal frontier now includes a hybrid chunked-order plus latest-cache
  strategy, which reaches `balanced_score` near `0.93-1.0` across the current
  test grid and removes the old "latest-state is basically collapsed" problem.
- The serious conversation benchmark improved again to `mean_score=0.986` /
  `pass_rate=0.957`, with `frontier=1.0` and `web_chat=1.0`.
- The remaining coding weakness is no longer the chat controller path; it is
  now mostly the structured-ingest codebase benchmark slice.

### 2026-04-24 - Frontier protocol alignment pass

Scope:

- Tightened `experiments/exp_d2846_sdm_nlocs.py` so the SDM-side reproduction
  now sweeps `gate_beta`, `route_top_k`, and sub-`64` `n_locs` values instead of
  reporting one fixed configuration.
- Extended `memory/sdm.py` query telemetry so the experiment can distinguish the
  routed shard from the candidate read path.
- Reworked `experiments/exp_d2872_dynamic_overwrite_scaling.py` around keyed
  memory semantics, with direct `no_reset` vs `perkey_reset` comparison and
  slot-scoped candidate decoding instead of a single global token pool.
- Refreshed the SDM and overwrite research artifacts and updated the repo docs
  to describe what is now closer to the lab protocol and what is still open.

Verification:

- `python -m pytest tests/test_experiments.py -k "sdm_nlocs or dynamic_overwrite_scaling"`
- `python experiments/exp_d2846_sdm_nlocs.py --n-locs 8 16 32 64 128 --gate-betas -3.0 -2.0 -1.0 --route-top-ks 1 3 --json-file research/results/d2846_sdm_nlocs.json --report-file research/results/d2846_sdm_nlocs.md`
- `python experiments/exp_d2872_dynamic_overwrite_scaling.py --json-file research/results/d2872_dynamic_overwrite_scaling.json --report-file research/results/d2872_dynamic_overwrite_scaling.md`
- `python -m pytest`
- `python experiments/exp_conversation_benchmark.py --preset roadmap_serious --output summary --results-file reports/conversation_benchmark_latest.json --report-file reports/conversation_benchmark_latest.md`

Observed repo-local behavior:

- The SDM-side harness is now informative in two ways instead of one:
  `route_top_k=1` shows the routing floor clearly, while `route_top_k=3`
  recovers near-exact or exact behavior across several low-`n_locs`
  configurations. In the current run, `route_top_k=3` reaches `retrieval_em=1.0`
  at `n_locs=8` and stays `>=0.99` through `16-64`, while the routed-shard hit
  rate remains materially lower.
- This means the repo can now separate "the router picked the wrong shard" from
  "the read path was strong enough once the right shard was in the candidate
  set," which is a real protocol-alignment improvement over the earlier single
  `route_miss_rate`.
- The overwrite scaling artifact now reflects actual keyed overwrite mechanics.
  `perkey_reset` is consistently better than `no_reset` across all tested grids,
  especially at higher entity/update loads, which is much closer to the
  `D-2857` / CI overwrite story than the earlier plain-vector proxy.
- Even after that alignment, the repo-local `D-2872` surface remains much
  harsher than the lab-positive result, so the remaining gap is no longer
  "missing experiment" but "remaining protocol mismatch or representation gap."
- The full repo test suite passes at `72 passed`, and the serious conversation
  benchmark stayed unchanged at `mean_score=0.768` / `pass_rate=0.696`, so the
  protocol-alignment pass improved research fidelity without regressing the
  project scorecard.

Implication:

- `D-2846` is now closer to a protocol-level reproduction and can support
  meaningful floor-comparison work rather than only proving the repo has an SDM
  harness.
- `D-2872` now has a better-shaped repo mirror, but it still should be treated
  as a partial alignment rather than a successful numeric reproduction.

### 2026-04-23 - HRR frontier research pass

Scope:

- Added `memory/sdm.py` plus `experiments/exp_d2846_sdm_nlocs.py` to close the
  repo's missing SDM-side `n_locs` experiment gap with a direct routed-memory
  harness and saved artifact.
- Added `experiments/exp_d2872_dynamic_overwrite_scaling.py` and
  `research/results/d2872_dynamic_overwrite_scaling.md` so dynamic overwrite now
  has an explicit repo-local scaling surface over entities, updates, properties,
  and dimension.
- Added `experiments/exp_sequential_unbinding_scaling.py` to unify relation-hop
  depth and hierarchical syntax depth on the same dimension sweep.
- Added `experiments/exp_temporal_ordering_frontier.py` to separate ordering,
  latest-state, chunking, and overwrite-only behavior in a single temporal
  probe.
- Revised `memory/chunked_kg.py` so the chunk-budget heuristic reflects the
  large-`d` `D-2860` correction instead of extending the simple `D-2858`
  scaling law unchanged above `d=4096`.
- Updated README and claim-boundary docs so `D-2854` and `D-2859` are treated as
  a closed HRR-internal generation branch rather than an open decoder-rescue
  hypothesis.

Verification:

- `python -m pytest`
- `python experiments/exp_d2846_sdm_nlocs.py --json-file research/results/d2846_sdm_nlocs.json --report-file research/results/d2846_sdm_nlocs.md`
- `python experiments/exp_d2872_dynamic_overwrite_scaling.py --json-file research/results/d2872_dynamic_overwrite_scaling.json --report-file research/results/d2872_dynamic_overwrite_scaling.md`
- `python experiments/exp_sequential_unbinding_scaling.py --dims 256 1024 2048 --hop-depths 1 2 3 --syntax-depths 1 2 3`
- `python experiments/exp_temporal_ordering_frontier.py --dims 1024 2048 4096 --events 50 100 200`
- `python experiments/exp_conversation_benchmark.py --preset roadmap_serious --output summary --results-file reports/conversation_benchmark_latest.json --report-file reports/conversation_benchmark_latest.md`

Observed repo-local behavior:

- The repo test suite now passes at `72 passed`.
- The new SDM-side harness is now exact at `n_locs={16,32}` and near-exact at
  `64` (`retrieval_em=0.995`) in the current proxy, which closes the "no repo
  SDM path at all" gap even though it is still not a perfect numeric clone of
  lab `D-2846`.
- The first dynamic overwrite scaling artifact exists and sweeps the intended
  axes, but its current per-key-reset proxy remains much harsher than the lab
  CI result; this is now a protocol-alignment problem rather than a missing
  experiment problem.
- The unified sequential-unbinding sweep now shows the same broad threshold
  pattern across two families: `d=256` is weak, `d=1024` is the first strong
  frontier, and `d=2048` is effectively exact for the relation-chain depth-3
  side while keeping depth-3 hierarchical syntax in the low-to-mid `0.9`s.
- The temporal ordering frontier makes the open problem sharper: overwrite-only
  storage collapses pairwise order to `0.0`, while flat or chunked temporal-role
  storage preserves substantial order signal, so ordering should be treated as a
  distinct representational target rather than folded into generic long-context
  work.
- Large-`d` chunk budgeting is now guarded against the old `D-2858` overread,
  so `d=8192` no longer implicitly inherits an unjustified 2x role-4 capacity
  assumption.
- The serious conversation benchmark stayed unchanged at `mean_score=0.768` /
  `pass_rate=0.696`, so the frontier-research pass did not regress the current
  scorecard while adding the new research surfaces.

Implication:

- The repo now has executable surfaces for all five frontier workstreams from
  the attached plan.
- The next iteration should focus on tightening protocol fidelity for SDM and
  dynamic overwrite, then deciding whether the sequential-unbinding sweep is
  strong enough to justify a data-fitted hop budget in the runtime.

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

1. Relation quality revisit
   - Push typed fallback materially above the current real-corpus `0.25`
     positive-case recovery before widening its rollout posture.
   - Turn alias proposal logs plus canned workbench fixtures into a repeatable
     benchmark-miss review loop rather than an ad hoc manual pass.
2. Structured coding / corpus lift
   - Improve the `codebase_dependency_memory_substrate` slice so the remaining
     coding weakness moves off the structured-ingest benchmark as well as the
     chat controller.
3. Workbench branching/report depth
   - Build richer report views and branch ranking on top of the new
     truth/history/branching endpoints so larger provenance investigations are
     easier to compare and explain.
