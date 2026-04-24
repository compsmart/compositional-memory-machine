# Compositional Memory Machine

This repository is a proof of concept for a language-memory architecture built
from Holographic Reduced Representations (HRR), append-oriented associative
memory (AMM), a small FactGraph revision layer, and optional Gemini 2.5 Flash
Lite extraction.

It is not a standalone GPT-style language model. Gemini currently handles raw
text extraction. The non-transformer core stores, retrieves, revises-adjacent,
predicts, and learns over structured HRR representations.

```text
raw text or structured facts
        |
        v
Gemini extraction, or direct triples
        |
        v
HRR encoders
        |
        +-------------------+--------------------+
        |                   |                    |
        v                   v                    v
AMM fact memory      FactGraph revision    HRR language primitives
sentence storage     local graph updates   n-gram + word learning
        |
        v
structured retrieval
        |
        v
template answer or optional generator
```

## What Works

- HRR SVO sentence encoding and retrieval.
- Append-oriented AMM fact storage.
- Chunked HRR knowledge-graph storage with per-chunk provenance and bridge-entity tracking.
- Dimension-aware chunk sizing derived from finding-backed HRR capacity budgets.
- Compositional routing for controlled SVO facts.
- FactGraph local revision via PerKey Reset.
- Multi-hop chain querying that combines symbolic graph traversal with chunk-local HRR evidence.
- Gemini 2.5 Flash Lite real-text extraction into triples.
- HRR bigram-context next-token prediction.
- Ranked probabilistic continuations for multi-modal next-token contexts.
- Context-based word learning by ACTION-role unbinding.
- Compositional value generation via HRR-native unbinding and a linear decoding head.
- Sequence-chain prefix disambiguation for variable-length HRR rule prefixes.
- A frozen generation adapter that can decode retrieved compositional value
  vectors into 2-token outputs in the controlled D-2838 setting.
- A browser UI demo with the `nexus-16` dashboard design and 3D fact
  visualization, adapted to HHR-native query and ingestion flows.
- Scripted memory-grounded conversation demo.
- Projected-address sweep harness for one-hot, HRR SVO, HRR n-gram, and
  continuous context keys.
- D-2836-style episodic conversation benchmark with persistent sessions,
  revision, cross-session recall, final retention checks, and dialogue-turn
  metadata / correction probes.
- A longitudinal conversation benchmark that separates current implemented
  capabilities from frontier challenge probes such as logic, coding,
  multilingual prompts, sentiment, and explanation quality.
- Relation registry and provenance payloads for extracted triples and episodic
  writes.
- An experimental typed relation fallback, gated behind
  `HHR_ENABLE_TYPED_RELATION_FALLBACK=1`, that can map disjoint unseen relation
  surfaces onto known canonical relations in the focused repo-local prototype.
- Alias proposal logs that track pending vs accepted relation candidates from
  pair overlap and typed fallback evidence.
- Current-truth vs historical-evidence queries with explicit provenance and
  competing-claim tracking.
- A larger extracted-corpus benchmark with contradiction and refusal checks.
- Controller-driven episodic episode ingestion in addition to the scripted
  D-2836-style dialogue benchmark path.
- Evidence-backed controller routing for explanation prompts plus narrow
  built-in helpers for benchmark-style coding, logic, puzzle, sentiment, and
  multilingual recall prompts.
- Python codebase ingestion into graph facts for dependency-style questions.
- A structural generalization suite that now combines prefix, hierarchical, and
  chat-surface pattern tasks.
- MemoryWorkbench snapshot export, checked-in scenario fixtures, truth/history
  query routes, provenance-aware branching chain routes, and a headless CLI
  runner with fixture selection.

## Research Grounding

The design is grounded in Compsmart AI Research Lab findings:

- D-2824: HRR sentence memory in AMM achieved 100% retrieval and 0% forgetting.
- D-2825: HRR+AMM achieved 100% compositional generalisation on held-out pairs.
- D-2829: AMM worked as a primitive n-gram predictor for seen/familiar contexts.
- D-2830: HRR+AMM learned pseudoword meanings from a few context examples.
- D-2820: append-only AMM protection blocks in-place fact revision.
- D-2821/D-2826: PerKey Reset supports no-cascade FactGraph revision.
- D-2823: HSM/kappa-gate CI protection was unreliable, so it is not used here.
- D-2827: low-dimensional HRR causes address collisions.
- D-2831/D-2832: SDM `addr_dim=64` is a critical projection bottleneck for
  HRR/embedding keys; `d_hrr=2048` alone is not enough for projected SDM CI.
- D-2835: full-vector HRR+AMM capacity saturates by `D=256` in the tested
  conditions; `D=128` is the robust minimum and `D=64` is exact/partial viable.
- D-2836: controlled multi-turn conversational memory achieved 100% EM across
  immediate, distant, cross-session, revision, and retention probes.
- D-2837: SDM+AMM `beta0` reaches zero forgetting at 10+ domains for one-hot
  keys, so projected-address experiments must separate key families.
- D-2838: HRR+AMM supports exact 2-token property generation from retrieved
  compositional values, both through HRR-native unbinding and a linear ridge
  head trained on held-out entities.
- D-2839: chained HRR sequence prefixes show a hard disambiguation threshold:
  1-2 tokens are ambiguous, while 3 tokens are sufficient for perfect rule
  identification in the tested setup.
- D-2847/D-2857: clean belief revision is supported both by pure HRR subtract
  and by PerKey reset in the CI-style language-memory setting.
- D-2849: HRR weighted superposition can represent probabilistic multi-modal
  continuations, not just a single retrieved next token.
- D-2850/D-2851/D-2852: temporal role binding, pragmatic roles, and chunked
  long-context narrative memory are all strong lab signals for HRR as a richer
  language-memory substrate.
- D-2854/D-2859: multi-step autoregressive generation fails even when
  single-step retrieval succeeds, and the follow-up decoder-rescue branch also
  fails, which closes the repo's HRR-internal generation path.
- D-2855/D-2856: recursive syntax and explicit failure-boundary / aliasing
  measurements are now part of the repo validation story rather than only lab
  motivation.
- D-2858/D-2860: HRR chunk capacity gives a concrete bounded chunk-design law,
  but the simple linear ratio needs a large-`d` revision above `4096`.
- D-2869: multi-hop accuracy follows a k-hop factorization law, which supports
  path confidence budgeting in the query engine.
- D-2862/D-2863: temporal role binding and hierarchical syntax are viable in the
  CI setting, motivating explicit temporal and nested-role extensions.
- D-2866/D-2870/D-2871/D-2874: chunked HRR memory enables reliable multi-hop
  retrieval across bounded local fact sets and many domains.
- D-2872/D-2873: dynamic state tracking and sequential unbinding both benefit
  from higher dimensions, with `D=1024` as the first strong frontier and
  `D=2048` as the robust target for 2-hop to 3-hop style behavior.
- L-932/L-933: benchmark design must avoid misleading full-tuple metrics and
  must use shared entity pools plus explicit chain construction for multi-hop
  evaluation.

See [docs/research-synthesis.md](docs/research-synthesis.md) for the living
research synthesis, completed roadmap items, claim boundary, and next research
tracks. See [reports/lab_claim_validation.md](reports/lab_claim_validation.md)
for the repo-vs-lab claim ledger and the validation status for the current
scorecard.

## Current Results

Verified locally:

```text
python -m pytest
75 passed
```

Representative outcomes:

- CI storage at `d=2048`: 100% top-1 retrieval across 3 seeds and 10 cycles.
- Composition at `d=2048`: 100% cluster-EM across 3 seeds.
- Address-routed stress: about 0.87-0.92 at `d=512` vs 0.98-0.99 at `d=2048`.
- D-2829-style next-token primitive: seen EM 1.0, familiar EM 1.0, novel hit
  rate 0.0.
- D-2849-style probabilistic continuation: ranked next-token alternatives now
  preserve weighted ordering for multi-modal contexts such as `the artist ...`.
- D-2830-style word learning: cluster routing 1.0 and retention 1.0.
- D-2858-informed chunk budgets: chunk sizing now derives from dimension and
  role complexity instead of a fixed heuristic.
- Large-`d` chunk budgeting now applies a `D-2860`-style revision for
  4-role memory above `d=4096`, so `d=8192` no longer incorrectly looks like a
  2x capacity jump over `d=4096`.
- D-2838-style compositional generation: repo benchmark reaches 1.0 exact
  retrieval, 1.0 HRR-native EM, and 1.0 linear-head EM across
  `D={64,128,256,512,2048}`.
- D-2839-style sequence chains: `K={1,2}` stays at 0.25 EM (chance across 4
  rules), while `K={3,5,7,10}` reaches 1.0 EM across 3 seeds.
- Chunked multi-hop benchmark: 2-hop, 3-hop, and cross-domain 4-step chain
  retrieval all reach 1.0 EM in the repo smoke test while preserving chunk
  provenance.
- Query budgeting now uses fitted hop bases from the sequential-unbinding
  frontier sweep, so weak multi-hop paths refuse instead of over-claiming a
  result.
- Temporal state tracking benchmark: current state, state history, and
  historical lookup all reach 1.0 EM in the repo smoke test.
- D-2850 mirror: flat 4-role temporal binding stays clean through `n=50`, then
  falls sharply by `n=200`, reproducing the same qualitative capacity wall as
  the lab.
- D-2851 mirror: pragmatic roles stay stronger than core roles at `n=50`
  (`nuanced_acc=1.0` vs `core_acc=0.98` in the repo-local mirror).
- D-2852 mirror: chunked narrative storage reaches `recall=1.0` and
  `latest_state=1.0` at `n=200`, while flat storage degrades materially.
- D-2854 mirror: multi-step generation remains below `seq_em=0.2` for all
  tested decoding strategies, preserving the retrieval-only architectural
  boundary.
- D-2859 boundary update: the README and research synthesis now treat
  decoder-assisted HRR generation as closed rather than still-open future work.
- D-2855 mirror: recursive syntax remains usable through depth 3 at `n=25`
  (`main_acc=0.92` in the repo-local mirror).
- D-2846 repo harness: `research/results/d2846_sdm_nlocs.md` now sweeps
  `n_locs`, `gate_beta`, and `route_top_k`, and it separates routed-shard
  quality from candidate-read quality. In the current run, `route_top_k=3`
  stays exact at `n_locs=8` and near-exact through `16-64`, while
  `route_top_k=1` exposes the routing-floor degradation the earlier proxy hid.
- Sequential unbinding frontier sweep: the new unified scaling run shows
  relation-chain depth `2` already strong at `d=1024` (`0.85-0.90` across
  seeds), while `d=2048` reaches `1.0` through relation depth `3` and keeps
  depth-3 hierarchical syntax in the `0.92-0.96` range.
- Temporal ordering frontier: the hybrid chunked-order plus latest-cache
  strategy now keeps latest-state and pairwise ordering measurable together,
  with balanced scores around `0.93-1.0` across the current grid.
- Dynamic overwrite scaling now has a keyed-memory repo artifact at
  `research/results/d2872_dynamic_overwrite_scaling.md`; it now compares
  `no_reset` against `perkey_reset` directly with slot-scoped decode candidates.
  `perkey_reset` is consistently better than `no_reset`, but the current repo
  protocol is still materially harsher than the lab-positive `D-2872` result.
- Temporal ordering now has a dedicated frontier protocol: overwrite-only memory
  drops pairwise order to `0.0`, while flat or chunked temporal-role storage
  retains substantial order signal, making the ordering problem more specific
  than generic long-context failure.
- D-2856 mirror: near-duplicate failure appears only at very high similarity
  (`sim=0.95`), while overwrite without reset collapses toward a `50/50` blend.
- D-2857 mirror: `perkey_reset` restores `revised_em=1.0` and
  `retained_em=1.0`, while no-reset revision stays far lower.
- D-2836 dialogue benchmark extension: the repo-local scripted dialogue run now
  keeps `speaker_intent_em=1.0`, `assistant_answer_em=1.0`, and
  `correction_em=1.0` across 3 seeds while preserving the original D-2836
  recall / revision / retention metrics at 1.0.
- Longitudinal conversation benchmark (`roadmap_serious` preset): implemented
  track currently scores `0.980`, frontier challenge track scores `1.000`, and
  the combined scorecard lands at `0.986` mean score / `0.957` pass rate; see
  [reports/conversation_benchmark_latest.md](reports/conversation_benchmark_latest.md).
- Relation concept memory research: in the synthetic disjoint-entity setting,
  exact pair overlap fell to `0.0` accuracy / `1.0` unresolved rate, while the
  typed relation memory prototype reached `1.0` accuracy across supports
  `1,2,4,8`; see
  [research/results/relation_concept_memory.md](research/results/relation_concept_memory.md).
- Curated real-corpus validation is more mixed: the current typed fallback
  improved unseen canonical recovery from `0.0` to `0.25` across eight positive
  corpus-style cases while keeping four negative safety cases clean; see
  [research/results/relation_fallback_real_corpus.md](research/results/relation_fallback_real_corpus.md).
- Truth-history benchmark: current truth, historical evidence, competing claims,
  and unresolved-refusal checks all reach `1.0` in the repo smoke test.
- Large-document benchmark: recall, chain retrieval, contradiction tracking, and
  refusal all reach `1.0` in the repo smoke test while preserving chunked
  provenance.
- Codebase memory benchmark: Python imports, calls, and symbol ownership queries
  all reach `1.0` in the repo smoke test.
- Structural generalization suite: prefix threshold, hierarchical syntax, and
  chat-surface pattern completion now report together as a single breadth score.
- The demo now includes a compositional value answer generated from a retrieved
  HRR value vector: `entity_demo has property silver signal.`
- The web UI now mirrors the `nexus-16` dashboard style while exposing HHR
  structured querying, text ingestion, resettable seed memory, chunk summaries,
  chain queries, and the same 3D fact visualization pattern.
- FactGraph chain3 revision: 100% exact match across tested positions.
- Projected-address roadmap sweep: top-1 stayed strong across the serious
  `dim=2048` run, but candidate contamination still separated one-hot, HRR SVO,
  HRR n-gram, and continuous key families; see
  [reports/projected_address_sweep_full.md](reports/projected_address_sweep_full.md).

Important boundary: the repo's core AMM is currently full-vector
nearest-neighbor memory. The serious projected-address sweep sharpened the same
claim boundary rather than removing it: one-hot and HRR families can stay clean
at higher `addr_dim`, while continuous keys still carry heavy stale-candidate
contamination even when noisy top-1 is perfect. D-2837's positive one-hot SDM
result should therefore stay separate from HRR and continuous embedding key
families. Relation normalization is now registry-based, with an experimental
typed fallback behind a feature flag. The next roadmap step is to validate that
fallback on real extracted corpora and benchmark misses before expanding it
beyond a conservative experimental path.

## Quick Start

Run tests:

```powershell
python -m pytest
```

Run the longitudinal conversation benchmark:

```powershell
python experiments/exp_conversation_benchmark.py --preset roadmap_serious --output summary --results-file reports/conversation_benchmark_latest.json --report-file reports/conversation_benchmark_latest.md
```

Use `--preset smoke` for a fast regression-oriented slice, or pass
`--compare-to <prior-results.json>` to show score deltas against an earlier run.

Run the scripted conversation demo:

```powershell
python conversation_demo.py
```

`conversation_demo.py` uses Gemini-backed text extraction, so it requires
`GOOGLE_API_KEY` or `GEMINI_API_KEY`.

Run the web UI:

```powershell
python web.py
```

Then open `http://127.0.0.1:8765`.

Run core experiments:

```powershell
python experiments/exp_d2824_ci_storage.py
python experiments/exp_d2825_composition.py
python experiments/exp_d2827_dimension_sweep.py
python experiments/exp_collision_stress.py
python experiments/exp_d2829_next_token.py
python experiments/exp_d2849_probabilistic_next_token.py
python experiments/exp_d2830_word_learning.py
python experiments/exp_d2838_compositional_generation.py --summary
python experiments/exp_d2839_sequence_chain.py --summary
python experiments/exp_chunked_multihop.py
python experiments/exp_conversation_benchmark.py --preset smoke --output summary
python experiments/exp_revision_chain3.py
python experiments/exp_projected_address_sweep.py
python experiments/exp_d2836_episodic_memory.py
python experiments/exp_temporal_state_tracking.py
python experiments/exp_d2850_temporal_role_binding.py
python experiments/exp_d2851_pragmatic_roles.py
python experiments/exp_d2852_narrative_chunking.py
python experiments/exp_d2854_generation_boundary.py
python experiments/exp_d2855_hierarchical_syntax.py
python experiments/exp_d2856_failure_boundary.py
python experiments/exp_d2857_language_revision.py
python experiments/exp_truth_provenance_conflicts.py
python experiments/exp_large_document_memory.py
python experiments/exp_codebase_memory.py
python experiments/exp_structural_generalization.py
```

Run the full projected-address roadmap sweep and write an aggregate report:

```powershell
python experiments/exp_projected_address_sweep.py --preset roadmap_serious --output summary --report-file reports/projected_address_sweep_full.md
```

Run real-text ingestion:

```powershell
python real_text_demo.py
```

Ingest arbitrary text:

```powershell
python ingest_text.py --text "Ada Lovelace worked with Charles Babbage." --domain history --probe-subject "Ada Lovelace" --probe-relation worked_with --probe-object "Charles Babbage"
```

Gemini ingestion requires `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

Generate structured fact archives from Hugging Face corpora:

1. Install the optional HF dependency:

```powershell
python -m pip install -e ".[hf]"
```

2. Run the batched ingest script. It maps supported dataset rows into
   `ExtractedFact` records, writes them to `facts.jsonl`, and optionally uses a
   SQLite ledger for deduplication and resume state. The
   `wikimedia/structured-wikipedia` path streams the published zip/jsonl shards
   directly so long runs are not blocked by nested-schema drift across shards.

Example with `wikimedia/structured-wikipedia`:

```powershell
python experiments/exp_hf_corpus_ingest.py `
  --dataset "wikimedia/structured-wikipedia" `
  --config-name "20240916.en" `
  --split train `
  --batch-rows 1000 `
  --max-total-rows 10000 `
  --output-dir reports/hf_ingest_runs/structured_wikipedia_10k `
  --sqlite-path reports/hf_ingest_runs/structured_wikipedia_10k/ledger.sqlite
```

Medical-only example with domain routing:

```powershell
python experiments/exp_hf_corpus_ingest.py `
  --dataset "wikimedia/structured-wikipedia" `
  --config-name "20240916.en" `
  --split train `
  --batch-rows 10000 `
  --max-total-rows 250000 `
  --medical-only `
  --qid-allowlist data/medical_wikidata_qid_map.json `
  --output-dir reports/hf_ingest_runs/structured_wikipedia_medical_250k `
  --sqlite-path reports/hf_ingest_runs/structured_wikipedia_medical_250k/ledger.sqlite
```

The medical mode keeps only rows with medical signals, assigns them to
subdomains such as `medical.disease`, `medical.drug`, `medical.pathogen`, and
`medical.specialty`, and writes the final domain onto each JSONL record so the
archive can be replayed faithfully later.

Important outputs:

- `reports/hf_ingest_runs/<run>/facts.jsonl`: replayable structured fact archive
  for HHR.
- `reports/hf_ingest_runs/<run>/progress.json`: batch-level ingest summary and
  resume metadata.
- `reports/hf_ingest_runs/<run>/ledger.sqlite`: optional deduplicated ingest
  ledger and progress cursor.

Supported dataset ids in the current adapter layer:

- `Jotschi/wikipedia_knowledge_base_en`
- `Jotschi/wikipedia_knowledge_graph_en`
- `Wikimedians/wikidata-all`
- `wikimedia/structured-wikipedia`

Current mapping behavior:

- Jotschi KB rows are converted into conservative `title described_by fact_text`
  records.
- Jotschi KG rows are converted into direct `entity_a rel entity_b` triples.
- Wikidata rows are converted from statement claims, with
  `--max-claims-per-entity` available to cap per-entity expansion.
- Structured Wikipedia rows currently emit `described_by` plus conservative
  `infobox_has` facts when available.
- Structured Wikipedia medical runs can filter to medical pages with
  `--medical-only`, use `data/medical_wikidata_qid_map.json` as a starter
  QID-to-domain override map, and preserve the emitted `medical.*` domain in
  both `facts.jsonl` and `ledger.sqlite`.

Then preload the generated facts into the web UI:

```powershell
python web.py `
  --preload-jsonl reports/hf_ingest_runs/structured_wikipedia_10k/facts.jsonl `
  --preload-limit 5000
```

Or run the conversation benchmark against the same fact archive:

```powershell
python experiments/exp_conversation_benchmark.py `
  --preset roadmap_serious `
  --preload-jsonl reports/hf_ingest_runs/structured_wikipedia_10k/facts.jsonl `
  --preload-limit 5000 `
  --output summary `
  --results-file reports/conversation_benchmark_hf_10k.json `
  --report-file reports/conversation_benchmark_hf_10k.md
```

Notes:

- Keep `--dim` and `--seed` aligned across ingest, web preload, and benchmark
  runs if you override the defaults.
- Large corpora are meant to stay on disk as JSONL and be replayed in bounded
  slices with `--preload-limit` rather than loaded fully into one web process.
- Keep `facts.jsonl` as the portable replay artifact for HHR and benchmarks.
  Use SQLite as the scalable deduplication and resume layer, not as a
  replacement for the JSONL archive.
- `HuggingFaceFW/finewiki` is intentionally not part of this structured ingest
  path; it would need raw-text extraction rather than direct triple mapping.

Run the headless MemoryWorkbench scenario CLI:

```powershell
python cli/workbench_cli.py --scenario path\\to\\scenario.json --output snapshot.json
python cli/workbench_cli.py --list-fixtures
python cli/workbench_cli.py --fixture ada_memory --output snapshot.json
```

Use the browser workbench snapshot and scenario APIs:

- `GET /api/snapshot`
- `POST /api/scenario/load`
- `POST /api/query/current-truth`
- `POST /api/query/history`
- `POST /api/query/branching-chain`

Enable the experimental typed relation fallback:

```powershell
$env:HHR_ENABLE_TYPED_RELATION_FALLBACK = "1"
python conversation_demo.py
```

The flag keeps exact registry hits first and only uses typed relation memory on
unresolved surfaces.

Run the relation concept memory feasibility study:

```powershell
python experiments/exp_relation_concept_memory.py --output summary --json-file research/results/relation_concept_memory.json --report-file research/results/relation_concept_memory.md
```

Run the curated real-corpus fallback validation:

```powershell
python experiments/exp_relation_fallback_real_corpus.py --output summary --json-file research/results/relation_fallback_real_corpus.json --report-file research/results/relation_fallback_real_corpus.md
```

## Example Output

```text
User: Read this short passage and remember it.
Assistant: I extracted 6 distinct facts and wrote 6 of them into HRR memory.
User: Who did Ada Lovelace work with?
Assistant: Ada Lovelace worked with Charles Babbage. Confidence: 1.000.
User: Complete this learned pattern: 'the doctor ...'
Assistant: The next token is 'treats' from context 'the doctor' (confidence 1.000).
User: Now learn a new word: dax. A child daxes an apple; a chef daxes soup; a bird daxes seed.
Assistant: I learned 'dax' as an ingest action. Nearest known action: consume (confidence 0.449).
User: What happens if I ask for something you did not learn?
Assistant: I do not have a reliable memory for that. Best confidence was 0.340.
```

## Layout

```text
hrr/              HRR vectors, binding, and SVO encoding
memory/           append-oriented associative memory, chunked KG memory, and metrics
factgraph/        local revision graph for chain facts
ingestion/        Gemini 2.5 Flash Lite text-to-triples ingestion
language/         n-gram prediction and context word-learning primitives
generation/       optional frozen-generator adapter interface
experiments/      reproducible PoC experiments
research/         standalone feasibility studies and research artifacts
tests/            focused unit tests
web_static/       browser dashboard assets
web.py            local HTTP server for the web UI demo
reports/          result notes
docs/             living research synthesis and design notes
```

## Limitations

- Gemini still performs raw-text extraction.
- Responses are mostly template-based.
- The conversation demo is scripted, not an autonomous chat loop.
- Syntax learning and general reasoning are not implemented.
- Word learning currently uses controlled context hints.
- Typed relation fallback is experimental, off by default, and currently backed
  by synthetic feasibility results, a first mixed curated real-corpus
  validation, and focused ingestion tests rather than a large extracted-corpus
  benchmark.
- The current `FactGraph` still stores one active target per `(source, relation)`,
  so branching graph queries remain intentionally conservative.
- The D-2858 chunk law is applied conservatively to this repo's full-vector
  chunk design; it should not be read as a solved projected-address or SDM law.
- Full SDM projected-address CI remains an open engineering target for HRR and
  continuous embedding keys; one-hot key results should be tracked separately.
- The new SDM `n_locs` harness is now much closer to a protocol-level
  reproduction, but it still should not be read as a full numeric clone of the
  lab recipe.
- Several new claim-validation mirrors are intentionally synthetic local
  reproductions of the lab tasks; they validate the repo's qualitative
  behavior, but they are not one-to-one replacements for the full lab protocol.
