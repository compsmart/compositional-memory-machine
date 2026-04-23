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
  revision, cross-session recall, and final retention checks.
- Relation registry and provenance payloads for extracted triples.

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
- D-2854: multi-step autoregressive generation fails even when single-step
  retrieval succeeds, which defines the repo's retrieval-vs-generation boundary.
- D-2855/D-2856: recursive syntax and explicit failure-boundary / aliasing
  measurements are now part of the repo validation story rather than only lab
  motivation.
- D-2858: HRR chunk capacity scales linearly with dimension, giving a concrete
  `n*` budget for bounded chunk design.
- D-2869: multi-hop accuracy follows a k-hop factorization law, which supports
  path confidence budgeting in the query engine.
- D-2862/D-2863: temporal role binding and hierarchical syntax are viable in the
  CI setting, motivating explicit temporal and nested-role extensions.
- D-2866/D-2870/D-2871/D-2874: chunked HRR memory enables reliable multi-hop
  retrieval across bounded local fact sets and many domains.
- D-2872/D-2873: dynamic state tracking and 2-hop relational chaining benefit
  from higher dimensions (`D>=1024`, with `D=2048` as the robust target).
- L-932/L-933: benchmark design must avoid misleading full-tuple metrics and
  must use shared entity pools plus explicit chain construction for multi-hop
  evaluation.

See [docs/research-roadmap.md](docs/research-roadmap.md) for the detailed lab
mapping, external VSA/HDC literature context, risk register, and next research
tracks. See [reports/lab_claim_validation.md](reports/lab_claim_validation.md)
for the repo-vs-lab claim ledger and the validation status for the current
scorecard.

## Current Results

Verified locally:

```text
python -m pytest
44 passed
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
- D-2838-style compositional generation: repo benchmark reaches 1.0 exact
  retrieval, 1.0 HRR-native EM, and 1.0 linear-head EM across
  `D={64,128,256,512,2048}`.
- D-2839-style sequence chains: `K={1,2}` stays at 0.25 EM (chance across 4
  rules), while `K={3,5,7,10}` reaches 1.0 EM across 3 seeds.
- Chunked multi-hop benchmark: 2-hop, 3-hop, and cross-domain 4-step chain
  retrieval all reach 1.0 EM in the repo smoke test while preserving chunk
  provenance.
- Query budgeting now uses D-2869/D-2873-style dimension and hop budgets so
  weak multi-hop paths refuse instead of over-claiming a result.
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
- D-2855 mirror: recursive syntax remains usable through depth 3 at `n=25`
  (`main_acc=0.92` in the repo-local mirror).
- D-2856 mirror: near-duplicate failure appears only at very high similarity
  (`sim=0.95`), while overwrite without reset collapses toward a `50/50` blend.
- D-2857 mirror: `perkey_reset` restores `revised_em=1.0` and
  `retained_em=1.0`, while no-reset revision stays far lower.
- The demo now includes a compositional value answer generated from a retrieved
  HRR value vector: `entity_demo has property silver signal.`
- The web UI now mirrors the `nexus-16` dashboard style while exposing HHR
  structured querying, text ingestion, resettable seed memory, chunk summaries,
  chain queries, and the same 3D fact visualization pattern.
- FactGraph chain3 revision: 100% exact match across tested positions.
- Bounded projected-address sweep: top-1 stayed high in the small repo run, but
  candidate contamination separated key families; see
  [reports/projected_address_sweep.md](reports/projected_address_sweep.md).

Important boundary: the repo's core AMM is currently full-vector
nearest-neighbor memory. The lab's D-2831/D-2832 results show that projected SDM
addressing needs a dedicated `addr_dim` sweep before making stronger CI claims.
D-2837's positive one-hot SDM result should be tracked separately from HRR and
continuous embedding key families. Relation normalization is now registry-based,
but the default alias set is intentionally small and should grow from benchmark
misses.

## Quick Start

Run tests:

```powershell
python -m pytest
```

Run the scripted conversation demo:

```powershell
python conversation_demo.py
```

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
tests/            focused unit tests
web_static/       browser dashboard assets
web.py            local HTTP server for the web UI demo
reports/          result notes
docs/             research roadmap and design notes
```

## Limitations

- Gemini still performs raw-text extraction.
- Responses are mostly template-based.
- The conversation demo is scripted, not an autonomous chat loop.
- Syntax learning and general reasoning are not implemented.
- Word learning currently uses controlled context hints.
- The current `FactGraph` still stores one active target per `(source, relation)`,
  so branching graph queries remain intentionally conservative.
- The D-2858 chunk law is applied conservatively to this repo's full-vector
  chunk design; it should not be read as a solved projected-address or SDM law.
- Full SDM projected-address CI remains an open engineering target for HRR and
  continuous embedding keys; one-hot key results should be tracked separately.
- Several new claim-validation mirrors are intentionally synthetic local
  reproductions of the lab tasks; they validate the repo's qualitative
  behavior, but they are not one-to-one replacements for the full lab protocol.
