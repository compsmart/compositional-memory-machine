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
- Compositional routing for controlled SVO facts.
- FactGraph local revision via PerKey Reset.
- Gemini 2.5 Flash Lite real-text extraction into triples.
- HRR bigram-context next-token prediction.
- Context-based word learning by ACTION-role unbinding.
- Scripted memory-grounded conversation demo.
- Projected-address sweep harness for one-hot, HRR SVO, HRR n-gram, and
  continuous context keys.

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

See [docs/research-roadmap.md](docs/research-roadmap.md) for the detailed lab
mapping, external VSA/HDC literature context, risk register, and next research
tracks.

## Current Results

Verified locally:

```text
python -m pytest
15 passed
```

Representative outcomes:

- CI storage at `d=2048`: 100% top-1 retrieval across 3 seeds and 10 cycles.
- Composition at `d=2048`: 100% cluster-EM across 3 seeds.
- Address-routed stress: about 0.87-0.92 at `d=512` vs 0.98-0.99 at `d=2048`.
- D-2829-style next-token primitive: seen EM 1.0, familiar EM 1.0, novel hit
  rate 0.0.
- D-2830-style word learning: cluster routing 1.0 and retention 1.0.
- FactGraph chain3 revision: 100% exact match across tested positions.

Important boundary: the repo's core AMM is currently full-vector
nearest-neighbor memory. The lab's D-2831/D-2832 results show that projected SDM
addressing needs a dedicated `addr_dim` sweep before making stronger CI claims.
D-2837's positive one-hot SDM result should be tracked separately from HRR and
continuous embedding key families.

## Quick Start

Run tests:

```powershell
python -m pytest
```

Run the scripted conversation demo:

```powershell
python conversation_demo.py
```

Run core experiments:

```powershell
python experiments/exp_d2824_ci_storage.py
python experiments/exp_d2825_composition.py
python experiments/exp_d2827_dimension_sweep.py
python experiments/exp_collision_stress.py
python experiments/exp_d2829_next_token.py
python experiments/exp_d2830_word_learning.py
python experiments/exp_revision_chain3.py
python experiments/exp_projected_address_sweep.py
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
memory/           append-oriented associative memory and metrics
factgraph/        local revision graph for chain facts
ingestion/        Gemini 2.5 Flash Lite text-to-triples ingestion
language/         n-gram prediction and context word-learning primitives
generation/       optional frozen-generator adapter interface
experiments/      reproducible PoC experiments
tests/            focused unit tests
reports/          result notes
docs/             research roadmap and design notes
```

## Limitations

- Gemini still performs raw-text extraction.
- Responses are mostly template-based.
- The conversation demo is scripted, not an autonomous chat loop.
- Syntax learning and general reasoning are not implemented.
- Word learning currently uses controlled context hints.
- Full SDM projected-address CI remains an open engineering target for HRR and
  continuous embedding keys; one-hot key results should be tracked separately.
