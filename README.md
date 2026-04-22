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

Lab result: new words can be learned from a few context examples by unbinding
the ACTION role, averaging extracted vectors into a semantic centroid, and
routing that centroid to known verb/action clusters.

Implementation:

- [language/word_learning.py](language/word_learning.py) creates context frames
  with subject, action, object, and property hints.
- It unbinds the action role, averages examples, stores the learned word in AMM,
  and queries known clusters.
- [experiments/exp_d2830_word_learning.py](experiments/exp_d2830_word_learning.py)
  learns `dax` and `blick` in separate cycles and verifies retention.

Current PoC result at `d=2048`:

```text
dax_cluster_correct: 1.0
blick_cluster_correct: 1.0
retention: 1.0
plausible action similarity: ~0.46-0.47
implausible action similarity: ~0 or negative
```

Why it matters: this is a concrete lexical acquisition primitive. It is still
controlled, but it demonstrates fast-mapping without weight updates.

### D-2833: Emergent Syntactic Composition

Lab result: syntactically different forms of the same meaning align in HRR
space without training cross-pattern mappings. The reported lab shape was high
within-pattern similarity, strong cross-pattern similarity, and low random
similarity.

Implementation:

- [language/syntax.py](language/syntax.py) encodes active, passive, relative,
  prepositional, and coordinated forms with shared semantic role bindings plus
  pattern/variant bindings.
- [experiments/exp_d2833_emergent_syntax.py](experiments/exp_d2833_emergent_syntax.py)
  evaluates 5 domains x 30 triples x 3 seeds.

Current PoC result at `d=2048`:

```text
mean_within_cosine: ~0.888-0.890
mean_cross_pattern_cosine: ~0.645-0.653
mean_random_cosine: ~-0.010-0.003
```

Why it matters: the syntax gap is no longer the immediate blocker. The same
role-filler geometry that stores facts also gives same-meaning syntactic forms
a shared vector neighborhood.

### D-2834: Closed-Loop HRR AMM QA

Lab result: a full question-answering loop can retrieve an HRR fact vector,
unbind the verb and object roles, cleanup against known vocabularies, and retain
facts across continual-learning cycles.

Implementation:

- [language/qa.py](language/qa.py) stores facts as subject/verb/object
  role-filler superpositions.
- Queries use only subject plus verb roles.
- Answers are recovered by unbinding the retrieved vector with the object role,
  not by reading a prompt.
- [experiments/exp_d2834_closed_loop_qa.py](experiments/exp_d2834_closed_loop_qa.py)
  evaluates 5 domains x 50 facts x 3 seeds.

Current PoC result at `d=2048`:

```text
answer_em: 1.0
verb_accuracy: 1.0
object_accuracy: 1.0
forgetting: 0.0
mean_fact_confidence: ~0.816-0.817
mean_object_confidence: ~0.577-0.579
```

Why it matters: this moves the project from isolated primitives to a closed
memory-grounded QA loop. The next research target is multi-turn conversation
with online fact updates.

## Architecture

### 1. HRR Vector Substrate

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

See [docs/research-roadmap.md](docs/research-roadmap.md) for the detailed lab
mapping, external VSA/HDC literature context, risk register, and next research
tracks.

## Current Results

Verified locally:

```text
python -m pytest
14 passed
```

Representative outcomes:

- CI storage at `d=2048`: 100% top-1 retrieval across 3 seeds and 10 cycles.
- Composition at `d=2048`: 100% cluster-EM across 3 seeds.
- Address-routed stress: about 0.87-0.92 at `d=512` vs 0.98-0.99 at `d=2048`.
- D-2829-style next-token primitive: seen EM 1.0, familiar EM 1.0, novel hit
  rate 0.0.
- D-2830-style word learning: cluster routing 1.0 and retention 1.0.
- FactGraph chain3 revision: 100% exact match across tested positions.

- [ingestion/gemini.py](ingestion/gemini.py)
- [real_text_demo.py](real_text_demo.py)
- [ingest_text.py](ingest_text.py)

Responsibilities:

- Use Gemini 2.5 Flash Lite to extract JSON triples from raw text.
- Run a two-pass extraction pattern adapted from Nexus-16:
  - Pass 1: explicit fact extraction and estimated count.
  - Pass 2: missed, implied, and derived facts.
- Deduplicate triples.
- Write retained triples into both HRR+AMM and FactGraph.

This is the transformer-dependent part of the current project.

### 5. Language Primitives

Files:

- [language/ngram.py](language/ngram.py)
- [language/qa.py](language/qa.py)
- [language/syntax.py](language/syntax.py)
- [language/word_learning.py](language/word_learning.py)

Responsibilities:

- Learn simple next-token patterns from HRR context keys.
- Learn new action words from structured context examples.
- Compose syntactic variants around shared semantic role bindings.
- Answer subject/relation questions by retrieving and unbinding HRR facts.
- Retain earlier learned words while learning later ones.

These primitives are deliberately small. They test whether HRR+AMM can support
language-like operations before any fluent generator is attached.

### 6. Conversation Demo

File:

- [conversation_demo.py](conversation_demo.py)

The conversation demo is scripted but end-to-end:

1. Gemini extracts facts from an Ada Lovelace passage.
2. Extracted facts are written to HRR+AMM and FactGraph.
3. The system answers memory-grounded questions.
4. The n-gram primitive completes a learned token pattern.
5. The word-learning primitive learns `dax`.
6. It learns `blick` and shows `dax` was retained.
7. It refuses an unsupported query when confidence is low.

## What This Is And Is Not

This is:

- A continual structured language-memory PoC.
- A non-transformer memory substrate after extraction.
- A testbed for HRR composition, AMM routing, local revision, n-gram prediction,
  fast lexical acquisition, syntax composition, and closed-loop QA.

This is not yet:

- A standalone LLM.
- A native raw-text parser.
- An open-ended chatbot.
- A general chain-of-thought reasoner.
- A replacement for a fluent generator such as Gemma/Gemini.

## Current Context Window

There is no transformer-style fixed context window in HRR+AMM memory. The
limits are different:

- Gemini extraction is limited by Gemini's input context.
- HRR+AMM storage is limited by memory size, vector dimension, address
  collisions, and retrieval quality.
- Conversation state is currently explicit and minimal.

The intended direction is to turn text into persistent memory instead of
keeping everything in a prompt window.

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
python experiments/exp_addr_dim_sweep.py
python experiments/exp_projected_sdm_capacity.py
python experiments/exp_projected_sdm_stress.py
python experiments/exp_projected_sdm_readout.py
python experiments/exp_projected_sdm_ngram.py
python experiments/exp_projected_sdm_trigram.py
python experiments/exp_d2829_next_token.py
python experiments/exp_d2830_word_learning.py
python experiments/exp_d2833_emergent_syntax.py
python experiments/exp_d2834_closed_loop_qa.py
python experiments/exp_revision_chain3.py
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
language/         n-gram, word-learning, syntax, and QA primitives
generation/       optional frozen-generator adapter interface
experiments/      reproducible PoC experiments
tests/            focused unit tests
reports/          result notes
docs/             research roadmap and design notes
```

## Limitations

- Gemini still performs raw-text extraction.
- There is no autonomous multi-turn dialogue loop.
- Responses are mostly template-based.
- Syntax composition is controlled rather than induced from raw corpora.
- Chain-of-thought, planning, and general logic are not learned capabilities.
- The word-learning experiment uses controlled property hints.
- The experiments are PoC-scale rather than corpus-scale.

Next work:

1. Implement Exp 7: multi-turn conversational QA with online fact updates.
2. Add a controller that routes between AMM, FactGraph, n-gram memory, word
   learning, and generation.
3. Add larger corpora and chunked ingestion.
4. Add confidence calibration and provenance display.
5. Add a generator adapter for fluent answers grounded strictly in retrieved
   memory.
