# HRR Language Memory PoC

This repository is a proof of concept for a Compsmart AI Research Labs-inspired
language-memory architecture built from Holographic Reduced Representations
(HRR), append-oriented associative memory (AMM), a small FactGraph revision
layer, and optional Gemini 2.5 Flash Lite extraction.

The goal is not to build another transformer. The goal is to test a different
language-capable substrate:

```text
raw text or structured facts
        |
        v
Gemini extraction, or direct structured input
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

The non-transformer core stores, retrieves, revises, predicts, and learns over
HRR vectors. Gemini is currently used only as a raw-text-to-triples front end.

## Research Grounding

This implementation was created from the following Compsmart AI Research Labs
findings and design constraints.

### D-2824: Holographic Reduced Representation (HRR) Sentence Memory With Continual Learning

Lab result:  HRR-encoded SVO sentence vectors stored in AMM achieved 100%
retrieval with 0% forgetting across repeated continual-learning cycles.

Implementation:

- [hrr/encoder.py](hrr/encoder.py) encodes SVO facts as:

```text
sentence = bind(role_S, subject) + bind(role_V, verb) + bind(role_O, object)
```

- [memory/amm.py](memory/amm.py) stores one normalized vector per key using a
  per-key accumulator.
- [experiments/exp_d2824_ci_storage.py](experiments/exp_d2824_ci_storage.py)
  repeats acquisition cycles and verifies top-1 retrieval.

Why it matters: this is the persistent linguistic memory layer. Facts are added
without gradient retraining and without overwriting earlier facts in the PoC
setting.

### D-2825: Compositional Generalisation

Lab result: held-out subject/context combinations routed correctly through AMM
nearest-key geometry, showing compositional generalisation rather than pure
table lookup.

Implementation:

- [experiments/exp_d2825_composition.py](experiments/exp_d2825_composition.py)
  holds out selected subjects and probes novel subject/relation/object
  combinations.
- The query routes through HRR vector similarity, not a hand-coded lookup table.

Why it matters: the system can reuse learned parts in new combinations, which
is the first step toward language-like systematicity.

### D-2827: Dimension Floor And Address Collisions

Lab result: low-dimensional HRR vectors collide in SDM/AMM address space; the
research constraint is that `d >= 2048` is the safe working regime.

Implementation:

- All serious demos default to `d=2048`.
- [experiments/exp_collision_stress.py](experiments/exp_collision_stress.py)
  separates full-vector nearest-neighbor lookup from compressed address-routed
  retrieval.

Current PoC finding:

```text
d=512   address_noisy_top1 ~= 0.87-0.92
d=1024  address_noisy_top1 ~= 0.945-0.975
d=2048  address_noisy_top1 ~= 0.98-0.99
```

Why it matters: full-vector lookup is forgiving, but address-routed retrieval
shows the dimension pressure expected from the lab results.

### D-2820: Append-Only Memory Is Not Enough For Revision

Lab result: the entropy gate that protects continual learning blocks in-place
fact revision. A familiar key causes the protective mechanism to close.

Implementation choice:

- AMM remains append-oriented and protective.
- Revisions are handled separately by FactGraph.

Why it matters: the architecture does not pretend that one memory mechanism
does everything. Stable accumulation and local revision are separated.

### D-2821 And D-2826: Per-Key Reset For Local FactGraph Revision

Lab result: no-cascade PerKey Reset supports local graph fact updates across
entry, middle, and terminal positions in chain reasoning.

Implementation:

- [factgraph/graph.py](factgraph/graph.py) implements `per_key_reset` and
  `revise`.
- [experiments/exp_revision_chain3.py](experiments/exp_revision_chain3.py)
  verifies chain3 updates.

Why it matters: factual correction does not require retraining or propagating a
cascade through unrelated memory.

### D-2823: HSM Branch Closed For This PoC

Lab result: the HSM/kappa-gate branch was architecturally unreliable in the lab
batch described by the user.

Implementation choice:

- This repository does not implement HSM.
- The PoC focuses on SDM/AMM-style HRR memory, FactGraph, and language
  primitives.

Why it matters: the implementation follows the surviving backbone rather than
keeping a known-bad branch alive.

### D-2829: HRR AMM As A Primitive Next-Token Predictor

Lab result: HRR bigram/trigram contexts can work as AMM keys for next-token
prediction. Seen and familiar contexts route correctly; novel contexts honestly
fail instead of hallucinating.

Implementation:

- [language/ngram.py](language/ngram.py) binds position roles to token vectors
  to form a context key.
- The AMM payload stores the predicted next token.
- [experiments/exp_d2829_next_token.py](experiments/exp_d2829_next_token.py)
  measures seen, familiar, and novel contexts separately.

Current PoC result at `d=2048`:

```text
seen_em: 1.0
familiar_em: 1.0
novel_hit_rate: 0.0
mean_familiar_cosine: ~0.486-0.509
```

Why it matters: this is not a GPT-like decoder, but it is a continual
non-transformer sequence prediction primitive.

### D-2830: Word Learning From Context

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

## Architecture

### 1. HRR Vector Substrate

Files:

- [hrr/binding.py](hrr/binding.py)
- [hrr/vectors.py](hrr/vectors.py)
- [hrr/encoder.py](hrr/encoder.py)

Responsibilities:

- Generate deterministic random token vectors.
- Generate unitary role vectors for cleaner circular binding/unbinding.
- Encode SVO facts into normalized HRR sentence vectors.

The core operation is circular convolution binding:

```text
bind(role, filler)
```

Roles such as `SUBJECT`, `VERB`, `OBJECT`, `ACTION`, `POSITION_1`, and
`POSITION_2` give the same filler different meanings depending on where it
appears.

### 2. AMM Fact Memory

Files:

- [memory/amm.py](memory/amm.py)
- [memory/metrics.py](memory/metrics.py)

Responsibilities:

- Store normalized HRR vectors under stable keys.
- Accumulate repeated writes per key.
- Retrieve nearest records by cosine similarity.
- Keep structured payloads alongside vectors.

AMM is used for:

- SVO fact retrieval.
- next-token context routing.
- learned word centroid storage.
- semantic cluster lookup.

### 3. FactGraph Revision Layer

File:

- [factgraph/graph.py](factgraph/graph.py)

Responsibilities:

- Store local directed facts as `(source, relation) -> target`.
- Revise a key by resetting only that key.
- Follow simple relation chains.

This exists because the research showed append-only AMM and local revision want
different mechanisms.

### 4. Gemini Extraction Front End

Files:

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
- [language/word_learning.py](language/word_learning.py)

Responsibilities:

- Learn simple next-token patterns from HRR context keys.
- Learn new action words from structured context examples.
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
  and fast lexical acquisition.

This is not yet:

- A standalone LLM.
- A native raw-text parser.
- An open-ended chatbot.
- A general chain-of-thought reasoner.
- A broad syntax learner.
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

Run the conversation demo:

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
python experiments/exp_d2829_next_token.py
python experiments/exp_d2830_word_learning.py
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

From `conversation_demo.py`:

```text
User: Read this short passage and remember it.
Assistant: I extracted 6 distinct facts and wrote 6 of them into HRR memory.
User: Who did Ada Lovelace work with?
Assistant: Ada Lovelace worked with Charles Babbage. Confidence: 1.000.
User: What did she publish notes about?
Assistant: Ada Lovelace published notes about Analytical Engine.
User: Complete this learned pattern: 'the doctor ...'
Assistant: The next token is 'treats' from context 'the doctor' (confidence 1.000).
User: Now learn a new word: dax. A child daxes an apple; a chef daxes soup; a bird daxes seed.
Assistant: I learned 'dax' as an ingest action. Nearest known action: consume (confidence 0.449).
User: Learn another word, then tell me if you still remember dax.
Assistant: Yes. 'dax' still routes to ingest; nearest action is consume (confidence 0.449).
User: What happens if I ask for something you did not learn?
Assistant: I do not have a reliable memory for that. Best confidence was 0.340.
```

## Repository Layout

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
demo.py           end-to-end memory-first demo
conversation_demo.py scripted memory-grounded conversation
real_text_demo.py live Gemini real-text extraction demo
ingest_text.py    CLI for arbitrary text or files
```

## Limitations And Next Work

Current limitations:

- Gemini still performs raw-text extraction.
- There is no autonomous dialogue loop.
- Responses are mostly template-based.
- Syntax learning is not implemented.
- Chain-of-thought, planning, and general logic are not learned capabilities.
- The word-learning experiment uses controlled property hints.
- The experiments are PoC-scale rather than corpus-scale.

Next work:

1. Implement Exp 5: emergent syntax from exposure.
2. Add a controller that routes between AMM, FactGraph, n-gram memory, word
   learning, and generation.
3. Add larger corpora and chunked ingestion.
4. Add confidence calibration and provenance display.
5. Add a generator adapter for fluent answers grounded strictly in retrieved
   memory.
