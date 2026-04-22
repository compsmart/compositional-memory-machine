# HRR Language Memory PoC

This project is a proof of concept for a language-capable memory substrate built
around Holographic Reduced Representations (HRR), append-oriented associative
memory (AMM), and a small FactGraph revision layer.

It is not a standalone GPT-style language model. The current system uses Gemini
2.5 Flash Lite as an optional transformer front end for raw-text fact
extraction. The non-transformer part of the system stores, retrieves, revises,
predicts, and learns over structured HRR representations.

## What Works

- HRR role/filler sentence encoding
- AMM continual fact storage with repeated acquisition cycles
- Compositional nearest-key routing for simple SVO facts
- Dimension/collision stress testing with compressed address routing
- Local FactGraph revision through per-key reset
- Gemini 2.5 Flash Lite real-text extraction into triples
- HRR bigram-context next-token prediction
- Context-based word learning by unbinding the ACTION role
- A scripted memory-grounded conversation demo

## Current Verdict

The PoC works as a continual language-memory engine, not as an open-ended
chatbot. It can ingest or receive structured language facts, keep them outside
gradient training, retrieve them later, revise graph facts locally, learn simple
sequence continuations, and fast-map new action words from structured contexts.

The current raw-text understanding step is transformer-based:

```text
Raw text -> Gemini extraction -> HRR+AMM/FactGraph memory -> structured/template answer
```

Without Gemini, the system still works on structured triples and controlled
examples. It does not yet parse arbitrary prose by itself.

## Results Snapshot

Verified locally:

```text
python -m pytest
14 passed
```

Key experiment outcomes:

- D-2824 style CI storage: 100% top-1 retrieval at `d=2048` across 3 seeds and 10 cycles.
- D-2825 style composition: 100% cluster-EM at `d=2048` across 3 seeds.
- Collision/address stress: compressed address routing improves from roughly 0.87-0.92 at `d=512` to roughly 0.98-0.99 at `d=2048`.
- D-2829 next-token primitive: seen EM 1.0, familiar EM 1.0, novel hit rate 0.0.
- D-2830 word learning: new-word cluster routing 1.0 and retention 1.0 across 3 seeds.
- FactGraph chain3 revision: 100% exact match across entry, middle, and terminal updates.
- Gemini real-text demo: extracted and stored Ada Lovelace facts, then retrieved `Ada Lovelace --worked_with--> Charles Babbage` at confidence 1.0.

## Quick Start

```powershell
python -m pytest
python conversation_demo.py
```

Run the full experiment set:

```powershell
python experiments/exp_d2824_ci_storage.py
python experiments/exp_d2825_composition.py
python experiments/exp_d2827_dimension_sweep.py
python experiments/exp_collision_stress.py
python experiments/exp_d2829_next_token.py
python experiments/exp_d2830_word_learning.py
python experiments/exp_revision_chain3.py
```

Run demos:

```powershell
python demo.py
python conversation_demo.py
python real_text_demo.py
```

Ingest arbitrary text with Gemini:

```powershell
python ingest_text.py --text "Ada Lovelace worked with Charles Babbage." --domain history --probe-subject "Ada Lovelace" --probe-relation worked_with --probe-object "Charles Babbage"
```

Gemini ingestion requires `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

## Example Conversation Output

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
demo.py           end-to-end memory-first demo
conversation_demo.py scripted memory-grounded conversation
real_text_demo.py live Gemini real-text extraction demo
ingest_text.py    CLI for arbitrary text or files
```

## Core Representation

The sentence representation is:

```text
sentence = bind(role_S, subject) + bind(role_V, verb) + bind(role_O, object)
```

Next-token prediction uses HRR bigram contexts as keys and stores the predicted
token in the AMM payload. Word learning unbinds an ACTION role from several
context examples, averages the extracted vectors into a centroid, and routes
that centroid to known semantic clusters.

## Limitations

- No native raw-text parser yet; Gemini handles extraction.
- No open-ended generation unless a generator is attached.
- No learned chain-of-thought or general logic engine.
- No broad syntax model yet.
- Dialogue state is explicit and minimal.
- The conversation demo is scripted and memory-grounded, not an autonomous chat loop.
- Current experiments are PoC-scale and synthetic except for the Gemini extraction demo.
