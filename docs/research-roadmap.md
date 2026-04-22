# Compositional Memory Machine Research Roadmap

Generated: 2026-04-22

This document captures the current research position for the HRR+AMM language
memory PoC and lays out the most useful next experiments. It separates three
things that should not be blurred:

- Compsmart AI Research Lab findings: validated lab records that motivated the
  architecture.
- Repo-local PoC behavior: what this repository currently implements and tests.
- External research context: HRR/VSA/HDC literature that supports or challenges
  the design.

## Current Thesis

The Compositional Memory Machine is not a transformer replacement today. It is a
continual memory substrate for language-shaped knowledge:

```text
raw text or structured input
        |
        v
Gemini extraction or direct triples
        |
        v
HRR role/filler encoders
        |
        +-----------------------+---------------------+
        |                       |                     |
        v                       v                     v
AMM retrieval memory      FactGraph revision    language primitives
sentence/fact vectors     local graph updates   n-gram + word learning
        |
        v
structured evidence
        |
        v
template answer or grounded generator
```

The strongest claim is narrower and more interesting than "we built an LLM":
HRR+AMM can accumulate, retrieve, revise-adjacent, and recombine structured
language memory without gradient retraining.

## Lab-Grounded Discoveries

| Finding | What The Lab Found | Design Consequence In This Repo |
| --- | --- | --- |
| D-2824 | HRR SVO sentence vectors in AMM reached 100% retrieval EM and 0% forgetting across 10 CI cycles at `d=2048`. | Implement HRR sentence storage with AMM retrieval as the base linguistic memory. |
| D-2825 | Novel subject/verb compositions reached 100% cluster-EM by exploiting semantic cluster geometry. | Test compositional routing, not only exact memorization. |
| D-2829 | HRR n-gram contexts can act as keys for next-token AMM prediction: seen/familiar EM 1.0, novel EM 0.0, zero forgetting. | Add a primitive non-transformer sequence memory and keep novel-context refusal explicit. |
| D-2830 | New word meanings can be learned from context by unbinding ACTION role vectors and averaging semantic centroids. | Add a lexical acquisition primitive based on role unbinding and cluster lookup. |
| D-2820 | SDM+AMM entropy protection blocks in-place revision when the same key receives a new value. | Keep AMM append-oriented; do not use it as the sole update mechanism. |
| D-2821 | PerKey Reset supports 3-hop middle-pointer revision with 100% chain3 EM. | Use local graph reset for structural revision. |
| D-2826 | PerKey/Global Reset support entry-pointer revision with 100% chain3 EM. | Treat no-cascade FactGraph revision as the current correction path. |
| D-2823 | HSM kappa-gate protection remained unreliable; investigation closed for CI memory protection. | Do not implement HSM as the CI backbone in this PoC. |
| D-2827 | HRR at `d=512` produced severe CI forgetting through address collision. | Default serious HRR experiments to high dimension and stress-test collisions. |
| D-2831 | With SDM projection `addr_dim=64`, all tested HRR dimensions failed CI; address projection was the bottleneck. | Do not claim `d_hrr=2048` alone solves SDM CI; address dimensionality must be swept. |
| D-2832 | Continuous embedding n-gram keys failed under `addr_dim=64` with near-total forgetting. | Keep repo n-gram results scoped to full-vector AMM; SDM-projected n-gram CI remains open. |

## External Research Context

The project sits inside the Vector Symbolic Architecture (VSA) /
Hyperdimensional Computing (HDC) family.

- Tony Plate's 1995 HRR paper introduced circular convolution for representing
  variable bindings, sequences, and frame-like structures in fixed-width
  distributed vectors. This is the direct ancestor of the repo's role/filler
  encoding. Source: [Plate 1995 HRR PDF](https://redwood.berkeley.edu/wp-content/uploads/2020/08/Plate-HRR-IEEE-TransNN.pdf),
  [DOI metadata](https://bibbase.org/network/publication/plate-holographicreducedrepresentations-1995).
- Kleyko et al.'s two-part HDC/VSA survey frames HRR, Tensor Product
  Representations, Binary Spatter Codes, MAP, and related models as variants of
  high-dimensional algebraic symbolic representation. Sources:
  [Part I](https://arxiv.org/abs/2111.06077),
  [Part II](https://arxiv.org/abs/2112.15424).
- Heddes et al. provide a 2024 Journal of Big Data survey of HDC/VSA components
  and design patterns. Source:
  [Journal of Big Data article](https://link.springer.com/article/10.1186/s40537-024-01010-8).
- Torchhd shows the field has mature enough tooling for GPU/PyTorch-backed HDC
  experimentation and reproducible VSA baselines. Sources:
  [JMLR Torchhd paper](https://www.jmlr.org/papers/v24/23-0300.html),
  [Torchhd GitHub](https://github.com/hyperdimensional-computing/torchhd).
- Symbolic Representation and Learning with HDC demonstrates the broader
  neuro-symbolic motivation: hypervectors can bind symbolic labels to learned
  perceptual structures and retrieve by similarity. Source:
  [Frontiers 2020](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2020.00063/full).
- Hyperseed is relevant because it uses Fourier HRR/VSA operations for fast
  few-shot/topology-preserving learning, including a language-identification
  task using n-gram statistics. Source:
  [Hyperseed arXiv](https://arxiv.org/abs/2110.08343).

The literature supports the basic machinery: binding, bundling, cleanup memory,
similarity retrieval, and compact compositional representation. The Compsmart
lab results add the project-specific CI and language-memory evidence.

## Current Repo Capabilities

The repo currently implements:

- HRR SVO encoding with circular convolution and unitary role vectors.
- Full-vector AMM storage and cosine nearest-neighbor retrieval.
- FactGraph local revision through `per_key_reset`.
- Gemini 2.5 Flash Lite two-pass extraction into triples.
- HRR bigram-context next-token memory.
- Context word learning with ACTION-role unbinding and semantic cluster lookup.
- A scripted memory-grounded conversation demo.

Verified locally:

```text
python -m pytest
14 passed
```

The important caveat: this repo's AMM is full-vector nearest-neighbor memory,
not the full SDM hard-location architecture from the negative `addr_dim=64`
findings. The repo includes an address-routed stress test, but it does not yet
implement the lab's full SDM projection/gating stack.

## Priority Research Tracks

### 1. Address-Dimension Critical Path

Goal: resolve the D-2831/D-2832 bottleneck before making strong SDM CI claims.

Experiment:

- Implement an SDM-style projected address layer alongside the current
  full-vector AMM.
- Sweep `addr_dim = {64, 128, 256, 512, 1024, 2048}` with `d_hrr=2048`.
- Re-run SVO CI and n-gram CI under projected address routing.
- Report forgetting, top-1 retrieval, stale contamination, and candidate counts.

Success:

- Identify the minimum `addr_dim` that keeps forgetting under the lab criterion.
- Clearly separate "full-vector AMM works" from "SDM-projected AMM works."

Why this is first: without it, the docs risk overstating the portability of the
current PoC to the production SDM/AMM recipe.

### 2. Large-Document Memory

Goal: move from toy passages to real documents.

Experiment:

- Ingest a full Wikipedia article, a technical paper section, and one project
  README.
- Chunk text, extract triples with Gemini, deduplicate, write to AMM+FactGraph.
- Generate a fixed benchmark of fact questions, relation questions, and unknown
  questions.

Measure:

- extracted facts, deduplicated facts, source provenance, query accuracy,
  refusal accuracy, contradiction count, and retrieval confidence calibration.

Why it matters: this tests whether persistent memory is useful once extraction
produces hundreds or thousands of facts.

### 3. Multilingual Fact Normalization

Goal: test language-agnostic memory.

Experiment:

- Ingest equivalent short documents in English, French, Spanish, and German.
- Ask queries in English against facts extracted from each language.
- Compare whether Gemini canonicalizes relation labels consistently enough for
  the same HRR/FactGraph memory to answer.

Success:

- Same normalized triples appear across languages.
- English queries retrieve facts from non-English sources.

Risk:

- Inconsistent relation labels such as `worked_with`, `collaborated_with`, and
  `worked_on_with` may fragment memory. This likely requires a relation
  registry.

### 4. Codebase Memory

Goal: turn code into graph facts.

Facts to extract:

```text
module imports module
function calls function
class inherits class
test covers function
cli command invokes handler
file defines symbol
```

Experiment:

- Parse this repo with Python AST, not Gemini.
- Store code facts in FactGraph and AMM.
- Ask dependency questions such as "what calls this function?" and "which tests
  cover this behavior?"

Why it matters: code has explicit structure, so it is a good domain for a
non-transformer memory substrate.

### 5. Contradiction And Temporal Revision

Goal: handle facts that change.

Experiment:

- Ingest current and historical facts with timestamps/source provenance.
- Detect collisions on `(subject, relation)` with different objects.
- Use FactGraph as current truth and AMM as historical evidence.

Example:

```text
Alice --works_at--> Clinic A
Alice --works_at--> Hospital B
```

Questions:

- Where does Alice work now?
- Where did Alice previously work?
- Which source caused the revision?

Why it matters: this directly exercises the D-2820 boundary and the D-2821/D-2826
revision solution.

### 6. Emergent Syntax

Goal: run the missing Directive #95 syntax experiment.

Experiment:

- Generate patterned sentence families with controlled substitutions:
  subjects, verbs, objects, adjectives, prepositional phrases, tense markers.
- Train only from exposure.
- Probe whether vectors cluster by syntactic role without explicit role labels.

Possible tests:

- subject-like slot retrieval,
- verb-like slot retrieval,
- object-like slot retrieval,
- modifier attachment,
- active/passive alternation,
- SVO vs SOV word order.

Success:

- syntactic-role clusters emerge above random baseline,
- novel sentence structures route correctly,
- no catastrophic forgetting across new grammar families.

### 7. Episodic Conversation Memory

Goal: give the system memory across dialogue turns without a prompt window.

Facts to store:

```text
turn_7 speaker user
turn_7 intent teach_word
turn_7 introduced_word dax
turn_8 assistant_answer dax means ingest
```

Questions:

- What did I teach you earlier?
- What did you say about dax?
- Did I correct you?

Why it matters: this turns the current scripted demo into a persistent dialogue
memory experiment.

### 8. Procedural And Tool-Routing Memory

Goal: store actions and route tasks.

Procedural memory:

```text
procedure reset_service step_1 stop_service
step_1 next step_2
step_2 action clear_cache
```

Tool routing:

```text
weather_question requires weather_api
repo_bug_question requires code_search
math_question requires calculator
```

Success:

- retrieve next steps,
- retrieve previous steps,
- route task classes to tools from learned examples.

This is a plausible bridge from memory to action without pretending the memory
itself is a full planner.

### 9. Analogical Retrieval

Goal: exploit role/filler geometry for relational similarity.

Experiment:

- Store relation patterns such as:

```text
doctor treats patient
mechanic repairs engine
teacher helps student
engineer debugs program
```

- Query for closest analogies by relational structure.

Success:

- Similar role structures retrieve above entity-overlap baselines.
- Analogies fail gracefully when no role pattern matches.

### 10. Memory-Grounded Generation

Goal: attach a generator only after retrieval.

Pipeline:

```text
question -> AMM/FactGraph evidence -> evidence packet -> generator answer
```

Rules:

- generator receives only retrieved evidence,
- answer cites evidence IDs,
- low-confidence retrieval returns uncertainty,
- hallucination checks compare generated answer back to evidence triples.

This keeps the architecture honest: HRR+AMM handles memory; the generator handles
surface language.

## Product Direction: MemoryWorkbench

The most useful next product artifact is a `MemoryWorkbench` CLI or minimal UI:

- ingest text,
- ingest code,
- list extracted facts,
- show FactGraph edges,
- query AMM,
- query FactGraph,
- revise facts,
- detect contradictions,
- learn words,
- run n-gram probes,
- export a run report.

This would turn the repo from experiment scripts into a repeatable research
instrument.

## Risk Register

| Risk | Why It Matters | Mitigation |
| --- | --- | --- |
| Overclaiming language-model capability | The system is not an open-ended LLM. | Keep "memory substrate" language; show transformer boundary. |
| Gemini extraction hides hard parsing problems | Raw text understanding is currently outsourced. | Track direct-structured vs Gemini-ingested experiments separately. |
| `d_hrr=2048` is mistaken for full CI safety | D-2831/D-2832 show `addr_dim` can dominate. | Add SDM address-dimension sweep. |
| Relation label fragmentation | Same relation may be extracted under many labels. | Add relation registry and alias normalization. |
| Template answers look conversational but are not autonomous | Current demo is scripted. | Add explicit dialogue memory and controller experiments. |
| Word learning uses controlled hints | D-2830-style repo test is not raw natural acquisition. | Move toward extracted contextual properties and no-hint ablations. |
| Full-vector AMM may not scale | Nearest-neighbor scan is simple but expensive. | Add vector index or SDM routing after address sweep. |

## Recommended Next Build Order

1. Add SDM projected address memory and run the `addr_dim` sweep.
2. Add relation registry and provenance model for extracted triples.
3. Add large-document ingestion benchmark.
4. Add codebase AST memory.
5. Add contradiction/current-vs-history revision experiment.
6. Add emergent syntax experiment.
7. Add MemoryWorkbench CLI.
8. Add evidence-grounded generation adapter.

## Claim Boundary

The defensible claim today:

> The Compositional Memory Machine is a working PoC for continual
> language-shaped memory: HRR+AMM stores and retrieves structured facts, supports
> simple compositional routing, sequence-memory primitives, controlled word
> learning, and graph-local revision, while preserving a clean boundary between
> transformer extraction/generation and non-transformer memory.

The claim not yet supported:

> This is a standalone language model with general reasoning, syntax, and
> open-ended conversation.

