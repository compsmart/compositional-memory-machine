# Longitudinal Conversation Benchmark

Generated on 2026-04-24 06:49 UTC.

## Configuration

- `preset=roadmap_serious`
- `cases=23`
- `chat_dim=2048`
- `episodic_dim=2048`
- `episodic_seeds={42,123,7}`
- `temporal_dim=2048`
- `temporal_seeds={42,123}`
- `preload_jsonl=C:\projects\agents\hhr\reports\hf_ingest_runs\structured_wikipedia_medical_250k\facts.jsonl`
- `preload_limit=5000`

## Overall Score

- Mean score: `0.986`
- Pass rate: `0.957`
- Delta vs previous: `+0.000`

## Track Rollup

| track | mean score | pass rate | delta |
| --- | ---: | ---: | ---: |
| frontier | 1.000 | 1.000 | +0.000 |
| implemented | 0.980 | 0.941 | +0.000 |

## Surface Rollup

| surface | mean score | pass rate | delta |
| --- | ---: | ---: | ---: |
| episodic_substrate | 1.000 | 1.000 | +0.000 |
| structured_ingest | 0.889 | 0.667 | +0.000 |
| web_chat | 1.000 | 1.000 | +0.000 |

## Category Scorecard

| category | mean score | pass rate | delta |
| --- | ---: | ---: | ---: |
| canonical_meanings | 1.000 | 1.000 | +0.000 |
| coding | 0.833 | 0.500 | +0.000 |
| explanation_understanding | 1.000 | 1.000 | +0.000 |
| general_context | 1.000 | 1.000 | +0.000 |
| language_patterning | 1.000 | 1.000 | +0.000 |
| logic | 1.000 | 1.000 | +0.000 |
| memory | 1.000 | 1.000 | +0.000 |
| multi_hop | 1.000 | 1.000 | +0.000 |
| multilingual | 1.000 | 1.000 | +0.000 |
| puzzles | 1.000 | 1.000 | +0.000 |
| sentiment | 1.000 | 1.000 | +0.000 |
| temporal | 1.000 | 1.000 | +0.000 |
| trick_questions | 1.000 | 1.000 | +0.000 |

## Lowest-Scoring Cases

### codebase_dependency_memory_substrate

- Category: `coding`
- Surface: `structured_ingest`
- Score: `0.667`
- Expected: Answer imports, calls, and symbol ownership queries from codebase ingestion.
- Notes: Python codebase structure can be ingested and queried as dependency-style facts.

### alias_normalization_ingest

- Category: `canonical_meanings`
- Surface: `structured_ingest`
- Score: `1.000`
- Expected: Map `collaborated with` onto canonical `worked_with` with provenance.
- Notes: Ingestion should normalize alias-equivalent relations to canonical registry forms.

### coding_python_function

- Category: `coding`
- Surface: `web_chat`
- Score: `1.000`
- Expected: Return a correct `add(a, b)` implementation.
- Notes: Exact-answer challenge expecting: def add, return, a, b.
- Prompts:
  - `Write a Python function add(a, b) that returns the sum.`
- Final route: `builtin_coding`
- Final reply: `def add(a, b):
    return a + b`

### context_pronoun_carryover

- Category: `general_context`
- Surface: `web_chat`
- Score: `1.000`
- Expected: Carry the subject from the prior factual turn and answer the pronoun question correctly.
- Notes: Pronoun carryover resolved to the last fact subject.
- Prompts:
  - `Remember Ada text`
  - `Who did Ada Lovelace work with?`
  - `Who did she work with?`
- Final route: `fact_query`
- Final reply: `Ada Lovelace worked with Charles Babbage. Confidence: 1.000.`

### episodic_dialogue_memory_substrate

- Category: `memory`
- Surface: `episodic_substrate`
- Score: `1.000`
- Expected: Maintain immediate, distant, cross-session, revision, and retention EM.
- Notes: D-2836-style episodic dialogue memory metrics averaged across seeds.
