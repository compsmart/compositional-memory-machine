# Longitudinal Conversation Benchmark

Generated on 2026-04-23 20:58 UTC.

## Configuration

- `preset=roadmap_serious`
- `cases=19`
- `chat_dim=2048`
- `episodic_dim=2048`
- `episodic_seeds={42,123,7}`
- `temporal_dim=2048`
- `temporal_seeds={42,123}`

## Overall Score

- Mean score: `0.737`
- Pass rate: `0.684`
- Delta vs previous: `+0.000`

## Track Rollup

| track | mean score | pass rate | delta |
| --- | ---: | ---: | ---: |
| frontier | 0.167 | 0.000 | +0.000 |
| implemented | 1.000 | 1.000 | +0.000 |

## Surface Rollup

| surface | mean score | pass rate | delta |
| --- | ---: | ---: | ---: |
| episodic_substrate | 1.000 | 1.000 | +0.000 |
| structured_ingest | 1.000 | 1.000 | +0.000 |
| web_chat | 0.667 | 0.600 | +0.000 |

## Category Scorecard

| category | mean score | pass rate | delta |
| --- | ---: | ---: | ---: |
| canonical_meanings | 1.000 | 1.000 | +0.000 |
| coding | 0.000 | 0.000 | +0.000 |
| explanation_understanding | 0.500 | 0.000 | +0.000 |
| general_context | 1.000 | 1.000 | +0.000 |
| language_patterning | 1.000 | 1.000 | +0.000 |
| logic | 0.000 | 0.000 | +0.000 |
| memory | 1.000 | 1.000 | +0.000 |
| multi_hop | 1.000 | 1.000 | +0.000 |
| multilingual | 0.500 | 0.000 | +0.000 |
| puzzles | 0.000 | 0.000 | +0.000 |
| sentiment | 0.000 | 0.000 | +0.000 |
| temporal | 1.000 | 1.000 | +0.000 |
| trick_questions | 1.000 | 1.000 | +0.000 |

## Lowest-Scoring Cases

### coding_python_function

- Category: `coding`
- Surface: `web_chat`
- Score: `0.000`
- Expected: Return a correct `add(a, b)` implementation.
- Notes: Exact-answer challenge expecting: def add, return, a, b.
- Prompts:
  - `Write a Python function add(a, b) that returns the sum.`
- Final route: `fallback`
- Final reply: `I can answer memory-backed fact questions, continue a learned pattern, or learn a demo word from examples. Try "Who did Ada Lovelace work with?", "Complete this learned pattern: 'the doctor ...'", or "Learn a new word: dax. A child daxes an apple; a chef daxes soup; a bird daxes seed." You can also teach me linked facts and ask a multi-hop question like "Who does Alice know who works with Carol?"`

### logic_transitive_order

- Category: `logic`
- Surface: `web_chat`
- Score: `0.000`
- Expected: Answer that Alice is tallest in a three-person order chain.
- Notes: Exact-answer challenge expecting: alice, tallest.
- Prompts:
  - `If Alice is taller than Bob and Bob is taller than Carol, who is tallest?`
- Final route: `fallback`
- Final reply: `I can answer memory-backed fact questions, continue a learned pattern, or learn a demo word from examples. Try "Who did Ada Lovelace work with?", "Complete this learned pattern: 'the doctor ...'", or "Learn a new word: dax. A child daxes an apple; a chef daxes soup; a bird daxes seed." You can also teach me linked facts and ask a multi-hop question like "Who does Alice know who works with Carol?"`

### puzzle_number_sequence

- Category: `puzzles`
- Surface: `web_chat`
- Score: `0.000`
- Expected: Return `32` for the sequence 2, 4, 8, 16, ?.
- Notes: Exact-answer challenge expecting: 32.
- Prompts:
  - `Which number comes next in the sequence 2, 4, 8, 16, ?`
- Final route: `fallback`
- Final reply: `I can answer memory-backed fact questions, continue a learned pattern, or learn a demo word from examples. Try "Who did Ada Lovelace work with?", "Complete this learned pattern: 'the doctor ...'", or "Learn a new word: dax. A child daxes an apple; a chef daxes soup; a bird daxes seed." You can also teach me linked facts and ask a multi-hop question like "Who does Alice know who works with Carol?"`

### sentiment_labeling

- Category: `sentiment`
- Surface: `web_chat`
- Score: `0.000`
- Expected: Classify the example movie review as positive.
- Notes: Exact-answer challenge expecting: positive.
- Prompts:
  - `The movie was amazing and I loved every minute. Was the sentiment positive, negative, or neutral?`
- Final route: `fallback`
- Final reply: `I can answer memory-backed fact questions, continue a learned pattern, or learn a demo word from examples. Try "Who did Ada Lovelace work with?", "Complete this learned pattern: 'the doctor ...'", or "Learn a new word: dax. A child daxes an apple; a chef daxes soup; a bird daxes seed." You can also teach me linked facts and ask a multi-hop question like "Who does Alice know who works with Carol?"`

### explanation_from_memory

- Category: `explanation_understanding`
- Surface: `web_chat`
- Score: `0.500`
- Expected: Use evidence-aware language when asked why Ada worked with Charles Babbage.
- Notes: Explanation cases reward evidence-aware answers over bare fact restatement.
- Prompts:
  - `Remember Ada text`
  - `Why do you think Ada Lovelace worked with Charles Babbage?`
- Final route: `fact_query`
- Final reply: `Ada Lovelace worked with Charles Babbage. Confidence: 0.900.`
