# Research

This directory holds research notes and generated result artifacts for
relation-memory experiments whose runnable scripts now live under
`experiments/`.

Current study:

- `research/results/relation_concept_memory.md`: synthetic feasibility report
  for relation concept memory.
- `research/results/relation_fallback_real_corpus.md`: curated real-corpus
  fallback validation report.

Run it with:

```powershell
python experiments/exp_relation_concept_memory.py --output summary --json-file research/results/relation_concept_memory.json --report-file research/results/relation_concept_memory.md
```

The generated markdown report summarizes whether context-based relation
clustering is strong enough to justify a deeper implementation in the main
ingestion/query stack.

Run the curated real-corpus validation with:

```powershell
python experiments/exp_relation_fallback_real_corpus.py --output summary --json-file research/results/relation_fallback_real_corpus.json --report-file research/results/relation_fallback_real_corpus.md
```

This second report is the reality check after the synthetic feasibility pass:
it measures whether the current fallback does enough on more realistic corpora
to justify further tuning.
