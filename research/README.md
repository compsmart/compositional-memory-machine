# Research

This directory holds standalone feasibility experiments that are intentionally
more exploratory than the main `experiments/` suite.

Current study:

- `exp_relation_concept_memory.py`: compares exact pair-overlap aliasing against
  `dax`-style relation concept memory prototypes on synthetic relation data.
- `exp_relation_fallback_real_corpus.py`: validates the experimental typed
  fallback on curated corpus-style positives and negative safety cases.

Run it with:

```powershell
python research/exp_relation_concept_memory.py --output summary --json-file research/results/relation_concept_memory.json --report-file research/results/relation_concept_memory.md
```

The generated markdown report summarizes whether context-based relation
clustering is strong enough to justify a deeper implementation in the main
ingestion/query stack.

Run the curated real-corpus validation with:

```powershell
python research/exp_relation_fallback_real_corpus.py --output summary --json-file research/results/relation_fallback_real_corpus.json --report-file research/results/relation_fallback_real_corpus.md
```

This second report is the reality check after the synthetic feasibility pass:
it measures whether the current fallback does enough on more realistic corpora
to justify further tuning.
