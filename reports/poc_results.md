# PoC Results

Generated on 2026-04-22.

## Verified Commands

```powershell
python -m pytest
python experiments/exp_d2824_ci_storage.py
python experiments/exp_d2825_composition.py
python experiments/exp_d2827_dimension_sweep.py
python experiments/exp_collision_stress.py
python experiments/exp_d2829_next_token.py
python experiments/exp_d2830_word_learning.py
python experiments/exp_revision_chain3.py
python demo.py
python conversation_demo.py
python real_text_demo.py
```

## Current Findings

- Unit tests: 14 passed.
- CI storage at `d=2048`: 100% top-1 retrieval across 3 seeds and 10 cycles.
- Composition holdout at `d=2048`: 100% cluster-EM across 3 seeds, 0.2 random baseline.
- Chain3 FactGraph revision: 100% exact match across entry, middle, and terminal updates.
- Full-vector AMM lookup is robust even at `d=512` on this synthetic corpus.
- Address-routed noisy retrieval shows the expected dimension pressure:
  - `d=512`: about 0.87-0.92 top-1
  - `d=1024`: about 0.945-0.975 top-1
  - `d=2048`: about 0.98-0.99 top-1
- Gemini 2.5 Flash Lite real-text ingestion works on the Ada Lovelace fixture:
  - Pass 1 extracted 4 facts.
  - Pass 2 added 3 facts.
  - Deduplication retained 6 facts.
  - HRR+AMM retrieved `Ada Lovelace --worked_with--> Charles Babbage` at 1.0 confidence.
- `conversation_demo.py` runs end to end:
  - Extracts 6 facts from real text.
  - Answers two memory-grounded questions.
  - Completes a learned next-token pattern.
  - Learns and retains a new word meaning.
  - Refuses an unknown fact when confidence is below threshold.
- D-2829 next-token primitive:
  - HRR bigram contexts route to next-token payloads in AMM.
  - Seen contexts and familiar noisy contexts are measured separately from novel contexts.
  - Current run at `d=2048`, 3 seeds: seen EM 1.0, familiar EM 1.0, novel hit rate 0.0.
  - Familiar-context cosine is about 0.486-0.509 in this implementation.
- D-2830 word-learning primitive:
  - Unknown action words are learned from context by unbinding the ACTION role.
  - Learned centroids are retained while adding later words.
  - Current run at `d=2048`, 3 seeds: `dax` cluster correct 1.0, `blick` cluster correct 1.0, retention 1.0.
  - Plausible action similarity is about 0.46-0.47; implausible action similarity is near zero or negative.
- Superseding lab nuance:
  - D-2831 showed that `addr_dim=64` is a bottleneck for projected SDM addressing: HRR dimensions 512, 1024, and 2048 all failed under that projection setting.
  - D-2832 showed continuous embedding n-gram keys also fail under the same `addr_dim=64` bottleneck.
  - This repo's positive n-gram and sentence-memory results use full-vector AMM-style retrieval, so they should not be over-read as proof that the lab SDM projection recipe is solved.

## Interpretation

The PoC is working as a memory-first HRR language substrate. The important
distinction is that full-vector nearest-neighbor AMM does not reproduce the
low-dimensional SDM failure by itself. The collision/capacity problem appears
when retrieval is routed through compressed address signatures before vector
scoring, which is closer to the SDM/AMM address-space constraint described in
the findings. D-2831 and D-2832 sharpen this further: the next critical
research step is an explicit `addr_dim` sweep for projected SDM retrieval.
