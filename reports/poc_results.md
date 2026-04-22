# PoC Results

Generated on 2026-04-22.

## Verified Commands

```powershell
python -m pytest
python -m pytest tests/test_web.py
python experiments/exp_d2824_ci_storage.py
python experiments/exp_d2825_composition.py
python experiments/exp_d2827_dimension_sweep.py
python experiments/exp_collision_stress.py
python experiments/exp_d2829_next_token.py
python experiments/exp_d2830_word_learning.py
python experiments/exp_d2838_compositional_generation.py --summary
python experiments/exp_d2839_sequence_chain.py --summary
python experiments/exp_revision_chain3.py
python demo.py
python conversation_demo.py
python real_text_demo.py
python experiments/exp_projected_address_sweep.py --dim 512 --addr-dims 64 128 256 512 --seeds 0 1 2 --items 300 --probes 120 --noise 0.5
python experiments/exp_projected_address_sweep.py --dim 512 --addr-dims 64 128 256 512 --seeds 0 1 2 --items 300 --probes 120 --noise 1.0
```

## Current Findings

- Unit tests: 23 passed.
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
- D-2838 compositional generation benchmark:
  - Current repo benchmark at `D={64,128,256,512,2048}`, 150 entities, and 3
    seeds reaches exact retrieval 1.0, HRR-native EM 1.0, and linear-head EM
    1.0.
  - This makes linear decoding over retrieved HRR value vectors a verified repo
    capability in the controlled 2-token setting.
- D-2839 sequence-chain benchmark:
  - Current repo benchmark at 5 families, 4 rules per family, sequence length
    10, and 3 seeds shows a hard prefix transition.
  - `K={1,2}` stays at 0.25 EM while `K={3,5,7,10}` reaches 1.0 EM.
- D-2838 generation adapter prototype:
  - The shared compositional decoder now lives in `generation/` and is used by
    both the benchmark and the frozen adapter path.
  - `demo.py` now answers a controlled compositional query with
    `entity_demo has property silver signal.`
- Web UI demo prototype:
  - `web.py` plus `web_static/` now serve a local browser dashboard using the
    same visual system and 3D fact graph pattern as `nexus-16`.
  - The UI exposes seeded fact memory, structured SVO querying, Gemini-backed
    text ingestion when configured, demo reset, and the controlled compositional
    value decode card.
- Projected address key-family sweep:
  - Bounded repo run at `dim=512`, 300 items, 3 seeds, `addr_dim={64,128,256,512}`.
  - At `noise=0.5`, all families reached 1.0 exact and noisy top-1.
  - At `noise=1.0`, low address dimensions showed modest top-1 degradation,
    especially at `addr_dim=64`.
  - Candidate-pool contamination separated key families: one-hot was cleanest,
    HRR SVO improved strongly with `addr_dim`, HRR n-gram stayed more ambiguous,
    and continuous keys carried the highest stale-candidate load.
- Superseding lab nuance:
  - D-2831 showed that `addr_dim=64` is a bottleneck for projected SDM addressing: HRR dimensions 512, 1024, and 2048 all failed under that projection setting.
  - D-2832 showed continuous embedding n-gram keys also fail under the same `addr_dim=64` bottleneck.
  - This repo's positive n-gram and sentence-memory results use full-vector AMM-style retrieval, so they should not be over-read as proof that the lab SDM projection recipe is solved.
  - D-2837's positive one-hot SDM result should be treated as a separate key-family condition, not as evidence that HRR or continuous projected-address routing is solved.

## Interpretation

The PoC is working as a memory-first HRR language substrate. The important
distinction is that full-vector nearest-neighbor AMM does not reproduce the
low-dimensional SDM failure by itself. The collision/capacity problem appears
when retrieval is routed through compressed address signatures before vector
scoring, which is closer to the SDM/AMM address-space constraint described in
the findings. D-2831 and D-2832 sharpen this further: the next critical
research step is an explicit `addr_dim` sweep for projected SDM retrieval.
