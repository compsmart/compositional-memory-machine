# Lab Claim Validation

This report compares the scorecard claims against two sources:

- live lab findings fetched through `cli/mcp_cli.py`
- repo-local experiments and tests in `experiments/` and `tests/`

The goal is to keep three things separate:

- what the lab has already established
- what this repo reproduced before this validation pass
- what was added here to close the benchmark gap

## Claim Ledger

| Scorecard item | Lab basis | Repo status before | Repo status now |
| --- | --- | --- | --- |
| Probabilistic next-token | `D-2848`, `D-2849` | Direct via `exp_d2849_probabilistic_next_token.py` | Direct and re-rerun locally |
| Language revision / overwrite | `D-2847`, `D-2857` | Partial via `exp_revision_chain3.py`, `exp_d2836_episodic_memory.py`, `FactGraph` revision | Direct repo mirror added with `exp_d2857_language_revision.py` |
| Temporal role binding | `D-2850` | Partial via `exp_temporal_state_tracking.py` and temporal encodings | Direct pure-HRR mirror added with `exp_d2850_temporal_role_binding.py` |
| Sentiment / modality / certainty / negation | `D-2851` | Missing benchmark | Direct repo mirror added with `exp_d2851_pragmatic_roles.py` |
| Hierarchical syntax / recursion | `D-2855` | Partial proxy via `exp_d2839_sequence_chain.py` | Direct recursive-role mirror added with `exp_d2855_hierarchical_syntax.py` |
| Long-context narrative chunking | `D-2852` | Partial proxy via chunked KG and multi-hop | Direct narrative mirror added with `exp_d2852_narrative_chunking.py` |
| Compression / capacity / frontier | `D-2835`, `D-2845`, `D-2846` | Partial via dimension sweep, projected-address sweep, collision stress | Still partial: full SDM `n_locs` lab recipe remains lab-only |
| Hybrid generative decoding boundary | `D-2854` | Missing; repo only had positive bounded `D-2838` decoding | Direct negative mirror added with `exp_d2854_generation_boundary.py` |
| Failure boundary / aliasing | `D-2856` | Partial via collision stress and query refusal budgets | Direct adversarial mirror added with `exp_d2856_failure_boundary.py` |

## Live Lab Evidence

The following findings were pulled directly from the lab and used as the source of truth for setup, metric names, and verdicts:

- `D-2835`: HRR+AMM capacity frontier, with `D>=128` as the robust minimum and `D=256+` saturating the tested conditions.
- `D-2845`: vocabulary-capacity phase transition for the linear decoding head at `D=256`.
- `D-2846`: SDM+AMM `n_locs` floor remains positive down to `n_locs=64`; true failure point not found.
- `D-2847`: pure HRR subtract gives perfect belief revision up to `n=100`.
- `D-2848`: CI-track probabilistic next-token at `D=2048` reaches `top1=0.922`, `top3=1.0`, `rho=0.842`.
- `D-2849`: HSM-track probabilistic multi-modal continuation reaches `top1=0.993` at `D=4096`.
- `D-2850`: flat 4-role temporal world model stays clean through about `n<=25`, remains usable at `n=50`, and collapses by `n=200`.
- `D-2851`: 7-role pragmatic encoding preserves nuanced roles better than core roles at `n=50`.
- `D-2852`: chunked narrative memory breaks the flat `n=50` capacity wall and stays perfect at `n=200`.
- `D-2854`: multi-step autoregressive generation fails decisively even though single-step retrieval is strong.
- `D-2855`: recursive role binding supports depth-3 syntax with usable main-clause recovery.
- `D-2856`: entity aliasing stays clean through moderate similarity and breaks near `sim=0.95`; overwrite without subtract leaves old/new blends.
- `D-2857`: PerKey reset gives perfect revision and retention at the language-memory level.

## Repo-Local Reproduction

### Existing experiments rerun

- `exp_d2849_probabilistic_next_token.py`:
  repo-local controlled setting still returns `top1_correct=1.0`, `top3_hit=1.0`, `probability_sum=1.0` across three seeds. This is stricter and smaller than the lab setup, but it confirms the repo still preserves ranked probabilistic continuations.
- `exp_temporal_state_tracking.py`:
  `latest_state_em=1.0`, `history_em=1.0`, `historical_em=1.0` across the default seeds on the graph-backed episodic path.
- `exp_chunked_multihop.py`:
  `hop2_em=1.0`, `hop3_em=1.0`, `cross_domain_em=1.0` with chunk provenance intact.
- `exp_d2838_compositional_generation.py --summary`:
  `hrr_native_em_mean=1.0`, `linear_head_em_mean=1.0`, `exact_retrieval_mean=1.0` for the bounded value-decoding task.
- `exp_d2839_sequence_chain.py --summary`:
  `K=1,2` stay at chance `0.25`, while `K=3,10` are `1.0`.
- `exp_revision_chain3.py`:
  `chain3_revision_em=1.0`.
- `exp_collision_stress.py`:
  the address-routed path is weaker at lower dimension while full-vector retrieval remains near-perfect, reinforcing the repo's existing projection caveat.

### New claim-validation mirrors added here

- `exp_d2850_temporal_role_binding.py`:
  `role_acc` stays at `1.0` through `n=50`, drops to `0.90` at `n=100`, and to `0.565` at `n=200`. This reproduces the same qualitative flat-memory capacity wall as the lab finding. The repo's existing graph-backed temporal benchmark remains stronger than the pure-HRR mirror on latest-state queries, so the exact latest-state curve is still not a one-for-one replication of the lab protocol.
- `exp_d2851_pragmatic_roles.py`:
  at `n=50`, `core_acc=0.98` and `nuanced_acc=1.0`, reproducing the central claim that pragmatic roles survive superposition at least as well as the core slots in this repo-local mirror.
- `exp_d2852_narrative_chunking.py`:
  at `n=200`, `chunked` reaches `recall=1.0` and `latest_state=1.0`, while `flat` falls to `recall=0.605`. This validates the chunking direction in-project, even though the repo-local mirror does not reproduce the lab's weak temporal-order metric.
- `exp_d2854_generation_boundary.py`:
  at `n=50`, all strategies remain below `seq_em=0.2`, with greedy and beam at `seq_em=0.08`. Token accuracy is higher than the lab's exact value because the repo-local synthetic task includes deterministic pivots, but exact-sequence generation still fails decisively enough to preserve the same architectural boundary: retrieval is not generation.
- `exp_d2855_hierarchical_syntax.py`:
  at `n=25`, `depth=2 main_acc=1.0` and `depth=3 main_acc=0.92`. This validates bounded recursive retrieval in the repo, though the exact depth-by-depth degradation pattern is not numerically identical to the lab result.
- `exp_d2856_failure_boundary.py`:
  similarity stays perfect at `sim=0.4`, falls to `0.567` correct at `sim=0.95`, and overwrite-without-reset converges to about a `50/50` old/new split at `n=50`.
- `exp_d2857_language_revision.py`:
  `perkey_reset` reaches `revised_em=1.0` and `retained_em=1.0` across `D={256,1024,2048}`, while `no_reset` stays in the `0.35-0.50` revised-EM range.

## What Is Now Strongly Supported

- The repo now has direct executable coverage for probabilistic next-token, belief revision, temporal role binding, pragmatic roles, hierarchical syntax, long-context chunking, failure boundaries, and the multi-step generation boundary.
- The strongest repo-local evidence still supports the narrower claim from the README: this project is a language-memory substrate with retrieval, revision, and bounded decoding, not a standalone open-ended language model.
- The lab scorecard claim that HRR is retrieval-only unless paired with a decoder is now reflected in the repo rather than only discussed in prose.

## Remaining Gaps

1. `D-2846` remains lab-only in substance.
   The repo has projected-address and collision stress experiments, but it does not implement the full SDM gating recipe from the positive `n_locs` finding. That claim should still be treated as lab motivation, not repo-local reproduction.

2. The temporal and narrative mirrors are qualitative, not exact protocol clones.
   The repo now mirrors the flat-memory cliff and the chunking rescue, but the exact latest-state and temporal-order numbers differ because the repo benchmark scaffolding is simpler than the lab's full protocol.

3. The hierarchical syntax mirror is a capability validation, not a perfect numeric reproduction.
   It now demonstrates recursive recovery in-project, but the exact depth-3 embedded-clause curve remains an open target.

4. `D-2838` and `D-2854` must stay separate in project messaging.
   The repo can still truthfully claim bounded compositional value decoding. What it cannot claim is successful multi-step autoregressive sequence generation from HRR retrieval alone.

## Capability Enhancements Added

- `hrr/encoder.py` now exposes pragmatic-role and hierarchical-clause encoders.
- `memory/amm.py` now supports explicit key deletion / prefix reset needed for clean overwrite workflows.
- `language/ngram.py` now has multi-step generation helpers, which make the retrieval-vs-generation boundary directly testable.
- New benchmark mirrors:
  - `exp_d2850_temporal_role_binding.py`
  - `exp_d2851_pragmatic_roles.py`
  - `exp_d2852_narrative_chunking.py`
  - `exp_d2854_generation_boundary.py`
  - `exp_d2855_hierarchical_syntax.py`
  - `exp_d2856_failure_boundary.py`
  - `exp_d2857_language_revision.py`
- `tests/test_experiments.py` now covers the new claim-validation experiments.

## Recommended Next Steps

1. Build a stricter SDM-side reproduction for `D-2846` rather than inferring it from the repo's full-vector or projected-address proxies.
2. Refine the temporal/narrative mirrors so latest-state and temporal-order queries match the lab protocol more closely.
3. Push the hierarchical mirror from capability validation toward a closer reproduction of the lab depth-3 degradation curve.
4. Keep README messaging split into:
   - validated here
   - observed in the lab
   - still open
