from __future__ import annotations

from pathlib import Path

from experiments.exp_d2824_ci_storage import run as run_ci
from experiments.exp_d2825_composition import run as run_composition
from experiments.exp_d2849_probabilistic_next_token import run as run_probabilistic_next_token
from experiments.exp_d2850_temporal_role_binding import run as run_temporal_role_binding
from experiments.exp_d2851_pragmatic_roles import run as run_pragmatic_roles
from experiments.exp_d2852_narrative_chunking import run as run_narrative_chunking
from experiments.exp_d2854_generation_boundary import run as run_generation_boundary
from experiments.exp_d2855_hierarchical_syntax import run as run_hierarchical_syntax
from experiments.exp_d2856_failure_boundary import run as run_failure_boundary
from experiments.exp_d2857_language_revision import run as run_language_revision
from experiments.exp_d2872_dynamic_overwrite_scaling import (
    run as run_dynamic_overwrite_scaling,
    summarize as summarize_dynamic_overwrite_scaling,
)
from experiments.exp_codebase_memory import run as run_codebase_memory
from experiments.exp_chunked_multihop import run as run_chunked_multihop
from experiments.exp_d2846_sdm_nlocs import run as run_sdm_nlocs, summarize as summarize_sdm_nlocs
from experiments.exp_d2838_compositional_generation import run as run_compositional_generation
from experiments.exp_d2839_sequence_chain import summarize as summarize_sequence_chain
from experiments.exp_d2839_sequence_chain import run as run_sequence_chain
from experiments.exp_d2836_episodic_memory import run as run_episodic_memory
from experiments.exp_large_document_memory import run as run_large_document_memory
from experiments.exp_sequential_unbinding_scaling import (
    fitted_relation_hop_bases,
    run as run_sequential_unbinding_scaling,
    summarize as summarize_sequential_unbinding_scaling,
)
from experiments.exp_structural_generalization import run as run_structural_generalization
from experiments.exp_temporal_state_tracking import run as run_temporal_state_tracking
from experiments.exp_temporal_ordering_frontier import (
    run as run_temporal_ordering_frontier,
    summarize as summarize_temporal_ordering_frontier,
)
from experiments.exp_truth_provenance_conflicts import run as run_truth_provenance_conflicts
from experiments.exp_conversation_benchmark import (
    build_results_payload as build_conversation_benchmark_payload,
    compare_summary_rows as compare_conversation_benchmark_summary,
    load_results as load_conversation_benchmark_results,
    render_markdown_report as render_conversation_benchmark_report,
    run as run_conversation_benchmark,
    save_results as save_conversation_benchmark_results,
    summarize as summarize_conversation_benchmark,
)
from experiments.exp_collision_stress import run as run_collision_stress
from experiments.exp_projected_address_sweep import (
    render_markdown_report,
    run as run_projected_address_sweep,
    summarize as summarize_projected_address_sweep,
)
from ingestion import ExtractedFact, write_fact_jsonl
from ingestion.hf_corpora import StructuredFactRecord


def test_ci_storage_experiment_smoke() -> None:
    rows = run_ci(dim=1024, seeds=(0,), cycles=2)

    assert rows[0]["top1"] == 1.0


def test_composition_experiment_beats_baseline() -> None:
    row = run_composition(dim=2048, seed=0)

    assert row["cluster_em"] > row["random_baseline"]


def test_collision_stress_smoke() -> None:
    rows = run_collision_stress(dims=(512,), seeds=(0,), domains=2, facts_per_domain=50, probes=20, noise=0.5)

    assert rows[0]["facts"] == 100.0
    assert 0.0 <= rows[0]["full_vector_noisy_top1"] <= 1.0
    assert 0.0 <= rows[0]["address_noisy_top1"] <= 1.0


def test_projected_address_sweep_smoke() -> None:
    rows = run_projected_address_sweep(
        dim=128,
        addr_dims=(16, 32),
        families=("one_hot", "hrr_svo"),
        seeds=(0, 1),
        n_items=(20, 40),
        probes=10,
        noise=(0.0, 0.25),
    )

    assert len(rows) == 32
    assert {row["family"] for row in rows} == {"one_hot", "hrr_svo"}
    assert all(0.0 <= float(row["exact_top1"]) <= 1.0 for row in rows)
    assert all(0.0 <= float(row["noisy_top1"]) <= 1.0 for row in rows)

    summary_rows = summarize_projected_address_sweep(rows)

    assert len(summary_rows) == 16
    assert all(float(row["runs"]) == 2.0 for row in summary_rows)

    report = render_markdown_report(
        summary_rows,
        dim=128,
        addr_dims=(16, 32),
        families=("one_hot", "hrr_svo"),
        seeds=(0, 1),
        item_counts=(20, 40),
        probes=10,
        noise_levels=(0.0, 0.25),
    )

    assert "## Aggregate Results" in report
    assert "| one_hot | 16 |" in report


def test_sdm_nlocs_experiment_smoke() -> None:
    rows = run_sdm_nlocs(
        dim=256,
        addr_dim=32,
        n_locs_values=(16, 32),
        n_domains=4,
        steps_per_domain=20,
        audit_stride=10,
        probe_samples=2,
        seeds=(0,),
        gate_betas=(-2.0,),
        route_top_ks=(1, 2),
    )

    assert len(rows) == 4
    assert all(0.0 <= row["retrieval_em"] <= 1.0 for row in rows)
    assert all(row["peak_mem_mb"] > 0.0 for row in rows)
    assert all(0.0 <= row["candidate_shard_hit_rate"] <= 1.0 for row in rows)

    summary_rows = summarize_sdm_nlocs(rows)

    assert len(summary_rows) == 4
    assert {int(row["n_locs"]) for row in summary_rows} == {16, 32}
    assert all("candidate_read_rescue_rate" in row for row in summary_rows)
    assert all("read_path_failure_rate" in row for row in summary_rows)
    assert all(0.0 <= row["candidate_read_rescue_rate"] <= 1.0 for row in summary_rows)
    assert all(0.0 <= row["read_path_failure_rate"] <= 1.0 for row in summary_rows)


def test_conversation_benchmark_smoke(tmp_path) -> None:
    rows = run_conversation_benchmark(
        preset="smoke",
        episodic_seeds=(0,),
        temporal_seeds=(0,),
    )
    by_case = {str(row["case_id"]): row for row in rows}

    assert len(rows) == 7
    assert by_case["memory_fact_recall"]["score"] == 1.0
    assert by_case["alias_normalization_ingest"]["score"] == 1.0
    assert by_case["coding_python_function"]["score"] == 1.0

    summary_rows = summarize_conversation_benchmark(rows)
    overall = next(row for row in summary_rows if row["summary_type"] == "overall")
    implemented = next(row for row in summary_rows if row["summary_type"] == "track" and row["track"] == "implemented")
    frontier = next(row for row in summary_rows if row["summary_type"] == "track" and row["track"] == "frontier")

    assert overall["mean_score"] > 0.8
    assert implemented["mean_score"] == 1.0
    assert frontier["mean_score"] == 1.0

    config = {
        "preset": "smoke",
        "chat_dim": 2048,
        "chat_seed": 0,
        "episodic_dim": 2048,
        "episodic_seeds": (0,),
        "episodic_sessions": 3,
        "episodic_turns": 10,
        "episodic_facts_per_turn": 3,
        "temporal_dim": 2048,
        "temporal_seeds": (0,),
        "case_ids": tuple(by_case),
    }
    payload = build_conversation_benchmark_payload(rows=rows, summary_rows=summary_rows, config=config)
    results_path = tmp_path / "conversation_benchmark.json"
    save_conversation_benchmark_results(results_path, payload)
    loaded = load_conversation_benchmark_results(results_path)
    compared = compare_conversation_benchmark_summary(summary_rows, loaded["summary"])

    assert loaded["config"]["preset"] == "smoke"
    assert next(row for row in compared if row["summary_key"] == "overall:overall")["mean_score_delta"] == 0.0

    report = render_conversation_benchmark_report(
        rows,
        summary_rows,
        config=config,
        previous_summary_rows=loaded["summary"],
    )

    assert "## Category Scorecard" in report
    assert "| implemented | 1.000 | 1.000 | +0.000 |" in report
    assert "coding_python_function" in report


def test_conversation_benchmark_records_preload_config(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "facts.jsonl"
    write_fact_jsonl(
        jsonl_path,
        [
            StructuredFactRecord(
                fact=ExtractedFact(
                    subject="Grossglockner",
                    relation="first_climbed_on",
                    object="1800",
                    confidence=0.8,
                    kind="explicit",
                    source="fixture",
                    source_id="fixture:1",
                ),
                domain="hf_wikipedia_kg",
            )
        ],
    )

    rows = run_conversation_benchmark(
        preset="smoke",
        episodic_seeds=(0,),
        temporal_seeds=(0,),
        preload_jsonl=jsonl_path,
        preload_limit=1,
    )
    summary_rows = summarize_conversation_benchmark(rows)
    payload = build_conversation_benchmark_payload(
        rows=rows,
        summary_rows=summary_rows,
        config={
            "preset": "smoke",
            "chat_dim": 2048,
            "chat_seed": 0,
            "episodic_dim": 2048,
            "episodic_seeds": (0,),
            "episodic_sessions": 3,
            "episodic_turns": 10,
            "episodic_facts_per_turn": 3,
            "temporal_dim": 2048,
            "temporal_seeds": (0,),
            "preload_jsonl": str(jsonl_path),
            "preload_limit": 1,
            "case_ids": tuple(str(row["case_id"]) for row in rows),
        },
    )

    assert len(rows) == 7
    assert payload["config"]["preload_jsonl"] == str(jsonl_path)
    assert payload["config"]["preload_limit"] == 1


def test_episodic_memory_experiment_smoke() -> None:
    rows = run_episodic_memory(dim=512, seeds=(0,), sessions=2, turns=4, facts_per_turn=3)

    assert rows[0]["immediate_em"] == 1.0
    assert rows[0]["distant_em"] == 1.0
    assert rows[0]["cross_session_em"] == 1.0
    assert rows[0]["revision_em"] == 1.0
    assert rows[0]["retention_em"] == 1.0
    assert rows[0]["speaker_intent_em"] == 1.0
    assert rows[0]["assistant_answer_em"] == 1.0
    assert rows[0]["correction_em"] == 1.0


def test_compositional_generation_experiment_smoke() -> None:
    rows = run_compositional_generation(dims=(64, 128), seeds=(0,), n_entities=60)

    assert len(rows) == 2
    assert all(row["exact_retrieval"] == 1.0 for row in rows)
    assert all(row["hrr_native_em"] >= 0.95 for row in rows)
    assert all(row["linear_head_em"] == 1.0 for row in rows)


def test_sequence_chain_experiment_smoke() -> None:
    rows = run_sequence_chain(seeds=(42,), prefix_lengths=(1, 2, 3, 10))
    summary = summarize_sequence_chain(rows)
    by_prefix = {int(row["prefix_len"]): row for row in summary}

    assert by_prefix[1]["mean_em"] == 0.25
    assert by_prefix[2]["mean_em"] == 0.25
    assert by_prefix[3]["mean_em"] == 1.0
    assert by_prefix[10]["mean_em"] == 1.0


def test_chunked_multihop_experiment_smoke() -> None:
    rows = run_chunked_multihop(dim=2048, seeds=(0,), chunk_sizes=(2,))

    assert rows[0]["hop2_em"] == 1.0
    assert rows[0]["hop3_em"] == 1.0
    assert rows[0]["cross_domain_em"] == 1.0
    assert rows[0]["chunk_count"] >= 2.0
    assert rows[0]["shared_entity_pool"] == 1.0
    assert rows[0]["explicit_chain_construction"] == 1.0


def test_temporal_state_tracking_experiment_smoke() -> None:
    rows = run_temporal_state_tracking(dim=2048, seeds=(0,))

    assert rows[0]["latest_state_em"] == 1.0
    assert rows[0]["history_em"] == 1.0
    assert rows[0]["historical_em"] == 1.0


def test_dynamic_overwrite_scaling_experiment_smoke() -> None:
    rows = run_dynamic_overwrite_scaling(
        dims=(256, 1024),
        entity_counts=(10,),
        update_counts=(3,),
        properties=("location", "action"),
        seeds=(0,),
    )

    assert len(rows) == 4
    assert all(0.0 <= row["mean_em"] <= 1.0 for row in rows)
    assert {str(row["condition"]) for row in rows} == {"no_reset", "perkey_reset"}

    summary_rows = summarize_dynamic_overwrite_scaling(rows)

    assert len(summary_rows) == 4
    assert {int(row["dim"]) for row in summary_rows} == {256, 1024}
    by_condition = {str(row["condition"]): row for row in summary_rows if int(row["dim"]) == 256}
    assert by_condition["perkey_reset"]["mean_em"] >= by_condition["no_reset"]["mean_em"]
    assert "mean_em_delta_vs_no_reset" in by_condition["perkey_reset"]
    assert by_condition["perkey_reset"]["mean_em_delta_vs_no_reset"] >= 0.0
    assert by_condition["no_reset"]["mean_em_delta_vs_no_reset"] == 0.0


def test_truth_provenance_conflicts_experiment_smoke() -> None:
    rows = run_truth_provenance_conflicts(dim=2048, seeds=(0,))

    assert rows[0]["current_truth_em"] == 1.0
    assert rows[0]["history_em"] == 1.0
    assert rows[0]["competing_evidence_em"] == 1.0
    assert rows[0]["provenance_em"] == 1.0
    assert rows[0]["unresolved_refusal_em"] == 1.0


def test_large_document_memory_experiment_smoke() -> None:
    rows = run_large_document_memory(dim=2048, seeds=(0,))

    assert rows[0]["written_facts"] >= 10.0
    assert rows[0]["chunk_count"] >= 1.0
    assert rows[0]["recall_em"] == 1.0
    assert rows[0]["chain_em"] == 1.0
    assert rows[0]["current_truth_em"] == 1.0
    assert rows[0]["history_em"] == 1.0
    assert rows[0]["competing_evidence_em"] == 1.0
    assert rows[0]["refusal_em"] == 1.0


def test_probabilistic_next_token_experiment_smoke() -> None:
    rows = run_probabilistic_next_token(dim=2048, seeds=(0,))

    assert rows[0]["top1_correct"] == 1.0
    assert rows[0]["top3_hit"] == 1.0
    assert abs(rows[0]["probability_sum"] - 1.0) < 1e-6


def test_temporal_role_binding_experiment_smoke() -> None:
    rows = run_temporal_role_binding(dim=4096, seeds=(0,), n_events_values=(25, 50, 200))
    by_events = {int(row["n_events"]): row for row in rows}

    assert by_events[25]["role_acc"] >= 0.95
    assert by_events[50]["role_acc"] >= 0.95
    assert by_events[200]["role_acc"] < by_events[50]["role_acc"]


def test_pragmatic_roles_experiment_smoke() -> None:
    rows = run_pragmatic_roles(dim=4096, seeds=(0,), sentence_counts=(50,))
    row = rows[0]

    assert row["core_acc"] >= 0.95
    assert row["nuanced_acc"] > row["core_acc"]


def test_narrative_chunking_experiment_smoke() -> None:
    rows = run_narrative_chunking(dim=4096, seeds=(0,), lengths=(200,))
    by_strategy = {str(row["strategy"]): row for row in rows}

    assert by_strategy["chunked"]["recall"] == 1.0
    assert by_strategy["chunked"]["latest_state"] == 1.0
    assert by_strategy["flat"]["recall"] < by_strategy["chunked"]["recall"]


def test_temporal_ordering_frontier_experiment_smoke() -> None:
    rows = run_temporal_ordering_frontier(dims=(1024,), event_counts=(50,), seeds=(0,))
    by_strategy = {str(row["strategy"]): row for row in rows}

    assert by_strategy["flat_temporal_roles"]["pairwise_order_em"] > 0.0
    assert by_strategy["explicit_pair_links"]["pairwise_order_em"] > by_strategy["overwrite_only"]["pairwise_order_em"]
    assert by_strategy["overwrite_only"]["pairwise_order_em"] == 0.0
    assert by_strategy["hybrid_chunked_plus_latest_cache"]["latest_state_em"] >= 0.9
    assert by_strategy["hybrid_chunked_plus_latest_cache"]["pairwise_order_em"] >= by_strategy["chunked_temporal_roles"]["pairwise_order_em"]

    summary_rows = summarize_temporal_ordering_frontier(rows)
    hybrid_summary = next(row for row in summary_rows if str(row["strategy"]) == "hybrid_chunked_plus_latest_cache")
    assert hybrid_summary["balanced_score"] >= 0.9


def test_generation_boundary_experiment_smoke() -> None:
    rows = run_generation_boundary(dim=4096, seeds=(0,), n_sequences_values=(50,))

    assert all(float(row["seq_em"]) < 0.2 for row in rows)
    assert all(float(row["tok_acc"]) < 0.8 for row in rows)


def test_hierarchical_syntax_experiment_smoke() -> None:
    rows = run_hierarchical_syntax(dim=4096, seeds=(0,), depths=(2, 3), sentence_counts=(25,))
    by_depth = {int(row["depth"]): row for row in rows}

    assert by_depth[2]["main_acc"] >= 0.9
    assert by_depth[3]["main_acc"] >= 0.8


def test_sequential_unbinding_scaling_experiment_smoke() -> None:
    rows = run_sequential_unbinding_scaling(
        dims=(256, 1024),
        hop_depths=(1, 2, 3),
        syntax_depths=(1, 2, 3),
        seeds=(0,),
        n_chains=8,
        n_sentences=8,
    )

    relation_rows = [row for row in rows if row["task"] == "relation_chain"]
    syntax_rows = [row for row in rows if row["task"] == "hierarchical_syntax"]

    assert len(relation_rows) == 6
    assert len(syntax_rows) == 6
    assert all(0.0 <= float(row["em"]) <= 1.0 for row in rows)

    summary_rows = summarize_sequential_unbinding_scaling(rows)
    fitted_rows = fitted_relation_hop_bases(summary_rows)
    by_dim = {int(row["dim"]): row for row in fitted_rows}
    assert by_dim[1024]["fitted_hop_base"] > by_dim[256]["fitted_hop_base"]
    assert by_dim[1024]["fitted_hop_base"] >= 0.85


def test_failure_boundary_experiment_smoke() -> None:
    rows = run_failure_boundary(dim=4096, seeds=(0,), similarities=(0.95, 0.4), conflict_sizes=(50,))
    similarity_rows = {float(row["similarity"]): row for row in rows if row["mode"] == "similarity"}
    overwrite_row = next(row for row in rows if row["mode"] == "overwrite")

    assert similarity_rows[0.4]["correct_retrieval"] == 1.0
    assert similarity_rows[0.95]["correct_retrieval"] < 0.7
    assert 0.4 <= overwrite_row["old_contamination"] <= 0.6


def test_language_revision_experiment_smoke() -> None:
    rows = run_language_revision(dims=(256, 1024), seeds=(0,), n_entities=20)
    by_condition = {str(row["condition"]): row for row in rows if int(row["dim"]) == 256}

    assert by_condition["no_reset"]["revised_em"] < 0.7
    assert by_condition["perkey_reset"]["revised_em"] == 1.0
    assert by_condition["perkey_reset"]["retained_em"] == 1.0


def test_codebase_memory_experiment_smoke() -> None:
    row = run_codebase_memory(dim=1024, seed=0)

    assert row["file_count"] == 3.0
    assert row["written_facts"] >= 5.0
    assert row["imports_em"] == 1.0
    assert row["calls_em"] == 1.0
    assert row["defined_in_em"] == 1.0


def test_structural_generalization_experiment_smoke() -> None:
    row = run_structural_generalization(seeds=(42,))

    assert row["prefix_threshold_em"] == 1.0
    assert row["hierarchical_depth3_acc"] >= 0.8
    assert row["pattern_surface_em"] == 1.0
