from __future__ import annotations

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
from experiments.exp_chunked_multihop import run as run_chunked_multihop
from experiments.exp_d2838_compositional_generation import run as run_compositional_generation
from experiments.exp_d2839_sequence_chain import summarize as summarize_sequence_chain
from experiments.exp_d2839_sequence_chain import run as run_sequence_chain
from experiments.exp_d2836_episodic_memory import run as run_episodic_memory
from experiments.exp_temporal_state_tracking import run as run_temporal_state_tracking
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
    assert by_case["coding_python_function"]["score"] == 0.0

    summary_rows = summarize_conversation_benchmark(rows)
    overall = next(row for row in summary_rows if row["summary_type"] == "overall")
    implemented = next(row for row in summary_rows if row["summary_type"] == "track" and row["track"] == "implemented")
    frontier = next(row for row in summary_rows if row["summary_type"] == "track" and row["track"] == "frontier")

    assert overall["mean_score"] > 0.8
    assert implemented["mean_score"] == 1.0
    assert frontier["mean_score"] == 0.0

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


def test_generation_boundary_experiment_smoke() -> None:
    rows = run_generation_boundary(dim=4096, seeds=(0,), n_sequences_values=(50,))

    assert all(float(row["seq_em"]) < 0.2 for row in rows)
    assert all(float(row["tok_acc"]) < 0.8 for row in rows)


def test_hierarchical_syntax_experiment_smoke() -> None:
    rows = run_hierarchical_syntax(dim=4096, seeds=(0,), depths=(2, 3), sentence_counts=(25,))
    by_depth = {int(row["depth"]): row for row in rows}

    assert by_depth[2]["main_acc"] >= 0.9
    assert by_depth[3]["main_acc"] >= 0.8


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
