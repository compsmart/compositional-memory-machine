from __future__ import annotations

from experiments.exp_d2824_ci_storage import run as run_ci
from experiments.exp_d2825_composition import run as run_composition
from experiments.exp_chunked_multihop import run as run_chunked_multihop
from experiments.exp_d2838_compositional_generation import run as run_compositional_generation
from experiments.exp_d2839_sequence_chain import summarize as summarize_sequence_chain
from experiments.exp_d2839_sequence_chain import run as run_sequence_chain
from experiments.exp_d2836_episodic_memory import run as run_episodic_memory
from experiments.exp_temporal_state_tracking import run as run_temporal_state_tracking
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


def test_episodic_memory_experiment_smoke() -> None:
    rows = run_episodic_memory(dim=512, seeds=(0,), sessions=2, turns=4, facts_per_turn=2)

    assert rows[0]["immediate_em"] == 1.0
    assert rows[0]["distant_em"] == 1.0
    assert rows[0]["cross_session_em"] == 1.0
    assert rows[0]["revision_em"] == 1.0
    assert rows[0]["retention_em"] == 1.0


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
    rows = run_chunked_multihop(dim=1024, seeds=(0,), chunk_sizes=(2,))

    assert rows[0]["hop2_em"] == 1.0
    assert rows[0]["hop3_em"] == 1.0
    assert rows[0]["cross_domain_em"] == 1.0
    assert rows[0]["chunk_count"] >= 2.0


def test_temporal_state_tracking_experiment_smoke() -> None:
    rows = run_temporal_state_tracking(dim=512, seeds=(0,))

    assert rows[0]["latest_state_em"] == 1.0
    assert rows[0]["history_em"] == 1.0
    assert rows[0]["historical_em"] == 1.0
