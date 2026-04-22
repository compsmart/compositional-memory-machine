from __future__ import annotations

from experiments.exp_d2824_ci_storage import run as run_ci
from experiments.exp_d2825_composition import run as run_composition
from experiments.exp_d2836_episodic_memory import run as run_episodic_memory
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
