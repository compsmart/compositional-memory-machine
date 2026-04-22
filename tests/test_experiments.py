from __future__ import annotations

from experiments.exp_d2824_ci_storage import run as run_ci
from experiments.exp_d2825_composition import run as run_composition
from experiments.exp_collision_stress import run as run_collision_stress
from experiments.exp_addr_dim_sweep import run as run_addr_dim_sweep


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


def test_addr_dim_sweep_smoke() -> None:
    rows = run_addr_dim_sweep(addr_dims=(64, 128), seeds=(0,), hrr_dim=256, domains=2, facts_per_domain=8)

    assert len(rows) == 2
    assert {row["addr_dim"] for row in rows} == {64.0, 128.0}
    assert all(0.0 <= row["forgetting"] <= 1.0 for row in rows)
    assert all(row["write_mode"] == "overwrite" for row in rows)
