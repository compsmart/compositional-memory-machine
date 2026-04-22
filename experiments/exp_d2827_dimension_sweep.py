from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import build_memory, evaluate_known
from experiments.exp_d2825_composition import run as run_composition


def run(dims: tuple[int, ...] = (512, 1024, 2048, 4096), seeds: tuple[int, ...] = (0, 1, 2)) -> list[dict[str, float]]:
    rows = []
    for dim in dims:
        for seed in seeds:
            encoder, memory = build_memory(dim=dim, seed=seed, cycles=10)
            known = evaluate_known(encoder, memory)
            comp = run_composition(dim=dim, seed=seed)
            rows.append(
                {
                    "dim": float(dim),
                    "seed": float(seed),
                    "known_top1": known["top1"],
                    "known_min_margin": known["min_margin"],
                    "composition_cluster_em": comp["cluster_em"],
                    "composition_margin": comp["mean_margin"],
                }
            )
    return rows


if __name__ == "__main__":
    for row in run():
        print(row)
