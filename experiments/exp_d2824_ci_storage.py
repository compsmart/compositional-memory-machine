from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import build_memory, evaluate_known


def run(dim: int = 2048, seeds: tuple[int, ...] = (0, 1, 2), cycles: int = 10) -> list[dict[str, float]]:
    results = []
    for seed in seeds:
        encoder, memory = build_memory(dim=dim, seed=seed, cycles=cycles)
        metrics = evaluate_known(encoder, memory)
        metrics.update({"seed": float(seed), "dim": float(dim), "cycles": float(cycles)})
        results.append(metrics)
    return results


if __name__ == "__main__":
    for row in run():
        print(row)
