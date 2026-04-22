from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factgraph import FactGraph


def build_chain() -> FactGraph:
    graph = FactGraph()
    graph.write("a", "r1", "b")
    graph.write("b", "r2", "c")
    graph.write("c", "r3", "d")
    return graph


def run() -> dict[str, float]:
    positions = {
        "entry": ("a", "r1", "b2", ["a", "b2"]),
        "middle": ("b", "r2", "c2", ["a", "b", "c2"]),
        "terminal": ("c", "r3", "d2", ["a", "b", "c", "d2"]),
    }
    correct = 0
    for _name, (source, relation, target, expected_prefix) in positions.items():
        graph = build_chain()
        graph.revise(source, relation, target)
        relations = ["r1", "r2", "r3"][: len(expected_prefix) - 1]
        path = graph.follow_chain("a", relations)
        correct += int(path == expected_prefix)
    return {"chain3_revision_em": correct / len(positions), "positions": float(len(positions))}


if __name__ == "__main__":
    print(run())
