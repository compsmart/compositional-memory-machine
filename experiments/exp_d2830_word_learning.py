from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from language import ContextExample, WordLearningMemory


def _build_memory(dim: int, seed: int) -> WordLearningMemory:
    memory = WordLearningMemory(dim=dim, seed=seed)
    for action in ["eat", "drink", "consume"]:
        memory.add_known_action(action, "ingest", "ingest")
    for action in ["run", "walk", "travel"]:
        memory.add_known_action(action, "move", "move")
    for action in ["say", "tell", "explain"]:
        memory.add_known_action(action, "communicate", "communicate")
    return memory


def run(dim: int = 2048, seeds: tuple[int, ...] = (0, 1, 2)) -> list[dict[str, float]]:
    rows = []
    examples = [
        ContextExample("child", "dax", "apple", "ingest"),
        ContextExample("student", "dax", "sandwich", "ingest"),
        ContextExample("doctor", "dax", "meal", "ingest"),
        ContextExample("bird", "dax", "seed", "ingest"),
        ContextExample("chef", "dax", "soup", "ingest"),
    ]
    cycle2_examples = [
        ContextExample("runner", "blick", "track", "move"),
        ContextExample("traveler", "blick", "road", "move"),
        ContextExample("child", "blick", "path", "move"),
        ContextExample("pilot", "blick", "route", "move"),
        ContextExample("hiker", "blick", "trail", "move"),
    ]

    for seed in seeds:
        memory = _build_memory(dim, seed)
        learned = memory.learn_word("dax", examples)
        retained_before = memory.retrieve_word("dax")
        learned2 = memory.learn_word("blick", cycle2_examples)
        retained_after = memory.retrieve_word("dax")

        rows.append(
            {
                "dim": float(dim),
                "seed": float(seed),
                "dax_cluster_correct": float(learned["cluster"] == "ingest"),
                "blick_cluster_correct": float(learned2["cluster"] == "move"),
                "retention": float(retained_before["cluster"] == retained_after["cluster"] == "ingest"),
                "dax_plausibility": memory.plausibility("dax", "eat"),
                "dax_implausibility": memory.plausibility("dax", "run"),
            }
        )
    return rows


if __name__ == "__main__":
    for row in run():
        print(row)
