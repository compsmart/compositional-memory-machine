from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.episodic import ConversationFact, EpisodicMemory


def run(*, dim: int = 2048, seeds: tuple[int, ...] = (42, 123)) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for seed in seeds:
        memory = EpisodicMemory(dim=dim, seed=seed)
        memory.state_fact(ConversationFact(0, 0, "robot", "location", "lab"))
        memory.state_fact(ConversationFact(0, 1, "robot", "action", "inspect"))
        memory.revise_fact(ConversationFact(0, 2, "robot", "location", "field"))

        location_history = memory.recall_history("robot", "location")
        location_revisions = memory.graph.history("robot", "location")
        first_revision = location_revisions[0].revision if location_revisions else 0

        rows.append(
            {
                "seed": float(seed),
                "latest_state_em": float(memory.recall_current("robot", "location") == "field"),
                "history_em": float(location_history == ["lab", "field"]),
                "historical_em": float(memory.recall_at_turn("robot", "location", revision=first_revision) == "lab"),
                "retention_em": float(
                    memory.recall_evidence(ConversationFact(0, 2, "robot", "location", "field"))
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    args = parser.parse_args()

    for row in run(dim=args.dim, seeds=tuple(args.seeds)):
        print(row)


if __name__ == "__main__":
    main()
