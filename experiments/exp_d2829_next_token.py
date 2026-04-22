from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from language import NGramLanguageMemory


TRAINING = [
    ["the", "doctor", "treats", "the", "patient"],
    ["the", "nurse", "monitors", "the", "patient"],
    ["the", "chef", "prepares", "the", "meal"],
    ["the", "pilot", "flies", "the", "plane"],
]


def run(dim: int = 2048, seeds: tuple[int, ...] = (0, 1, 2), cycles: int = 5) -> list[dict[str, float]]:
    rows = []
    for seed in seeds:
        model = NGramLanguageMemory(dim=dim, seed=seed)
        for sequence in TRAINING:
            model.learn_sequence(sequence, cycles=cycles)

        seen = [
            ("the", "doctor", "treats"),
            ("doctor", "treats", "the"),
            ("the", "chef", "prepares"),
            ("chef", "prepares", "the"),
        ]
        familiar = [
            ("a", "doctor", "treats"),
            ("skilled", "chef", "prepares"),
        ]
        novel = [
            ("doctor", "prepares"),
            ("chef", "flies"),
        ]

        seen_correct = sum(int(model.predict(left, right).token == expected) for left, right, expected in seen)
        familiar_correct = sum(
            int(model.predict(left, right, min_confidence=0.25).token == expected)
            for left, right, expected in familiar
        )
        novel_hits = sum(int(model.predict(left, right, min_confidence=0.6).token is not None) for left, right in novel)
        familiar_scores = [model.predict(left, right, min_confidence=0.0).confidence for left, right, _expected in familiar]

        rows.append(
            {
                "dim": float(dim),
                "seed": float(seed),
                "cycles": float(cycles),
                "seen_em": seen_correct / len(seen),
                "familiar_em": familiar_correct / len(familiar),
                "novel_hit_rate": novel_hits / len(novel),
                "mean_familiar_cosine": sum(familiar_scores) / len(familiar_scores),
            }
        )
    return rows


if __name__ == "__main__":
    for row in run():
        print(row)
