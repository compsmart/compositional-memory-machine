from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from language import ProjectedNGramLanguageMemory


TRAINING = [
    ["the", "doctor", "treats", "the", "patient"],
    ["the", "nurse", "monitors", "the", "patient"],
    ["the", "chef", "prepares", "the", "meal"],
    ["the", "pilot", "flies", "the", "plane"],
    ["the", "teacher", "explains", "the", "lesson"],
    ["the", "judge", "reviews", "the", "evidence"],
    ["the", "mechanic", "inspects", "the", "engine"],
    ["the", "baker", "bakes", "the", "bread"],
]


def run(
    *,
    dim: int = 2048,
    addr_dim: int = 512,
    seeds: tuple[int, ...] = (0, 1, 2),
    cycles: int = 5,
    n_locations: int = 2048,
    write_k: int = 8,
    read_k: int = 128,
) -> list[dict[str, float]]:
    rows = []
    for seed in seeds:
        model = ProjectedNGramLanguageMemory(
            dim=dim,
            seed=seed,
            addr_dim=addr_dim,
            n_locations=n_locations,
            write_k=write_k,
            read_k=read_k,
        )
        for _cycle in range(cycles):
            for sequence in TRAINING:
                model.learn_sequence(sequence, cycles=1)

        seen = [
            ("the", "doctor", "treats"),
            ("doctor", "treats", "the"),
            ("the", "chef", "prepares"),
            ("chef", "prepares", "the"),
            ("the", "teacher", "explains"),
            ("judge", "reviews", "the"),
        ]
        familiar = [
            ("a", "doctor", "treats"),
            ("skilled", "chef", "prepares"),
            ("strict", "judge", "reviews"),
            ("careful", "mechanic", "inspects"),
        ]
        novel = [
            ("doctor", "prepares"),
            ("chef", "flies"),
            ("judge", "bakes"),
            ("teacher", "inspects"),
        ]

        seen_correct = sum(int(model.predict(left, right).token == expected) for left, right, expected in seen)
        seen_predictions = [model.predict(left, right, min_confidence=0.0) for left, right, _expected in seen]
        familiar_predictions = [model.predict(left, right, min_confidence=0.0) for left, right, _expected in familiar]
        novel_predictions = [model.predict(left, right, min_confidence=0.0) for left, right in novel]
        familiar_correct = sum(
            int(prediction.token == expected)
            for prediction, (_left, _right, expected) in zip(familiar_predictions, familiar)
        )
        novel_hits = sum(int(prediction.token is not None) for prediction in novel_predictions)
        calibrated_threshold = (
            sum(pred.confidence for pred in familiar_predictions) / len(familiar_predictions)
            + sum(pred.confidence for pred in novel_predictions) / len(novel_predictions)
        ) / 2
        calibrated_familiar_correct = sum(
            int(prediction.confidence >= calibrated_threshold and prediction.token == expected)
            for prediction, (_left, _right, expected) in zip(familiar_predictions, familiar)
        )
        calibrated_novel_hits = sum(
            int(prediction.confidence >= calibrated_threshold)
            for prediction in novel_predictions
        )

        rows.append(
            {
                "dim": float(dim),
                "addr_dim": float(addr_dim),
                "seed": float(seed),
                "cycles": float(cycles),
                "n_locations": float(n_locations),
                "write_k": float(write_k),
                "read_k": float(read_k),
                "seen_em": seen_correct / len(seen),
                "familiar_em": familiar_correct / len(familiar),
                "novel_hit_rate": novel_hits / len(novel),
                "calibrated_threshold": calibrated_threshold,
                "calibrated_familiar_em": calibrated_familiar_correct / len(familiar),
                "calibrated_novel_hit_rate": calibrated_novel_hits / len(novel),
                "mean_seen_score": sum(pred.confidence for pred in seen_predictions) / len(seen_predictions),
                "mean_familiar_score": sum(pred.confidence for pred in familiar_predictions)
                / len(familiar_predictions),
                "mean_novel_score": sum(pred.confidence for pred in novel_predictions) / len(novel_predictions),
            }
        )
    return rows


if __name__ == "__main__":
    for row in run():
        print(row)
