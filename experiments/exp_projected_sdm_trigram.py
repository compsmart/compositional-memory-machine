from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from language import ProjectedTrigramLanguageMemory


RULES = [
    ("doctor", "treats", "patient"),
    ("nurse", "monitors", "patient"),
    ("chef", "prepares", "meal"),
    ("pilot", "flies", "plane"),
    ("teacher", "explains", "lesson"),
    ("judge", "reviews", "evidence"),
    ("mechanic", "inspects", "engine"),
    ("baker", "bakes", "bread"),
]

TRAIN_FILLERS = ["near", "beside", "during"]
FAMILIAR_FILLERS = ["after", "before", "around"]
NOVEL_PAIRS = [
    ("doctor", "prepares", "x1"),
    ("chef", "flies", "x2"),
    ("judge", "bakes", "x3"),
    ("teacher", "inspects", "x4"),
]


def run(
    *,
    dim: int = 2048,
    addr_dim: int = 512,
    seeds: tuple[int, ...] = (0, 1, 2),
    cycles: int = 5,
    n_locations: int = 2048,
    write_k: int = 8,
    read_k: int = 16,
) -> list[dict[str, float]]:
    rows = []
    for seed in seeds:
        model = ProjectedTrigramLanguageMemory(
            dim=dim,
            seed=seed,
            addr_dim=addr_dim,
            n_locations=n_locations,
            write_k=write_k,
            read_k=read_k,
        )
        for _cycle in range(cycles):
            for left, right, next_token in RULES:
                for filler in TRAIN_FILLERS:
                    model.learn(left, right, filler, next_token)

        seen = [(left, right, TRAIN_FILLERS[0], next_token) for left, right, next_token in RULES]
        familiar = [
            (left, right, FAMILIAR_FILLERS[idx % len(FAMILIAR_FILLERS)], next_token)
            for idx, (left, right, next_token) in enumerate(RULES)
        ]

        seen_scores = [model.score(left, right, filler) for left, right, filler, _expected in seen]
        familiar_scores = [model.score(left, right, filler) for left, right, filler, _expected in familiar]
        novel_scores = [model.score(left, right, filler) for left, right, filler in NOVEL_PAIRS]

        threshold = _midpoint(_mean(familiar_scores, "score"), _mean(novel_scores, "score"))
        margin_threshold = _midpoint(_mean(familiar_scores, "margin"), _mean(novel_scores, "margin"))

        rows.append(
            {
                "dim": float(dim),
                "addr_dim": float(addr_dim),
                "seed": float(seed),
                "cycles": float(cycles),
                "n_locations": float(n_locations),
                "write_k": float(write_k),
                "read_k": float(read_k),
                "seen_em": _em(seen_scores, seen),
                "familiar_em": _em(familiar_scores, familiar),
                "novel_hit_rate": _hit_rate(novel_scores, min_score=0.0, min_margin=0.0),
                "threshold": threshold,
                "margin_threshold": margin_threshold,
                "score_calibrated_familiar_em": _em(
                    familiar_scores,
                    familiar,
                    min_score=threshold,
                    min_margin=0.0,
                ),
                "score_calibrated_novel_hit_rate": _hit_rate(
                    novel_scores,
                    min_score=threshold,
                    min_margin=0.0,
                ),
                "calibrated_familiar_em": _em(
                    familiar_scores,
                    familiar,
                    min_score=threshold,
                    min_margin=margin_threshold,
                ),
                "calibrated_novel_hit_rate": _hit_rate(
                    novel_scores,
                    min_score=threshold,
                    min_margin=margin_threshold,
                ),
                "mean_seen_score": _mean(seen_scores, "score"),
                "mean_familiar_score": _mean(familiar_scores, "score"),
                "mean_novel_score": _mean(novel_scores, "score"),
                "mean_familiar_margin": _mean(familiar_scores, "margin"),
                "mean_novel_margin": _mean(novel_scores, "margin"),
            }
        )
    return rows


def _mean(rows: list[dict[str, float | str | None]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _midpoint(left: float, right: float) -> float:
    return (left + right) / 2


def _em(
    scores: list[dict[str, float | str | None]],
    expected: list[tuple[str, str, str, str]],
    *,
    min_score: float = 0.0,
    min_margin: float = 0.0,
) -> float:
    correct = 0
    for score, (_left, _right, _filler, expected_token) in zip(scores, expected):
        correct += int(
            score["token"] == expected_token
            and float(score["score"]) >= min_score
            and float(score["margin"]) >= min_margin
        )
    return correct / len(expected)


def _hit_rate(
    scores: list[dict[str, float | str | None]],
    *,
    min_score: float,
    min_margin: float,
) -> float:
    return sum(
        int(score["token"] is not None and float(score["score"]) >= min_score and float(score["margin"]) >= min_margin)
        for score in scores
    ) / len(scores)


if __name__ == "__main__":
    for row in run():
        print(row)
