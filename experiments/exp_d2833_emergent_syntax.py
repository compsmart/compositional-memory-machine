from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from language.syntax import SyntaxComposer, build_syntax_triples


def run(
    dim: int = 2048,
    seeds: tuple[int, ...] = (0, 1, 2),
    domains: int = 5,
    triples_per_domain: int = 30,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    triples = build_syntax_triples(domains=domains, triples_per_domain=triples_per_domain)
    for seed in seeds:
        composer = SyntaxComposer(dim=dim, seed=seed)
        within_scores: list[float] = []
        cross_scores: list[float] = []
        random_scores: list[float] = []

        for idx, (_domain, triple) in enumerate(triples):
            active_a = composer.encode(triple, "active", variant="a")
            active_b = composer.encode(triple, "active", variant="b")
            within_scores.append(composer.similarity(active_a, active_b))

            pattern_vectors = {
                pattern: composer.encode(triple, pattern, variant="a") for pattern in SyntaxComposer.PATTERNS
            }
            for left_pattern, right_pattern in combinations(SyntaxComposer.PATTERNS, 2):
                cross_scores.append(
                    composer.similarity(pattern_vectors[left_pattern], pattern_vectors[right_pattern])
                )

            other_triple = triples[(idx + triples_per_domain + 7) % len(triples)][1]
            random_scores.append(
                composer.similarity(
                    composer.encode(triple, "active", variant="a"),
                    composer.encode(other_triple, "coordinated", variant="b"),
                )
            )

        mean_within = sum(within_scores) / len(within_scores)
        mean_cross = sum(cross_scores) / len(cross_scores)
        mean_random = sum(random_scores) / len(random_scores)
        rows.append(
            {
                "dim": float(dim),
                "seed": float(seed),
                "domains": float(domains),
                "triples": float(len(triples)),
                "patterns": float(len(SyntaxComposer.PATTERNS)),
                "mean_within_cosine": mean_within,
                "mean_cross_pattern_cosine": mean_cross,
                "mean_random_cosine": mean_random,
                "cross_over_random_margin": mean_cross - mean_random,
                "cross_within_ratio": mean_cross / mean_within if mean_within else 0.0,
            }
        )
    return rows


if __name__ == "__main__":
    for row in run():
        print(row)
