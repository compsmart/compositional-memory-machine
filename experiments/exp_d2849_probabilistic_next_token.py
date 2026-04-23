from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from language import NGramLanguageMemory


def run(dim: int = 2048, seeds: tuple[int, ...] = (0, 1, 2)) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    weighted = {"paints": 4.0, "sketches": 2.0, "draws": 1.0, "illustrates": 1.0}
    for seed in seeds:
        model = NGramLanguageMemory(dim=dim, seed=seed)
        model.learn_distribution("the", "artist", weighted)
        prediction = model.predict("the", "artist", min_confidence=0.25, top_k=4)
        alt_tokens = [candidate.token for candidate in prediction.alternatives]
        alt_probs = [candidate.probability for candidate in prediction.alternatives]
        rows.append(
            {
                "dim": float(dim),
                "seed": float(seed),
                "top1_correct": float(prediction.token == "paints"),
                "top3_hit": float("sketches" in alt_tokens[:3] and "draws" in alt_tokens[:4]),
                "probability_sum": sum(alt_probs),
                "top1_probability": alt_probs[0] if alt_probs else 0.0,
            }
        )
    return rows


if __name__ == "__main__":
    for row in run():
        print(row)
