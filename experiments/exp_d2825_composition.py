from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrr.datasets import DOMAINS, fact_key
from hrr.encoder import SVOEncoder, SVOFact
from memory.amm import AMM

from experiments.common import payload


def run(dim: int = 2048, seed: int = 0) -> dict[str, float]:
    encoder = SVOEncoder(dim=dim, seed=seed)
    memory = AMM()

    held_out_subjects = {"doctor", "teacher", "pilot", "chef", "lawyer"}
    held_out: list[tuple[str, SVOFact]] = []

    for domain, facts in DOMAINS.items():
        for fact in facts:
            if fact.subject in held_out_subjects:
                held_out.append((domain, fact))
                continue
            memory.write(fact_key(domain, fact), encoder.encode_fact(fact), payload(domain, fact))

    # Add known domain prototypes with new subjects but familiar verb/object slots.
    probes = [
        ("medical", SVOFact("doctor", "monitors", "patient")),
        ("education", SVOFact("teacher", "reads", "book")),
        ("aviation", SVOFact("pilot", "inspects", "engine")),
        ("kitchen", SVOFact("chef", "bakes", "bread")),
        ("legal", SVOFact("lawyer", "reviews", "evidence")),
    ]

    correct_cluster = 0
    margins = []
    for expected_domain, probe in probes:
        nearest = memory.nearest(encoder.encode_fact(probe), top_k=2)
        predicted_domain = nearest[0][0].payload["domain"] if nearest else None
        correct_cluster += int(predicted_domain == expected_domain)
        if len(nearest) == 2:
            margins.append(nearest[0][1] - nearest[1][1])

    return {
        "dim": float(dim),
        "seed": float(seed),
        "cluster_em": correct_cluster / len(probes),
        "random_baseline": 1 / len(DOMAINS),
        "held_out_count": float(len(held_out)),
        "mean_margin": sum(margins) / len(margins) if margins else 0.0,
    }


if __name__ == "__main__":
    for seed in (0, 1, 2):
        print(run(seed=seed))
