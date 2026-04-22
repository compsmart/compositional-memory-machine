from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrr.binding import normalize
from hrr.datasets import fact_key, synthetic_facts
from hrr.encoder import SVOEncoder
from memory.amm import AMM


def _payload(domain: str, fact: object) -> dict[str, str]:
    return {
        "domain": domain,
        "subject": fact.subject,
        "verb": fact.verb,
        "object": fact.object,
    }


def _noisy(vec: np.ndarray, rng: np.random.Generator, noise: float) -> np.ndarray:
    if noise <= 0:
        return vec
    direction = normalize(rng.normal(0.0, 1.0, len(vec)))
    return normalize(vec + noise * direction)


def _address_top1(
    probes: np.ndarray,
    matrix: np.ndarray,
    rng: np.random.Generator,
    dim: int,
    radius_fraction: float = 0.35,
) -> tuple[np.ndarray, float, float]:
    bits = max(8, dim // 32)
    projection = rng.normal(0.0, 1.0, (dim, bits))
    record_signatures = matrix @ projection >= 0.0
    probe_signatures = probes @ projection >= 0.0
    radius = max(1, int(bits * radius_fraction))

    winners = np.full(len(probes), -1, dtype=int)
    candidate_counts: list[int] = []
    for row_idx, probe_signature in enumerate(probe_signatures):
        hamming = np.count_nonzero(record_signatures != probe_signature, axis=1)
        candidates = np.flatnonzero(hamming <= radius)
        candidate_counts.append(int(len(candidates)))
        if len(candidates) == 0:
            continue
        scores = matrix[candidates] @ probes[row_idx]
        winners[row_idx] = int(candidates[int(np.argmax(scores))])
    return winners, float(bits), float(np.mean(candidate_counts))


def run(
    dims: tuple[int, ...] = (512, 1024, 2048),
    seeds: tuple[int, ...] = (0, 1, 2),
    domains: int = 8,
    facts_per_domain: int = 500,
    probes: int = 200,
    noise: float = 0.85,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    total_facts = domains * facts_per_domain

    for dim in dims:
        for seed in seeds:
            rng = np.random.default_rng(seed + 1000)
            facts = synthetic_facts(domains=domains, facts_per_domain=facts_per_domain, seed=seed)
            encoder = SVOEncoder(dim=dim, seed=seed)
            memory = AMM()
            encoded: list[tuple[str, np.ndarray]] = []

            for domain, fact in facts:
                vector = encoder.encode_fact(fact)
                key = fact_key(domain, fact)
                memory.write(key, vector, _payload(domain, fact))
                encoded.append((key, vector))

            sample_indices = rng.choice(len(encoded), size=min(probes, len(encoded)), replace=False)
            matrix = np.vstack([vector for _key, vector in encoded])
            clean_probes = matrix[sample_indices]
            noisy_probes = np.vstack([_noisy(vector, rng, noise) for vector in clean_probes])

            clean_scores = clean_probes @ matrix.T
            noisy_scores_matrix = noisy_probes @ matrix.T

            clean_top1 = np.argmax(clean_scores, axis=1)
            noisy_top1 = np.argmax(noisy_scores_matrix, axis=1)
            expected = sample_indices
            address_top1, address_bits, address_candidates = _address_top1(
                noisy_probes,
                matrix,
                np.random.default_rng(seed + dim + 9000),
                dim,
            )

            clean_correct = int(np.sum(clean_top1 == expected))
            noisy_correct = int(np.sum(noisy_top1 == expected))
            address_correct = int(np.sum(address_top1 == expected))

            top2 = np.partition(clean_scores, -2, axis=1)[:, -2:]
            top2.sort(axis=1)
            margins = top2[:, 1] - top2[:, 0]
            noisy_best_scores = np.max(noisy_scores_matrix, axis=1)

            rows.append(
                {
                    "dim": float(dim),
                    "seed": float(seed),
                    "facts": float(total_facts),
                    "noise": float(noise),
                    "clean_top1": clean_correct / len(sample_indices),
                    "full_vector_noisy_top1": noisy_correct / len(sample_indices),
                    "address_noisy_top1": address_correct / len(sample_indices),
                    "address_bits": address_bits,
                    "mean_address_candidates": address_candidates,
                    "mean_clean_margin": float(np.mean(margins)) if len(margins) else 0.0,
                    "p05_clean_margin": float(np.quantile(margins, 0.05)) if len(margins) else 0.0,
                    "mean_noisy_score": float(np.mean(noisy_best_scores)) if len(noisy_best_scores) else 0.0,
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts-per-domain", type=int, default=500)
    parser.add_argument("--domains", type=int, default=8)
    parser.add_argument("--probes", type=int, default=200)
    parser.add_argument("--noise", type=float, default=0.85)
    args = parser.parse_args()

    for row in run(
        domains=args.domains,
        facts_per_domain=args.facts_per_domain,
        probes=args.probes,
        noise=args.noise,
    ):
        print(row)


if __name__ == "__main__":
    main()
