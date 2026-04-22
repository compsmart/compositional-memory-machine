from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrr.binding import normalize
from hrr.datasets import fact_key, synthetic_facts
from hrr.encoder import SVOEncoder
from memory.projected_sdm import ProjectedSDM


def _payload(domain: str, fact: object) -> dict[str, str]:
    return {
        "domain": domain,
        "subject": fact.subject,
        "verb": fact.verb,
        "object": fact.object,
    }


def _noisy(vector: np.ndarray, rng: np.random.Generator, noise: float) -> np.ndarray:
    if noise <= 0.0:
        return vector
    return normalize(vector + noise * normalize(rng.normal(0.0, 1.0, len(vector))))


def _evaluate(
    memory: ProjectedSDM,
    examples: list[tuple[str, object]],
    encoder: SVOEncoder,
    rng: np.random.Generator,
    *,
    cleanup: str,
    noise: float,
) -> float:
    if not examples:
        return 0.0
    correct = 0
    for domain, fact in examples:
        expected = fact_key(domain, fact)
        vector = _noisy(encoder.encode_fact(fact), rng, noise)
        record, _score = memory.query(vector, cleanup=cleanup)
        correct += int(record is not None and record.key == expected)
    return correct / len(examples)


def run(
    *,
    hrr_dim: int = 2048,
    addr_dim: int = 512,
    seeds: tuple[int, ...] = (0, 1, 2),
    domains: int = 8,
    facts_per_domain: int = 100,
    n_locations: int = 2048,
    k: int = 8,
    write_mode: str = "sum",
    cleanup_modes: tuple[str, ...] = ("global", "address"),
    noise_values: tuple[float, ...] = (0.0, 0.25, 0.5, 0.85),
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for cleanup in cleanup_modes:
        for noise in noise_values:
            for seed in seeds:
                rng = np.random.default_rng(seed + int(noise * 1000))
                encoder = SVOEncoder(dim=hrr_dim, seed=seed)
                memory = ProjectedSDM(
                    vector_dim=hrr_dim,
                    addr_dim=addr_dim,
                    n_locations=n_locations,
                    k=k,
                    write_mode=write_mode,
                    seed=seed + addr_dim,
                )
                facts = synthetic_facts(
                    domains=domains,
                    facts_per_domain=facts_per_domain,
                    subjects=max(160, domains * facts_per_domain),
                    verbs=160,
                    objects=max(160, domains * facts_per_domain),
                    seed=seed,
                )
                by_domain = {
                    f"domain{domain_idx}": [
                        (domain, fact) for domain, fact in facts if domain == f"domain{domain_idx}"
                    ]
                    for domain_idx in range(domains)
                }
                d1_after_first = 0.0
                for domain_idx in range(domains):
                    domain_name = f"domain{domain_idx}"
                    for domain, fact in by_domain[domain_name]:
                        memory.write(fact_key(domain, fact), encoder.encode_fact(fact), _payload(domain, fact))
                    if domain_idx == 0:
                        d1_after_first = _evaluate(
                            memory,
                            by_domain["domain0"],
                            encoder,
                            rng,
                            cleanup=cleanup,
                            noise=0.0,
                        )

                d1_final = _evaluate(memory, by_domain["domain0"], encoder, rng, cleanup=cleanup, noise=noise)
                all_final = _evaluate(memory, facts, encoder, rng, cleanup=cleanup, noise=noise)
                rows.append(
                    {
                        "hrr_dim": float(hrr_dim),
                        "addr_dim": float(addr_dim),
                        "seed": float(seed),
                        "domains": float(domains),
                        "facts": float(len(facts)),
                        "n_locations": float(n_locations),
                        "k": float(k),
                        "write_mode": write_mode,
                        "cleanup": cleanup,
                        "noise": float(noise),
                        "d1_after_first": d1_after_first,
                        "d1_final": d1_final,
                        "all_final": all_final,
                        "forgetting": max(0.0, d1_after_first - d1_final),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hrr-dim", type=int, default=2048)
    parser.add_argument("--addr-dim", type=int, default=512)
    parser.add_argument("--domains", type=int, default=8)
    parser.add_argument("--facts-per-domain", type=int, default=100)
    parser.add_argument("--n-locations", type=int, default=2048)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()
    for row in run(
        hrr_dim=args.hrr_dim,
        addr_dim=args.addr_dim,
        domains=args.domains,
        facts_per_domain=args.facts_per_domain,
        n_locations=args.n_locations,
        k=args.k,
    ):
        print(row)


if __name__ == "__main__":
    main()
