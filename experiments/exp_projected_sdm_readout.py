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


def run(
    *,
    hrr_dim: int = 2048,
    addr_dim: int = 512,
    seeds: tuple[int, ...] = (0, 1),
    domains: int = 5,
    facts_per_domain: int = 60,
    n_locations: int = 2048,
    write_k: int = 8,
    read_k_values: tuple[int, ...] = (8, 32, 128),
    noise_values: tuple[float, ...] = (0.25, 0.5, 0.85),
    cleanup: str = "address",
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for read_k in read_k_values:
        for noise in noise_values:
            for seed in seeds:
                rng = np.random.default_rng(seed + read_k + int(noise * 1000))
                encoder = SVOEncoder(dim=hrr_dim, seed=seed)
                memory = ProjectedSDM(
                    vector_dim=hrr_dim,
                    addr_dim=addr_dim,
                    n_locations=n_locations,
                    k=write_k,
                    write_mode="sum",
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
                for domain, fact in facts:
                    memory.write(fact_key(domain, fact), encoder.encode_fact(fact), _payload(domain, fact))

                correct = 0
                for domain, fact in facts:
                    expected = fact_key(domain, fact)
                    vector = _noisy(encoder.encode_fact(fact), rng, noise)
                    record, _score = memory.query(vector, cleanup=cleanup, read_k=read_k)
                    correct += int(record is not None and record.key == expected)

                rows.append(
                    {
                        "hrr_dim": float(hrr_dim),
                        "addr_dim": float(addr_dim),
                        "seed": float(seed),
                        "domains": float(domains),
                        "facts": float(len(facts)),
                        "n_locations": float(n_locations),
                        "write_k": float(write_k),
                        "read_k": float(read_k),
                        "cleanup": cleanup,
                        "noise": float(noise),
                        "top1": correct / len(facts),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hrr-dim", type=int, default=2048)
    parser.add_argument("--addr-dim", type=int, default=512)
    parser.add_argument("--domains", type=int, default=5)
    parser.add_argument("--facts-per-domain", type=int, default=60)
    parser.add_argument("--n-locations", type=int, default=2048)
    parser.add_argument("--write-k", type=int, default=8)
    args = parser.parse_args()
    for row in run(
        hrr_dim=args.hrr_dim,
        addr_dim=args.addr_dim,
        domains=args.domains,
        facts_per_domain=args.facts_per_domain,
        n_locations=args.n_locations,
        write_k=args.write_k,
    ):
        print(row)


if __name__ == "__main__":
    main()
