from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def _evaluate_domain(memory: ProjectedSDM, examples: list[tuple[str, object]], encoder: SVOEncoder) -> float:
    if not examples:
        return 0.0
    vectors = []
    expected = []
    for domain, fact in examples:
        expected.append(fact_key(domain, fact))
        vectors.append(encoder.encode_fact(fact))
    results = memory.query_many(np.vstack(vectors))
    correct = sum(int(record is not None and record.key == key) for (record, _score), key in zip(results, expected))
    return correct / len(examples)


def run(
    *,
    addr_dims: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048),
    seeds: tuple[int, ...] = (0, 1, 2),
    hrr_dim: int = 2048,
    domains: int = 5,
    facts_per_domain: int = 40,
    n_locations: int = 512,
    k: int = 8,
    write_mode: str = "overwrite",
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for addr_dim in addr_dims:
        for seed in seeds:
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
                verbs=80,
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
            d1_final = 0.0
            all_final = 0.0
            for domain_idx in range(domains):
                domain_name = f"domain{domain_idx}"
                for domain, fact in by_domain[domain_name]:
                    memory.write(fact_key(domain, fact), encoder.encode_fact(fact), _payload(domain, fact))
                if domain_idx == 0:
                    d1_after_first = _evaluate_domain(memory, by_domain["domain0"], encoder)

            d1_final = _evaluate_domain(memory, by_domain["domain0"], encoder)
            all_final = _evaluate_domain(memory, facts, encoder)
            forgetting = max(0.0, d1_after_first - d1_final)
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
                    "d1_after_first": d1_after_first,
                    "d1_final": d1_final,
                    "all_final": all_final,
                    "forgetting": forgetting,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hrr-dim", type=int, default=2048)
    parser.add_argument("--domains", type=int, default=5)
    parser.add_argument("--facts-per-domain", type=int, default=40)
    parser.add_argument("--n-locations", type=int, default=512)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--write-mode", choices=("sum", "overwrite"), default="overwrite")
    args = parser.parse_args()
    for row in run(
        hrr_dim=args.hrr_dim,
        domains=args.domains,
        facts_per_domain=args.facts_per_domain,
        n_locations=args.n_locations,
        k=args.k,
        write_mode=args.write_mode,
    ):
        print(row)


if __name__ == "__main__":
    main()
