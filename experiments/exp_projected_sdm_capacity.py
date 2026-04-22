from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp_addr_dim_sweep import run as run_addr_dim_protocol


def run(
    *,
    hrr_dim: int = 2048,
    addr_dim: int = 512,
    seeds: tuple[int, ...] = (0, 1, 2),
    domains: int = 5,
    facts_per_domain: int = 40,
    n_locations_values: tuple[int, ...] = (512, 1024, 2048),
    k_values: tuple[int, ...] = (1, 4, 8, 16),
    write_modes: tuple[str, ...] = ("overwrite", "sum"),
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for write_mode in write_modes:
        for n_locations in n_locations_values:
            for k in k_values:
                raw = run_addr_dim_protocol(
                    addr_dims=(addr_dim,),
                    seeds=seeds,
                    hrr_dim=hrr_dim,
                    domains=domains,
                    facts_per_domain=facts_per_domain,
                    n_locations=n_locations,
                    k=k,
                    write_mode=write_mode,
                )
                rows.append(
                    {
                        "hrr_dim": float(hrr_dim),
                        "addr_dim": float(addr_dim),
                        "domains": float(domains),
                        "facts_per_domain": float(facts_per_domain),
                        "n_locations": float(n_locations),
                        "k": float(k),
                        "write_mode": write_mode,
                        "mean_d1_after_first": _mean(raw, "d1_after_first"),
                        "mean_d1_final": _mean(raw, "d1_final"),
                        "mean_all_final": _mean(raw, "all_final"),
                        "mean_forgetting": _mean(raw, "forgetting"),
                    }
                )
    return rows


def _mean(rows: list[dict[str, float | str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hrr-dim", type=int, default=2048)
    parser.add_argument("--addr-dim", type=int, default=512)
    parser.add_argument("--domains", type=int, default=5)
    parser.add_argument("--facts-per-domain", type=int, default=40)
    args = parser.parse_args()
    for row in run(
        hrr_dim=args.hrr_dim,
        addr_dim=args.addr_dim,
        domains=args.domains,
        facts_per_domain=args.facts_per_domain,
    ):
        print(row)


if __name__ == "__main__":
    main()
