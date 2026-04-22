from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrr.binding import bind, normalize
from hrr.datasets import fact_key, synthetic_facts
from hrr.encoder import SVOEncoder
from hrr.vectors import VectorStore
from memory.projected import ProjectedAddressIndex


KeyRows = list[tuple[str, np.ndarray, dict[str, str]]]


def _noisy(vector: np.ndarray, rng: np.random.Generator, noise: float) -> np.ndarray:
    if noise <= 0.0:
        return vector
    return normalize(vector + noise * normalize(rng.normal(0.0, 1.0, len(vector))))


def _one_hot_rows(dim: int, n_items: int) -> KeyRows:
    if n_items > dim:
        raise ValueError("one_hot key family requires n_items <= dim")
    rows: KeyRows = []
    for idx in range(n_items):
        vector = np.zeros(dim)
        vector[idx] = 1.0
        key = f"one_hot:{idx}"
        rows.append((key, vector, {"family": "one_hot", "item": str(idx)}))
    return rows


def _hrr_svo_rows(dim: int, seed: int, n_items: int) -> KeyRows:
    encoder = SVOEncoder(dim=dim, seed=seed)
    facts = synthetic_facts(domains=5, facts_per_domain=max(1, (n_items + 4) // 5), seed=seed)[:n_items]
    return [
        (
            fact_key(domain, fact),
            encoder.encode_fact(fact),
            {
                "family": "hrr_svo",
                "domain": domain,
                "subject": fact.subject,
                "verb": fact.verb,
                "object": fact.object,
            },
        )
        for domain, fact in facts
    ]


def _hrr_ngram_rows(dim: int, seed: int, n_items: int) -> KeyRows:
    store = VectorStore(dim=dim, seed=seed)
    role_left = store.get_unitary("__SWEEP_LEFT__")
    role_right = store.get_unitary("__SWEEP_RIGHT__")
    rows: KeyRows = []
    for idx in range(n_items):
        left = f"tok{idx % 97}"
        right = f"tok{(idx * 37 + 11) % 193}"
        key = f"hrr_ngram:{left}:{right}:{idx}"
        vector = normalize(
            bind(role_left, store.get(f"tok:{left}"))
            + bind(role_right, store.get(f"tok:{right}"))
        )
        rows.append((key, vector, {"family": "hrr_ngram", "left": left, "right": right}))
    return rows


def _continuous_context_rows(dim: int, seed: int, n_items: int) -> KeyRows:
    store = VectorStore(dim=dim, seed=seed)
    rows: KeyRows = []
    for idx in range(n_items):
        topic = f"topic{idx % 31}"
        actor = f"actor{idx % 79}"
        action = f"action{(idx * 13) % 53}"
        key = f"continuous:{topic}:{actor}:{action}:{idx}"
        vector = normalize(
            0.55 * store.get(f"topic:{topic}")
            + 0.30 * store.get(f"actor:{actor}")
            + 0.15 * store.get(f"action:{action}")
        )
        rows.append((key, vector, {"family": "continuous", "topic": topic, "actor": actor, "action": action}))
    return rows


def _build_rows(family: str, dim: int, seed: int, n_items: int) -> KeyRows:
    if family == "one_hot":
        return _one_hot_rows(dim, n_items)
    if family == "hrr_svo":
        return _hrr_svo_rows(dim, seed, n_items)
    if family == "hrr_ngram":
        return _hrr_ngram_rows(dim, seed, n_items)
    if family == "continuous":
        return _continuous_context_rows(dim, seed, n_items)
    raise ValueError(f"unknown key family: {family}")


def _evaluate(
    rows: KeyRows,
    *,
    dim: int,
    addr_dim: int,
    seed: int,
    probes: int,
    noise: float,
) -> dict[str, float]:
    rng = np.random.default_rng(seed + addr_dim + 17)
    sample_indices = rng.choice(len(rows), size=min(probes, len(rows)), replace=False)
    index = ProjectedAddressIndex(dim, addr_dim, seed=seed + addr_dim + 1000)
    index.build(rows)

    exact_hits = 0
    noisy_hits = 0
    expected_candidate_hits = 0
    candidate_counts: list[int] = []
    stale_rates: list[float] = []
    empty_queries = 0

    for idx in sample_indices:
        key, vector, _payload = rows[int(idx)]
        exact = index.query(vector, expected_key=key)
        if exact.key == key:
            exact_hits += 1

        noisy = index.query(_noisy(vector, rng, noise), expected_key=key)
        if noisy.key == key:
            noisy_hits += 1
        if noisy.expected_in_candidates:
            expected_candidate_hits += 1
        if noisy.candidate_count == 0:
            empty_queries += 1
            stale_rates.append(0.0)
        else:
            candidate_counts.append(noisy.candidate_count)
            stale_count = noisy.candidate_count - int(bool(noisy.expected_in_candidates))
            stale_rates.append(stale_count / noisy.candidate_count)

    probe_count = len(sample_indices)
    return {
        "exact_top1": exact_hits / probe_count,
        "noisy_top1": noisy_hits / probe_count,
        "expected_candidate_rate": expected_candidate_hits / probe_count,
        "empty_query_rate": empty_queries / probe_count,
        "mean_candidates": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "stale_contamination": float(np.mean(stale_rates)) if stale_rates else 0.0,
    }


def run(
    *,
    dim: int = 2048,
    addr_dims: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048),
    families: tuple[str, ...] = ("one_hot", "hrr_svo", "hrr_ngram", "continuous"),
    seeds: tuple[int, ...] = (0, 1, 2),
    n_items: int = 500,
    probes: int = 200,
    noise: float = 0.5,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for family in families:
        for seed in seeds:
            family_rows = _build_rows(family, dim, seed, n_items)
            for addr_dim in addr_dims:
                metrics = _evaluate(
                    family_rows,
                    dim=dim,
                    addr_dim=addr_dim,
                    seed=seed,
                    probes=probes,
                    noise=noise,
                )
                rows.append(
                    {
                        "family": family,
                        "dim": float(dim),
                        "addr_dim": float(addr_dim),
                        "seed": float(seed),
                        "items": float(len(family_rows)),
                        "noise": float(noise),
                        **metrics,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--addr-dims", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--families", nargs="+", default=["one_hot", "hrr_svo", "hrr_ngram", "continuous"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--items", type=int, default=500)
    parser.add_argument("--probes", type=int, default=200)
    parser.add_argument("--noise", type=float, default=0.5)
    args = parser.parse_args()

    for row in run(
        dim=args.dim,
        addr_dims=tuple(args.addr_dims),
        families=tuple(args.families),
        seeds=tuple(args.seeds),
        n_items=args.items,
        probes=args.probes,
        noise=args.noise,
    ):
        print(row)


if __name__ == "__main__":
    main()

