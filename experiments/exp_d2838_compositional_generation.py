from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generation import ADJECTIVES, NOUNS, CompositionalValueDecoder, make_value_vector
from hrr.binding import normalize
from hrr.vectors import VectorStore
from memory.amm import AMM


def _entity_rows(
    *,
    dim: int,
    seed: int,
    n_entities: int,
    cycles: int,
) -> tuple[VectorStore, AMM, list[tuple[str, int, int]]]:
    store = VectorStore(dim=dim, seed=seed)
    memory = AMM()
    rows: list[tuple[str, int, int]] = []
    for idx in range(n_entities):
        entity = f"entity_{idx:03d}"
        adj_idx = idx % len(ADJECTIVES)
        noun_idx = (idx * 7 + seed) % len(NOUNS)
        property_vector = make_value_vector(store, ADJECTIVES[adj_idx], NOUNS[noun_idx])
        entity_key = store.get(f"entity:{entity}")
        payload = {
            "entity": entity,
            "adj": ADJECTIVES[adj_idx],
            "noun": NOUNS[noun_idx],
            "adj_idx": str(adj_idx),
            "noun_idx": str(noun_idx),
            "value_vector": property_vector,
        }
        for _cycle in range(cycles):
            memory.write(entity, entity_key, payload)
        rows.append((entity, adj_idx, noun_idx))
    return store, memory, rows


def run(
    *,
    dims: tuple[int, ...] = (64, 128, 256, 512, 2048),
    seeds: tuple[int, ...] = (0, 1, 2),
    n_entities: int = 150,
    train_fraction: float = 0.8,
    cycles: int = 3,
    ridge_alpha: float = 0.1,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []

    for dim in dims:
        for seed in seeds:
            store, memory, entities = _entity_rows(dim=dim, seed=seed, n_entities=n_entities, cycles=cycles)
            permutation = np.random.default_rng(seed + dim + 2838).permutation(len(entities))
            train_count = int(round(len(entities) * train_fraction))
            test_idx = permutation[train_count:]

            value_vectors: list[np.ndarray] = []
            adj_targets: list[int] = []
            noun_targets: list[int] = []
            hrr_hits = 0
            exact_retrieval_hits = 0
            linear_examples: list[tuple[np.ndarray, str, str]] = []

            for row_idx in permutation:
                entity, adj_idx, noun_idx = entities[int(row_idx)]
                record, _score = memory.query(store.get(f"entity:{entity}"))
                if record is None:
                    raise ValueError("expected stored entity record")
                exact_retrieval_hits += int(record.key == entity)
                value_vector = np.asarray(record.payload["value_vector"], dtype=float)
                value_vectors.append(value_vector)
                adj_targets.append(adj_idx)
                noun_targets.append(noun_idx)
                adjective = ADJECTIVES[adj_idx]
                noun = NOUNS[noun_idx]
                linear_examples.append((value_vector, adjective, noun))

            decoder = CompositionalValueDecoder(store=store)
            decoder.fit_linear_head(linear_examples[:train_count], ridge_alpha=ridge_alpha)
            for idx, value_vector in enumerate(value_vectors):
                decoded = decoder.decode_hrr(value_vector)
                hrr_hits += int(
                    decoded.adjective == ADJECTIVES[adj_targets[idx]]
                    and decoded.noun == NOUNS[noun_targets[idx]]
                )

            linear_hits = 0
            for row_idx in range(train_count, len(value_vectors)):
                decoded = decoder.decode_linear(value_vectors[row_idx])
                linear_hits += int(
                    decoded.adjective == ADJECTIVES[adj_targets[row_idx]]
                    and decoded.noun == NOUNS[noun_targets[row_idx]]
                )

            rows.append(
                {
                    "dim": float(dim),
                    "seed": float(seed),
                    "entities": float(n_entities),
                    "train_entities": float(train_count),
                    "test_entities": float(len(test_idx)),
                    "cycles": float(cycles),
                    "hrr_native_em": hrr_hits / len(entities),
                    "linear_head_em": linear_hits / len(test_idx) if len(test_idx) else 1.0,
                    "adj_vocab": float(len(ADJECTIVES)),
                    "noun_vocab": float(len(NOUNS)),
                    "exact_retrieval": exact_retrieval_hits / len(entities),
                }
            )
    return rows


def summarize(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    dims = sorted({int(row["dim"]) for row in rows})
    summary: list[dict[str, float]] = []
    for dim in dims:
        bucket = [row for row in rows if int(row["dim"]) == dim]
        summary.append(
            {
                "dim": float(dim),
                "runs": float(len(bucket)),
                "hrr_native_em_mean": float(np.mean([row["hrr_native_em"] for row in bucket])),
                "linear_head_em_mean": float(np.mean([row["linear_head_em"] for row in bucket])),
                "exact_retrieval_mean": float(np.mean([row["exact_retrieval"] for row in bucket])),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256, 512, 2048])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--entities", type=int, default=150)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=0.1)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    rows = run(
        dims=tuple(args.dims),
        seeds=tuple(args.seeds),
        n_entities=args.entities,
        train_fraction=args.train_fraction,
        cycles=args.cycles,
        ridge_alpha=args.ridge_alpha,
    )
    if args.summary:
        for row in summarize(rows):
            print(row)
        return
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
