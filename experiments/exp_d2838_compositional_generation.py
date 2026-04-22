from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrr.binding import bind, cosine, normalize, unbind
from hrr.vectors import VectorStore
from memory.amm import AMM


ADJECTIVES = tuple(
    [
        "amber",
        "ancient",
        "brisk",
        "calm",
        "crisp",
        "distant",
        "eager",
        "faint",
        "gentle",
        "golden",
        "hidden",
        "icy",
        "jagged",
        "lively",
        "mellow",
        "narrow",
        "quiet",
        "rapid",
        "silver",
        "warm",
    ]
)
NOUNS = tuple(
    [
        "bridge",
        "cedar",
        "cloud",
        "comet",
        "field",
        "forest",
        "garden",
        "harbor",
        "meadow",
        "mirror",
        "mountain",
        "orchard",
        "planet",
        "river",
        "signal",
        "station",
        "stone",
        "temple",
        "thunder",
        "valley",
    ]
)


def _ridge_fit(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> np.ndarray:
    gram = train_x.T @ train_x
    identity = np.eye(gram.shape[0], dtype=train_x.dtype)
    return np.linalg.solve(gram + alpha * identity, train_x.T @ train_y)


def _one_hot(indices: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros((len(indices), size), dtype=float)
    out[np.arange(len(indices)), indices] = 1.0
    return out


def _decode_nearest(vector: np.ndarray, candidates: list[np.ndarray]) -> int:
    scores = [cosine(vector, candidate) for candidate in candidates]
    return int(np.argmax(scores))


def _entity_rows(
    *,
    dim: int,
    seed: int,
    n_entities: int,
    cycles: int,
) -> tuple[VectorStore, AMM, list[tuple[str, int, int]]]:
    store = VectorStore(dim=dim, seed=seed)
    memory = AMM()
    role_adj = store.get_unitary("__ROLE_ADJ__")
    role_noun = store.get_unitary("__ROLE_NOUN__")
    rows: list[tuple[str, int, int]] = []
    for idx in range(n_entities):
        entity = f"entity_{idx:03d}"
        adj_idx = idx % len(ADJECTIVES)
        noun_idx = (idx * 7 + seed) % len(NOUNS)
        property_vector = normalize(
            bind(role_adj, store.get(f"adj:{ADJECTIVES[adj_idx]}"))
            + bind(role_noun, store.get(f"noun:{NOUNS[noun_idx]}"))
        )
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
            role_adj = store.get_unitary("__ROLE_ADJ__")
            role_noun = store.get_unitary("__ROLE_NOUN__")
            adj_vectors = [store.get(f"adj:{token}") for token in ADJECTIVES]
            noun_vectors = [store.get(f"noun:{token}") for token in NOUNS]
            entity_vectors = np.vstack([store.get(f"entity:{entity}") for entity, _adj_idx, _noun_idx in entities])
            permutation = np.random.default_rng(seed + dim + 2838).permutation(len(entities))
            train_count = int(round(len(entities) * train_fraction))
            train_idx = permutation[:train_count]
            test_idx = permutation[train_count:]

            value_vectors: list[np.ndarray] = []
            adj_targets: list[int] = []
            noun_targets: list[int] = []
            hrr_hits = 0
            exact_retrieval_hits = 0

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

                decoded_adj = unbind(value_vector, role_adj)
                decoded_noun = unbind(value_vector, role_noun)
                pred_adj = _decode_nearest(decoded_adj, adj_vectors)
                pred_noun = _decode_nearest(decoded_noun, noun_vectors)
                hrr_hits += int(pred_adj == adj_idx and pred_noun == noun_idx)

            value_matrix = np.vstack(value_vectors)
            adj_targets_arr = np.asarray(adj_targets, dtype=int)
            noun_targets_arr = np.asarray(noun_targets, dtype=int)

            train_x = value_matrix[: len(train_idx)]
            test_x = value_matrix[len(train_idx) :]
            train_adj = _one_hot(adj_targets_arr[: len(train_idx)], len(ADJECTIVES))
            train_noun = _one_hot(noun_targets_arr[: len(train_idx)], len(NOUNS))
            test_adj_idx = adj_targets_arr[len(train_idx) :]
            test_noun_idx = noun_targets_arr[len(train_idx) :]

            adj_head = _ridge_fit(train_x, train_adj, ridge_alpha)
            noun_head = _ridge_fit(train_x, train_noun, ridge_alpha)
            adj_logits = test_x @ adj_head
            noun_logits = test_x @ noun_head
            linear_hits = int(
                np.sum(
                    (np.argmax(adj_logits, axis=1) == test_adj_idx)
                    & (np.argmax(noun_logits, axis=1) == test_noun_idx)
                )
            )

            rows.append(
                {
                    "dim": float(dim),
                    "seed": float(seed),
                    "entities": float(n_entities),
                    "train_entities": float(len(train_idx)),
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
