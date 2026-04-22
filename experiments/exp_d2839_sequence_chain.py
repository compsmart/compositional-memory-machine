from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrr.binding import bind, cosine, normalize
from hrr.vectors import VectorStore


PREFIX_LENGTHS = (1, 2, 3, 5, 7, 10)


def _sequence_for_rule(family: int, rule: int) -> list[str]:
    shared = [f"family{family}_start", f"family{family}_shared"]
    unique = f"family{family}_rule{rule}"
    tail = [f"family{family}_rule{rule}_step{idx}" for idx in range(3, 10)]
    return [*shared, unique, *tail]


def _prefix_vector(store: VectorStore, sequence: list[str], prefix_len: int) -> np.ndarray:
    if prefix_len <= 0:
        raise ValueError("prefix_len must be positive")
    vector = bind(store.get_unitary("__SEQ_BASE__"), store.get(f"tok:{sequence[0]}"))
    for idx, token in enumerate(sequence[1:prefix_len], start=1):
        step_role = store.get_unitary(f"__SEQ_STEP_{idx}__")
        vector = normalize(bind(vector, bind(step_role, store.get(f"tok:{token}"))))
    return normalize(vector)


def run(
    *,
    dim: int = 2048,
    seeds: tuple[int, ...] = (42, 123, 7),
    families: int = 5,
    rules_per_family: int = 4,
    sequence_length: int = 10,
    prefix_lengths: tuple[int, ...] = PREFIX_LENGTHS,
) -> list[dict[str, float]]:
    if sequence_length != 10:
        raise ValueError("this benchmark currently expects sequence_length=10")

    rows: list[dict[str, float]] = []
    for seed in seeds:
        store = VectorStore(dim=dim, seed=seed)
        sequences = {
            (family, rule): _sequence_for_rule(family, rule)
            for family in range(families)
            for rule in range(rules_per_family)
        }
        for prefix_len in prefix_lengths:
            hits = 0
            total = 0
            for family in range(families):
                rule_vectors = [
                    _prefix_vector(store, sequences[(family, rule)], prefix_len)
                    for rule in range(rules_per_family)
                ]
                for rule in range(rules_per_family):
                    probe = _prefix_vector(store, sequences[(family, rule)], prefix_len)
                    scores = [cosine(probe, candidate) for candidate in rule_vectors]
                    predicted_rule = int(np.argmax(scores))
                    hits += int(predicted_rule == rule)
                    total += 1
            rows.append(
                {
                    "dim": float(dim),
                    "seed": float(seed),
                    "families": float(families),
                    "rules_per_family": float(rules_per_family),
                    "sequence_length": float(sequence_length),
                    "prefix_len": float(prefix_len),
                    "em": hits / total if total else 0.0,
                }
            )
    return rows


def summarize(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    summary: list[dict[str, float]] = []
    for prefix_len in sorted({int(row["prefix_len"]) for row in rows}):
        bucket = [row for row in rows if int(row["prefix_len"]) == prefix_len]
        summary.append(
            {
                "prefix_len": float(prefix_len),
                "runs": float(len(bucket)),
                "mean_em": float(np.mean([row["em"] for row in bucket])),
                "min_em": float(np.min([row["em"] for row in bucket])),
                "max_em": float(np.max([row["em"] for row in bucket])),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--families", type=int, default=5)
    parser.add_argument("--rules-per-family", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=list(PREFIX_LENGTHS))
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    rows = run(
        dim=args.dim,
        seeds=tuple(args.seeds),
        families=args.families,
        rules_per_family=args.rules_per_family,
        sequence_length=args.sequence_length,
        prefix_lengths=tuple(args.prefix_lengths),
    )
    if args.summary:
        for row in summarize(rows):
            print(row)
        return
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
