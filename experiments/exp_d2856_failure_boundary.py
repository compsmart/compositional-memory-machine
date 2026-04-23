from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrr.binding import bind, unbind
from hrr.vectors import VectorStore

from experiments.hrr_claim_utils import bind_all, bound_token, bundle, make_similar_vector, nearest_token


def run(
    *,
    dim: int = 4096,
    seeds: tuple[int, ...] = (0, 1, 2),
    similarities: tuple[float, ...] = (0.95, 0.85, 0.75, 0.60, 0.40, 0.10),
    n_pairs: int = 30,
    conflict_sizes: tuple[int, ...] = (5, 10, 25, 50),
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed + dim)
        store = VectorStore(dim=dim, seed=seed)
        role_subject = store.get_unitary("__ROLE_SUBJECT__")
        role_verb = store.get_unitary("__ROLE_VERB__")
        role_object = store.get_unitary("__ROLE_OBJECT__")

        for similarity in similarities:
            pair_rows: list[tuple[object, str]] = []
            fact_vectors = []
            candidate_objects: dict[str, object] = {}
            for idx in range(n_pairs):
                left_vec, right_vec = make_similar_vector(rng, dim, similarity)
                left_object = f"left_obj_{idx}"
                right_object = f"right_obj_{idx}"
                candidate_objects[left_object] = store.get(f"obj:{left_object}")
                candidate_objects[right_object] = store.get(f"obj:{right_object}")
                pair_rows.append((left_vec, left_object))
                fact_vectors.extend(
                    [
                        bind_all(
                            [
                                bind(role_subject, left_vec),
                                bound_token(store, role_verb, "verb", "maps_to"),
                                bound_token(store, role_object, "obj", left_object),
                            ]
                        ),
                        bind_all(
                            [
                                bind(role_subject, right_vec),
                                bound_token(store, role_verb, "verb", "maps_to"),
                                bound_token(store, role_object, "obj", right_object),
                            ]
                        ),
                    ]
                )
            memory = bundle(fact_vectors)
            correct = 0
            total = 0
            for left_vec, left_object in pair_rows:
                recovered = unbind(
                    unbind(
                        memory,
                        bind_all([bind(role_subject, left_vec), bound_token(store, role_verb, "verb", "maps_to")]),
                    ),
                    role_object,
                )
                predicted, _score = nearest_token(recovered, candidate_objects)
                correct += int(predicted == left_object)
                total += 1
            correct_rate = correct / max(total, 1)
            rows.append(
                {
                    "mode": "similarity",
                    "seed": float(seed),
                    "similarity": float(similarity),
                    "correct_retrieval": correct_rate,
                    "confusion": 1.0 - correct_rate,
                }
            )

        for n_facts in conflict_sizes:
            fact_vectors = []
            candidates: dict[str, dict[str, object]] = {}
            subjects = [f"entity_{idx:03d}" for idx in range(n_facts)]
            for idx, subject in enumerate(subjects):
                verb = "status"
                old_object = f"old_{idx:03d}"
                new_object = f"new_{idx:03d}"
                old_vec = bind_all(
                    [
                        bound_token(store, role_subject, "subj", subject),
                        bound_token(store, role_verb, "verb", verb),
                        bound_token(store, role_object, "obj", old_object),
                    ]
                )
                new_vec = bind_all(
                    [
                        bound_token(store, role_subject, "subj", subject),
                        bound_token(store, role_verb, "verb", verb),
                        bound_token(store, role_object, "obj", new_object),
                    ]
                )
                fact_vectors.extend([old_vec, new_vec])
                candidates[subject] = {
                    old_object: store.get(f"obj:{old_object}"),
                    new_object: store.get(f"obj:{new_object}"),
                }
            memory = bundle(fact_vectors)

            new_hits = 0
            old_hits = 0
            for idx, subject in enumerate(subjects):
                verb = "status"
                old_object = f"old_{idx:03d}"
                new_object = f"new_{idx:03d}"
                cue = bind_all(
                    [
                        bound_token(store, role_subject, "subj", subject),
                        bound_token(store, role_verb, "verb", verb),
                    ]
                )
                recovered = unbind(unbind(memory, cue), role_object)
                predicted, _score = nearest_token(recovered, candidates[subject])
                new_hits += int(predicted == new_object)
                old_hits += int(predicted == old_object)

            rows.append(
                {
                    "mode": "overwrite",
                    "seed": float(seed),
                    "n_facts": float(n_facts),
                    "new_fact_recall": new_hits / n_facts,
                    "old_contamination": old_hits / n_facts,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    for row in run(dim=args.dim, seeds=tuple(args.seeds)):
        print(row)


if __name__ == "__main__":
    main()
