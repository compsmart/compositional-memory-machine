from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrr.binding import unbind
from hrr.vectors import VectorStore
from memory.amm import AMM

from experiments.hrr_claim_utils import bind_all, bound_token, bundle, nearest_token


PREDICATES = ("location", "status", "tool")


def _entity_composite(store: VectorStore, role_verb, role_object, entity_idx: int, revision: bool) -> tuple[object, dict[str, str]]:
    assignments = {
        "location": f"{'new' if revision else 'old'}_location_{entity_idx:03d}",
        "status": f"status_{entity_idx:03d}",
        "tool": f"tool_{entity_idx:03d}",
    }
    vectors = [
        bind_all(
            [
                bound_token(store, role_verb, "verb", predicate),
                bound_token(store, role_object, "obj", assignments[predicate]),
            ]
        )
        for predicate in PREDICATES
    ]
    return bundle(vectors), assignments


def _decode_object(vector, store: VectorStore, role_verb, role_object, predicate: str, candidates: dict[str, object]) -> str:
    recovered = unbind(unbind(vector, bound_token(store, role_verb, "verb", predicate)), role_object)
    predicted, _score = nearest_token(recovered, candidates)
    return predicted


def run(
    *,
    dims: tuple[int, ...] = (256, 1024, 2048),
    seeds: tuple[int, ...] = (0, 1, 2),
    n_entities: int = 20,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for dim in dims:
        for seed in seeds:
            store = VectorStore(dim=dim, seed=seed)
            role_verb = store.get_unitary("__ROLE_VERB__")
            role_object = store.get_unitary("__ROLE_OBJECT__")
            candidate_vectors = {
                predicate: {
                    **{f"old_location_{idx:03d}": store.get(f"obj:old_location_{idx:03d}") for idx in range(n_entities)},
                    **{f"new_location_{idx:03d}": store.get(f"obj:new_location_{idx:03d}") for idx in range(n_entities)},
                    **{f"status_{idx:03d}": store.get(f"obj:status_{idx:03d}") for idx in range(n_entities)},
                    **{f"tool_{idx:03d}": store.get(f"obj:tool_{idx:03d}") for idx in range(n_entities)},
                }
                for predicate in PREDICATES
            }

            conditions = {"no_reset": AMM(), "perkey_reset": AMM()}
            revised_targets: dict[str, dict[str, str]] = {}

            for idx in range(n_entities):
                entity_key = f"entity:{idx:03d}"
                initial_vector, initial_assignments = _entity_composite(store, role_verb, role_object, idx, False)
                revised_vector, revised_assignments = _entity_composite(store, role_verb, role_object, idx, True)
                revised_targets[entity_key] = revised_assignments

                conditions["no_reset"].write(entity_key, initial_vector, initial_assignments)
                conditions["no_reset"].write(entity_key, revised_vector, revised_assignments)

                conditions["perkey_reset"].write(entity_key, initial_vector, initial_assignments)
                conditions["perkey_reset"].delete(entity_key)
                conditions["perkey_reset"].write(entity_key, revised_vector, revised_assignments)

            for condition, memory in conditions.items():
                revised_hits = 0
                retained_hits = 0
                retained_total = 0
                for idx in range(n_entities):
                    entity_key = f"entity:{idx:03d}"
                    stored = memory.get(entity_key)
                    if stored is None:
                        continue
                    revised_hits += int(
                        _decode_object(
                            stored.vector,
                            store,
                            role_verb,
                            role_object,
                            "location",
                            candidate_vectors["location"],
                        )
                        == revised_targets[entity_key]["location"]
                    )
                    for predicate in ("status", "tool"):
                        retained_total += 1
                        retained_hits += int(
                            _decode_object(
                                stored.vector,
                                store,
                                role_verb,
                                role_object,
                                predicate,
                                candidate_vectors[predicate],
                            )
                            == revised_targets[entity_key][predicate]
                        )

                rows.append(
                    {
                        "dim": float(dim),
                        "seed": float(seed),
                        "revised_em": revised_hits / n_entities,
                        "retained_em": retained_hits / max(retained_total, 1),
                        "condition": condition,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[256, 1024, 2048])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--entities", type=int, default=20)
    args = parser.parse_args()

    for row in run(dims=tuple(args.dims), seeds=tuple(args.seeds), n_entities=args.entities):
        print(row)


if __name__ == "__main__":
    main()
