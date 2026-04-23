from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrr.binding import bind, unbind
from hrr.encoder import HierarchicalClause
from hrr.vectors import VectorStore

from experiments.hrr_claim_utils import bind_all, bound_token, bundle, nearest_token


def _build_clause(depth: int, idx: int) -> HierarchicalClause:
    vocab = 12 if depth == 1 else 6
    clause = HierarchicalClause(
        subject=f"subj_d{depth}_{idx:03d}",
        verb=f"verb_d{depth}_{idx % vocab:02d}",
        object=f"obj_d{depth}_{(idx * 3) % vocab:02d}",
    )
    if depth <= 1:
        return clause
    return HierarchicalClause(
        subject=f"subj_d{depth}_{idx:03d}",
        verb=f"verb_d{depth}_{idx % vocab:02d}",
        object=f"obj_d{depth}_{(idx * 3) % vocab:02d}",
        embedded=_build_clause(depth - 1, idx),
    )


def _roles(store: VectorStore) -> dict[str, object]:
    return {
        "subject": store.get_unitary("__ROLE_SUBJECT__"),
        "verb": store.get_unitary("__ROLE_VERB__"),
        "object": store.get_unitary("__ROLE_OBJECT__"),
        "rel_clause": store.get_unitary("__ROLE_REL_CLAUSE__"),
    }


def _encode_clause(store: VectorStore, roles: dict[str, object], clause: HierarchicalClause):
    parts = [
        bound_token(store, roles["subject"], "subj", clause.subject),
        bound_token(store, roles["verb"], "verb", clause.verb),
        bound_token(store, roles["object"], "obj", clause.object),
    ]
    if clause.embedded is not None:
        parts.append(bind(roles["rel_clause"], _encode_clause(store, roles, clause.embedded)))
    return bind_all(parts)


def _recover_subject(memory_vector, store: VectorStore, roles: dict[str, object], clause: HierarchicalClause, candidates: dict[str, object]) -> str:
    cue_parts = [
        bound_token(store, roles["verb"], "verb", clause.verb),
        bound_token(store, roles["object"], "obj", clause.object),
    ]
    if clause.embedded is not None:
        cue_parts.append(bind(roles["rel_clause"], _encode_clause(store, roles, clause.embedded)))
    recovered = unbind(unbind(memory_vector, bind_all(cue_parts)), roles["subject"])
    predicted, _score = nearest_token(recovered, candidates)
    return predicted


def _recover_embedded(memory_vector, store: VectorStore, roles: dict[str, object], clause: HierarchicalClause):
    isolated = memory_vector
    current = clause
    while current.embedded is not None:
        outer_cue = bind_all(
            [
                bound_token(store, roles["subject"], "subj", current.subject),
                bound_token(store, roles["verb"], "verb", current.verb),
                bound_token(store, roles["object"], "obj", current.object),
            ]
        )
        isolated = unbind(unbind(isolated, outer_cue), roles["rel_clause"])
        current = current.embedded
    return isolated, current


def run(
    *,
    dim: int = 4096,
    seeds: tuple[int, ...] = (0, 1, 2),
    depths: tuple[int, ...] = (1, 2, 3),
    sentence_counts: tuple[int, ...] = (5, 10, 25),
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for seed in seeds:
        store = VectorStore(dim=dim, seed=seed)
        roles = _roles(store)
        for depth in depths:
            for n_sentences in sentence_counts:
                clauses = [_build_clause(depth, idx) for idx in range(n_sentences)]
                memory = bundle(_encode_clause(store, roles, clause) for clause in clauses)
                candidates = {}
                for clause in clauses:
                    cursor = clause
                    while True:
                        candidates[cursor.subject] = store.get(f"subj:{cursor.subject}")
                        if cursor.embedded is None:
                            break
                        cursor = cursor.embedded

                main_hits = 0
                embedded_hits = 0
                embedded_total = 0
                for clause in clauses:
                    predicted_main = _recover_subject(memory, store, roles, clause, candidates)
                    main_hits += int(predicted_main == clause.subject)
                    if clause.embedded is not None:
                        embedded_total += 1
                        embedded_vector, embedded_clause = _recover_embedded(memory, store, roles, clause)
                        predicted_embedded = _recover_subject(embedded_vector, store, roles, embedded_clause, candidates)
                        embedded_hits += int(predicted_embedded == embedded_clause.subject)

                rows.append(
                    {
                        "dim": float(dim),
                        "seed": float(seed),
                        "depth": float(depth),
                        "n_sentences": float(n_sentences),
                        "main_acc": main_hits / len(clauses),
                        "embedded_acc": embedded_hits / embedded_total if embedded_total else 1.0,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sentences", type=int, nargs="+", default=[5, 10, 25])
    args = parser.parse_args()

    for row in run(
        dim=args.dim,
        seeds=tuple(args.seeds),
        depths=tuple(args.depths),
        sentence_counts=tuple(args.sentences),
    ):
        print(row)


if __name__ == "__main__":
    main()
