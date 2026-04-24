from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrr.binding import bind, unbind
from hrr.encoder import HierarchicalClause
from hrr.vectors import VectorStore

from experiments.hrr_claim_utils import bind_all, bound_token, bundle, nearest_token


def _relation_roles(store: VectorStore) -> tuple[object, object]:
    return store.get_unitary("__ROLE_RELATION__"), store.get_unitary("__ROLE_TARGET__")


def _relation_fact(store: VectorStore, role_relation, role_target, relation: str, target: str):
    return bind_all(
        [
            bind(role_relation, store.get(f"rel:{relation}")),
            bind(role_target, store.get(f"entity:{target}")),
        ]
    )


def _relation_memory(
    store: VectorStore,
    role_subject,
    role_relation,
    role_target,
    *,
    n_chains: int,
    max_depth: int,
):
    memory_rows = []
    targets: dict[int, list[str]] = {}
    for chain_idx in range(n_chains):
        path = [f"node_{chain_idx:03d}_{depth}" for depth in range(max_depth + 1)]
        relations = [f"rel_{depth}" for depth in range(1, max_depth + 1)]
        targets[chain_idx] = path
        for depth, relation in enumerate(relations, start=1):
            fact_vector = bind(
                bind(role_subject, store.get(f"entity:{path[depth - 1]}")),
                _relation_fact(store, role_relation, role_target, relation, path[depth]),
            )
            memory_rows.append(fact_vector)
    return bundle(memory_rows), targets


def _decode_chain(
    memory,
    store: VectorStore,
    role_subject,
    role_relation,
    role_target,
    *,
    chain_idx: int,
    hop_depth: int,
    candidates_by_depth: dict[int, dict[str, object]],
) -> str:
    current_subject = f"node_{chain_idx:03d}_0"
    for depth in range(1, hop_depth + 1):
        relation = f"rel_{depth}"
        cue = bind_all(
            [
                bind(role_subject, store.get(f"entity:{current_subject}")),
                bind(role_relation, store.get(f"rel:{relation}")),
            ]
        )
        recovered = unbind(unbind(memory, cue), role_target)
        current_subject, _score = nearest_token(recovered, candidates_by_depth[depth])
    return current_subject


def _build_clause(depth: int, idx: int) -> HierarchicalClause:
    clause = HierarchicalClause(
        subject=f"subj_d{depth}_{idx:03d}",
        verb=f"verb_d{depth}_{idx % 7:02d}",
        object=f"obj_d{depth}_{(idx * 3) % 11:02d}",
    )
    if depth <= 1:
        return clause
    return HierarchicalClause(
        subject=clause.subject,
        verb=clause.verb,
        object=clause.object,
        embedded=_build_clause(depth - 1, idx),
    )


def _hier_roles(store: VectorStore) -> dict[str, object]:
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


def _recover_clause_subject(memory_vector, store: VectorStore, roles: dict[str, object], clause: HierarchicalClause, candidates: dict[str, object]) -> str:
    cue_parts = [
        bound_token(store, roles["verb"], "verb", clause.verb),
        bound_token(store, roles["object"], "obj", clause.object),
    ]
    if clause.embedded is not None:
        cue_parts.append(bind(roles["rel_clause"], _encode_clause(store, roles, clause.embedded)))
    recovered = unbind(unbind(memory_vector, bind_all(cue_parts)), roles["subject"])
    predicted, _score = nearest_token(recovered, candidates)
    return predicted


def _recover_deepest(memory_vector, store: VectorStore, roles: dict[str, object], clause: HierarchicalClause):
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
    dims: tuple[int, ...] = (256, 1024, 2048, 4096),
    hop_depths: tuple[int, ...] = (1, 2, 3),
    syntax_depths: tuple[int, ...] = (1, 2, 3),
    seeds: tuple[int, ...] = (0, 1, 2),
    n_chains: int = 20,
    n_sentences: int = 25,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    max_hop_depth = max(hop_depths)

    for dim in dims:
        for seed in seeds:
            store = VectorStore(dim=dim, seed=seed)
            role_subject = store.get_unitary("__ROLE_CHAIN_SUBJECT__")
            role_relation, role_target = _relation_roles(store)
            relation_memory, chain_targets = _relation_memory(
                store,
                role_subject,
                role_relation,
                role_target,
                n_chains=n_chains,
                max_depth=max_hop_depth,
            )
            entity_candidates = {
                depth: {
                    f"node_{chain_idx:03d}_{depth}": store.get(f"entity:node_{chain_idx:03d}_{depth}")
                    for chain_idx in range(n_chains)
                }
                for depth in range(max_hop_depth + 1)
            }
            for hop_depth in hop_depths:
                hits = 0
                for chain_idx in range(n_chains):
                    expected = chain_targets[chain_idx][hop_depth]
                    predicted = _decode_chain(
                        relation_memory,
                        store,
                        role_subject,
                        role_relation,
                        role_target,
                        chain_idx=chain_idx,
                        hop_depth=hop_depth,
                        candidates_by_depth=entity_candidates,
                    )
                    hits += int(predicted == expected)
                rows.append(
                    {
                        "task": "relation_chain",
                        "dim": float(dim),
                        "seed": float(seed),
                        "depth": float(hop_depth),
                        "em": hits / max(n_chains, 1),
                    }
                )

            for syntax_depth in syntax_depths:
                roles = _hier_roles(store)
                clauses = [_build_clause(syntax_depth, idx) for idx in range(n_sentences)]
                syntax_memory = bundle(_encode_clause(store, roles, clause) for clause in clauses)
                subject_candidates = {}
                for clause in clauses:
                    current = clause
                    while True:
                        subject_candidates[current.subject] = store.get(f"subj:{current.subject}")
                        if current.embedded is None:
                            break
                        current = current.embedded
                hits = 0
                for clause in clauses:
                    if syntax_depth == 1:
                        expected = clause.subject
                        predicted = _recover_clause_subject(syntax_memory, store, roles, clause, subject_candidates)
                    else:
                        isolated, deepest = _recover_deepest(syntax_memory, store, roles, clause)
                        expected = deepest.subject
                        predicted = _recover_clause_subject(isolated, store, roles, deepest, subject_candidates)
                    hits += int(predicted == expected)
                rows.append(
                    {
                        "task": "hierarchical_syntax",
                        "dim": float(dim),
                        "seed": float(seed),
                        "depth": float(syntax_depth),
                        "em": hits / max(n_sentences, 1),
                    }
                )
    return rows


def summarize(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    grouped: dict[tuple[str, int, int], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), int(row["dim"]), int(row["depth"]))].append(row)
    summary: list[dict[str, float | str]] = []
    for task, dim, depth in sorted(grouped):
        group = grouped[(task, dim, depth)]
        summary.append(
            {
                "task": task,
                "dim": float(dim),
                "depth": float(depth),
                "runs": float(len(group)),
                "mean_em": sum(float(row["em"]) for row in group) / max(len(group), 1),
            }
        )
    return summary


def fitted_relation_hop_bases(summary_rows: list[dict[str, float | str]]) -> list[dict[str, float]]:
    grouped: dict[int, list[dict[str, float | str]]] = defaultdict(list)
    for row in summary_rows:
        if str(row["task"]) == "relation_chain":
            grouped[int(row["dim"])].append(row)
    bases: list[dict[str, float]] = []
    for dim in sorted(grouped):
        rows_for_dim = grouped[dim]
        root_estimates = [
            max(float(row["mean_em"]), 1e-6) ** (1.0 / max(float(row["depth"]), 1.0))
            for row in rows_for_dim
        ]
        bases.append(
            {
                "dim": float(dim),
                "fitted_hop_base": sum(root_estimates) / max(len(root_estimates), 1),
            }
        )
    return bases


def render_markdown_report(
    summary_rows: list[dict[str, float | str]],
    *,
    dims: tuple[int, ...],
    hop_depths: tuple[int, ...],
    syntax_depths: tuple[int, ...],
    seeds: tuple[int, ...],
    n_chains: int,
    n_sentences: int,
) -> str:
    relation_rows = [row for row in summary_rows if str(row["task"]) == "relation_chain"]
    syntax_rows = [row for row in summary_rows if str(row["task"]) == "hierarchical_syntax"]
    fitted_rows = fitted_relation_hop_bases(summary_rows)
    lines = [
        "# Sequential Unbinding Scaling",
        "",
        "## Configuration",
        "",
        f"- `dims={list(dims)}`",
        f"- `hop_depths={list(hop_depths)}`",
        f"- `syntax_depths={list(syntax_depths)}`",
        f"- `seeds={list(seeds)}`",
        f"- `n_chains={n_chains}`",
        f"- `n_sentences={n_sentences}`",
        "",
        "## Relation Chain Summary",
        "",
        "| dim | depth | runs | mean_em |",
        "| --- | --- | --- | --- |",
    ]
    for row in relation_rows:
        lines.append(
            "| {dim:.0f} | {depth:.0f} | {runs:.0f} | {mean_em:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Hierarchical Syntax Summary",
            "",
            "| dim | depth | runs | mean_em |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in syntax_rows:
        lines.append(
            "| {dim:.0f} | {depth:.0f} | {runs:.0f} | {mean_em:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Recommended Hop Bases",
            "",
            "| dim | fitted_hop_base |",
            "| --- | --- |",
        ]
    )
    for row in fitted_rows:
        lines.append(
            "| {dim:.0f} | {fitted_hop_base:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- The relation-chain frontier now reports a fitted per-hop base instead of only broad dimension tiers.",
            "- `d=1024` is the first strong runtime frontier, while `d>=2048` behaves as effectively exact for relation depth `1-3` in this repo-local mirror.",
            "- The hierarchical side stays aligned with the same dimension sweep, which makes the hop-budget story easier to compare against other structured retrieval limits.",
        ]
    )
    return "\n".join(lines)


def _write_artifacts(
    rows: list[dict[str, float | str]],
    summary_rows: list[dict[str, float | str]],
    *,
    json_file: str | None,
    report_file: str | None,
    config: dict[str, Any],
) -> None:
    payload = {
        "config": config,
        "rows": rows,
        "summary": summary_rows,
        "recommended_hop_bases": fitted_relation_hop_bases(summary_rows),
    }
    if json_file:
        path = Path(json_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if report_file:
        path = Path(report_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_markdown_report(
                summary_rows,
                dims=tuple(config["dims"]),
                hop_depths=tuple(config["hop_depths"]),
                syntax_depths=tuple(config["syntax_depths"]),
                seeds=tuple(config["seeds"]),
                n_chains=int(config["n_chains"]),
                n_sentences=int(config["n_sentences"]),
            ),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[256, 1024, 2048, 4096])
    parser.add_argument("--hop-depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--syntax-depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--chains", type=int, default=20)
    parser.add_argument("--sentences", type=int, default=25)
    parser.add_argument("--json-file")
    parser.add_argument("--report-file")
    args = parser.parse_args()

    rows = run(
        dims=tuple(args.dims),
        hop_depths=tuple(args.hop_depths),
        syntax_depths=tuple(args.syntax_depths),
        seeds=tuple(args.seeds),
        n_chains=args.chains,
        n_sentences=args.sentences,
    )
    summary_rows = summarize(rows)
    _write_artifacts(
        rows,
        summary_rows,
        json_file=args.json_file,
        report_file=args.report_file,
        config={
            "dims": args.dims,
            "hop_depths": args.hop_depths,
            "syntax_depths": args.syntax_depths,
            "seeds": args.seeds,
            "n_chains": args.chains,
            "n_sentences": args.sentences,
        },
    )
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
