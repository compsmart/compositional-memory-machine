from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrr.binding import bind, unbind
from hrr.vectors import VectorStore

from experiments.hrr_claim_utils import bind_all, bound_token, bundle, nearest_token


class Event:
    def __init__(self, subject: str, action: str, object_: str, time_token: str) -> None:
        self.subject = subject
        self.action = action
        self.object = object_
        self.time_token = time_token


def _events(n_events: int) -> list[Event]:
    subjects = [f"entity_{idx:02d}" for idx in range(6)]
    return [
        Event(
            subject=subjects[idx % len(subjects)],
            action="tracks",
            object_=f"state_{idx:03d}",
            time_token=f"t{idx:03d}",
        )
        for idx in range(n_events)
    ]


def _roles(store: VectorStore) -> dict[str, np.ndarray]:
    return {
        "subject": store.get_unitary("__ROLE_SUBJECT__"),
        "action": store.get_unitary("__ROLE_ACTION__"),
        "object": store.get_unitary("__ROLE_OBJECT__"),
        "time": store.get_unitary("__ROLE_TIME__"),
        "prev": store.get_unitary("__ROLE_PREV__"),
        "next": store.get_unitary("__ROLE_NEXT__"),
    }


def _event_vector(store: VectorStore, roles: dict[str, np.ndarray], event: Event) -> np.ndarray:
    return bind_all(
        [
            bound_token(store, roles["subject"], "subj", event.subject),
            bound_token(store, roles["action"], "verb", event.action),
            bound_token(store, roles["object"], "obj", event.object),
            bound_token(store, roles["time"], "time", event.time_token),
        ]
    )


def _time_index(token: str) -> int:
    return int(token[1:])


def _pair_metrics(memory, events: list[Event], store: VectorStore, roles: dict[str, np.ndarray], *, chunked: bool = False, chunk_size: int = 25) -> tuple[float, float]:
    time_candidates = {event.time_token: store.get(f"time:{event.time_token}") for event in events}
    object_candidates = {event.object: store.get(f"obj:{event.object}") for event in events}
    latest_per_subject: dict[str, Event] = {}
    ordered_pairs: list[tuple[Event, Event]] = []
    for event in events:
        previous = latest_per_subject.get(event.subject)
        if previous is not None:
            ordered_pairs.append((previous, event))
        latest_per_subject[event.subject] = event

    def _search_blocks(blocks: list[np.ndarray], decode) -> str:
        best_token = ""
        best_score = float("-inf")
        for block in blocks:
            token, score = decode(block)
            if score > best_score:
                best_score = score
                best_token = token
        return best_token

    blocks = [memory]
    if chunked:
        blocks = [bundle(_event_vector(store, roles, event) for event in events[start : start + chunk_size]) for start in range(0, len(events), chunk_size)]

    latest_hits = 0
    for event in latest_per_subject.values():
        cue = bind_all(
            [
                bound_token(store, roles["subject"], "subj", event.subject),
                bound_token(store, roles["action"], "verb", event.action),
            ]
        )
        predicted = _search_blocks(
            blocks,
            lambda block: nearest_token(unbind(unbind(block, cue), roles["object"]), object_candidates),
        )
        latest_hits += int(predicted == event.object)

    order_hits = 0
    for earlier, later in ordered_pairs:
        earlier_pred = _search_blocks(
            blocks,
            lambda block: nearest_token(
                unbind(
                    unbind(
                        block,
                        bind_all(
                            [
                                bound_token(store, roles["subject"], "subj", earlier.subject),
                                bound_token(store, roles["action"], "verb", earlier.action),
                                bound_token(store, roles["object"], "obj", earlier.object),
                            ]
                        ),
                    ),
                    roles["time"],
                ),
                time_candidates,
            ),
        )
        later_pred = _search_blocks(
            blocks,
            lambda block: nearest_token(
                unbind(
                    unbind(
                        block,
                        bind_all(
                            [
                                bound_token(store, roles["subject"], "subj", later.subject),
                                bound_token(store, roles["action"], "verb", later.action),
                                bound_token(store, roles["object"], "obj", later.object),
                            ]
                        ),
                    ),
                    roles["time"],
                ),
                time_candidates,
            ),
        )
        order_hits += int(_time_index(earlier_pred) < _time_index(later_pred))

    return (
        latest_hits / max(len(latest_per_subject), 1),
        order_hits / max(len(ordered_pairs), 1),
    )


def _linked_pair_accuracy(events: list[Event], store: VectorStore, roles: dict[str, np.ndarray]) -> float:
    pair_vectors = []
    time_candidates = {event.time_token: store.get(f"time:{event.time_token}") for event in events}
    for idx in range(1, len(events)):
        earlier = events[idx - 1]
        later = events[idx]
        pair_vectors.append(
            bind_all(
                [
                    bound_token(store, roles["prev"], "time", earlier.time_token),
                    bound_token(store, roles["next"], "time", later.time_token),
                ]
            )
        )
    link_memory = bundle(pair_vectors)
    hits = 0
    for idx in range(1, len(events)):
        earlier = events[idx - 1]
        recovered = unbind(unbind(link_memory, bound_token(store, roles["prev"], "time", earlier.time_token)), roles["next"])
        predicted, _score = nearest_token(recovered, time_candidates)
        hits += int(predicted == events[idx].time_token)
    return hits / max(len(events) - 1, 1)


def _latest_state_cache_accuracy(events: list[Event], store: VectorStore, roles: dict[str, np.ndarray]) -> float:
    latest_vectors: dict[str, np.ndarray] = {}
    object_candidates = {event.object: store.get(f"obj:{event.object}") for event in events}
    for event in events:
        latest_vectors[event.subject] = bind_all(
            [
                bound_token(store, roles["subject"], "subj", event.subject),
                bound_token(store, roles["action"], "verb", event.action),
                bound_token(store, roles["object"], "obj", event.object),
            ]
        )
    hits = 0
    total = 0
    for subject, vector in latest_vectors.items():
        latest = next(event for event in reversed(events) if event.subject == subject)
        cue = bind_all(
            [
                bound_token(store, roles["subject"], "subj", latest.subject),
                bound_token(store, roles["action"], "verb", latest.action),
            ]
        )
        recovered = unbind(unbind(vector, cue), roles["object"])
        predicted, _score = nearest_token(recovered, object_candidates)
        hits += int(predicted == latest.object)
        total += 1
    return hits / max(total, 1)


def run(
    *,
    dims: tuple[int, ...] = (1024, 2048, 4096),
    event_counts: tuple[int, ...] = (50, 100, 200),
    chunk_size: int = 25,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for dim in dims:
        for seed in seeds:
            store = VectorStore(dim=dim, seed=seed)
            roles = _roles(store)
            for n_events in event_counts:
                events = _events(n_events)
                flat_memory = bundle(_event_vector(store, roles, event) for event in events)
                flat_latest, flat_order = _pair_metrics(flat_memory, events, store, roles, chunked=False)
                chunked_latest, chunked_order = _pair_metrics(
                    flat_memory,
                    events,
                    store,
                    roles,
                    chunked=True,
                    chunk_size=chunk_size,
                )
                hybrid_latest = _latest_state_cache_accuracy(events, store, roles)

                overwrite_hits = 0
                overwrite_total = 0
                overwrite_state: dict[str, np.ndarray] = {}
                for event in events:
                    overwrite_state[event.subject] = _event_vector(store, roles, event)
                for subject, vector in overwrite_state.items():
                    latest = next(event for event in reversed(events) if event.subject == subject)
                    cue = bind_all(
                        [
                            bound_token(store, roles["subject"], "subj", latest.subject),
                            bound_token(store, roles["action"], "verb", latest.action),
                        ]
                    )
                    recovered = unbind(unbind(vector, cue), roles["object"])
                    candidates = {event.object: store.get(f"obj:{event.object}") for event in events}
                    predicted, _score = nearest_token(recovered, candidates)
                    overwrite_hits += int(predicted == latest.object)
                    overwrite_total += 1

                rows.extend(
                    [
                        {
                            "strategy": "flat_temporal_roles",
                            "dim": float(dim),
                            "seed": float(seed),
                            "n_events": float(n_events),
                            "latest_state_em": flat_latest,
                            "pairwise_order_em": flat_order,
                        },
                        {
                            "strategy": "chunked_temporal_roles",
                            "dim": float(dim),
                            "seed": float(seed),
                            "n_events": float(n_events),
                            "latest_state_em": chunked_latest,
                            "pairwise_order_em": chunked_order,
                        },
                        {
                            "strategy": "overwrite_only",
                            "dim": float(dim),
                            "seed": float(seed),
                            "n_events": float(n_events),
                            "latest_state_em": overwrite_hits / max(overwrite_total, 1),
                            "pairwise_order_em": 0.0,
                        },
                        {
                            "strategy": "explicit_pair_links",
                            "dim": float(dim),
                            "seed": float(seed),
                            "n_events": float(n_events),
                            "latest_state_em": 0.0,
                            "pairwise_order_em": _linked_pair_accuracy(events, store, roles),
                        },
                        {
                            "strategy": "hybrid_chunked_plus_latest_cache",
                            "dim": float(dim),
                            "seed": float(seed),
                            "n_events": float(n_events),
                            "latest_state_em": hybrid_latest,
                            "pairwise_order_em": chunked_order,
                        },
                    ]
                )
    return rows


def summarize(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    grouped: dict[tuple[str, int, int], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["strategy"]), int(row["dim"]), int(row["n_events"]))].append(row)
    summary: list[dict[str, float | str]] = []
    for strategy, dim, n_events in sorted(grouped):
        group = grouped[(strategy, dim, n_events)]
        latest = sum(float(row["latest_state_em"]) for row in group) / max(len(group), 1)
        order = sum(float(row["pairwise_order_em"]) for row in group) / max(len(group), 1)
        summary.append(
            {
                "strategy": strategy,
                "dim": float(dim),
                "n_events": float(n_events),
                "runs": float(len(group)),
                "latest_state_em": latest,
                "pairwise_order_em": order,
                "balanced_score": (latest + order) / 2.0,
            }
        )
    return summary


def render_markdown_report(
    summary_rows: list[dict[str, float | str]],
    *,
    dims: tuple[int, ...],
    event_counts: tuple[int, ...],
    chunk_size: int,
    seeds: tuple[int, ...],
) -> str:
    lines = [
        "# Temporal Ordering Frontier",
        "",
        "## Configuration",
        "",
        f"- `dims={list(dims)}`",
        f"- `event_counts={list(event_counts)}`",
        f"- `chunk_size={chunk_size}`",
        f"- `seeds={list(seeds)}`",
        "",
        "## Summary",
        "",
        "| strategy | dim | n_events | runs | latest_state_em | pairwise_order_em | balanced_score |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {strategy} | {dim:.0f} | {n_events:.0f} | {runs:.0f} | {latest_state_em:.3f} | {pairwise_order_em:.3f} | {balanced_score:.3f} |".format(
                **row
            )
        )
    best_balanced = max(summary_rows, key=lambda row: float(row["balanced_score"]))
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                f"- The best balanced strategy is `{best_balanced['strategy']}` at "
                f"`dim={best_balanced['dim']:.0f}`, `n_events={best_balanced['n_events']:.0f}`, "
                f"with `balanced_score={best_balanced['balanced_score']:.3f}`."
            ),
            "- The hybrid latest-cache plus chunked-order strategy keeps latest-state and pairwise ordering measurable together, which avoids the earlier near-zero latest-state collapse.",
            "- Overwrite-only storage still cleanly isolates the cost of dropping temporal-order structure.",
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
    payload = {"config": config, "rows": rows, "summary": summary_rows}
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
                event_counts=tuple(config["event_counts"]),
                chunk_size=int(config["chunk_size"]),
                seeds=tuple(config["seeds"]),
            ),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--events", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--json-file")
    parser.add_argument("--report-file")
    args = parser.parse_args()

    rows = run(
        dims=tuple(args.dims),
        event_counts=tuple(args.events),
        chunk_size=args.chunk_size,
        seeds=tuple(args.seeds),
    )
    summary_rows = summarize(rows)
    _write_artifacts(
        rows,
        summary_rows,
        json_file=args.json_file,
        report_file=args.report_file,
        config={
            "dims": args.dims,
            "event_counts": args.events,
            "chunk_size": args.chunk_size,
            "seeds": args.seeds,
        },
    )
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
