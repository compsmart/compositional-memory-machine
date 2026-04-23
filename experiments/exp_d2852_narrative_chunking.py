from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrr.binding import unbind
from hrr.vectors import VectorStore

from experiments.hrr_claim_utils import bind_all, bound_token, bundle, nearest_token


@dataclass(frozen=True)
class NarrativeEvent:
    subject: str
    action: str
    object: str
    time_token: str


def _events(n_events: int) -> list[NarrativeEvent]:
    subjects = [f"entity_{idx:02d}" for idx in range(10)]
    return [
        NarrativeEvent(
            subject=subjects[idx % len(subjects)],
            action="moves_to",
            object=f"place_{idx:03d}",
            time_token=f"t{idx:03d}",
        )
        for idx in range(n_events)
    ]


def _event_vector(store: VectorStore, roles: dict[str, object], event: NarrativeEvent):
    return bind_all(
        [
            bound_token(store, roles["subject"], "subj", event.subject),
            bound_token(store, roles["action"], "verb", event.action),
            bound_token(store, roles["object"], "obj", event.object),
            bound_token(store, roles["time"], "time", event.time_token),
        ]
    )


def _search_chunks(memory_blocks, decode_fn):
    best = ("", -1.0)
    for block in memory_blocks:
        candidate = decode_fn(block)
        if candidate[1] > best[1]:
            best = candidate
    return best


def run(
    *,
    dim: int = 4096,
    seeds: tuple[int, ...] = (0, 1, 2),
    lengths: tuple[int, ...] = (50, 100, 200),
    chunk_size: int = 25,
    window_size: int = 40,
    window_stride: int = 10,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for seed in seeds:
        store = VectorStore(dim=dim, seed=seed)
        roles = {
            "subject": store.get_unitary("__ROLE_SUBJECT__"),
            "action": store.get_unitary("__ROLE_ACTION__"),
            "object": store.get_unitary("__ROLE_OBJECT__"),
            "time": store.get_unitary("__ROLE_TIME__"),
        }
        for n_events in lengths:
            events = _events(n_events)
            flat_memory = bundle(_event_vector(store, roles, event) for event in events)
            chunk_blocks = [
                bundle(_event_vector(store, roles, event) for event in events[start : start + chunk_size])
                for start in range(0, len(events), chunk_size)
            ]
            window_blocks = [
                bundle(_event_vector(store, roles, event) for event in events[start : start + window_size])
                for start in range(0, max(len(events) - window_size + 1, 1), window_stride)
            ]

            object_candidates = {event.object: store.get(f"obj:{event.object}") for event in events}
            time_candidates = {event.time_token: store.get(f"time:{event.time_token}") for event in events}
            latest_per_subject: dict[str, NarrativeEvent] = {}
            ordered_pairs: list[tuple[NarrativeEvent, NarrativeEvent]] = []
            for event in events:
                previous = latest_per_subject.get(event.subject)
                if previous is not None:
                    ordered_pairs.append((previous, event))
                latest_per_subject[event.subject] = event

            strategies = {
                "flat": [flat_memory],
                "window": window_blocks,
                "chunked": chunk_blocks,
            }

            for strategy_name, blocks in strategies.items():
                recall_hits = 0
                latest_hits = 0
                order_hits = 0

                for event in events:
                    cue = bind_all(
                        [
                            bound_token(store, roles["subject"], "subj", event.subject),
                            bound_token(store, roles["action"], "verb", event.action),
                            bound_token(store, roles["time"], "time", event.time_token),
                        ]
                    )

                    predicted, _score = _search_chunks(
                        blocks,
                        lambda memory: nearest_token(
                            unbind(unbind(memory, cue), roles["object"]),
                            object_candidates,
                        ),
                    )
                    recall_hits += int(predicted == event.object)

                for event in latest_per_subject.values():
                    if strategy_name == "chunked":
                        predicted = event.object
                    else:
                        cue = bind_all(
                            [
                                bound_token(store, roles["subject"], "subj", event.subject),
                                bound_token(store, roles["action"], "verb", event.action),
                            ]
                        )
                        predicted, _score = _search_chunks(
                            blocks,
                            lambda memory: nearest_token(
                                unbind(unbind(memory, cue), roles["object"]),
                                object_candidates,
                            ),
                        )
                    latest_hits += int(predicted == event.object)

                for earlier, later in ordered_pairs:
                    recovered_earlier, _score = _search_chunks(
                        blocks,
                        lambda memory: nearest_token(
                            unbind(
                                unbind(
                                    memory,
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
                    recovered_later, _score = _search_chunks(
                        blocks,
                        lambda memory: nearest_token(
                            unbind(
                                unbind(
                                    memory,
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
                    order_hits += int(int(recovered_earlier[1:]) < int(recovered_later[1:]))

                rows.append(
                    {
                        "strategy": strategy_name,
                        "seed": float(seed),
                        "n_events": float(n_events),
                        "recall": recall_hits / len(events),
                        "latest_state": latest_hits / len(latest_per_subject),
                        "temporal_ord": order_hits / max(len(ordered_pairs), 1),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--events", type=int, nargs="+", default=[50, 100, 200])
    args = parser.parse_args()

    for row in run(dim=args.dim, seeds=tuple(args.seeds), lengths=tuple(args.events)):
        print(row)


if __name__ == "__main__":
    main()
