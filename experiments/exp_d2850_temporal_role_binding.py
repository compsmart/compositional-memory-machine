from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrr.binding import unbind
from hrr.vectors import VectorStore

from experiments.hrr_claim_utils import bind_all, bound_token, bundle, nearest_token


@dataclass(frozen=True)
class Event:
    subject: str
    action: str
    object: str
    time_token: str


def _events(n_events: int) -> list[Event]:
    subjects = [f"entity_{idx:02d}" for idx in range(5)]
    return [
        Event(
            subject=subjects[idx % len(subjects)],
            action="tracks",
            object=f"state_{idx:03d}",
            time_token=f"t{idx:03d}",
        )
        for idx in range(n_events)
    ]


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


def run(
    *,
    dim: int = 4096,
    seeds: tuple[int, ...] = (0, 1, 2),
    n_events_values: tuple[int, ...] = (10, 25, 50, 100, 200),
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for seed in seeds:
        store = VectorStore(dim=dim, seed=seed)
        roles = {
            "subject": store.get_unitary("__ROLE_SUBJECT__"),
            "action": store.get_unitary("__ROLE_ACTION__"),
            "object": store.get_unitary("__ROLE_OBJECT__"),
            "time": store.get_unitary("__ROLE_TIME__"),
        }
        for n_events in n_events_values:
            events = _events(n_events)
            memory = bundle(_event_vector(store, roles, event) for event in events)
            object_candidates = {event.object: store.get(f"obj:{event.object}") for event in events}
            time_candidates = {event.time_token: store.get(f"time:{event.time_token}") for event in events}

            role_hits = 0
            latest_hits = 0
            temporal_hits = 0
            latest_total = 0
            temporal_total = 0

            latest_per_subject: dict[str, Event] = {}
            ordered_pairs: list[tuple[Event, Event]] = []

            for event in events:
                cue = bind_all(
                    [
                        bound_token(store, roles["subject"], "subj", event.subject),
                        bound_token(store, roles["action"], "verb", event.action),
                        bound_token(store, roles["time"], "time", event.time_token),
                    ]
                )
                recovered = unbind(unbind(memory, cue), roles["object"])
                predicted_object, _score = nearest_token(recovered, object_candidates)
                role_hits += int(predicted_object == event.object)

                previous = latest_per_subject.get(event.subject)
                if previous is not None:
                    ordered_pairs.append((previous, event))
                latest_per_subject[event.subject] = event

            for event in latest_per_subject.values():
                latest_total += 1
                cue = bind_all(
                    [
                        bound_token(store, roles["subject"], "subj", event.subject),
                        bound_token(store, roles["action"], "verb", event.action),
                    ]
                )
                recovered = unbind(unbind(memory, cue), roles["object"])
                predicted_object, _score = nearest_token(recovered, object_candidates)
                latest_hits += int(predicted_object == event.object)

            for earlier, later in ordered_pairs:
                temporal_total += 1
                earlier_time = unbind(
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
                )
                later_time = unbind(
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
                )
                earlier_pred, _score = nearest_token(earlier_time, time_candidates)
                later_pred, _score = nearest_token(later_time, time_candidates)
                temporal_hits += int(_time_index(earlier_pred) < _time_index(later_pred))

            rows.append(
                {
                    "dim": float(dim),
                    "seed": float(seed),
                    "n_events": float(n_events),
                    "role_acc": role_hits / len(events),
                    "latest_state": latest_hits / max(latest_total, 1),
                    "temporal_ord": temporal_hits / max(temporal_total, 1),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--events", type=int, nargs="+", default=[10, 25, 50, 100, 200])
    args = parser.parse_args()

    for row in run(dim=args.dim, seeds=tuple(args.seeds), n_events_values=tuple(args.events)):
        print(row)


if __name__ == "__main__":
    main()
